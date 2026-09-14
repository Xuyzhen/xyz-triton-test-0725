# vLLM-Ascend patched kernel: _post_update_kernel from
# vllm-ascend/vllm_ascend/worker/v2/input_batch.py:112
# PATCH NOTE: This is an Ascend NPU adaptation of the original vLLM Triton kernel

"""
Precision test for patched _post_update_kernel (Ascend NPU version).

Patch differences vs original vllm:
- Uses load-balanced grid based on get_vectorcore_num() instead of (num_rows,)
- Uses rows_per_program distribution for better load balancing
- Uses tl.range for loop over num_sampled tokens
- Uses tl.load/tl.store for increment pattern instead of atomic_add
- Uses tl.minimum for bounds clamping

Kernel signature:
    _post_update_kernel(
        idx_mapping_ptr,                # [num_reqs x 1] request idx mapping (strided)
        idx_mapping_stride,             # stride(0) of idx_mapping
        num_computed_tokens_ptr,        # [max_num_reqs] num computed tokens
        last_sampled_tokens_ptr,        # [max_num_reqs] last sampled token ID
        output_bin_counts_ptr,          # [max_num_reqs, vocab_size] output bin counts
        output_bin_counts_stride,       # stride(0) of output_bin_counts
        sampled_tokens_ptr,             # [num_reqs, num_spec_steps+1] sampled tokens
        sampled_tokens_stride,          # stride(0) of sampled_tokens
        num_rows,                       # scalar: number of rows
        num_sampled_ptr,                # [num_reqs] number sampled per request
        num_rejected_ptr,               # [num_reqs] number rejected per request
        query_start_loc_ptr,            # [num_reqs + 1] query start locations
        all_token_ids_ptr,              # [max_num_reqs, max_model_len] all token IDs
        all_token_ids_stride,           # stride(0) of all_token_ids
        total_len_ptr,                  # [max_num_reqs] total length per request
    )

Post-update state after speculative decoding step:
- Updates last_sampled_tokens and total_len
- Appends sampled tokens to all_token_ids
- Updates output_bin_counts with new tokens
- Updates num_computed_tokens based on accepted tokens
"""

import torch

from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

import pytest


def _post_update_ref(
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    output_bin_counts: torch.Tensor,
    sampled_tokens: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
):
    """CPU reference for _post_update_kernel."""
    num_rows = idx_mapping.shape[0]

    for row_idx in range(num_rows):
        req_state_idx = idx_mapping[row_idx].item()
        total_len_ = total_len[req_state_idx].item()
        ns = num_sampled[row_idx].item()

        if ns > 0:
            token_id = sampled_tokens[row_idx, ns - 1].item()
            last_sampled_tokens[req_state_idx] = token_id
            total_len[req_state_idx] = total_len_ + ns

        for i in range(ns):
            token_id = sampled_tokens[row_idx, i].item()
            output_bin_counts[req_state_idx, token_id] += 1
            all_token_ids[req_state_idx, total_len_ + i] = token_id

        query_start = query_start_loc[row_idx].item()
        query_end = query_start_loc[row_idx + 1].item()
        query_len = query_end - query_start
        nrej = num_rejected[row_idx].item()
        num_computed_tokens[req_state_idx] += query_len - nrej


class TestPostUpdateKernelPatch:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    def _run_kernel(
        self,
        idx_mapping,
        num_computed_tokens,
        last_sampled_tokens,
        output_bin_counts,
        sampled_tokens,
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        total_len,
    ):
        from vllm_ascend.worker.v2.input_batch import _post_update_kernel

        num_rows = idx_mapping.shape[0]
        core_num = get_vectorcore_num()
        grid = (min(num_rows, core_num),)

        _post_update_kernel[grid](
            idx_mapping,
            idx_mapping.stride(0),
            num_computed_tokens,
            last_sampled_tokens,
            output_bin_counts,
            output_bin_counts.stride(0),
            sampled_tokens,
            sampled_tokens.stride(0),
            num_rows,
            num_sampled,
            num_rejected,
            query_start_loc,
            all_token_ids,
            all_token_ids.stride(0),
            total_len,
        )
        torch.npu.synchronize()

    def test_basic_update(self):
        """Verify basic post-update: append sampled tokens, update counts."""
        num_reqs = 2
        max_num_reqs = 2
        vocab_size = 64
        max_model_len = 32
        num_spec_steps = 3

        idx_mapping = torch.tensor([[0], [1]], dtype=torch.int32, device=self.device)
        num_computed_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=self.device)
        last_sampled_tokens = torch.full((max_num_reqs,), -1, dtype=torch.int32, device=self.device)
        output_bin_counts = torch.zeros(max_num_reqs, vocab_size, dtype=torch.int32, device=self.device)
        total_len = torch.tensor([5, 3], dtype=torch.int32, device=self.device)

        # Each req sampled 2 tokens
        sampled_tokens = torch.tensor([
            [10, 20, 0],
            [30, 40, 0],
        ], dtype=torch.int32, device=self.device)
        num_sampled = torch.tensor([2, 2], dtype=torch.int32, device=self.device)
        num_rejected = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
        query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32, device=self.device)

        all_token_ids = torch.full((max_num_reqs, max_model_len), -1, dtype=torch.int32, device=self.device)

        ref_args = (
            idx_mapping.cpu().clone(),
            num_computed_tokens.cpu().clone(),
            last_sampled_tokens.cpu().clone(),
            output_bin_counts.cpu().clone(),
            sampled_tokens.cpu().clone(),
            num_sampled.cpu().clone(),
            num_rejected.cpu().clone(),
            query_start_loc.cpu().clone(),
            all_token_ids.cpu().clone(),
            total_len.cpu().clone(),
        )
        expected = list(ref_args)
        _post_update_ref(*expected)

        self._run_kernel(
            idx_mapping, num_computed_tokens, last_sampled_tokens,
            output_bin_counts, sampled_tokens, num_sampled, num_rejected,
            query_start_loc, all_token_ids, total_len,
        )

        torch.testing.assert_close(last_sampled_tokens.cpu(), expected[2], rtol=0, atol=0)
        torch.testing.assert_close(total_len.cpu(), expected[9], rtol=0, atol=0)
        torch.testing.assert_close(output_bin_counts.cpu(), expected[4], rtol=0, atol=0)
        torch.testing.assert_close(num_computed_tokens.cpu(), expected[1], rtol=0, atol=0)

    def test_no_sampled_tokens(self):
        """When no tokens are sampled, only num_computed_tokens should change."""
        num_reqs = 1
        max_num_reqs = 1
        vocab_size = 32
        max_model_len = 16

        idx_mapping = torch.zeros((num_reqs, 1), dtype=torch.int32, device=self.device)
        num_computed_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=self.device)
        last_sampled_tokens = torch.full((max_num_reqs,), -1, dtype=torch.int32, device=self.device)
        output_bin_counts = torch.zeros(max_num_reqs, vocab_size, dtype=torch.int32, device=self.device)
        total_len = torch.tensor([5], dtype=torch.int32, device=self.device)

        sampled_tokens = torch.zeros((num_reqs, 3), dtype=torch.int32, device=self.device)
        num_sampled = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        num_rejected = torch.tensor([1], dtype=torch.int32, device=self.device)
        query_start_loc = torch.tensor([0, 2], dtype=torch.int32, device=self.device)

        all_token_ids = torch.full((max_num_reqs, max_model_len), -1, dtype=torch.int32, device=self.device)

        nct_before = num_computed_tokens.clone()

        self._run_kernel(
            idx_mapping, num_computed_tokens, last_sampled_tokens,
            output_bin_counts, sampled_tokens, num_sampled, num_rejected,
            query_start_loc, all_token_ids, total_len,
        )

        # No sampled: last_sampled unchanged, total_len unchanged, bin_counts unchanged
        assert last_sampled_tokens[0].item() == -1, "last_sampled should stay -1"
        assert total_len[0].item() == 5, "total_len should stay 5"
        # num_computed should increase by query_len - rejected = 2 - 1 = 1
        assert num_computed_tokens[0].item() == nct_before[0].item() + 1, \
            f"Expected computed={nct_before[0].item() + 1}, got {num_computed_tokens[0].item()}"
