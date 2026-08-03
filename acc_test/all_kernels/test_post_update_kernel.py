# vLLM vanilla kernel: _post_update_kernel from
# vllm/vllm/v1/worker/gpu/input_batch.py

"""
Precision test for _post_update_kernel (vanilla vLLM version).

Post-update state after a speculative decoding step:
- Updates last_sampled_tokens and total_len
- Appends sampled tokens to all_token_ids
- Updates output_bin_counts with new tokens (when not None)
- Updates num_computed_tokens based on accepted tokens

Kernel signature:
    _post_update_kernel(
        idx_mapping_ptr,               # [num_reqs] int32
        num_computed_tokens_ptr,       # [max_num_reqs] int32
        last_sampled_tokens_ptr,       # [max_num_reqs] int32
        output_bin_counts_ptr,         # [max_num_reqs, vocab_size] int32 or None
        output_bin_counts_stride,
        sampled_tokens_ptr,            # [num_reqs, num_spec_steps+1] int32
        sampled_tokens_stride,
        num_sampled_ptr,               # [num_reqs] int32
        num_rejected_ptr,              # [num_reqs] int32
        query_start_loc_ptr,           # [num_reqs + 1] int32 or None
        all_token_ids_ptr,            # [max_num_reqs, max_model_len] int32
        all_token_ids_stride,
        total_len_ptr,                 # [max_num_reqs] int32
    )
"""

import torch
import pytest

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import _post_update_kernel
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _post_update_ref(
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
    """CPU reference for _post_update_kernel."""
    out_nct = num_computed_tokens.clone()
    out_lst = last_sampled_tokens.clone()
    out_tl = total_len.clone()
    out_abc = all_token_ids.clone()
    out_binc = output_bin_counts.clone() if output_bin_counts is not None else None

    num_reqs = idx_mapping.shape[0]
    for req_id in range(num_reqs):
        req_state_idx = int(idx_mapping[req_id].item())
        if req_state_idx < 0:
            continue

        total_len_ = int(out_tl[req_state_idx].item())
        n_sampled = int(num_sampled[req_id].item())

        if n_sampled > 0:
            token_id = int(sampled_tokens[req_id, n_sampled - 1].item())
            out_lst[req_state_idx] = token_id
            out_tl[req_state_idx] = total_len_ + n_sampled

        for i in range(n_sampled):
            token_id = int(sampled_tokens[req_id, i].item())
            out_abc[req_state_idx, total_len_ + i] = token_id
            if out_binc is not None:
                out_binc[req_state_idx, token_id] += 1

        if query_start_loc is not None:
            qs = int(query_start_loc[req_id].item())
            qe = int(query_start_loc[req_id + 1].item())
            query_len = qe - qs
        else:
            query_len = 0

        n_rej = int(num_rejected[req_id].item())
        computed_delta = query_len - n_rej
        if computed_delta != 0:
            out_nct[req_state_idx] += computed_delta

    return out_nct, out_lst, out_tl, out_abc, out_binc


class TestPostUpdateKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_reqs", [1, 2])
    @pytest.mark.parametrize("num_spec_steps", [1, 3])
    def test_basic_update(self, num_reqs, num_spec_steps):
        """Verify basic post-update: append sampled tokens, update counts."""
        max_num_reqs = num_reqs
        vocab_size = 64
        max_model_len = 32

        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        num_computed_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=self.device)
        last_sampled_tokens = torch.full((max_num_reqs,), -1, dtype=torch.int32, device=self.device)
        output_bin_counts = torch.zeros(max_num_reqs, vocab_size, dtype=torch.int32, device=self.device)
        total_len = torch.full((max_num_reqs,), 5, dtype=torch.int32, device=self.device)

        sampled_tokens = torch.randint(0, vocab_size, (num_reqs, num_spec_steps + 1),
                                       dtype=torch.int32, device=self.device)
        n_sampled_vals = torch.randint(1, num_spec_steps + 1, (num_reqs,), dtype=torch.int32, device=self.device)

        # Ensure at least 1 sampled
        n_sampled_vals = torch.clamp(n_sampled_vals, min=1)
        num_rejected = torch.randint(0, 2, (num_reqs,), dtype=torch.int32, device=self.device)
        num_rejected = torch.minimum(num_rejected,
                                     (num_spec_steps + 1) - n_sampled_vals)
        query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32, device=self.device) * 2

        all_token_ids = torch.full((max_num_reqs, max_model_len), -1, dtype=torch.int32, device=self.device)

        ref_nct, ref_lst, ref_tl, ref_abc, ref_binc = _post_update_ref(
            idx_mapping.cpu(), num_computed_tokens.cpu(), last_sampled_tokens.cpu(),
            output_bin_counts.cpu(), sampled_tokens.cpu(), n_sampled_vals.cpu(),
            num_rejected.cpu(), query_start_loc.cpu(), all_token_ids.cpu(), total_len.cpu(),
        )

        _post_update_kernel[(num_reqs,)](
            idx_mapping,
            num_computed_tokens,
            last_sampled_tokens,
            output_bin_counts,
            output_bin_counts.stride(0),
            sampled_tokens,
            sampled_tokens.stride(0),
            n_sampled_vals,
            num_rejected,
            query_start_loc,
            all_token_ids,
            all_token_ids.stride(0),
            total_len,
            num_warps=1,
        )
        torch.npu.synchronize()

        torch.testing.assert_close(num_computed_tokens.cpu(), ref_nct, rtol=0, atol=0)
        torch.testing.assert_close(last_sampled_tokens.cpu(), ref_lst, rtol=0, atol=0)
        torch.testing.assert_close(total_len.cpu(), ref_tl, rtol=0, atol=0)
        torch.testing.assert_close(output_bin_counts.cpu(), ref_binc, rtol=0, atol=0)

    def test_negative_idx_mapping(self):
        """Rows with negative idx_mapping should be skipped."""
        num_reqs = 3
        max_num_reqs = 3
        vocab_size = 32
        max_model_len = 16

        idx_mapping = torch.tensor([-1, 0, 1], dtype=torch.int32, device=self.device)
        num_computed_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=self.device)
        last_sampled_tokens = torch.full((max_num_reqs,), -1, dtype=torch.int32, device=self.device)
        output_bin_counts = torch.zeros(max_num_reqs, vocab_size, dtype=torch.int32, device=self.device)
        total_len = torch.tensor([0, 5, 3], dtype=torch.int32, device=self.device)

        sampled_tokens = torch.tensor([[10, 20, 0], [30, 40, 0], [50, 60, 0]],
                                      dtype=torch.int32, device=self.device)
        n_sampled = torch.tensor([2, 2, 2], dtype=torch.int32, device=self.device)
        num_rejected = torch.tensor([0, 0, 0], dtype=torch.int32, device=self.device)
        query_start_loc = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device=self.device)
        all_token_ids = torch.full((max_num_reqs, max_model_len), -1, dtype=torch.int32, device=self.device)

        _post_update_kernel[(num_reqs,)](
            idx_mapping,
            num_computed_tokens,
            last_sampled_tokens,
            output_bin_counts,
            output_bin_counts.stride(0),
            sampled_tokens,
            sampled_tokens.stride(0),
            n_sampled,
            num_rejected,
            query_start_loc,
            all_token_ids,
            all_token_ids.stride(0),
            total_len,
            num_warps=1,
        )
        torch.npu.synchronize()

        # Req 0 (idx=-1): skipped - no changes
        # Req 1 (idx=0): updated
        # Req 2 (idx=1): updated
        assert last_sampled_tokens[0].item() == 20, "req_state 0 last_sampled should be 20"
        assert last_sampled_tokens[1].item() == 40, "req_state 1 last_sampled should be 40"

        assert num_computed_tokens[0].item() == 2, "req_state 0 computed should increase by 2"
        assert num_computed_tokens[1].item() == 2, "req_state 1 computed should increase by 2"

    def test_no_output_bin_counts(self):
        """When output_bin_counts is None, kernel should work without it."""
        num_reqs = 1
        max_num_reqs = 1
        max_model_len = 16

        idx_mapping = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        num_computed_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=self.device)
        last_sampled_tokens = torch.full((max_num_reqs,), -1, dtype=torch.int32, device=self.device)
        total_len = torch.tensor([0], dtype=torch.int32, device=self.device)
        sampled_tokens = torch.tensor([[42, 0, 0]], dtype=torch.int32, device=self.device)
        n_sampled = torch.tensor([1], dtype=torch.int32, device=self.device)
        num_rejected = torch.tensor([0], dtype=torch.int32, device=self.device)
        query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
        all_token_ids = torch.full((max_num_reqs, max_model_len), -1, dtype=torch.int32, device=self.device)

        _post_update_kernel[(num_reqs,)](
            idx_mapping,
            num_computed_tokens,
            last_sampled_tokens,
            None,  # output_bin_counts
            0,  # output_bin_counts_stride (dummy)
            sampled_tokens,
            sampled_tokens.stride(0),
            n_sampled,
            num_rejected,
            query_start_loc,
            all_token_ids,
            all_token_ids.stride(0),
            total_len,
            num_warps=1,
        )
        torch.npu.synchronize()

        assert last_sampled_tokens[0].item() == 42
        assert total_len[0].item() == 1
        assert all_token_ids[0, 0].item() == 42

    def test_query_start_loc_none(self):
        """When query_start_loc is None, computed_delta = 0 - num_rejected."""
        num_reqs = 1
        max_num_reqs = 1
        max_model_len = 16

        idx_mapping = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        num_computed_tokens = torch.tensor([10], dtype=torch.int32, device=self.device)
        last_sampled_tokens = torch.full((max_num_reqs,), -1, dtype=torch.int32, device=self.device)
        total_len = torch.tensor([5], dtype=torch.int32, device=self.device)
        sampled_tokens = torch.tensor([[99, 0, 0]], dtype=torch.int32, device=self.device)
        n_sampled = torch.tensor([1], dtype=torch.int32, device=self.device)
        num_rejected = torch.tensor([2], dtype=torch.int32, device=self.device)
        all_token_ids = torch.full((max_num_reqs, max_model_len), -1, dtype=torch.int32, device=self.device)

        _post_update_kernel[(num_reqs,)](
            idx_mapping,
            num_computed_tokens,
            last_sampled_tokens,
            None,
            0,
            sampled_tokens,
            sampled_tokens.stride(0),
            n_sampled,
            num_rejected,
            None,  # query_start_loc
            all_token_ids,
            all_token_ids.stride(0),
            total_len,
            num_warps=1,
        )
        torch.npu.synchronize()

        # computed_delta = 0 - 2 = -2, so num_computed should DECREASE by 2
        assert num_computed_tokens[0].item() == 8, \
            f"Expected 8, got {num_computed_tokens[0].item()}"
