# GENERATED STRICT UT. Source: accuracy_test/codex/missing_accuracy_tests/test_prepare_prefill_inputs_kernel.py
# Do not edit mechanically; update the reviewed Codex source or strict generator.
from accuracy_test.strict_ut.runtime_npu import STRICT_DEVICE as _STRICT_DEVICE
# vLLM vanilla kernel: _prepare_prefill_inputs_kernel from vllm/vllm/v1/worker/gpu/input_batch.py

"""
Precision test for _prepare_prefill_inputs_kernel.

Kernel signature:
    _prepare_prefill_inputs_kernel(
        input_ids_ptr,               # int32 output [max_num_tokens]
        next_prefill_tokens_ptr,     # int32 output [num_lookahead, max_num_reqs]
        next_prefill_tokens_stride,  # stride(0) of next_prefill_tokens
        num_lookahead,               # number of lookahead tokens to write
        idx_mapping_ptr,             # int32 [num_reqs] batch_idx -> req_state_idx
        query_start_loc_ptr,          # int32 [num_reqs + 1]
        all_token_ids_ptr,            # int32 [max_num_reqs, max_model_len]
        all_token_ids_stride,         # stride(0) of all_token_ids
        prefill_lens_ptr,             # int32 [max_num_reqs]
        num_computed_tokens_ptr,      # int32 [max_num_reqs]
        BLOCK_SIZE: tl.constexpr,     # block size for iteration
        LOOKAHEAD_BLOCK: tl.constexpr,# padded lookahead block size
    )

Copies prefill token IDs from all_token_ids to input_ids.
Stores up to num_lookahead following prefill tokens in next_prefill_tokens.
When num_computed >= prefill_len, returns early (not a prefill step).
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import _prepare_prefill_inputs_kernel
from accuracy_test.strict_ut.runtime_npu import init_device_properties_triton

import pytest


def _prepare_prefill_inputs_ref(
    input_ids: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    prefill_lens: torch.Tensor,
    num_computed_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference for prepare_prefill_inputs."""
    input_ids_out = input_ids.clone()
    next_prefill_tokens_out = next_prefill_tokens.clone()
    num_reqs = idx_mapping.shape[0]

    for batch_idx in range(num_reqs):
        rs_idx = idx_mapping[batch_idx].item()
        prefill_len = prefill_lens[rs_idx].item()
        num_computed = num_computed_tokens[rs_idx].item()
        if num_computed >= prefill_len:
            continue

        qs = query_start_loc[batch_idx].item()
        qe = query_start_loc[batch_idx + 1].item()
        qlen = qe - qs

        for i in range(qlen):
            tok = all_token_ids[rs_idx, num_computed + i].item()
            input_ids_out[qs + i] = tok

        for lookahead in range(next_prefill_tokens.shape[0]):
            next_pos = num_computed + qlen + lookahead
            next_prefill_tokens_out[lookahead, rs_idx] = (
                all_token_ids[rs_idx, next_pos].item()
                if next_pos < prefill_len
                else 0
            )

    return input_ids_out, next_prefill_tokens_out


class TestPreparePrefillInputsKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_reqs", [1, 2, 4])
    @pytest.mark.parametrize("query_len", [1, 4, 16])
    def test_prepare_prefill_inputs(self, num_reqs, query_len):
        """Compare kernel output with CPU reference."""
        max_model_len = 128
        max_num_reqs = 8
        max_num_tokens = num_reqs * query_len
        num_lookahead = 3

        all_token_ids = torch.arange(max_model_len, dtype=torch.int32, device=self.device).unsqueeze(0).repeat(max_num_reqs, 1)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32, device=self.device) * query_len
        num_computed_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=self.device)
        prefill_lens = torch.full((max_num_reqs,), 64, dtype=torch.int32, device=self.device)

        input_ids = torch.zeros(max_num_tokens, dtype=torch.int32, device=self.device)
        next_prefill_tokens = torch.zeros(num_lookahead, max_num_reqs, dtype=torch.int32, device=self.device)

        _prepare_prefill_inputs_kernel[(num_reqs,)](
            input_ids,
            next_prefill_tokens,
            next_prefill_tokens.stride(0),
            num_lookahead,
            idx_mapping,
            query_start_loc,
            all_token_ids,
            all_token_ids.stride(0),
            prefill_lens,
            num_computed_tokens,
            BLOCK_SIZE=1024,
            LOOKAHEAD_BLOCK=triton.next_power_of_2(num_lookahead),
        )
        torch.npu.synchronize()

        input_ids_exp, next_prefill_exp = _prepare_prefill_inputs_ref(
            torch.zeros(max_num_tokens, dtype=torch.int32),
            torch.zeros(num_lookahead, max_num_reqs, dtype=torch.int32),
            idx_mapping.cpu(), query_start_loc.cpu(),
            all_token_ids.cpu(), prefill_lens.cpu(), num_computed_tokens.cpu(),
        )
        torch.testing.assert_close(input_ids.cpu(), input_ids_exp, rtol=0, atol=0)
        torch.testing.assert_close(next_prefill_tokens.cpu(), next_prefill_exp, rtol=0, atol=0)

    def test_early_return_when_prefill_done(self):
        """When num_computed >= prefill_len, kernel should be a no-op."""
        num_reqs = 2
        query_len = 4
        max_num_tokens = num_reqs * query_len
        max_num_reqs = 4
        max_model_len = 32
        num_lookahead = 1

        all_token_ids = torch.randint(0, 100, (max_num_reqs, max_model_len), dtype=torch.int32, device=self.device)
        idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
        query_start_loc = torch.tensor([0, 4, 8], dtype=torch.int32, device=self.device)
        num_computed_tokens = torch.tensor([20, 15], dtype=torch.int32, device=self.device)
        prefill_lens = torch.tensor([10, 20], dtype=torch.int32, device=self.device)

        input_ids = torch.full((max_num_tokens,), -1, dtype=torch.int32, device=self.device)
        next_prefill_tokens = torch.full((num_lookahead, max_num_reqs), -1, dtype=torch.int32, device=self.device)
        expected_input_ids = input_ids.clone().cpu()
        expected_next_prefill = next_prefill_tokens.clone().cpu()

        _prepare_prefill_inputs_kernel[(num_reqs,)](
            input_ids,
            next_prefill_tokens,
            next_prefill_tokens.stride(0),
            num_lookahead,
            idx_mapping,
            query_start_loc,
            all_token_ids,
            all_token_ids.stride(0),
            prefill_lens,
            num_computed_tokens,
            BLOCK_SIZE=1024,
            LOOKAHEAD_BLOCK=triton.next_power_of_2(num_lookahead),
        )
        torch.npu.synchronize()

        # req 0: prefill_lens[0]=10, num_computed[0]=20 -> done -> early return
        # req 1: prefill_lens[1]=20, num_computed[1]=15 -> still prefilling
        token_ids_cpu = all_token_ids.cpu()
        for i in range(query_len):
            expected_input_ids[query_len + i] = token_ids_cpu[1, 15 + i]
        expected_next_prefill[0, 1] = token_ids_cpu[1, 15 + 4]

        torch.testing.assert_close(input_ids.cpu(), expected_input_ids, rtol=0, atol=0)
        torch.testing.assert_close(next_prefill_tokens.cpu(), expected_next_prefill, rtol=0, atol=0)

    def test_exact_prefill_boundary(self):
        """When num_computed + query_len == prefill_len, no next_prefill_token stored."""
        num_reqs = 1
        query_len = 5
        max_num_tokens = query_len
        max_num_reqs = 2
        max_model_len = 16
        num_lookahead = 2

        all_token_ids = torch.arange(max_model_len, dtype=torch.int32, device=self.device).unsqueeze(0).repeat(max_num_reqs, 1)
        idx_mapping = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        query_start_loc = torch.tensor([0, 5], dtype=torch.int32, device=self.device)
        num_computed_tokens = torch.tensor([5], dtype=torch.int32, device=self.device)
        prefill_lens = torch.tensor([10, 10], dtype=torch.int32, device=self.device)

        input_ids = torch.full((max_num_tokens,), -1, dtype=torch.int32, device=self.device)
        next_prefill_tokens = torch.full((num_lookahead, max_num_reqs), -1, dtype=torch.int32, device=self.device)

        _prepare_prefill_inputs_kernel[(num_reqs,)](
            input_ids,
            next_prefill_tokens,
            next_prefill_tokens.stride(0),
            num_lookahead,
            idx_mapping,
            query_start_loc,
            all_token_ids,
            all_token_ids.stride(0),
            prefill_lens,
            num_computed_tokens,
            BLOCK_SIZE=1024,
            LOOKAHEAD_BLOCK=triton.next_power_of_2(num_lookahead),
        )
        torch.npu.synchronize()

        # num_computed=5, query_len=5, prefill_len=10 => next_pos=10 == prefill_len, no next stored
        expected_input_ids = torch.tensor([5, 6, 7, 8, 9], dtype=torch.int32)
        expected_next = torch.full((num_lookahead, max_num_reqs), -1, dtype=torch.int32)
        expected_next[:, 0] = 0

        torch.testing.assert_close(input_ids.cpu(), expected_input_ids, rtol=0, atol=0)
        torch.testing.assert_close(next_prefill_tokens.cpu(), expected_next, rtol=0, atol=0)
