# GENERATED STRICT UT. Source: accuracy_test/codex/missing_accuracy_tests/test_prepare_decode_inputs_kernel.py
# Do not edit mechanically; update the reviewed Codex source or strict generator.
from accuracy_test.strict_ut.runtime_npu import STRICT_DEVICE as _STRICT_DEVICE
# vLLM vanilla kernel: _prepare_decode_inputs_kernel from
# vllm/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py

"""
Precision test for _prepare_decode_inputs_kernel.

Kernel signature:
    _prepare_decode_inputs_kernel(
        draft_tokens_ptr,              # int32 [num_reqs]
        draft_tokens_stride,           # stride(0) of draft_tokens
        target_seq_lens_ptr,           # int32 [num_reqs]
        num_rejected_ptr,              # int32 [num_reqs]
        input_ids_ptr,                 # int32 output [max_num_tokens]
        positions_ptr,                 # int64 output [max_num_tokens]
        query_start_loc_ptr,           # int32 output [max_num_reqs + 1]
        seq_lens_ptr,                  # int32 output [max_num_reqs]
        max_model_len,                 # scalar
        max_num_reqs,                  # scalar
        BLOCK_SIZE: tl.constexpr,      # block size for iteration
        ADVANCE_DRAFT_POSITIONS: tl.constexpr,  # flag
    )

Prepares decode inputs for speculative decoding. With (num_reqs + 1) programs:
the last thread block pads query_start_loc and seq_lens for CUDA graphs.
Each req block copies the draft token to input_ids and optionally advances
positions and sequence lengths.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    _prepare_decode_inputs_kernel,
)
from accuracy_test.strict_ut.runtime_npu import init_device_properties_triton

import pytest


class TestPrepareDecodeInputsKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_reqs", [1, 2, 4])
    @pytest.mark.parametrize("advance_pos", [False, True])
    def test_prepare_decode_inputs(self, num_reqs, advance_pos):
        """Compare with CPU reference."""
        max_num_tokens = num_reqs
        max_num_reqs = 8
        max_model_len = 2048

        draft_tokens = torch.randint(0, 100, (num_reqs, 1), dtype=torch.int32, device=self.device)
        target_seq_lens = torch.randint(10, 100, (num_reqs,), dtype=torch.int32, device=self.device)
        num_rejected = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        input_ids = torch.full((max_num_tokens,), -1, dtype=torch.int32, device=self.device)
        positions = torch.full((max_num_tokens,), -1, dtype=torch.int64, device=self.device)
        query_start_loc = torch.full((max_num_reqs + 1,), -1, dtype=torch.int32, device=self.device)
        seq_lens = torch.full((max_num_reqs,), -1, dtype=torch.int32, device=self.device)

        if advance_pos:
            # Seed positions with start values
            positions.copy_(torch.randint(0, 50, (max_num_tokens,), dtype=torch.int64, device=self.device))

        expected_input_ids = input_ids.clone().cpu()
        expected_positions = positions.clone().cpu()
        expected_seq_lens = seq_lens.clone().cpu()
        expected_query_start_loc = query_start_loc.clone().cpu()

        _prepare_decode_inputs_kernel[(num_reqs + 1,)](
            draft_tokens,
            draft_tokens.stride(0),
            target_seq_lens,
            num_rejected,
            input_ids,
            positions,
            query_start_loc,
            seq_lens,
            max_model_len,
            max_num_reqs,
            BLOCK_SIZE=1024,
            ADVANCE_DRAFT_POSITIONS=advance_pos,
        )
        torch.npu.synchronize()

        # CPU reference
        for req_idx in range(num_reqs):
            dt = draft_tokens[req_idx, 0].item()
            expected_input_ids[req_idx] = dt
            if advance_pos:
                old_pos = expected_positions[req_idx].item()
                expected_positions[req_idx] = min(old_pos + 1, max_model_len - 1)
                tsl = target_seq_lens[req_idx].item()
                nr = num_rejected[req_idx].item()
                expected_seq_lens[req_idx] = min(tsl - nr + 1, max_model_len)
        # Padding block at req_idx == num_reqs
        for i in range(num_reqs):
            expected_query_start_loc[i] = i
        expected_query_start_loc[num_reqs] = num_reqs
        for i in range(num_reqs, max_num_reqs + 1):
            expected_query_start_loc[i] = num_reqs
        for i in range(num_reqs, max_num_reqs):
            expected_seq_lens[i] = 0

        torch.testing.assert_close(input_ids.cpu(), expected_input_ids, rtol=0, atol=0)
        torch.testing.assert_close(seq_lens.cpu(), expected_seq_lens, rtol=0, atol=0)
        torch.testing.assert_close(query_start_loc.cpu(), expected_query_start_loc, rtol=0, atol=0)
        if advance_pos:
            torch.testing.assert_close(positions.cpu(), expected_positions, rtol=0, atol=0)

    def test_with_rejected_tokens(self):
        """Verify seq_len computation with num_rejected > 0."""
        num_reqs = 2
        max_num_tokens = num_reqs
        max_num_reqs = 4
        max_model_len = 64

        draft_tokens = torch.tensor([[42], [99]], dtype=torch.int32, device=self.device)
        target_seq_lens = torch.tensor([20, 30], dtype=torch.int32, device=self.device)
        num_rejected = torch.tensor([0, 3], dtype=torch.int32, device=self.device)
        input_ids = torch.full((max_num_tokens,), -1, dtype=torch.int32, device=self.device)
        positions = torch.full((max_num_tokens,), 10, dtype=torch.int64, device=self.device)
        query_start_loc = torch.full((max_num_reqs + 1,), -1, dtype=torch.int32, device=self.device)
        seq_lens = torch.full((max_num_reqs,), -1, dtype=torch.int32, device=self.device)

        _prepare_decode_inputs_kernel[(num_reqs + 1,)](
            draft_tokens,
            draft_tokens.stride(0),
            target_seq_lens,
            num_rejected,
            input_ids,
            positions,
            query_start_loc,
            seq_lens,
            max_model_len,
            max_num_reqs,
            BLOCK_SIZE=1024,
            ADVANCE_DRAFT_POSITIONS=True,
        )
        torch.npu.synchronize()

        expected_input_ids = torch.tensor([42, 99], dtype=torch.int32)
        expected_seq_lens_0 = min(20 - 0 + 1, max_model_len)  # 21
        expected_seq_lens_1 = min(30 - 3 + 1, max_model_len)  # 28
        torch.testing.assert_close(input_ids.cpu()[:num_reqs], expected_input_ids, rtol=0, atol=0)
        torch.testing.assert_close(seq_lens[0].item(), expected_seq_lens_0, rtol=0, atol=0)
        torch.testing.assert_close(seq_lens[1].item(), expected_seq_lens_1, rtol=0, atol=0)

    def test_model_len_clamp(self):
        """Positions and seq_lens should be clamped to max_model_len."""
        num_reqs = 2
        max_num_tokens = num_reqs
        max_num_reqs = 4
        max_model_len = 8

        draft_tokens = torch.tensor([[5], [7]], dtype=torch.int32, device=self.device)
        target_seq_lens = torch.tensor([20, 8], dtype=torch.int32, device=self.device)
        num_rejected = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        input_ids = torch.full((max_num_tokens,), -1, dtype=torch.int32, device=self.device)
        positions = torch.tensor([7, 7], dtype=torch.int64, device=self.device)
        query_start_loc = torch.full((max_num_reqs + 1,), -1, dtype=torch.int32, device=self.device)
        seq_lens = torch.full((max_num_reqs,), -1, dtype=torch.int32, device=self.device)

        _prepare_decode_inputs_kernel[(num_reqs + 1,)](
            draft_tokens,
            draft_tokens.stride(0),
            target_seq_lens,
            num_rejected,
            input_ids,
            positions,
            query_start_loc,
            seq_lens,
            max_model_len,
            max_num_reqs,
            BLOCK_SIZE=1024,
            ADVANCE_DRAFT_POSITIONS=True,
        )
        torch.npu.synchronize()

        # position 7+1=8, clamped to max_model_len-1=7
        assert positions[0].item() == 7, f"Expected 7, got {positions[0].item()}"
        assert positions[1].item() == 7, f"Expected 7, got {positions[1].item()}"
        # seq_lens 20+1=21 clamped to 8
        assert seq_lens[0].item() == 8, f"Expected 8, got {seq_lens[0].item()}"
