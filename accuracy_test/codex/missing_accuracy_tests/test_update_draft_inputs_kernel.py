# vLLM vanilla kernel: _update_draft_inputs_kernel from
# vllm/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py

"""
Precision test for _update_draft_inputs_kernel.

Kernel signature:
    _update_draft_inputs_kernel(
        output_draft_tokens_ptr,            # int64 output [num_reqs, num_speculative_steps + 1]
        output_draft_tokens_stride,         # stride(0) of output_draft_tokens
        next_input_hidden_states_ptr,        # fp16/bf16 output [num_reqs, hidden_size]
        next_input_hidden_states_stride,     # stride(0) of next_input_hidden_states
        input_ids_ptr,                       # int32 output [max_num_tokens]
        positions_ptr,                       # int64 output [max_num_tokens]
        seq_lens_ptr,                        # int32 output [max_num_reqs]
        draft_tokens_ptr,                    # int32 [num_reqs] current draft tokens
        current_draft_step_ptr,              # int64 scalar [1]
        hidden_states_ptr,                   # fp16/bf16 [num_reqs, hidden_size]
        hidden_states_stride,                # stride(0) of hidden_states
        hidden_size,                         # scalar
        max_model_len,                       # scalar
        num_speculative_steps,               # scalar
        BLOCK_SIZE: tl.constexpr,            # block size
        ADVANCE_DRAFT_POSITIONS: tl.constexpr,# flag
    )

Updates draft inputs for each step of speculative decoding.
Writes the sampled draft token, copies hidden states, and optionally
advances positions and seq_lens.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    _update_draft_inputs_kernel,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

import pytest


class TestUpdateDraftInputsKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_reqs", [1, 2, 4])
    @pytest.mark.parametrize("hidden_size", [128, 512])
    @pytest.mark.parametrize("advance_pos", [False, True])
    def test_update_draft_inputs(self, num_reqs, hidden_size, advance_pos):
        """Compare kernel output with CPU reference."""
        num_speculative_steps = 3
        max_num_reqs = 8
        max_num_tokens = max_num_reqs
        max_model_len = 2048

        output_draft_tokens = torch.full(
            (max_num_reqs, num_speculative_steps + 1), -1, dtype=torch.int64, device=self.device
        )
        next_input_hidden_states = torch.zeros(num_reqs, hidden_size, dtype=torch.float16, device=self.device)
        input_ids = torch.full((max_num_tokens,), -1, dtype=torch.int32, device=self.device)
        positions = torch.full((max_num_tokens,), 5, dtype=torch.int64, device=self.device)
        seq_lens = torch.full((max_num_reqs,), 10, dtype=torch.int32, device=self.device)
        draft_tokens = torch.randint(0, 100, (num_reqs,), dtype=torch.int32, device=self.device)
        current_draft_step = torch.tensor(1, dtype=torch.int64, device=self.device)  # not the final step
        hidden_states = torch.randn(num_reqs, hidden_size, dtype=torch.float16, device=self.device)

        expected_output_draft = output_draft_tokens.clone().cpu()
        expected_input_hidden = next_input_hidden_states.clone().cpu()
        expected_input_ids = input_ids.clone().cpu()
        expected_positions = positions.clone().cpu()
        expected_seq_lens = seq_lens.clone().cpu()

        _update_draft_inputs_kernel[(num_reqs,)](
            output_draft_tokens,
            output_draft_tokens.stride(0),
            next_input_hidden_states,
            next_input_hidden_states.stride(0),
            input_ids,
            positions,
            seq_lens,
            draft_tokens,
            current_draft_step,
            hidden_states,
            hidden_states.stride(0),
            hidden_size,
            max_model_len,
            num_speculative_steps,
            BLOCK_SIZE=1024,
            ADVANCE_DRAFT_POSITIONS=advance_pos,
        )
        torch.npu.synchronize()

        # CPU reference
        step = current_draft_step.item()
        for req_idx in range(num_reqs):
            dt = draft_tokens[req_idx].item()
            expected_output_draft[req_idx, step] = dt
            if step < num_speculative_steps - 1:
                expected_input_ids[req_idx] = dt
                expected_input_hidden[req_idx] = hidden_states.cpu()[req_idx]
                if advance_pos:
                    old_pos = positions[req_idx].item()
                    expected_positions[req_idx] = min(old_pos + 1, max_model_len - 1)
                    old_seq = seq_lens[req_idx].item()
                    expected_seq_lens[req_idx] = min(old_seq + 1, max_model_len)

        torch.testing.assert_close(output_draft_tokens.cpu(), expected_output_draft, rtol=0, atol=0)
        torch.testing.assert_close(next_input_hidden_states.cpu(), expected_input_hidden, rtol=1e-3, atol=1e-3)
        if step < num_speculative_steps - 1:
            torch.testing.assert_close(input_ids.cpu(), expected_input_ids, rtol=0, atol=0)
        if advance_pos and step < num_speculative_steps - 1:
            torch.testing.assert_close(positions.cpu(), expected_positions, rtol=0, atol=0)
            torch.testing.assert_close(seq_lens.cpu(), expected_seq_lens, rtol=0, atol=0)

    def test_final_step_skips_update(self):
        """On the last speculative step (step >= num_speculative_steps - 1),
        the kernel should only write the draft token and return early."""
        num_reqs = 2
        hidden_size = 64
        num_speculative_steps = 3
        max_num_reqs = 4
        max_num_tokens = max_num_reqs
        max_model_len = 128

        output_draft_tokens = torch.full(
            (max_num_reqs, num_speculative_steps + 1), -1, dtype=torch.int64, device=self.device
        )
        next_input_hidden_states = torch.zeros(num_reqs, hidden_size, dtype=torch.float16, device=self.device)
        input_ids = torch.full((max_num_tokens,), -1, dtype=torch.int32, device=self.device)
        positions = torch.full((max_num_tokens,), 5, dtype=torch.int64, device=self.device)
        seq_lens = torch.full((max_num_reqs,), 10, dtype=torch.int32, device=self.device)
        draft_tokens = torch.tensor([42, 99], dtype=torch.int32, device=self.device)
        current_draft_step = torch.tensor(2, dtype=torch.int64, device=self.device)  # final step (2 >= 3-1)
        hidden_states = torch.randn(num_reqs, hidden_size, dtype=torch.float16, device=self.device)

        expected_output_draft = output_draft_tokens.clone().cpu()
        expected_output_draft[0, 2] = 42
        expected_output_draft[1, 2] = 99

        _update_draft_inputs_kernel[(num_reqs,)](
            output_draft_tokens,
            output_draft_tokens.stride(0),
            next_input_hidden_states,
            next_input_hidden_states.stride(0),
            input_ids,
            positions,
            seq_lens,
            draft_tokens,
            current_draft_step,
            hidden_states,
            hidden_states.stride(0),
            hidden_size,
            max_model_len,
            num_speculative_steps,
            BLOCK_SIZE=1024,
            ADVANCE_DRAFT_POSITIONS=True,
        )
        torch.npu.synchronize()

        # Only output_draft_tokens should change
        torch.testing.assert_close(output_draft_tokens.cpu(), expected_output_draft, rtol=0, atol=0)
        # Everything else remains as initialized
        torch.testing.assert_close(input_ids.cpu(), torch.full((max_num_tokens,), -1, dtype=torch.int32), rtol=0, atol=0)
        torch.testing.assert_close(positions.cpu(), torch.full((max_num_tokens,), 5, dtype=torch.int64), rtol=0, atol=0)
