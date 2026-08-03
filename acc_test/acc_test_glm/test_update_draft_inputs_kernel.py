import pytest
import torch


def _update_draft_inputs_cpu(
    output_draft_tokens: torch.Tensor,
    next_input_hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    draft_tokens: torch.Tensor,
    current_draft_step: int,
    hidden_states: torch.Tensor,
    max_model_len: int,
    num_speculative_steps: int,
    advance_draft_positions: bool,
):
    num_reqs = draft_tokens.shape[0]
    for req_idx in range(num_reqs):
        draft_token = int(draft_tokens[req_idx])
        step = current_draft_step
        output_draft_tokens[req_idx, step] = draft_token

        if step >= num_speculative_steps - 1:
            continue

        input_ids[req_idx] = draft_token

        next_input_hidden_states[req_idx] = hidden_states[req_idx]

        if advance_draft_positions:
            position = int(positions[req_idx])
            position = min(position + 1, max_model_len - 1)
            positions[req_idx] = position

            seq_len = int(seq_lens[req_idx])
            seq_len = min(seq_len + 1, max_model_len)
            seq_lens[req_idx] = seq_len


def test_update_draft_inputs_kernel():
    torch.manual_seed(42)
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
        _update_draft_inputs_kernel,
    )

    num_reqs = 2
    num_speculative_steps = 4
    hidden_size = 8
    max_model_len = 1024

    output_draft_tokens = torch.zeros(num_reqs, num_speculative_steps, dtype=torch.int32)
    next_input_hidden_states = torch.zeros(num_reqs, hidden_size, dtype=torch.float32)
    input_ids = torch.zeros(num_reqs, dtype=torch.int32)
    positions = torch.tensor([10, 20], dtype=torch.int64)
    seq_lens = torch.tensor([15, 25], dtype=torch.int32)
    draft_tokens = torch.tensor([42, 77], dtype=torch.int32)
    current_draft_step = 1
    hidden_states = torch.randn(num_reqs, hidden_size, dtype=torch.float32)

    expected_output = output_draft_tokens.clone()
    expected_next_hs = next_input_hidden_states.clone()
    expected_input_ids = input_ids.clone()
    expected_positions = positions.clone()
    expected_seq_lens = seq_lens.clone()

    _update_draft_inputs_cpu(
        expected_output,
        expected_next_hs,
        expected_input_ids,
        expected_positions,
        expected_seq_lens,
        draft_tokens,
        current_draft_step,
        hidden_states,
        max_model_len,
        num_speculative_steps,
        advance_draft_positions=True,
    )

    device = torch.device("npu")

    _update_draft_inputs_kernel[(num_reqs,)](
        output_draft_tokens.to(device),
        output_draft_tokens.stride(0),
        next_input_hidden_states.to(device),
        next_input_hidden_states.stride(0),
        input_ids.to(device),
        positions.to(device),
        seq_lens.to(device),
        draft_tokens.to(device),
        torch.tensor(current_draft_step, dtype=torch.int32, device=device),
        hidden_states.to(device),
        hidden_states.stride(0),
        hidden_size,
        max_model_len,
        num_speculative_steps,
        BLOCK_SIZE=1024,
        ADVANCE_DRAFT_POSITIONS=True,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        output_draft_tokens.cpu(), expected_output, rtol=0, atol=0
    )
    torch.testing.assert_close(
        next_input_hidden_states.cpu(), expected_next_hs, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(input_ids.cpu(), expected_input_ids, rtol=0, atol=0)
    torch.testing.assert_close(positions.cpu(), expected_positions, rtol=0, atol=0)


def test_update_draft_inputs_kernel_final_step():
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
        _update_draft_inputs_kernel,
    )

    num_reqs = 1
    num_speculative_steps = 3
    hidden_size = 4
    max_model_len = 512

    output_draft_tokens = torch.zeros(num_reqs, num_speculative_steps, dtype=torch.int32)
    next_input_hidden_states = torch.randn(num_reqs, hidden_size, dtype=torch.float32)
    input_ids = torch.tensor([99], dtype=torch.int32)
    positions = torch.tensor([5], dtype=torch.int64)
    seq_lens = torch.tensor([10], dtype=torch.int32)
    draft_tokens = torch.tensor([55], dtype=torch.int32)
    current_draft_step = 2
    hidden_states = torch.randn(num_reqs, hidden_size, dtype=torch.float32)

    expected_output = output_draft_tokens.clone()
    expected_next_hs = next_input_hidden_states.clone()
    expected_input_ids = input_ids.clone()
    expected_positions = positions.clone()
    expected_seq_lens = seq_lens.clone()

    _update_draft_inputs_cpu(
        expected_output,
        expected_next_hs,
        expected_input_ids,
        expected_positions,
        expected_seq_lens,
        draft_tokens,
        current_draft_step,
        hidden_states,
        max_model_len,
        num_speculative_steps,
        advance_draft_positions=True,
    )

    device = torch.device("npu")

    _update_draft_inputs_kernel[(num_reqs,)](
        output_draft_tokens.to(device),
        output_draft_tokens.stride(0),
        next_input_hidden_states.to(device),
        next_input_hidden_states.stride(0),
        input_ids.to(device),
        positions.to(device),
        seq_lens.to(device),
        draft_tokens.to(device),
        torch.tensor(current_draft_step, dtype=torch.int32, device=device),
        hidden_states.to(device),
        hidden_states.stride(0),
        hidden_size,
        max_model_len,
        num_speculative_steps,
        BLOCK_SIZE=1024,
        ADVANCE_DRAFT_POSITIONS=True,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        output_draft_tokens.cpu(), expected_output, rtol=0, atol=0
    )
    torch.testing.assert_close(input_ids.cpu(), expected_input_ids, rtol=0, atol=0)
