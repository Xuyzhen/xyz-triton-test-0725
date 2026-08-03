# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    _prepare_decode_inputs_kernel,
    _update_draft_inputs_kernel,
)

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


class _InputBuffersStub:
    """Minimal stub mimicking InputBuffers for kernel testing."""

    def __init__(self, max_num_reqs: int, max_model_len: int, device: torch.device):
        self.input_ids = torch.empty(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self.positions = torch.empty(
            max_num_reqs, dtype=torch.int64, device=device
        )
        self.seq_lens = torch.empty(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self.query_start_loc = torch.empty(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )


# ---------------------------------------------------------------------------
# _prepare_decode_inputs_kernel tests
# ---------------------------------------------------------------------------


def _prepare_decode_inputs_cpu(
    draft_tokens: torch.Tensor,
    target_seq_lens: torch.Tensor,
    num_rejected: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    max_model_len: int,
    max_num_reqs: int,
    advance_draft_positions: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure PyTorch CPU reference for prepare_decode_inputs.

    Returns (input_ids, positions, seq_lens, query_start_loc).
    """
    num_reqs = len(draft_tokens)
    out_input_ids = input_ids.clone()
    out_positions = positions.clone()
    out_seq_lens = seq_lens.clone()
    out_query_start_loc = torch.empty(max_num_reqs + 1, dtype=torch.int32)

    for req_idx in range(num_reqs):
        out_input_ids[req_idx] = int(draft_tokens[req_idx])

        if advance_draft_positions:
            pos = int(positions[req_idx])
            pos = min(pos + 1, max_model_len - 1)
            out_positions[req_idx] = pos

            target_len = int(target_seq_lens[req_idx])
            rejected = int(num_rejected[req_idx])
            seq_len = target_len - rejected
            seq_len = min(seq_len + 1, max_model_len)
            out_seq_lens[req_idx] = seq_len

    # Pad query_start_loc for CUDA graphs.
    for i in range(max_num_reqs + 1):
        out_query_start_loc[i] = min(i, num_reqs) if i < max_num_reqs + 1 else num_reqs

    # Pad seq_lens for CUDA graphs (positions beyond num_reqs get 0).
    for i in range(num_reqs, max_num_reqs):
        out_seq_lens[i] = 0

    return out_input_ids, out_positions, out_seq_lens, out_query_start_loc


@pytest.mark.parametrize("advance_draft_positions", [True, False])
def test_prepare_decode_inputs_basic(advance_draft_positions: bool) -> None:
    """Prepare decode inputs kernel: basic functionality.

    Verifies that draft tokens become input IDs, and positions/seq_lens
    are updated accordingly when advance_draft_positions is enabled.
    """
    init_device_properties_triton()
    torch.manual_seed(42)

    num_reqs = 3
    max_num_reqs = 6
    max_model_len = 128
    BLOCK_SIZE = 1024

    device = torch.device("npu")

    draft_tokens = torch.randint(0, 32000, (num_reqs,), dtype=torch.int32, device=device)
    target_seq_lens = torch.randint(10, 50, (num_reqs,), dtype=torch.int32, device=device)
    num_rejected = torch.randint(0, 5, (num_reqs,), dtype=torch.int32, device=device)

    buffers = _InputBuffersStub(max_num_reqs, max_model_len, device)
    buffers.positions[:num_reqs] = torch.randint(5, 20, (num_reqs,), dtype=torch.int64, device=device)
    buffers.seq_lens[:num_reqs] = torch.randint(10, 30, (num_reqs,), dtype=torch.int32, device=device)

    input_ids_initial = buffers.input_ids.clone()
    positions_initial = buffers.positions.clone()
    seq_lens_initial = buffers.seq_lens.clone()

    grid = (num_reqs + 1,)
    _prepare_decode_inputs_kernel[grid](
        draft_tokens,
        draft_tokens.stride(0),
        target_seq_lens,
        num_rejected,
        buffers.input_ids,
        buffers.positions,
        buffers.query_start_loc,
        buffers.seq_lens,
        max_model_len,
        max_num_reqs,
        BLOCK_SIZE=BLOCK_SIZE,
        ADVANCE_DRAFT_POSITIONS=advance_draft_positions,
    )
    torch.npu.synchronize()

    expected = _prepare_decode_inputs_cpu(
        draft_tokens.cpu(),
        target_seq_lens.cpu(),
        num_rejected.cpu(),
        input_ids_initial.cpu(),
        positions_initial.cpu(),
        seq_lens_initial.cpu(),
        max_model_len,
        max_num_reqs,
        advance_draft_positions=advance_draft_positions,
    )

    torch.testing.assert_close(buffers.input_ids.cpu(), expected[0], rtol=0, atol=0)
    torch.testing.assert_close(buffers.positions.cpu(), expected[1], rtol=0, atol=0)
    torch.testing.assert_close(buffers.seq_lens.cpu(), expected[2], rtol=0, atol=0)
    torch.testing.assert_close(buffers.query_start_loc.cpu(), expected[3], rtol=0, atol=0)


def test_prepare_decode_inputs_padding() -> None:
    """Prepare decode inputs: padding for CUDA graphs.

    The last program (idx == num_reqs) should fill query_start_loc and
    pad seq_lens for the remaining slots.
    """
    init_device_properties_triton()

    num_reqs = 2
    max_num_reqs = 5
    max_model_len = 64
    BLOCK_SIZE = 1024

    device = torch.device("npu")

    draft_tokens = torch.randint(100, 200, (num_reqs,), dtype=torch.int32, device=device)
    target_seq_lens = torch.tensor([20, 30], dtype=torch.int32, device=device)
    num_rejected = torch.tensor([1, 2], dtype=torch.int32, device=device)

    buffers = _InputBuffersStub(max_num_reqs, max_model_len, device)
    buffers.positions[:num_reqs] = torch.tensor([10, 15], dtype=torch.int64, device=device)
    buffers.seq_lens[:num_reqs] = torch.tensor([15, 20], dtype=torch.int32, device=device)

    grid = (num_reqs + 1,)
    _prepare_decode_inputs_kernel[grid](
        draft_tokens,
        draft_tokens.stride(0),
        target_seq_lens,
        num_rejected,
        buffers.input_ids,
        buffers.positions,
        buffers.query_start_loc,
        buffers.seq_lens,
        max_model_len,
        max_num_reqs,
        BLOCK_SIZE=BLOCK_SIZE,
        ADVANCE_DRAFT_POSITIONS=True,
    )
    torch.npu.synchronize()

    # query_start_loc: first num_reqs entries = [0, 1, ..., num_reqs-1],
    # remaining = num_reqs (for CUDA graph padding)
    expected_qsl = torch.full((max_num_reqs + 1,), num_reqs, dtype=torch.int32)
    for i in range(num_reqs):
        expected_qsl[i] = i
    torch.testing.assert_close(
        buffers.query_start_loc.cpu(), expected_qsl, rtol=0, atol=0
    )

    # seq_lens: padded entries should be 0
    assert (buffers.seq_lens[num_reqs:].cpu() == 0).all()


# ---------------------------------------------------------------------------
# _update_draft_inputs_kernel tests
# ---------------------------------------------------------------------------


def test_update_draft_inputs_basic() -> None:
    """Update draft inputs kernel: basic functionality.

    Verifies that the sampled draft token is written to the output tensor,
    input_ids are updated, hidden states are copied, and positions/seq_lens
    are advanced.
    """
    init_device_properties_triton()
    torch.manual_seed(42)

    num_reqs = 2
    hidden_size = 16
    max_model_len = 128
    num_speculative_steps = 4
    BLOCK_SIZE = 1024

    device = torch.device("npu")

    output_draft_tokens = torch.zeros(
        num_reqs, num_speculative_steps, dtype=torch.int32, device=device
    )
    next_input_hidden_states = torch.zeros(
        num_reqs, hidden_size, dtype=torch.float32, device=device
    )
    input_ids = torch.randint(0, 100, (num_reqs,), dtype=torch.int32, device=device)
    positions = torch.randint(5, 20, (num_reqs,), dtype=torch.int64, device=device)
    seq_lens = torch.randint(15, 25, (num_reqs,), dtype=torch.int32, device=device)
    draft_tokens = torch.randint(0, 32000, (num_reqs,), dtype=torch.int32, device=device)
    current_draft_step = torch.tensor([1], dtype=torch.int32, device=device)
    hidden_states = torch.randn(num_reqs, hidden_size, dtype=torch.float32, device=device)

    input_ids_initial = input_ids.clone()
    positions_initial = positions.clone()
    seq_lens_initial = seq_lens.clone()

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
        BLOCK_SIZE=BLOCK_SIZE,
        ADVANCE_DRAFT_POSITIONS=True,
    )
    torch.npu.synchronize()

    # Verify draft token was written at the correct step.
    step = int(current_draft_step[0])
    for req_idx in range(num_reqs):
        expected_token = int(draft_tokens[req_idx])
        assert int(output_draft_tokens[req_idx, step]) == expected_token, (
            f"req {req_idx}, step {step}: got {int(output_draft_tokens[req_idx, step])}"
        )

    # Verify hidden states were copied.
    torch.testing.assert_close(
        next_input_hidden_states.cpu(), hidden_states.cpu(), rtol=1e-5, atol=1e-5
    )

    # Verify input_ids updated to draft token.
    for req_idx in range(num_reqs):
        assert int(input_ids[req_idx]) == int(draft_tokens[req_idx])

    # Verify positions advanced by 1.
    expected_pos = positions_initial.cpu() + 1
    expected_pos = torch.clamp(expected_pos, max=max_model_len - 1)
    torch.testing.assert_close(positions.cpu(), expected_pos, rtol=0, atol=0)

    # Verify seq_lens advanced by 1.
    expected_sl = seq_lens_initial.cpu() + 1
    expected_sl = torch.clamp(expected_sl, max=max_model_len)
    torch.testing.assert_close(seq_lens.cpu(), expected_sl, rtol=0, atol=0)


def test_update_draft_inputs_final_step() -> None:
    """Update draft inputs: final spec step should skip advance.

    When step >= num_speculative_steps - 1, the kernel should only write
    the draft token to the output and skip all other updates.
    """
    init_device_properties_triton()

    num_reqs = 2
    hidden_size = 8
    max_model_len = 128
    num_speculative_steps = 3
    BLOCK_SIZE = 1024

    device = torch.device("npu")

    output_draft_tokens = torch.zeros(
        num_reqs, num_speculative_steps, dtype=torch.int32, device=device
    )
    next_input_hidden_states = torch.full(
        (num_reqs, hidden_size), -1.0, dtype=torch.float32, device=device
    )
    input_ids = torch.tensor([99, 99], dtype=torch.int32, device=device)
    positions = torch.tensor([5, 10], dtype=torch.int64, device=device)
    seq_lens = torch.tensor([10, 20], dtype=torch.int32, device=device)
    draft_tokens = torch.tensor([42, 77], dtype=torch.int32, device=device)
    current_draft_step = torch.tensor([2], dtype=torch.int32, device=device)
    hidden_states = torch.randn(num_reqs, hidden_size, dtype=torch.float32, device=device)

    expected_input_ids = input_ids.clone()
    expected_positions = positions.clone()
    expected_seq_lens = seq_lens.clone()
    expected_next_hidden = next_input_hidden_states.clone()

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
        BLOCK_SIZE=BLOCK_SIZE,
        ADVANCE_DRAFT_POSITIONS=True,
    )
    torch.npu.synchronize()

    # Draft token was written to output at step=2
    for req_idx in range(num_reqs):
        assert int(output_draft_tokens[req_idx, 2]) == int(draft_tokens[req_idx])

    # input_ids, positions, seq_lens, hidden_states should be unchanged
    # because it's the final step.
    torch.testing.assert_close(input_ids.cpu(), expected_input_ids.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(positions.cpu(), expected_positions.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(seq_lens.cpu(), expected_seq_lens.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(
        next_input_hidden_states.cpu(), expected_next_hidden.cpu(), rtol=0, atol=0
    )


def test_update_draft_inputs_no_advance() -> None:
    """Update draft inputs: ADVANCE_DRAFT_POSITIONS=False.

    Verifies that positions and seq_lens are not updated when the flag is
    False.
    """
    init_device_properties_triton()

    num_reqs = 1
    hidden_size = 4
    max_model_len = 128
    num_speculative_steps = 3
    BLOCK_SIZE = 1024

    device = torch.device("npu")

    output_draft_tokens = torch.zeros(
        num_reqs, num_speculative_steps, dtype=torch.int32, device=device
    )
    next_input_hidden_states = torch.full(
        (num_reqs, hidden_size), -1.0, dtype=torch.float32, device=device
    )
    input_ids = torch.tensor([99], dtype=torch.int32, device=device)
    positions = torch.tensor([5], dtype=torch.int64, device=device)
    seq_lens = torch.tensor([10], dtype=torch.int32, device=device)
    draft_tokens = torch.tensor([42], dtype=torch.int32, device=device)
    current_draft_step = torch.tensor([1], dtype=torch.int32, device=device)
    hidden_states = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32, device=device)

    expected_positions = positions.clone()
    expected_seq_lens = seq_lens.clone()

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
        BLOCK_SIZE=BLOCK_SIZE,
        ADVANCE_DRAFT_POSITIONS=False,
    )
    torch.npu.synchronize()

    # Draft token written.
    assert int(output_draft_tokens[0, 1]) == 42
    # input_ids updated.
    assert int(input_ids[0]) == 42
    # Hidden states copied.
    torch.testing.assert_close(next_input_hidden_states.cpu(), hidden_states.cpu(),
                               rtol=1e-5, atol=1e-5)
    # Positions and seq_lens unchanged.
    torch.testing.assert_close(positions.cpu(), expected_positions.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(seq_lens.cpu(), expected_seq_lens.cpu(), rtol=0, atol=0)
