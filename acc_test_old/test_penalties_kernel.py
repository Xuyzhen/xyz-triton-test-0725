# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.gpu.sample.penalties import _penalties_kernel

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _apply_penalties_cpu(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    token_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
) -> None:
    """Pure PyTorch CPU reference implementation of penalties.

    Applies repetition, frequency, and presence penalties to logits.
    """
    logits = logits.clone()
    num_tokens, vocab_size = logits.shape

    for token_idx in range(num_tokens):
        req_state_idx = int(expanded_idx_mapping[token_idx])
        rep = float(repetition_penalty[req_state_idx])
        freq = float(frequency_penalty[req_state_idx])
        pres = float(presence_penalty[req_state_idx])

        if rep == 1.0 and freq == 0.0 and pres == 0.0:
            continue

        pos = int(expanded_local_pos[token_idx])
        start_idx = token_idx - pos

        # Accumulate draft token counts from previous positions.
        counts = output_bin_counts[req_state_idx].clone()
        for prev_pos in range(pos):
            prev_token = int(token_ids[start_idx + prev_pos + 1])
            counts[prev_token] += 1
        output_mask = counts > 0

        # Repetition penalty.
        if rep != 1.0:
            bin_mask = prompt_bin_mask[req_state_idx]
            for v in range(vocab_size):
                bit_idx = v % 32
                bin_idx = v // 32
                in_prompt = bool((bin_mask[bin_idx] >> bit_idx) & 1)
                should_scale = in_prompt or bool(output_mask[v])
                scale = rep if should_scale else 1.0
                if logits[token_idx, v] > 0:
                    logits[token_idx, v] /= scale
                else:
                    logits[token_idx, v] *= scale

        # Frequency and presence penalties.
        logits[token_idx] -= freq * counts.to(logits.dtype)
        logits[token_idx] -= pres * output_mask.to(logits.dtype)

    return logits


@pytest.mark.parametrize("num_tokens", [1, 4, 8])
def test_penalties_kernel_basic(num_tokens: int) -> None:
    """Penalties kernel: repetition / frequency / presence penalties applied correctly.

    Verifies that the Triton kernel matches the CPU reference for a small
    vocabulary with various penalty configurations.
    """
    init_device_properties_triton()
    torch.manual_seed(42)

    vocab_size = 256
    max_num_reqs = 4
    BLOCK_SIZE = 8192
    num_blocks = (vocab_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    device = torch.device("npu")

    # CPU data for reference.
    expanded_idx_mapping = torch.randint(0, max_num_reqs, (num_tokens,),
                                         dtype=torch.int64)
    token_ids = torch.randint(0, vocab_size, (num_tokens + 8,), dtype=torch.int32)
    expanded_local_pos = torch.zeros(num_tokens, dtype=torch.int64)

    repetition_penalty = torch.full((max_num_reqs,), 1.0, dtype=torch.float32)
    frequency_penalty = torch.zeros(max_num_reqs, dtype=torch.float32)
    presence_penalty = torch.zeros(max_num_reqs, dtype=torch.float32)

    # Apply penalties to different requests.
    repetition_penalty[1] = 1.2
    frequency_penalty[2] = 0.1
    presence_penalty[0] = 0.5

    prompt_bin_size = (vocab_size + 31) // 32
    prompt_bin_mask = torch.zeros(max_num_reqs, prompt_bin_size, dtype=torch.int32)
    # Mark a few tokens as seen in prompt for request 3.
    prompt_bin_mask[3, 0] = (1 << 0) | (1 << 5) | (1 << 10)

    output_bin_counts = torch.zeros(max_num_reqs, vocab_size, dtype=torch.int32)
    # Mark some output tokens for request 1.
    output_bin_counts[1, 10] = 2
    output_bin_counts[1, 20] = 1

    logits_cpu = torch.randn(num_tokens, vocab_size, dtype=torch.float32)

    # CPU reference.
    expected = _apply_penalties_cpu(
        logits_cpu, expanded_idx_mapping, token_ids, expanded_local_pos,
        repetition_penalty, frequency_penalty, presence_penalty,
        prompt_bin_mask, output_bin_counts,
    )

    # NPU kernel.
    logits_npu = logits_cpu.to(device)
    idx_mapping_npu = expanded_idx_mapping.to(device)
    token_ids_npu = token_ids.to(device)
    local_pos_npu = expanded_local_pos.to(device)
    rep_penalty_npu = repetition_penalty.to(device)
    freq_penalty_npu = frequency_penalty.to(device)
    pres_penalty_npu = presence_penalty.to(device)
    prompt_bin_mask_npu = prompt_bin_mask.to(device)
    output_bin_counts_npu = output_bin_counts.to(device)

    _penalties_kernel[(num_tokens, num_blocks)](
        logits_npu,
        logits_npu.stride(0),
        idx_mapping_npu,
        token_ids_npu,
        local_pos_npu,
        rep_penalty_npu,
        freq_penalty_npu,
        pres_penalty_npu,
        prompt_bin_mask_npu,
        prompt_bin_mask_npu.stride(0),
        output_bin_counts_npu,
        output_bin_counts_npu.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(logits_npu.cpu(), expected, rtol=1e-5, atol=1e-5)


def test_penalties_all_noops() -> None:
    """When all penalties are at default values, kernel returns early (no-op).

    Verifies that when rep=1.0, freq=0.0, pres=0.0 for all requests, the
    kernel does not modify logits.
    """
    init_device_properties_triton()
    torch.manual_seed(7)

    num_tokens = 2
    vocab_size = 128
    max_num_reqs = 2
    BLOCK_SIZE = 8192
    num_blocks = (vocab_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    device = torch.device("npu")

    expanded_idx_mapping = torch.tensor([0, 1], dtype=torch.int64)
    token_ids = torch.zeros(num_tokens + 8, dtype=torch.int32)
    expanded_local_pos = torch.zeros(num_tokens, dtype=torch.int64)
    repetition_penalty = torch.ones(max_num_reqs, dtype=torch.float32)
    frequency_penalty = torch.zeros(max_num_reqs, dtype=torch.float32)
    presence_penalty = torch.zeros(max_num_reqs, dtype=torch.float32)
    prompt_bin_size = (vocab_size + 31) // 32
    prompt_bin_mask = torch.zeros(max_num_reqs, prompt_bin_size, dtype=torch.int32)
    output_bin_counts = torch.zeros(max_num_reqs, vocab_size, dtype=torch.int32)
    logits_cpu = torch.randn(num_tokens, vocab_size, dtype=torch.float32)

    logits_npu = logits_cpu.clone().to(device)
    _penalties_kernel[(num_tokens, num_blocks)](
        logits_npu,
        logits_npu.stride(0),
        expanded_idx_mapping.to(device),
        token_ids.to(device),
        expanded_local_pos.to(device),
        repetition_penalty.to(device),
        frequency_penalty.to(device),
        presence_penalty.to(device),
        prompt_bin_mask.to(device),
        prompt_bin_mask.stride(0),
        output_bin_counts.to(device),
        output_bin_counts.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(logits_npu.cpu(), logits_cpu, rtol=0, atol=0)


def test_penalties_with_draft_tokens() -> None:
    """Penalties kernel with draft tokens (expanded_local_pos > 0).

    Verifies that the kernel correctly accumulates draft token counts
    from previous positions when expanded_local_pos is non-zero.
    """
    init_device_properties_triton()
    torch.manual_seed(3)

    num_tokens = 3
    vocab_size = 64
    max_num_reqs = 2
    BLOCK_SIZE = 8192
    num_blocks = (vocab_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    device = torch.device("npu")

    # expanded_local_pos indicates how many draft tokens precede each token.
    expanded_idx_mapping = torch.tensor([0, 0, 1], dtype=torch.int64)
    token_ids = torch.tensor([0, 5, 10, 20, 30, 0, 0, 0, 0, 0, 0],
                             dtype=torch.int32)
    expanded_local_pos = torch.tensor([0, 1, 0], dtype=torch.int64)
    # For token_idx=1: pos=1, start_idx=0, prev_token=token_ids[1]=5.
    # output_bin_counts[0, 5] should increment by 1.

    repetition_penalty = torch.ones(max_num_reqs, dtype=torch.float32)
    frequency_penalty = torch.tensor([0.1, 0.2], dtype=torch.float32)
    presence_penalty = torch.zeros(max_num_reqs, dtype=torch.float32)

    prompt_bin_size = (vocab_size + 31) // 32
    prompt_bin_mask = torch.zeros(max_num_reqs, prompt_bin_size, dtype=torch.int32)
    output_bin_counts = torch.zeros(max_num_reqs, vocab_size, dtype=torch.int32)

    logits_cpu = torch.randn(num_tokens, vocab_size, dtype=torch.float32)

    expected = _apply_penalties_cpu(
        logits_cpu, expanded_idx_mapping, token_ids, expanded_local_pos,
        repetition_penalty, frequency_penalty, presence_penalty,
        prompt_bin_mask, output_bin_counts,
    )

    logits_npu = logits_cpu.to(device)
    _penalties_kernel[(num_tokens, num_blocks)](
        logits_npu,
        logits_npu.stride(0),
        expanded_idx_mapping.to(device),
        token_ids.to(device),
        expanded_local_pos.to(device),
        repetition_penalty.to(device),
        frequency_penalty.to(device),
        presence_penalty.to(device),
        prompt_bin_mask.to(device),
        prompt_bin_mask.stride(0),
        output_bin_counts.to(device),
        output_bin_counts.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(logits_npu.cpu(), expected, rtol=1e-5, atol=1e-5)
