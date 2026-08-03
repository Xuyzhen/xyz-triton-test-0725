# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.gpu.sample.penalties import _bincount_kernel

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _bincount_cpu(
    expanded_idx_mapping: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    prefill_len: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
) -> None:
    """Pure PyTorch CPU reference implementation of bincount.

    For each request state:
      - Sets prompt bits for all prompt tokens (bitmap).
      - Increments output counts for tokens in the prefill window
        beyond prompt_len.
    """
    prompt_bin_mask = prompt_bin_mask.clone()
    output_bin_counts = output_bin_counts.clone()
    max_num_reqs = prompt_bin_mask.shape[0]

    for token_idx in range(len(expanded_idx_mapping)):
        req_state_idx = int(expanded_idx_mapping[token_idx])
        plen = int(prompt_len[req_state_idx])
        prefill = int(prefill_len[req_state_idx])

        # Reset.
        prompt_bin_mask[req_state_idx].zero_()
        output_bin_counts[req_state_idx].zero_()

        # Prompt tokens: set bits.
        for pos in range(plen):
            token = int(all_token_ids[req_state_idx, pos])
            bin_idx = token // 32
            bit_idx = token % 32
            prompt_bin_mask[req_state_idx, bin_idx] |= 1 << bit_idx

        # Output tokens from prefill window beyond prompt length.
        for pos in range(plen, prefill):
            token = int(all_token_ids[req_state_idx, pos])
            output_bin_counts[req_state_idx, token] += 1

    return prompt_bin_mask, output_bin_counts


@pytest.mark.parametrize("vocab_size", [256, 1024, 32000])
def test_bincount_kernel_basic(vocab_size: int) -> None:
    """Bincount kernel: count prompt / output token occurrences.

    Verifies the kernel correctly builds the prompt bin mask bitmap and
    accumulates output bin counts for each request state.
    """
    init_device_properties_triton()
    torch.manual_seed(42)

    num_tokens = 3
    max_num_reqs = 3
    max_prefill_len = 32
    BLOCK_SIZE = 1024
    num_blocks = (max_prefill_len + BLOCK_SIZE - 1) // BLOCK_SIZE  # 1

    device = torch.device("npu")

    expanded_idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int32)
    prompt_len = torch.tensor([10, 5, 20], dtype=torch.int32)
    prefill_len = torch.tensor([15, 10, 25], dtype=torch.int32)
    all_token_ids = torch.randint(
        0, vocab_size, (max_num_reqs, max_prefill_len), dtype=torch.int32
    )

    prompt_bin_size = (vocab_size + 31) // 32
    prompt_bin_mask = torch.zeros(
        max_num_reqs, prompt_bin_size, dtype=torch.int32, device=device
    )
    output_bin_counts = torch.zeros(
        max_num_reqs, vocab_size, dtype=torch.int32, device=device
    )

    _bincount_kernel[(num_tokens, num_blocks)](
        expanded_idx_mapping.to(device),
        all_token_ids.to(device),
        all_token_ids.stride(0),
        prompt_len.to(device),
        prefill_len.to(device),
        prompt_bin_mask,
        prompt_bin_mask.stride(0),
        output_bin_counts,
        output_bin_counts.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    expected_mask, expected_counts = _bincount_cpu(
        expanded_idx_mapping, all_token_ids, prompt_len, prefill_len,
        torch.zeros_like(prompt_bin_mask.cpu()),
        torch.zeros_like(output_bin_counts.cpu()),
    )

    torch.testing.assert_close(
        prompt_bin_mask.cpu(), expected_mask, rtol=0, atol=0
    )
    torch.testing.assert_close(
        output_bin_counts.cpu(), expected_counts, rtol=0, atol=0
    )


def test_bincount_empty_prompt() -> None:
    """Bincount with zero prompt length (prompt_len == 0).

    No prompt bits should be set, and output counts should be populated
    for all tokens in the prefill window.
    """
    init_device_properties_triton()
    torch.manual_seed(7)

    num_tokens = 1
    max_num_reqs = 1
    vocab_size = 128
    max_prefill_len = 16
    BLOCK_SIZE = 1024
    num_blocks = (max_prefill_len + BLOCK_SIZE - 1) // BLOCK_SIZE  # 1

    device = torch.device("npu")

    expanded_idx_mapping = torch.tensor([0], dtype=torch.int32)
    prompt_len = torch.tensor([0], dtype=torch.int32)
    prefill_len = torch.tensor([8], dtype=torch.int32)
    all_token_ids = torch.randint(
        0, vocab_size, (max_num_reqs, max_prefill_len), dtype=torch.int32
    )

    prompt_bin_size = (vocab_size + 31) // 32
    prompt_bin_mask = torch.zeros(
        max_num_reqs, prompt_bin_size, dtype=torch.int32, device=device
    )
    output_bin_counts = torch.zeros(
        max_num_reqs, vocab_size, dtype=torch.int32, device=device
    )

    _bincount_kernel[(num_tokens, num_blocks)](
        expanded_idx_mapping.to(device),
        all_token_ids.to(device),
        all_token_ids.stride(0),
        prompt_len.to(device),
        prefill_len.to(device),
        prompt_bin_mask,
        prompt_bin_mask.stride(0),
        output_bin_counts,
        output_bin_counts.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    # Prompt bin mask should be all zeros.
    assert prompt_bin_mask.cpu().sum().item() == 0

    # Output counts should sum to prefill_len (all tokens are output).
    actual_sum = int(output_bin_counts.sum().item())
    assert actual_sum == int(prefill_len[0]) - int(prompt_len[0])


def test_bincount_multiple_blocks() -> None:
    """Bincount kernel with multiple grid blocks.

    Verifies correctness when more than one BLOCK_SIZE worth of work
    exists (prefill_len > BLOCK_SIZE).
    """
    init_device_properties_triton()
    torch.manual_seed(99)

    vocab_size = 256
    num_tokens = 1
    max_num_reqs = 1
    max_prefill_len = 4096
    BLOCK_SIZE = 1024
    num_blocks = (max_prefill_len + BLOCK_SIZE - 1) // BLOCK_SIZE  # 4

    device = torch.device("npu")

    expanded_idx_mapping = torch.tensor([0], dtype=torch.int32)
    prompt_len = torch.tensor([100], dtype=torch.int32)
    prefill_len = torch.tensor([3000], dtype=torch.int32)
    all_token_ids = torch.randint(
        0, vocab_size, (max_num_reqs, max_prefill_len), dtype=torch.int32
    )

    prompt_bin_size = (vocab_size + 31) // 32
    prompt_bin_mask = torch.zeros(
        max_num_reqs, prompt_bin_size, dtype=torch.int32, device=device
    )
    output_bin_counts = torch.zeros(
        max_num_reqs, vocab_size, dtype=torch.int32, device=device
    )

    _bincount_kernel[(num_tokens, num_blocks)](
        expanded_idx_mapping.to(device),
        all_token_ids.to(device),
        all_token_ids.stride(0),
        prompt_len.to(device),
        prefill_len.to(device),
        prompt_bin_mask,
        prompt_bin_mask.stride(0),
        output_bin_counts,
        output_bin_counts.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    expected_mask, expected_counts = _bincount_cpu(
        expanded_idx_mapping, all_token_ids, prompt_len, prefill_len,
        torch.zeros_like(prompt_bin_mask.cpu()),
        torch.zeros_like(output_bin_counts.cpu()),
    )

    torch.testing.assert_close(
        prompt_bin_mask.cpu(), expected_mask, rtol=0, atol=0
    )
    torch.testing.assert_close(
        output_bin_counts.cpu(), expected_counts, rtol=0, atol=0
    )
