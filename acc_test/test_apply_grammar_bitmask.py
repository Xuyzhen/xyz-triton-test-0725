# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.gpu.structured_outputs import _apply_grammar_bitmask_kernel

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _apply_grammar_bitmask_cpu(
    logits: torch.Tensor,
    logits_indices: torch.Tensor,
    bitmask: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch CPU reference for apply_grammar_bitmask.

    For each (bitmask_idx, logits_idx) pair:
      1. Load the packed bitmask (each int32 = 32 bits).
      2. Unpack it: bitmask[j] == 0 means the token at position j
         should be masked (set to -inf).
      3. Apply to logits[logits_idx, :].

    Returns the modified logits tensor.
    """
    logits = logits.clone()
    num_masks = len(logits_indices)
    vocab_size = logits.shape[-1]

    for bitmask_idx in range(num_masks):
        logits_idx = int(logits_indices[bitmask_idx])
        row = logits[logits_idx]

        for token_id in range(vocab_size):
            word_idx = token_id // 32
            bit_idx = token_id % 32
            packed = int(bitmask[bitmask_idx, word_idx])
            allowed = ((packed >> bit_idx) & 1) != 0
            if not allowed:
                row[token_id] = float("-inf")

    return logits


def test_apply_grammar_bitmask_basic() -> None:
    """Apply grammar bitmask kernel: basic functionality.

    Verifies that the bitmask correctly masks out disallowed tokens by
    setting their logits to -inf.
    """
    init_device_properties_triton()
    torch.manual_seed(42)

    num_masks = 2
    vocab_size = 64
    BLOCK_SIZE = 8192

    device = torch.device("npu")

    logits = torch.randn(num_masks, vocab_size, dtype=torch.float32, device=device)
    logits_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)

    # Build custom bitmask for each entry.
    # Bitmask shape: [num_masks, ceil(vocab_size / 32)]
    bitmask = torch.zeros(num_masks, (vocab_size + 31) // 32, dtype=torch.int32, device=device)
    # For mask 0: allow tokens 0..31 (first 32 tokens).
    bitmask[0, 0] = 0xFFFFFFFF  # all 32 bits set
    # For mask 1: allow token 16 and 48.
    bitmask[1, 0] = 1 << 16  # token 16
    bitmask[1, 1] = 1 << (48 % 32)  # token 48 inside second word

    expected = _apply_grammar_bitmask_cpu(
        logits.cpu(), logits_indices.cpu(), bitmask.cpu()
    )

    grid = (num_masks, (vocab_size + BLOCK_SIZE - 1) // BLOCK_SIZE)
    _apply_grammar_bitmask_kernel[grid](
        logits,
        logits.stride(0),
        logits_indices,
        bitmask,
        bitmask.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(logits.cpu(), expected, rtol=0, atol=0)


def test_apply_grammar_bitmask_all_allowed() -> None:
    """Apply grammar bitmask: all tokens allowed (full mask).

    When all bits are set, logits should remain unchanged.
    """
    init_device_properties_triton()
    torch.manual_seed(7)

    num_masks = 1
    vocab_size = 32
    BLOCK_SIZE = 8192

    device = torch.device("npu")

    logits_orig = torch.randn(num_masks, vocab_size, dtype=torch.float32)
    logits = logits_orig.clone().to(device)
    logits_indices = torch.tensor([0], dtype=torch.int32, device=device)

    # All bits set.
    bitmask = torch.full(
        (num_masks, (vocab_size + 31) // 32), -1, dtype=torch.int32, device=device
    )

    grid = (num_masks, (vocab_size + BLOCK_SIZE - 1) // BLOCK_SIZE)
    _apply_grammar_bitmask_kernel[grid](
        logits,
        logits.stride(0),
        logits_indices,
        bitmask,
        bitmask.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(logits.cpu(), logits_orig, rtol=0, atol=0)


def test_apply_grammar_bitmask_none_allowed() -> None:
    """Apply grammar bitmask: no tokens allowed (zero mask).

    When no bits are set, all logits should become -inf.
    """
    init_device_properties_triton()

    num_masks = 1
    vocab_size = 32
    BLOCK_SIZE = 8192

    device = torch.device("npu")

    logits = torch.randn(num_masks, vocab_size, dtype=torch.float32, device=device)
    logits_indices = torch.tensor([0], dtype=torch.int32, device=device)

    bitmask = torch.zeros(
        num_masks, (vocab_size + 31) // 32, dtype=torch.int32, device=device
    )

    grid = (num_masks, (vocab_size + BLOCK_SIZE - 1) // BLOCK_SIZE)
    _apply_grammar_bitmask_kernel[grid](
        logits,
        logits.stride(0),
        logits_indices,
        bitmask,
        bitmask.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    assert (logits.cpu() == float("-inf")).all()


@pytest.mark.parametrize("vocab_size", [32, 128, 32000])
def test_apply_grammar_bitmask_varied_vocab(vocab_size: int) -> None:
    """Apply grammar bitmask with different vocabulary sizes.

    Tests edge cases where vocab_size is not a multiple of 32 or 8192.
    """
    init_device_properties_triton()
    torch.manual_seed(42)

    num_masks = 2
    BLOCK_SIZE = 8192

    device = torch.device("npu")

    logits = torch.randn(num_masks, vocab_size, dtype=torch.float32, device=device)
    logits_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)

    bitmask_size = (vocab_size + 31) // 32
    bitmask = torch.randint(0, 2**32, (num_masks, bitmask_size), dtype=torch.int64, device=device).to(dtype=torch.int32)

    expected = _apply_grammar_bitmask_cpu(
        logits.cpu(), logits_indices.cpu(), bitmask.cpu()
    )

    grid = (num_masks, (vocab_size + BLOCK_SIZE - 1) // BLOCK_SIZE)
    _apply_grammar_bitmask_kernel[grid](
        logits,
        logits.stride(0),
        logits_indices,
        bitmask,
        bitmask.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(logits.cpu(), expected, rtol=0, atol=0)
