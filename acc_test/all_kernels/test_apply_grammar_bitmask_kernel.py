# vLLM vanilla kernel: _apply_grammar_bitmask_kernel from
# vllm/vllm/v1/worker/gpu/structured_outputs.py

"""
Precision test for _apply_grammar_bitmask_kernel (vanilla vLLM version).

For each bitmask, loads a packed 32-bit-per-word bitmask, unpacks it, and
sets logits to -inf for positions where the bitmask is 0 (blocked tokens).

Kernel signature:
    _apply_grammar_bitmask_kernel(
        logits_ptr,         # [num_logits, vocab_size] fp32
        logits_stride,
        logits_indices_ptr, # [num_bitmasks] int32
        bitmask_ptr,        # [num_bitmasks, padded_vocab//32] int32
        bitmask_stride,
        vocab_size,
        BLOCK_SIZE: tl.constexpr,
    )
"""

import torch
import pytest

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.structured_outputs import _apply_grammar_bitmask_kernel
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _apply_grammar_bitmask_ref(
    logits,
    logits_indices,
    bitmask,
):
    """CPU reference for _apply_grammar_bitmask_kernel."""
    out = logits.clone()
    for bm_idx in range(logits_indices.shape[0]):
        logits_idx = int(logits_indices[bm_idx].item())
        for v in range(out.shape[1]):
            word_idx = v // 32
            bit_idx = v % 32
            if word_idx < bitmask.shape[1]:
                packed = int(bitmask[bm_idx, word_idx])
                blocked = ((packed >> bit_idx) & 1) == 0
                if blocked:
                    out[logits_idx, v] = float("-inf")
    return out


class TestApplyGrammarBitmaskKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")
        self.BLOCK_SIZE = 8192

    @pytest.mark.parametrize("vocab_size", [128, 1024, 8192])
    def test_basic_bitmask(self, vocab_size):
        """Verify bitmask correctly blocks tokens."""
        num_bitmasks = 2
        num_logits = 4

        logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=self.device)
        logits_indices = torch.tensor([0, 2], dtype=torch.int32, device=self.device)

        padded_vocab_words = triton.cdiv(vocab_size, 32)
        bitmask = torch.full((num_bitmasks, padded_vocab_words), -1, dtype=torch.int32, device=self.device)

        # Block first half of vocabulary for first bitmask
        half = vocab_size // 2
        for v in range(half):
            w, b = v // 32, v % 32
            bitmask[0, w] &= ~(1 << b)

        logits_copy = logits.clone().cpu()

        num_blocks = triton.cdiv(vocab_size, self.BLOCK_SIZE)
        _apply_grammar_bitmask_kernel[(num_bitmasks, num_blocks)](
            logits,
            logits.stride(0),
            logits_indices,
            bitmask,
            bitmask.stride(0),
            vocab_size,
            BLOCK_SIZE=self.BLOCK_SIZE,
        )
        torch.npu.synchronize()

        expected = _apply_grammar_bitmask_ref(
            logits_copy, logits_indices.cpu(), bitmask.cpu(),
        )

        torch.testing.assert_close(logits.cpu(), expected, rtol=0, atol=0)

    @pytest.mark.parametrize("vocab_size", [128, 512, 4096])
    def test_all_allowed(self, vocab_size):
        """When all bits are 1 (allowed), logits should be unchanged."""
        num_bitmasks = 1
        num_logits = 2

        logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=self.device)
        logits_indices = torch.tensor([0], dtype=torch.int32, device=self.device)

        padded_vocab_words = triton.cdiv(vocab_size, 32)
        bitmask = torch.full((num_bitmasks, padded_vocab_words), -1, dtype=torch.int32, device=self.device)

        logits_copy = logits.clone().cpu()

        num_blocks = triton.cdiv(vocab_size, self.BLOCK_SIZE)
        _apply_grammar_bitmask_kernel[(num_bitmasks, num_blocks)](
            logits,
            logits.stride(0),
            logits_indices,
            bitmask,
            bitmask.stride(0),
            vocab_size,
            BLOCK_SIZE=self.BLOCK_SIZE,
        )
        torch.npu.synchronize()

        torch.testing.assert_close(logits.cpu(), logits_copy, rtol=0, atol=0)

    def test_all_blocked(self):
        """When all bits are 0 (blocked), logits should all be -inf."""
        vocab_size = 256
        num_bitmasks = 1
        num_logits = 1

        logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=self.device)
        logits_indices = torch.tensor([0], dtype=torch.int32, device=self.device)

        padded_vocab_words = triton.cdiv(vocab_size, 32)
        bitmask = torch.zeros((num_bitmasks, padded_vocab_words), dtype=torch.int32, device=self.device)

        num_blocks = triton.cdiv(vocab_size, self.BLOCK_SIZE)
        _apply_grammar_bitmask_kernel[(num_bitmasks, num_blocks)](
            logits,
            logits.stride(0),
            logits_indices,
            bitmask,
            bitmask.stride(0),
            vocab_size,
            BLOCK_SIZE=self.BLOCK_SIZE,
        )
        torch.npu.synchronize()

        assert torch.all(logits[0] == float("-inf")).item(), \
            "All logits should be -inf when bitmask is all zeros"

    def test_multiple_bitmasks_same_logits_row(self):
        """Multiple bitmasks can target different logits rows."""
        vocab_size = 64
        num_logits = 2
        num_bitmasks = 2

        logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=self.device)
        logits_indices = torch.tensor([0, 0], dtype=torch.int32, device=self.device)

        padded_vocab_words = triton.cdiv(vocab_size, 32)
        bitmask = torch.full((num_bitmasks, padded_vocab_words), -1, dtype=torch.int32, device=self.device)
        # Block odd positions in first bitmask
        for v in range(1, vocab_size, 2):
            w, b = v // 32, v % 32
            bitmask[0, w] &= ~(1 << b)
        # Block even positions in second bitmask
        for v in range(0, vocab_size, 2):
            w, b = v // 32, v % 32
            bitmask[1, w] &= ~(1 << b)

        logits_copy = logits.clone().cpu()

        num_blocks = triton.cdiv(vocab_size, self.BLOCK_SIZE)
        _apply_grammar_bitmask_kernel[(num_bitmasks, num_blocks)](
            logits,
            logits.stride(0),
            logits_indices,
            bitmask,
            bitmask.stride(0),
            vocab_size,
            BLOCK_SIZE=self.BLOCK_SIZE,
        )
        torch.npu.synchronize()

        expected = _apply_grammar_bitmask_ref(
            logits_copy, logits_indices.cpu(), bitmask.cpu(),
        )

        torch.testing.assert_close(logits.cpu(), expected, rtol=0, atol=0)
