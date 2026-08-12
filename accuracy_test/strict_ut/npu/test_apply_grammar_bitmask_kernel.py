# GENERATED STRICT UT. Source: accuracy_test/codex/missing_accuracy_tests/test_apply_grammar_bitmask_kernel_patch.py
# Do not edit mechanically; update the reviewed Codex source or strict generator.
from accuracy_test.strict_ut.runtime_npu import STRICT_DEVICE as _STRICT_DEVICE
# vLLM-Ascend patched kernel: _apply_grammar_bitmask_kernel from
# vllm-ascend/vllm_ascend/worker/v2/structured_outputs.py:35
# PATCH NOTE: This is an Ascend NPU adaptation of the original vLLM Triton kernel

"""
Precision test for patched _apply_grammar_bitmask_kernel (Ascend NPU version).

Patch differences vs original vllm:
- Uses BLOCK_SIZE_SUB=1024 sub-block tiling to avoid UB overflow with BLOCK_SIZE=8192
- Iterates over sub-blocks with tl.range for NPU compatibility
- Uses packed bitmask with word-level loading (32 bits per word)
- Applies bitmask via ((packed >> bit_idx) & 1) == 0 pattern
- Stores -inf for blocked positions using mask pattern

Kernel signature:
    _apply_grammar_bitmask_kernel(
        logits_ptr,         # [num_logits, vocab_size] fp32 logits
        logits_stride,      # stride(0) of logits
        logits_indices_ptr, # [num_bitmasks] logits index for each bitmask
        bitmask_ptr,        # [num_bitmasks, padded_vocab//32] packed bitmasks
        bitmask_stride,     # stride of bitmask (padded_vocab // 32)
        vocab_size,         # scalar: vocab size
        BLOCK_SIZE: tl.constexpr,   # block size (8192)
    )

For each bitmask, sets logits to -inf for positions where the bitmask is 0
(blocked tokens).
"""

import torch

from vllm.triton_utils import tl, triton
from accuracy_test.strict_ut.runtime_npu import init_device_properties_triton

import pytest


def _apply_grammar_bitmask_ref(
    logits: torch.Tensor,
    logits_indices: torch.Tensor,
    bitmask: torch.Tensor,
) -> torch.Tensor:
    """CPU reference for _apply_grammar_bitmask_kernel."""
    out = logits.clone()
    bitmask_np = bitmask.cpu().numpy()

    for bm_idx, logits_idx in enumerate(logits_indices):
        idx = logits_idx.item()
        for v in range(out.shape[1]):
            word_idx = v // 32
            bit_idx = v % 32
            if word_idx < bitmask.shape[1]:
                packed = bitmask_np[bm_idx, word_idx]
                blocked = ((packed >> bit_idx) & 1) == 0
                if blocked:
                    out[idx, v] = float("-inf")
    return out


class TestApplyGrammarBitmaskKernelPatch:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")
        self.BLOCK_SIZE = 8192
        self.BLOCK_SIZE_SUB = 1024

    def _run_kernel(self, logits, logits_indices, bitmask):
        from vllm_ascend.worker.v2.structured_outputs import _apply_grammar_bitmask_kernel

        num_bitmasks = logits_indices.shape[0]
        vocab_size = logits.shape[1]
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

        self._run_kernel(logits, logits_indices, bitmask)

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

        self._run_kernel(logits, logits_indices, bitmask)

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

        self._run_kernel(logits, logits_indices, bitmask)

        assert torch.all(logits[0] == float("-inf")).item(), \
            "All logits should be -inf when bitmask is all zeros"
