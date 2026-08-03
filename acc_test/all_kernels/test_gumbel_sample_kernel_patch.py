# vLLM-Ascend patched kernel: _gumbel_sample_kernel from
# vllm-ascend/vllm_ascend/worker/v2/sample/gumbel.py:85
# PATCH NOTE: This is an Ascend NPU adaptation of the original vLLM Triton kernel

"""
Precision test for patched _gumbel_sample_kernel (Ascend NPU version).

Patch differences vs original vllm:
- Uses do_not_specialize with explicit parameter list (including local_argmax_stride, etc.)
- Uses tl.rand (float32) instead of tl.rand64 / tl_rand64 (float64 not supported on NPU)
- Casts pos to tl.int32 instead of uint64 (NPU umulhi limitation)
- Uses tl.float32 for r instead of tl.float64
- Adds 1e-20 epsilon to log arguments for numerical stability
- Uses tl.where(mask, ...) pattern for masked operations
- Uses single-output tl.argmax with axis=0 instead of return_indices=True
- Uses APPLY_TEMPERATURE constexpr flag
- Outputs processed_logits with mask pattern

Kernel signature:
    _gumbel_sample_kernel(
        local_argmax_ptr,           # [num_tokens, num_blocks] int64 local argmax per block
        local_argmax_stride,        # stride(0) of local_argmax
        local_max_ptr,              # [num_tokens, num_blocks] fp32 local max per block
        local_max_stride,           # stride(0) of local_max
        processed_logits_ptr,       # optional [max_num_reqs, ...] processed logits output
        processed_logits_stride,    # stride(0) of processed_logits
        processed_logits_col_ptr,   # optional column index into processed_logits
        logits_ptr,                 # [num_tokens, vocab_size] input logits
        logits_stride,              # stride(0) of logits
        expanded_idx_mapping_ptr,   # [num_tokens] token->request mapping
        seeds_ptr,                  # [max_num_reqs] random seeds
        pos_ptr,                    # [num_tokens] positions for seeding
        temp_ptr,                   # [max_num_reqs] temperatures
        vocab_size,                 # scalar: vocab size
        BLOCK_SIZE: tl.constexpr,   # block size (1024)
        APPLY_TEMPERATURE: tl.constexpr,  # whether to apply temperature
    )
"""

import torch

from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

import pytest


def _gumbel_sample_ref(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    apply_temperature: bool,
    vocab_size: int,
    BLOCK_SIZE: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference: compute block-level argmax and max with Gumbel noise."""
    import math
    import random

    num_tokens, _ = logits.shape
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)

    local_argmax = torch.empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max = torch.empty(num_tokens, num_blocks, dtype=torch.float32)

    for token_idx in range(num_tokens):
        req_state_idx = expanded_idx_mapping[token_idx].item()
        temp = temperature[req_state_idx].item()
        seed_val = seed[req_state_idx].item()
        pos_val = pos[token_idx].item()

        for block_idx in range(num_blocks):
            start = block_idx * BLOCK_SIZE
            end = min(start + BLOCK_SIZE, vocab_size)
            block_logits = logits[token_idx, start:end].clone().to(torch.float32).numpy()

            # Apply temperature (same logic as kernel)
            if temp != 0.0 and apply_temperature:
                block_logits = block_logits / temp

            # Apply Gumbel noise (simulate with deterministic seed)
            if temp != 0.0:
                rng = random.Random(seed_val + pos_val + block_idx)
                for i in range(len(block_logits)):
                    r = rng.random()  # [0, 1)
                    r = max(r, 1e-20)
                    gumbel_noise = -math.log(-math.log(r) + 1e-20)
                    block_logits[i] = block_logits[i] + gumbel_noise

            # Replace -inf padding
            block_logits_full = block_logits.copy()
            if len(block_logits_full) < BLOCK_SIZE:
                block_logits_full = list(block_logits_full) + [float("-inf")] * (BLOCK_SIZE - len(block_logits_full))

            max_idx = max(range(BLOCK_SIZE), key=lambda i: block_logits_full[i])
            local_argmax[token_idx, block_idx] = start + max_idx
            local_max[token_idx, block_idx] = block_logits_full[max_idx]

    return local_argmax, local_max


class TestGumbelSampleKernelPatch:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")
        self.BLOCK_SIZE = 1024

    def _run_kernel(
        self,
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature,
        output_processed_logits=None,
        output_processed_logits_col=None,
    ):
        from vllm_ascend.worker.v2.sample.gumbel import _gumbel_sample_kernel

        num_tokens, vocab_size = logits.shape
        num_blocks = triton.cdiv(vocab_size, self.BLOCK_SIZE)

        local_argmax = torch.empty(num_tokens, num_blocks, dtype=torch.int64, device=self.device)
        local_max = torch.empty(num_tokens, num_blocks, dtype=torch.float32, device=self.device)

        _gumbel_sample_kernel[(num_tokens, num_blocks)](
            local_argmax,
            local_argmax.stride(0),
            local_max,
            local_max.stride(0),
            output_processed_logits,
            output_processed_logits.stride(0) if output_processed_logits is not None else 0,
            output_processed_logits_col,
            logits,
            logits.stride(0),
            expanded_idx_mapping,
            seed,
            pos,
            temperature,
            vocab_size,
            BLOCK_SIZE=self.BLOCK_SIZE,
            APPLY_TEMPERATURE=apply_temperature,
        )
        torch.npu.synchronize()

        return local_argmax, local_max

    def test_no_temperature_greedy(self):
        """When temp=0, no Gumbel noise: argmax picks the max element in each block."""
        num_tokens, vocab_size = 2, 256
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        temperature = torch.zeros(1, dtype=torch.float32, device=self.device)
        seed = torch.zeros(1, dtype=torch.int64, device=self.device)
        pos = torch.zeros(num_tokens, dtype=torch.int64, device=self.device)

        local_argmax_gpu, local_max_gpu = self._run_kernel(
            logits, expanded_idx_mapping, temperature, seed, pos,
            apply_temperature=True,
        )

        # CPU check: without noise, argmax should match
        for token_idx in range(num_tokens):
            for block_idx in range(triton.cdiv(vocab_size, self.BLOCK_SIZE)):
                start = block_idx * self.BLOCK_SIZE
                end = min(start + self.BLOCK_SIZE, vocab_size)
                block = logits[token_idx, start:end]
                cpu_max_idx = start + torch.argmax(block).item()
                assert local_argmax_gpu[token_idx, block_idx].item() == cpu_max_idx, \
                    f"Argmax mismatch at token {token_idx}, block {block_idx}"

    def test_with_temperature_and_gumbel(self):
        """With temp != 0, Gumbel noise is added, and argmax should still be valid indices."""
        num_tokens, vocab_size = 2, 256
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
        temperature = torch.tensor([1.0, 1.0], dtype=torch.float32, device=self.device)
        seed = torch.tensor([42, 99], dtype=torch.int64, device=self.device)
        pos = torch.tensor([0, 1], dtype=torch.int64, device=self.device)

        local_argmax_gpu, local_max_gpu = self._run_kernel(
            logits, expanded_idx_mapping, temperature, seed, pos,
            apply_temperature=True,
        )

        num_blocks = triton.cdiv(vocab_size, self.BLOCK_SIZE)

        # Verify all argmax indices are in valid ranges
        for token_idx in range(num_tokens):
            for block_idx in range(num_blocks):
                idx_val = local_argmax_gpu[token_idx, block_idx].item()
                start = block_idx * self.BLOCK_SIZE
                end = min(start + self.BLOCK_SIZE, vocab_size)
                assert start <= idx_val < end, \
                    f"Argmax {idx_val} out of range [{start}, {end})"

    def test_processed_logits_output(self):
        """When processed_logits_ptr is provided, temperature-applied logits are stored."""
        num_tokens, vocab_size = 2, 128
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.tensor([0, 0], dtype=torch.int32, device=self.device)
        max_num_reqs = 1
        temperature = torch.tensor([2.0], dtype=torch.float32, device=self.device)
        seed = torch.tensor([42], dtype=torch.int64, device=self.device)
        pos = torch.zeros(num_tokens, dtype=torch.int64, device=self.device)

        processed_logits = torch.empty(max_num_reqs, vocab_size, dtype=torch.float32, device=self.device)

        self._run_kernel(
            logits, expanded_idx_mapping, temperature, seed, pos,
            apply_temperature=True,
            output_processed_logits=processed_logits,
        )

        # Verify processed_logits = logits / temperature
        expected = (logits / 2.0).cpu()
        torch.testing.assert_close(processed_logits[0].cpu(), expected[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(processed_logits[0].cpu(), expected[1], rtol=1e-5, atol=1e-5)

    def test_gumbel_sample_full_pipeline(self):
        """Test the full gumbel_sample wrapper."""
        from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample

        num_tokens, vocab_size = 4, 512
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=self.device)
        temperature = torch.tensor([1.0, 0.0], dtype=torch.float32, device=self.device)
        seed = torch.tensor([42, 99], dtype=torch.int64, device=self.device)
        pos = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device=self.device)

        sampled = gumbel_sample(
            logits, expanded_idx_mapping, temperature, seed, pos,
            apply_temperature=True,
        )

        assert sampled.shape == (num_tokens,), f"Expected shape ({num_tokens},), got {sampled.shape}"
        assert sampled.dtype == torch.int64, f"Expected int64, got {sampled.dtype}"
        assert torch.all(sampled >= 0).item(), "Sampled tokens must be non-negative"
        assert torch.all(sampled < vocab_size).item(), "Sampled tokens must be in vocab range"
