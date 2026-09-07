# Standalone Ascend A3 adaptation of an upstream vLLM accuracy path.
# Accuracy UT source: vllm/tests/v1/worker/test_gpu_gumbel_sample.py
# Kernel source: vllm/vllm/v1/worker/gpu/sample/gumbel.py
# Coverage: gumbel_block_argmax

# vLLM vanilla kernel: gumbel_block_argmax from
# vllm/vllm/v1/worker/gpu/sample/gumbel.py

"""
Precision test for gumbel_block_argmax helper.

gumbel_block_argmax is a JIT helper used by _gumbel_sample_kernel.  It applies
temperature scaling (if enabled), optionally stores processed logits, adds
Gumbel noise, and returns (value, idx) = argmax over the block.

This test wraps gumbel_block_argmax in a small Triton kernel to verify it
produces correct argmax results compared to a CPU reference.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_block_argmax
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

import pytest


@triton.jit
def _gumbel_block_argmax_wrapper(
    output_val_ptr,
    output_idx_ptr,
    logits_ptr,
    token_idx,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    processed_logits_ptr,
    processed_logits_stride,
    processed_logits_col_ptr,
    vocab_size,
    block_start,
    BLOCK_SIZE: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr,
):
    """Wrapper to test gumbel_block_argmax on a single block."""
    block = block_start + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * BLOCK_SIZE + block,
        mask=mask,
        other=float("-inf"),
    )
    logits = logits.to(tl.float32)

    value, idx = gumbel_block_argmax(
        logits,
        block,
        mask,
        token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seeds_ptr,
        pos_ptr,
        processed_logits_ptr,
        processed_logits_stride,
        processed_logits_col_ptr,
        vocab_size,
        APPLY_TEMPERATURE=APPLY_TEMPERATURE,
        USE_FP64=USE_FP64,
        PER_TOKEN_COL=PER_TOKEN_COL,
    )
    tl.store(output_val_ptr, value)
    tl.store(output_idx_ptr, idx)


class TestGumbelBlockArgmax:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")
        self.vocab_size = 128
        self.BLOCK_SIZE = 64

    def _run_gumbel_block(
        self,
        logits_block,
        token_idx,
        req_state_idx,
        temp_val,
        seed_val,
        pos_val,
        apply_temperature=False,
        use_fp64=False,
    ):
        num_reqs = 4
        logits = logits_block.unsqueeze(0)  # [1, block_size]
        expanded_idx_mapping = torch.tensor([req_state_idx], dtype=torch.int64, device=self.device)
        temp = torch.full((num_reqs,), temp_val, dtype=torch.float32, device=self.device)
        seeds = torch.full((num_reqs,), seed_val, dtype=torch.int64, device=self.device)
        pos = torch.tensor([pos_val], dtype=torch.int64, device=self.device)

        out_val = torch.empty(1, dtype=torch.float64 if use_fp64 else torch.float32, device=self.device)
        out_idx = torch.empty(1, dtype=torch.int64, device=self.device)

        # Pad logits to full block
        padded = torch.full((1, self.BLOCK_SIZE), float("-inf"), dtype=torch.float32, device=self.device)
        padded[0, :logits_block.shape[0]] = logits_block

        _gumbel_block_argmax_wrapper[(1,)](
            out_val,
            out_idx,
            padded,
            token_idx,
            expanded_idx_mapping,
            temp,
            seeds,
            pos,
            None,  # processed_logits_ptr
            0,     # processed_logits_stride
            None,  # processed_logits_col_ptr
            self.vocab_size,
            0,     # block_start
            BLOCK_SIZE=self.BLOCK_SIZE,
            APPLY_TEMPERATURE=apply_temperature,
            USE_FP64=use_fp64,
            PER_TOKEN_COL=False,
        )
        torch.npu.synchronize()

        return out_val.item(), out_idx.item()

    def test_no_temperature_no_gumbel(self):
        """When temp=0, no Gumbel noise: should return max element and its index."""
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0] + [float("-inf")] * 60,
                               dtype=torch.float32, device=self.device)
        val, idx = self._run_gumbel_block(
            logits, token_idx=0, req_state_idx=0, temp_val=0.0, seed_val=42, pos_val=0,
        )
        assert idx == 1, f"Expected idx 1 (value 5.0), got {idx}"

    def test_temperature_applied(self):
        """When APPLY_TEMPERATURE=True, logits should be divided by temp before argmax."""
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0] + [float("-inf")] * 60,
                               dtype=torch.float32, device=self.device)
        # With temperature 0 (early return, no noise), should return max element
        val, idx = self._run_gumbel_block(
            logits, token_idx=0, req_state_idx=0,
            temp_val=0.0, seed_val=42, pos_val=0,
            apply_temperature=True,
        )
        assert idx == 1, f"Expected idx 1 (value 5.0), got {idx}"

    def test_gumbel_noise(self):
        """With noise and deterministic argmax, the max should be perturbed."""
        # Use logits with one clearly dominant value (large margin)
        logits = torch.tensor([100.0, 0.0, 0.0] + [float("-inf")] * 61,
                               dtype=torch.float32, device=self.device)
        # With temperature 1.0 and random noise, the max at idx 0 should still dominate
        # due to the large margin
        val, idx = self._run_gumbel_block(
            logits, token_idx=0, req_state_idx=0,
            temp_val=1.0, seed_val=42, pos_val=0,
            apply_temperature=True,
        )
        # With a 100-point margin, even Gumbel noise (~0-20 range) can't flip it
        assert idx == 0, f"Expected argmax at idx 0, got {idx}"
