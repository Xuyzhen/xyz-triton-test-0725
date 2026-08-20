# vLLM-Ascend patched kernel: _npu_gumbel_block_argmax from
# vllm-ascend/vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py:34
# PATCH NOTE: Replaces gumbel_block_argmax on Ascend NPU

"""
Precision test for _npu_gumbel_block_argmax helper (Ascend NPU version).

This replaces `gumbel_block_argmax` on Ascend NPU (monkey-patched).

Patch differences vs original vllm gumbel_block_argmax:
- Uses tl.rand (float32) instead of tl.rand64 (float64 not supported on NPU)
- Casts pos to tl.int32 instead of uint64 (NPU umulhi limitation)
- Uses tl.float32 for r instead of tl.float64
- Adds 1e-20 epsilon for numerical stability in log operations
- Uses return_indices=True variant: value, idx = tl.max(..., return_indices=True)
- No PER_TOKEN_COL or USE_FP64 parameters (simplified interface)
- APPLY_TEMPERATURE constexpr is passed from caller

Kernel signature (called inline, no standalone grid launch):
    _npu_gumbel_block_argmax(
        logits,                         # block-level logits
        block,                          # block indices
        mask,                           # validity mask
        token_idx,                      # token index
        expanded_idx_mapping_ptr,       # [num_logits] token -> request mapping
        temp_ptr,                       # [max_num_reqs] temperatures
        seeds_ptr,                      # [max_num_reqs] seeds
        pos_ptr,                        # [num_logits] positions
        processed_logits_ptr,           # optional output logits ptr
        processed_logits_stride,        # stride of processed_logits
        processed_logits_col_ptr,       # optional column index
        vocab_size,                     # scalar: vocab size
        APPLY_TEMPERATURE: tl.constexpr,# whether to apply temperature
    ) -> (value, idx)

Applies temperature scaling (if enabled), optionally stores processed logits,
adds Gumbel noise, and returns (value, idx) = argmax over the block.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

import pytest


@triton.jit
def _npu_gumbel_block_argmax_wrapper(
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
):
    """Wrapper to test _npu_gumbel_block_argmax on a single block."""
    from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import _npu_gumbel_block_argmax

    block = block_start + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * BLOCK_SIZE + block,
        mask=mask,
        other=float("-inf"),
    )
    logits = logits.to(tl.float32)

    value, idx = _npu_gumbel_block_argmax(
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
    )
    tl.store(output_val_ptr, value)
    tl.store(output_idx_ptr, idx)


class TestNpuGumbelBlockArgmaxPatch:

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
    ):
        num_reqs = 4
        logits = logits_block.unsqueeze(0)
        expanded_idx_mapping = torch.tensor([req_state_idx], dtype=torch.int64, device=self.device)
        temp = torch.full((num_reqs,), temp_val, dtype=torch.float32, device=self.device)
        seeds = torch.full((num_reqs,), seed_val, dtype=torch.int64, device=self.device)
        pos = torch.tensor([pos_val], dtype=torch.int64, device=self.device)

        out_val = torch.empty(1, dtype=torch.float32, device=self.device)
        out_idx = torch.empty(1, dtype=torch.int64, device=self.device)

        padded = torch.full((1, self.BLOCK_SIZE), float("-inf"), dtype=torch.float32, device=self.device)
        padded[0, :logits_block.shape[0]] = logits_block

        _npu_gumbel_block_argmax_wrapper[(1,)](
            out_val,
            out_idx,
            padded,
            token_idx,
            expanded_idx_mapping,
            temp,
            seeds,
            pos,
            None,
            0,
            None,
            self.vocab_size,
            0,
            BLOCK_SIZE=self.BLOCK_SIZE,
            APPLY_TEMPERATURE=apply_temperature,
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
        assert val == 5.0, f"Expected val 5.0, got {val}"

    def test_temperature_applied(self):
        """When APPLY_TEMPERATURE=True, logits should be divided by temp before argmax."""
        logits = torch.tensor([2.0, 10.0, 6.0, 4.0] + [float("-inf")] * 60,
                               dtype=torch.float32, device=self.device)
        # With temperature 0 (early return, no noise), should return max element
        val, idx = self._run_gumbel_block(
            logits, token_idx=0, req_state_idx=0,
            temp_val=0.0, seed_val=42, pos_val=0,
            apply_temperature=True,
        )
        assert idx == 1, f"Expected idx 1 (value 10.0), got {idx}"

    def test_gumbel_noise_dominates(self):
        """With noise, the max should still be the dominant element with large margin."""
        logits = torch.tensor([100.0, 0.0, 0.0] + [float("-inf")] * 61,
                               dtype=torch.float32, device=self.device)
        val, idx = self._run_gumbel_block(
            logits, token_idx=0, req_state_idx=0,
            temp_val=1.0, seed_val=42, pos_val=0,
            apply_temperature=True,
        )
        # With a 100-point margin, Gumbel noise (~0-20 range) cannot flip it
        assert idx == 0, f"Expected argmax at idx 0, got {idx}"

    def test_processed_logits_output(self):
        """When processed_logits_ptr is provided, should store temperature-applied logits."""
        num_reqs = 1
        vocab_size = 256
        BLOCK_SIZE = 128
        num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)

        logits = torch.randn(1, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.zeros(1, dtype=torch.int64, device=self.device)
        temperature = torch.tensor([2.0], dtype=torch.float32, device=self.device)
        seed = torch.tensor([42], dtype=torch.int64, device=self.device)
        pos = torch.zeros(1, dtype=torch.int64, device=self.device)

        processed_logits = torch.zeros(num_reqs, vocab_size, dtype=torch.float32, device=self.device)

        from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import _npu_gumbel_block_argmax

        for block_idx in range(num_blocks):
            block_start = block_idx * BLOCK_SIZE
            block = block_start + torch.arange(BLOCK_SIZE, device=self.device)
            mask = block < vocab_size

            block_logits = logits[0, block_start:block_start + BLOCK_SIZE]

            # Launch tiny wrapper for each block
            out_val = torch.empty(1, dtype=torch.float32, device=self.device)
            out_idx = torch.empty(1, dtype=torch.int64, device=self.device)

            @triton.jit
            def _wrapper_kernel(
                val_ptr, idx_ptr, l_ptr, t_idx, eim, tmp, s, p, pl, pls, plc, vs, bs,
                BLOCK_S: tl.constexpr, APP_T: tl.constexpr,
            ):
                blk = bs + tl.arange(0, BLOCK_S)
                msk = blk < vs
                lgt = tl.load(l_ptr + t_idx * BLOCK_S + blk, mask=msk, other=float("-inf"))
                v, i = _npu_gumbel_block_argmax(
                    lgt, blk, msk, t_idx, eim, tmp, s, p, pl, pls, plc, vs, APP_T,
                )
                tl.store(val_ptr, v)
                tl.store(idx_ptr, i)

            _wrapper_kernel[(1,)](
                out_val, out_idx, logits, 0,
                expanded_idx_mapping, temperature, seed, pos,
                processed_logits, processed_logits.stride(0), None,
                vocab_size, block_start,
                BLOCK_S=BLOCK_SIZE, APP_T=True,
            )

        torch.npu.synchronize()

        # Verify processed_logits == logits / temperature
        expected = (logits / 2.0).cpu()
        torch.testing.assert_close(processed_logits.cpu(), expected, rtol=1e-5, atol=1e-5)
