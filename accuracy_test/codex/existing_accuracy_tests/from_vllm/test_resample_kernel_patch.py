# Accuracy UT source: no direct Ascend kernel UT; adapted from vllm/tests/v1/spec_decode/test_rejection_sampler_utils.py
# vLLM-Ascend patched kernel: _resample_kernel from
# vllm-ascend-xyz/vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py:82
# PATCH NOTE: This is an Ascend NPU adaptation of the original vLLM Triton kernel

"""
Precision test for patched _resample_kernel (Ascend NPU version).

Patch differences vs original vllm:
- Uses _npu_gumbel_block_argmax instead of gumbel_block_argmax for block-level argmax
- APPLY_TEMPERATURE=False is passed to _npu_gumbel_block_argmax (temperature already applied)
- Uses tl.float32 for target_logits cast (explicit npu dtype)

Kernel signature:
    _resample_kernel(
        resampled_local_argmax_ptr,     # [num_reqs, num_blocks] int64 local argmax
        resampled_local_argmax_stride,  # stride(0) of resampled_local_argmax
        resampled_local_max_ptr,        # [num_reqs, num_blocks] fp32 local max
        resampled_local_max_stride,     # stride(0) of resampled_local_max
        target_logits_ptr,              # [num_logits, V] target model logits
        target_logits_stride,           # stride(0) of target_logits
        target_rejected_logsumexp_ptr,  # [num_reqs] target LSE for rejected tokens
        draft_logits_ptr,               # [max_num_reqs, num_spec_steps, V] draft logits
        draft_logits_stride_0,          # stride(0) of draft_logits
        draft_logits_stride_1,          # stride(1) of draft_logits
        draft_rejected_logsumexp_ptr,   # [num_reqs] draft LSE for rejected tokens
        rejected_step_ptr,              # [num_reqs] int32: num accepted steps
        cu_num_logits_ptr,              # [num_reqs+1] cumulative logit counts
        expanded_idx_mapping_ptr,       # [num_logits] logit_idx -> req_state_idx
        draft_sampled_ptr,              # [num_logits] draft sampled token IDs
        temp_ptr,                       # [max_num_reqs] temperatures
        seed_ptr,                       # [max_num_reqs] seeds
        pos_ptr,                        # [num_logits] positions
        vocab_size,                     # scalar: vocab size
        BLOCK_SIZE: tl.constexpr,       # block size (1024)
        HAS_DRAFT_LOGITS: tl.constexpr, # whether draft logits are available
    )

Resamples tokens for rejected/bonus positions in speculative decoding.
When temp=0 (greedy) and not a bonus token, early-returns.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

import pytest


class TestResampleKernelPatch:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")
        self.BLOCK_SIZE = 1024

    def test_greedy_bonus_token(self):
        """For a bonus token with temp=0, resample with Gumbel noise."""
        from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import _resample_kernel

        num_reqs = 1
        vocab_size = 512
        num_blocks = triton.cdiv(vocab_size, self.BLOCK_SIZE)
        padded_resample_num_blocks = triton.next_power_of_2(num_blocks)

        # num_logits = 2 tokens (req0 has 2 tokens, last is bonus)
        num_logits = 2

        resampled_local_argmax = torch.zeros(num_reqs, padded_resample_num_blocks, dtype=torch.int64, device=self.device)
        resampled_local_max = torch.zeros(num_reqs, padded_resample_num_blocks, dtype=torch.float32, device=self.device)

        target_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=self.device)

        target_rejected_logsumexp = torch.zeros(num_reqs, dtype=torch.float32, device=self.device)

        draft_logits = target_logits.new_empty(1, 1, 1)
        draft_rejected_logsumexp = torch.zeros(num_reqs, dtype=torch.float32, device=self.device)

        # Rejected step = 1 means the bonus (last) token was rejected
        rejected_step = torch.ones(num_reqs, dtype=torch.int32, device=self.device)

        # cum counts: req0 has 2 logits (0 and 1; bonus is index 1)
        cu_num_logits = torch.tensor([0, 2], dtype=torch.int32, device=self.device)
        expanded_idx_mapping = torch.zeros(num_logits, dtype=torch.int32, device=self.device)

        draft_sampled = torch.zeros(num_logits, dtype=torch.int32, device=self.device)
        temp = torch.tensor([0.0], dtype=torch.float32, device=self.device)
        seed = torch.tensor([42], dtype=torch.int64, device=self.device)
        pos = torch.tensor([0, 1], dtype=torch.int64, device=self.device)

        _resample_kernel[(num_reqs, num_blocks)](
            resampled_local_argmax,
            resampled_local_argmax.stride(0),
            resampled_local_max,
            resampled_local_max.stride(0),
            target_logits,
            target_logits.stride(0),
            target_rejected_logsumexp,
            draft_logits,
            draft_logits.stride(0),
            draft_logits.stride(1),
            draft_rejected_logsumexp,
            rejected_step,
            cu_num_logits,
            expanded_idx_mapping,
            draft_sampled,
            temp,
            seed,
            pos,
            vocab_size,
            BLOCK_SIZE=self.BLOCK_SIZE,
            HAS_DRAFT_LOGITS=False,
        )
        torch.npu.synchronize()

        # Verify results: local_argmax should have valid token IDs
        for b in range(num_blocks):
            token_id = resampled_local_argmax[0, b].item()
            start = b * self.BLOCK_SIZE
            end = min(start + self.BLOCK_SIZE, vocab_size)
            assert start <= token_id < end, \
                f"Token ID {token_id} out of range [{start}, {end})"

    def test_non_bonus_greedy(self):
        """For non-bonus greedy (temp=0), the kernel should early-return (no-op)."""
        from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import _resample_kernel

        num_reqs = 1
        vocab_size = 128
        num_blocks = triton.cdiv(vocab_size, self.BLOCK_SIZE)
        num_logits = 2  # 2 tokens, non-bonus

        resampled_local_argmax = torch.full((1, 1), -1, dtype=torch.int64, device=self.device)
        resampled_local_max = torch.full((1, 1), -999.0, dtype=torch.float32, device=self.device)

        target_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=self.device)
        target_rejected_logsumexp = torch.zeros(num_reqs, dtype=torch.float32, device=self.device)
        draft_logits = target_logits.new_empty(1, 1, 1)
        draft_rejected_logsumexp = torch.zeros(num_reqs, dtype=torch.float32, device=self.device)

        # rejected_step = 0 => resample idx 0, which is start_idx + 0 = first token (not bonus since end_idx is 2)
        rejected_step = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        cu_num_logits = torch.tensor([0, 2], dtype=torch.int32, device=self.device)
        expanded_idx_mapping = torch.zeros(num_logits, dtype=torch.int32, device=self.device)
        draft_sampled = torch.zeros(num_logits, dtype=torch.int32, device=self.device)
        temp = torch.tensor([0.0], dtype=torch.float32, device=self.device)
        seed = torch.tensor([42], dtype=torch.int64, device=self.device)
        pos = torch.tensor([0, 1], dtype=torch.int64, device=self.device)

        argmax_before = resampled_local_argmax.clone()
        max_before = resampled_local_max.clone()

        _resample_kernel[(num_reqs, num_blocks)](
            resampled_local_argmax,
            resampled_local_argmax.stride(0),
            resampled_local_max,
            resampled_local_max.stride(0),
            target_logits,
            target_logits.stride(0),
            target_rejected_logsumexp,
            draft_logits,
            draft_logits.stride(0),
            draft_logits.stride(1),
            draft_rejected_logsumexp,
            rejected_step,
            cu_num_logits,
            expanded_idx_mapping,
            draft_sampled,
            temp,
            seed,
            pos,
            vocab_size,
            BLOCK_SIZE=self.BLOCK_SIZE,
            HAS_DRAFT_LOGITS=False,
        )
        torch.npu.synchronize()

        # Greedy non-bonus: early return, outputs unchanged
        torch.testing.assert_close(resampled_local_argmax, argmax_before, rtol=0, atol=0)
        torch.testing.assert_close(resampled_local_max, max_before, rtol=0, atol=0)
