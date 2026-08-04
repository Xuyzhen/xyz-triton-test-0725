# Standalone Ascend A3 adaptation of an upstream vLLM accuracy path.
# Accuracy UT source: vllm/tests/v1/spec_decode/test_rejection_sampler_utils.py
# Kernel source: vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
# Coverage: _resample_kernel

# vLLM vanilla kernel: _resample_kernel from
# vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py

"""
Precision test for _resample_kernel.

Resamples a rejected (or bonus) token from the residual distribution
after speculative decoding rejection.

Kernel signature:
    _resample_kernel(
        resampled_local_argmax_ptr,   # [num_reqs, num_blocks] int64 output
        resampled_local_argmax_stride,
        resampled_local_max_ptr,      # [num_reqs, num_blocks] fp32/fp64 output
        resampled_local_max_stride,
        target_logits_ptr,            # [num_logits, V] fp32
        target_logits_stride,
        target_rejected_logsumexp_ptr,# [num_reqs] fp32
        draft_logits_ptr,             # [max_num_reqs, num_spec_steps, V] fp32 or None
        draft_logits_stride_0,
        draft_logits_stride_1,
        draft_rejected_logsumexp_ptr, # [num_reqs] fp32
        rejected_step_ptr,            # [num_reqs] int64 (num_sampled per req)
        cu_num_logits_ptr,            # [num_reqs+1] int64
        expanded_idx_mapping_ptr,     # [num_logits] int64
        draft_sampled_ptr,            # [num_logits] int64
        temp_ptr,                     # [max_num_reqs] fp32
        seed_ptr,                     # [max_num_reqs] int32
        pos_ptr,                      # [num_logits] int32
        cumulative_log_p_ptr,         # [num_logits] fp32 or None
        vocab_size,
        BLOCK_SIZE: tl.constexpr,
        HAS_DRAFT_LOGITS: tl.constexpr,
        USE_FP64: tl.constexpr,
        USE_BLOCK_VERIFICATION: tl.constexpr,
    )
"""

import torch
import pytest

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _resample_kernel,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


_KERNEL_ARG_NAMES = set(_resample_kernel.arg_names)
_HAS_BLOCK_VERIFICATION = {
    "cumulative_log_p_ptr",
    "USE_BLOCK_VERIFICATION",
}.issubset(_KERNEL_ARG_NAMES)


def _launch_resample(
    grid,
    args_before_optional,
    cumulative_log_p,
    vocab_size,
    block_size,
    has_draft_logits,
    use_block_verification,
):
    """Launch either the legacy or block-verification kernel signature."""
    args = list(args_before_optional)
    if _HAS_BLOCK_VERIFICATION:
        args.append(cumulative_log_p)
    args.append(vocab_size)
    kwargs = {
        "BLOCK_SIZE": block_size,
        "HAS_DRAFT_LOGITS": has_draft_logits,
        "USE_FP64": False,
    }
    if _HAS_BLOCK_VERIFICATION:
        kwargs["USE_BLOCK_VERIFICATION"] = use_block_verification
    _resample_kernel[grid](*args, **kwargs)


def _resample_ref(
    target_logits,          # [num_logits, V]
    target_rejected_lse,    # [num_reqs]
    draft_logits,           # [max_num_reqs, num_spec_steps, V] or None
    draft_rejected_lse,     # [num_reqs]
    rejected_step,          # [num_reqs] num_sampled per req
    cu_num_logits,          # [num_reqs+1]
    expanded_idx_mapping,   # [num_logits]
    draft_sampled,          # [num_logits]
    temp,                   # [max_num_reqs]
    seed,                   # [max_num_reqs]
    pos,                    # [num_logits]
    cumulative_log_p,       # [num_logits] or None
    vocab_size,
    has_draft_logits=False,
    use_block_verification=False,
):
    """
    CPU reference for _resample_kernel.

    For each req, determines the rejected token index and resamples using
    a simple softmax-based resampling from the residual distribution.
    """
    num_reqs = rejected_step.shape[0]
    resample_num_blocks = triton.cdiv(vocab_size, 1024)
    BLOCK_SIZE = 1024

    resampled_local_argmax = torch.zeros(num_reqs, resample_num_blocks, dtype=torch.int64)
    resampled_local_max = torch.zeros(num_reqs, resample_num_blocks, dtype=torch.float32)

    for req_idx in range(num_reqs):
        resample_idx = int(rejected_step[req_idx].item())
        start_idx = int(cu_num_logits[req_idx].item())
        end_idx = int(cu_num_logits[req_idx + 1].item())
        resample_token_idx = start_idx + resample_idx
        req_state_idx = int(expanded_idx_mapping[resample_token_idx].item())

        t = float(temp[req_state_idx].item())
        is_bonus = (resample_token_idx == end_idx - 1)

        if t == 0.0 and not is_bonus:
            # Greedy + non-bonus: already has argmax, skip.
            continue

        def _softmax_logits(logits):
            maxv = torch.max(logits)
            exps = torch.exp(logits - maxv)
            return exps / torch.sum(exps)

        target_row = target_logits[resample_token_idx].float()

        if is_bonus:
            residual_logits = target_row
        elif has_draft_logits:
            draft_row = draft_logits[req_state_idx, resample_idx].float()
            tgt_lse_val = float(target_rejected_lse[req_idx].item())
            drf_lse_val = float(draft_rejected_lse[req_idx].item())
            target_log_probs = target_row - tgt_lse_val
            if use_block_verification:
                log_p_tau = 0.0
                if resample_idx > 0:
                    log_p_tau = float(cumulative_log_p[resample_token_idx - 1].item())
                target_log_probs += log_p_tau
            draft_log_probs = draft_row - drf_lse_val
            ratio = torch.exp(draft_log_probs - target_log_probs)
            residual_logits = torch.where(
                ratio < 1.0,
                target_log_probs + torch.log1p(-ratio),
                torch.tensor(float("-inf")),
            )
        else:
            rejected_draft_token = int(draft_sampled[resample_token_idx + 1].item())
            residual_logits = target_row.clone()
            residual_logits[rejected_draft_token] = float("-inf")

        # PyTorch and Triton use different PRNG streams. Exact comparison is
        # valid only for temperature zero, where the kernel adds no noise.
        if t != 0.0:
            raise ValueError(
                "Exact CPU reference is only valid for deterministic temp=0 cases"
            )
        gumbel_logits = residual_logits

        for block_idx in range(resample_num_blocks):
            block_start = block_idx * BLOCK_SIZE
            block_end = min(block_start + BLOCK_SIZE, vocab_size)
            block_slice = gumbel_logits[block_start:block_end]
            token_offset = torch.argmax(block_slice).item()
            max_val = block_slice[token_offset].item()
            token_id = block_start + token_offset
            resampled_local_argmax[req_idx, block_idx] = token_id
            resampled_local_max[req_idx, block_idx] = max_val

    return resampled_local_argmax, resampled_local_max


class TestResampleKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_reqs", [1, 2, 4])
    @pytest.mark.parametrize("num_spec_steps", [1, 3])
    @pytest.mark.parametrize("has_draft_logits", [False, True])
    @pytest.mark.parametrize("use_block_verification", [False, True])
    def test_resample_basic(self, num_reqs, num_spec_steps, has_draft_logits, use_block_verification):
        """Test resample kernel with simple inputs."""
        vocab_size = 1024
        num_logits = num_reqs * (num_spec_steps + 1)
        max_num_reqs = num_reqs
        BLOCK_SIZE = 1024
        RESAMPLE_BLOCK_SIZE = 1024
        resample_num_blocks = triton.cdiv(vocab_size, RESAMPLE_BLOCK_SIZE)

        # Use bonus positions with temp=0 so no Gumbel noise is added. This
        # permits an exact CPU comparison without assuming identical PRNGs.
        target_logits = torch.randn(
            num_logits, vocab_size, dtype=torch.float32, device=self.device
        )
        cu_num_logits = (
            torch.arange(num_reqs + 1, device=self.device, dtype=torch.int32)
            * (num_spec_steps + 1)
        )
        expanded_idx_mapping = torch.arange(
            num_reqs, device=self.device, dtype=torch.int32
        ).repeat_interleave(num_spec_steps + 1)

        # draught sampled tokens (one per logit + 1)
        draft_sampled = torch.randint(0, vocab_size, (num_logits + 1,), dtype=torch.int64, device=self.device)

        temp = torch.zeros(
            max_num_reqs, dtype=torch.float32, device=self.device
        )
        seed = torch.full(
            (max_num_reqs,), 42, dtype=torch.int32, device=self.device
        )
        pos = torch.arange(num_logits, dtype=torch.int32, device=self.device)

        rejected_step = torch.full(
            (num_reqs,), num_spec_steps, dtype=torch.int64, device=self.device
        )
        target_rejected_lse = torch.zeros(num_reqs, dtype=torch.float32, device=self.device)
        draft_rejected_lse = torch.zeros(num_reqs, dtype=torch.float32, device=self.device)

        draft_logits = None
        if has_draft_logits:
            draft_logits = torch.randn(max_num_reqs, num_spec_steps, vocab_size, dtype=torch.float32, device=self.device)

        if use_block_verification and not _HAS_BLOCK_VERIFICATION:
            pytest.skip(
                "installed vLLM predates block-verification _resample_kernel; "
                "precision was not tested"
            )
        cumulative_log_p = torch.zeros(
            num_logits, dtype=torch.float32, device=self.device
        )

        resampled_local_argmax = torch.zeros(num_reqs, resample_num_blocks, dtype=torch.int64, device=self.device)
        resampled_local_max = torch.zeros(num_reqs, resample_num_blocks, dtype=torch.float32, device=self.device)

        draft_logits_arg = (
            draft_logits
            if draft_logits is not None
            else torch.empty(1, 1, 1, dtype=torch.float32, device=self.device)
        )
        _launch_resample(
            (num_reqs, resample_num_blocks),
            [
                resampled_local_argmax,
                resampled_local_argmax.stride(0),
                resampled_local_max,
                resampled_local_max.stride(0),
                target_logits,
                target_logits.stride(0),
                target_rejected_lse,
                draft_logits_arg,
                draft_logits.stride(0) if draft_logits is not None else 0,
                draft_logits.stride(1) if draft_logits is not None else 0,
                draft_rejected_lse,
                rejected_step,
                cu_num_logits,
                expanded_idx_mapping,
                draft_sampled,
                temp,
                seed,
                pos,
            ],
            cumulative_log_p,
            vocab_size,
            RESAMPLE_BLOCK_SIZE,
            has_draft_logits,
            use_block_verification,
        )
        torch.npu.synchronize()

        # CPU reference
        ref_argmax, ref_max = _resample_ref(
            target_logits.cpu(),
            target_rejected_lse.cpu(),
            draft_logits.cpu() if draft_logits is not None else None,
            draft_rejected_lse.cpu(),
            rejected_step.cpu(),
            cu_num_logits.cpu(),
            expanded_idx_mapping.cpu(),
            draft_sampled.cpu(),
            temp.cpu(),
            seed.cpu(),
            pos.cpu(),
            cumulative_log_p.cpu(),
            vocab_size,
            has_draft_logits=has_draft_logits,
            use_block_verification=use_block_verification,
        )

        # Compare argmax outputs (the actual token IDs selected per block)
        # The resampled_local_argmax stores the block-local argmax token index.
        torch.testing.assert_close(
            resampled_local_argmax.cpu().to(torch.int64),
            ref_argmax.to(torch.int64),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            resampled_local_max.cpu(),
            ref_max,
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.parametrize("has_draft_logits", [False, True])
    def test_greedy_no_resample(self, has_draft_logits):
        """When temperature == 0 and not bonus token, kernel should be a no-op."""
        num_reqs = 2
        num_spec_steps = 2
        vocab_size = 512
        num_logits = num_reqs * (num_spec_steps + 1)
        max_num_reqs = num_reqs
        BLOCK_SIZE = 512
        resample_num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)

        target_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=self.device)
        cu_num_logits = (
            torch.arange(num_reqs + 1, device=self.device, dtype=torch.int32)
            * (num_spec_steps + 1)
        )
        expanded_idx_mapping = torch.arange(
            num_reqs, device=self.device, dtype=torch.int32
        ).repeat_interleave(num_spec_steps + 1)
        draft_sampled = torch.randint(0, vocab_size, (num_logits + 1,), dtype=torch.int64, device=self.device)
        temp = torch.zeros(max_num_reqs, dtype=torch.float32, device=self.device)
        seed = torch.full((max_num_reqs,), 42, dtype=torch.int32, device=self.device)
        pos = torch.arange(num_logits, dtype=torch.int32, device=self.device)
        rejected_step = torch.full((num_reqs,), 1, dtype=torch.int64, device=self.device)
        target_rejected_lse = torch.zeros(num_reqs, dtype=torch.float32, device=self.device)
        draft_rejected_lse = torch.zeros(num_reqs, dtype=torch.float32, device=self.device)

        draft_logits = None
        if has_draft_logits:
            draft_logits = torch.randn(max_num_reqs, num_spec_steps, vocab_size, dtype=torch.float32, device=self.device)

        resampled_local_argmax = -torch.ones(num_reqs, resample_num_blocks, dtype=torch.int64, device=self.device)
        resampled_local_max = -torch.ones(num_reqs, resample_num_blocks, dtype=torch.float32, device=self.device)

        draft_logits_arg = (
            draft_logits
            if draft_logits is not None
            else torch.empty(1, 1, 1, dtype=torch.float32, device=self.device)
        )
        cumulative_log_p = torch.zeros(
            num_logits, dtype=torch.float32, device=self.device
        )
        _launch_resample(
            (num_reqs, resample_num_blocks),
            [
                resampled_local_argmax,
                resampled_local_argmax.stride(0),
                resampled_local_max,
                resampled_local_max.stride(0),
                target_logits,
                target_logits.stride(0),
                target_rejected_lse,
                draft_logits_arg,
                draft_logits.stride(0) if draft_logits is not None else 0,
                draft_logits.stride(1) if draft_logits is not None else 0,
                draft_rejected_lse,
                rejected_step,
                cu_num_logits,
                expanded_idx_mapping,
                draft_sampled,
                temp,
                seed,
                pos,
            ],
            cumulative_log_p,
            vocab_size,
            BLOCK_SIZE,
            has_draft_logits,
            False,
        )
        torch.npu.synchronize()

        # For greedy non-bonus, kernel skips storing, so values stay as initial -1
        torch.testing.assert_close(
            resampled_local_argmax.cpu(),
            -torch.ones(num_reqs, resample_num_blocks, dtype=torch.int64),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            resampled_local_max.cpu(),
            -torch.ones(num_reqs, resample_num_blocks, dtype=torch.float32),
            rtol=0,
            atol=0,
        )

    def test_bonus_token(self):
        """When the resample token is the bonus token, always resample."""
        num_reqs = 1
        num_spec_steps = 2
        vocab_size = 256
        num_logits = num_reqs * (num_spec_steps + 1)
        BLOCK_SIZE = 256
        resample_num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)

        target_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=self.device)
        cu_num_logits = torch.tensor(
            [0, num_logits], device=self.device, dtype=torch.int32
        )
        expanded_idx_mapping = torch.zeros(
            num_logits, device=self.device, dtype=torch.int32
        )
        draft_sampled = torch.randint(0, vocab_size, (num_logits + 1,), dtype=torch.int64, device=self.device)
        # Bonus token: rejected_step points to end_idx - 1 = last token
        rejected_step = torch.tensor([num_spec_steps], device=self.device, dtype=torch.int64)
        temp = torch.zeros(1, dtype=torch.float32, device=self.device)
        seed = torch.full((1,), 42, dtype=torch.int32, device=self.device)
        pos = torch.arange(num_logits, dtype=torch.int32, device=self.device)
        target_rejected_lse = torch.zeros(1, dtype=torch.float32, device=self.device)
        draft_rejected_lse = torch.zeros(1, dtype=torch.float32, device=self.device)

        resampled_local_argmax = -torch.ones(num_reqs, resample_num_blocks, dtype=torch.int64, device=self.device)
        resampled_local_max = -torch.ones(num_reqs, resample_num_blocks, dtype=torch.float32, device=self.device)

        draft_logits_arg = torch.empty(
            1, 1, 1, dtype=torch.float32, device=self.device
        )
        cumulative_log_p = torch.zeros(
            num_logits, dtype=torch.float32, device=self.device
        )
        _launch_resample(
            (num_reqs, resample_num_blocks),
            [
                resampled_local_argmax,
                resampled_local_argmax.stride(0),
                resampled_local_max,
                resampled_local_max.stride(0),
                target_logits,
                target_logits.stride(0),
                target_rejected_lse,
                draft_logits_arg,
                0,
                0,
                draft_rejected_lse,
                rejected_step,
                cu_num_logits,
                expanded_idx_mapping,
                draft_sampled,
                temp,
                seed,
                pos,
            ],
            cumulative_log_p,
            vocab_size,
            BLOCK_SIZE,
            False,
            False,
        )
        torch.npu.synchronize()

        # With temp=0 there is no Gumbel noise, so the result is exact.
        expected_token = int(torch.argmax(target_logits[-1]).item())
        assert resampled_local_argmax.cpu().item() == expected_token
        torch.testing.assert_close(
            resampled_local_max.cpu().item(),
            target_logits[-1, expected_token].cpu().item(),
            rtol=1e-5,
            atol=1e-5,
        )
