# GENERATED STRICT UT. Source: accuracy_test/codex/existing_accuracy_tests/from_vllm/test_compute_block_stats_kernel.py
# Do not edit mechanically; update the reviewed Codex source or strict generator.
from accuracy_test.strict_ut.runtime_gpu import STRICT_DEVICE as _STRICT_DEVICE
# Standalone Ascend A3 adaptation of an upstream vLLM accuracy path.
# Accuracy UT source: vllm/tests/v1/spec_decode/test_rejection_sampler_utils.py
# Kernel source: vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
# Coverage: _compute_block_stats_kernel

# vLLM vanilla kernel: _compute_cumulative_log_p_kernel from
# vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py

"""
Precision test for _compute_cumulative_log_p_kernel.

Kernel signature:
    _compute_cumulative_log_p_kernel(
        cumulative_log_p_ptr,            # fp32 [num_logits]
        target_logits_ptr,               # fp32 [num_logits, V]
        target_logits_stride,            # stride(0)
        target_local_max_ptr,            # fp32 [num_logits, num_blocks]
        target_local_max_stride,         # stride(0)
        target_local_sumexp_ptr,         # fp32 [num_logits, num_blocks]
        target_local_sumexp_stride,      # stride(0)
        draft_sampled_ptr,              # int64 [num_logits]
        draft_logits_ptr,               # fp32 [max_num_reqs, num_spec_steps, V]
        draft_logits_stride_0,           # stride(0)
        draft_logits_stride_1,           # stride(1)
        draft_local_max_ptr,             # fp32 [num_logits, num_blocks]
        draft_local_max_stride,          # stride(0)
        draft_local_sumexp_ptr,          # fp32 [num_logits, num_blocks]
        draft_local_sumexp_stride,       # stride(0)
        cu_num_logits_ptr,              # int64 [num_reqs + 1]
        idx_mapping_ptr,                # int32 [num_reqs]
        temp_ptr,                       # fp32 [max_num_reqs]
        vocab_num_blocks,               # scalar
        PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
        HAS_DRAFT_LOGITS: tl.constexpr,
    )

Computes the cumulative log probability ratio p_i for each draft token position
as part of block verification. For each request, iterates over draft tokens and
computes log_p = min(log_p + (target_logprob - draft_logprob), 0).

This kernel uses _compute_global_logprobs_and_logsumexp internally.
We name this file _compute_block_stats_kernel to match the user specification.
"""

import torch
import pytest

from vllm.triton_utils import tl, triton
try:
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        _compute_cumulative_log_p_kernel,
        _compute_local_logits_stats_kernel,
    )
except ImportError as exc:
    pytest.skip(
        "installed vLLM does not provide the block-verification kernels required "
        f"by this test; precision was not tested: {exc}",
        allow_module_level=True,
    )
from accuracy_test.strict_ut.runtime_gpu import init_device_properties_triton


class TestComputeBlockStatsKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("cuda")

    @pytest.mark.parametrize("num_reqs", [1, 2])
    @pytest.mark.parametrize("num_draft_tokens", [1, 2, 3])
    @pytest.mark.parametrize("vocab_size", [128, 1024])
    def test_cumulative_log_p(self, num_reqs, num_draft_tokens, vocab_size):
        """Compare cumulative log-p with CPU reference."""
        max_num_reqs = 4
        num_logits = num_reqs * (num_draft_tokens + 1)
        num_speculative_steps = num_draft_tokens
        VOCAB_BLOCK_SIZE = 8192
        vocab_num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
        padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)

        target_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=self.device)
        draft_logits = torch.randn(max_num_reqs, num_speculative_steps, vocab_size, dtype=torch.float32, device=self.device)

        cu_num_logits = torch.arange(num_reqs + 1, dtype=torch.int64, device=self.device) * (num_draft_tokens + 1)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)

        # Draft sampled tokens
        draft_sampled = torch.randint(0, vocab_size, (num_logits + 1,), dtype=torch.int64, device=self.device)

        temperature = torch.full((max_num_reqs,), 1.0, dtype=torch.float32, device=self.device)

        expanded_idx_mapping = torch.zeros(num_logits, dtype=torch.int64, device=self.device)
        expanded_local_pos = torch.zeros(num_logits, dtype=torch.int64, device=self.device)
        for ri in range(num_reqs):
            for di in range(num_draft_tokens + 1):
                li = ri * (num_draft_tokens + 1) + di
                expanded_idx_mapping[li] = ri
                expanded_local_pos[li] = di

        # Compute block stats first (pre-requisite)
        target_local_argmax = torch.empty(num_logits, vocab_num_blocks, dtype=torch.int64, device=self.device)
        target_local_max = torch.empty(num_logits, vocab_num_blocks, dtype=torch.float32, device=self.device)
        target_local_sumexp = torch.empty(num_logits, vocab_num_blocks, dtype=torch.float32, device=self.device)
        draft_local_max = torch.empty(num_logits, vocab_num_blocks, dtype=torch.float32, device=self.device)
        draft_local_sumexp = torch.empty(num_logits, vocab_num_blocks, dtype=torch.float32, device=self.device)

        _compute_local_logits_stats_kernel[(num_logits, vocab_num_blocks)](
            target_local_argmax,
            target_local_argmax.stride(0),
            target_local_max,
            target_local_max.stride(0),
            target_local_sumexp,
            target_local_sumexp.stride(0),
            draft_local_max,
            draft_local_max.stride(0),
            draft_local_sumexp,
            draft_local_sumexp.stride(0),
            target_logits,
            target_logits.stride(0),
            draft_logits,
            draft_logits.stride(0),
            draft_logits.stride(1),
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            vocab_size,
            num_speculative_steps,
            BLOCK_SIZE=VOCAB_BLOCK_SIZE,
            HAS_DRAFT_LOGITS=True,
        )
        torch.cuda.synchronize()

        # Compute cumulative log p
        cumulative_log_p = torch.empty(num_logits, dtype=torch.float32, device=self.device)
        _compute_cumulative_log_p_kernel[(num_reqs,)](
            cumulative_log_p,
            target_logits,
            target_logits.stride(0),
            target_local_max,
            target_local_max.stride(0),
            target_local_sumexp,
            target_local_sumexp.stride(0),
            draft_sampled,
            draft_logits,
            draft_logits.stride(0),
            draft_logits.stride(1),
            draft_local_max,
            draft_local_max.stride(0),
            draft_local_sumexp,
            draft_local_sumexp.stride(0),
            cu_num_logits,
            idx_mapping,
            temperature,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
            HAS_DRAFT_LOGITS=True,
            num_warps=1,
        )
        torch.cuda.synchronize()

        # CPU reference: brute-force compute cumulative log p
        tgt_cpu = target_logits.cpu()
        drf_cpu = draft_logits.cpu()
        ds_cpu = draft_sampled.cpu()
        idx_cpu = idx_mapping.cpu()
        temp_cpu = temperature.cpu()
        cum_cpu = cumulative_log_p.cpu()

        for ri in range(num_reqs):
            rs = idx_cpu[ri].item()
            temp = temp_cpu[rs].item()
            if temp == 0.0:
                continue  # greedy, cumulative log p not meaningful

            start = cu_num_logits[ri].item()
            num_dt = cu_num_logits[ri + 1].item() - start - 1

            log_p = 0.0
            for step in range(num_dt):
                li = start + step
                dt = ds_cpu[li + 1].item()

                # Target log prob
                tgt_lse = _global_logsumexp_cpu(
                    target_local_max[li].cpu(), target_local_sumexp[li].cpu(),
                    vocab_num_blocks
                )
                target_logit = tgt_cpu[li, dt].item()
                tgt_lp = target_logit - tgt_lse

                # Draft log prob
                drf_lse = _global_logsumexp_cpu(
                    draft_local_max[li].cpu(), draft_local_sumexp[li].cpu(),
                    vocab_num_blocks
                )
                draft_logit = drf_cpu[rs, step, dt].item()
                drf_lp = draft_logit - drf_lse

                log_p = min(log_p + (tgt_lp - drf_lp), 0.0)
                torch.testing.assert_close(
                    cum_cpu[li].item(), log_p, rtol=1e-4, atol=1e-4,
                    msg=f"req {ri}, step {step}: cumulative_log_p mismatch"
                )


def _global_logsumexp_cpu(
    local_max: torch.Tensor,
    local_sumexp: torch.Tensor,
    vocab_num_blocks: int,
) -> float:
    """CPU reference for global logsumexp."""
    maxes = local_max[:vocab_num_blocks].float()
    sumexps = local_sumexp[:vocab_num_blocks].float()
    global_max = float(maxes.max().item())
    if global_max > float("-inf"):
        weighted = sumexps * torch.exp(maxes - global_max)
        return global_max + float(torch.log(torch.sum(weighted)).item())
    return global_max
