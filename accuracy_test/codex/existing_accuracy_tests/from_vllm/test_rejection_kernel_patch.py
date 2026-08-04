# Accuracy UT source: no direct worker-v2 Ascend UT; adapted from
# vllm/tests/v1/spec_decode/test_rejection_sampler_utils.py
# vLLM-Ascend patched kernel: _probabilistic_rejection_kernel from
# vllm-ascend-xyz/vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py:192
# PATCH NOTE: Replaces _rejection_kernel on Ascend NPU

"""
Precision test for patched _probabilistic_rejection_kernel (Ascend NPU version).

This replaces `_rejection_kernel` on Ascend NPU (renamed/reimplemented).

Patch differences vs original vllm _rejection_kernel:
- No synthetic_conditional_rates_ptr, cumulative_log_p_ptr, local_residual_mass_ptr
- No SYNTHETIC_MODE or USE_BLOCK_VERIFICATION constexpr parameters
- Uses u = tl.full([], 0.0, dtype=tl.float32) instead of tl_rand32 (NPU does not
  support tl_rand64 / float64) -- always accepts draft token on non-greedy paths
- Greedy path loads blocks and computes argmax inline with tl.argmax over blocks
- Non-greedy path computes LSE with _compute_global_lse helper
- Rejection uses target_log_prob > tl.log(u) + draft_log_prob (with u=0 always true)
- HAS_DRAFT_LOGITS switches between draft-logits path and one-hot (zero log prob)
- Extra draft_logits_stride_0 and draft_logits_stride_1 in kernel signature directly

Kernel signature:
    _probabilistic_rejection_kernel(
        sampled_ptr,                        # [num_reqs, num_spec_steps + 1] int64 sampled
        sampled_stride,                     # stride(0) of sampled
        rejected_steps_ptr,                 # [num_reqs] int32 accepted count
        target_rejected_logsumexp_ptr,      # [num_reqs] fp32 target LSE
        draft_rejected_logsumexp_ptr,       # [num_reqs] fp32 draft LSE
        target_logits_ptr,                  # [num_logits, V] target logits
        target_logits_stride,               # stride(0) of target_logits
        target_local_argmax_ptr,            # [num_logits, num_blocks] int64 local argmax
        target_local_argmax_stride,         # stride(0)
        target_local_max_ptr,               # [num_logits, num_blocks] fp32 local max
        target_local_max_stride,            # stride(0)
        target_local_sumexp_ptr,            # [num_logits, num_blocks] fp32 local sumexp
        target_local_sumexp_stride,         # stride(0)
        draft_sampled_ptr,                  # [num_logits] int32 draft sampled tokens
        draft_logits_ptr,                   # [max_num_reqs, num_spec_steps, V] or dummy
        draft_logits_stride_0,              # stride(0) of draft_logits
        draft_logits_stride_1,              # stride(1) of draft_logits
        draft_local_max_ptr,                # [num_logits, num_blocks] fp32
        draft_local_max_stride,             # stride(0)
        draft_local_sumexp_ptr,             # [num_logits, num_blocks] fp32
        draft_local_sumexp_stride,          # stride(0)
        cu_num_logits_ptr,                  # [num_reqs+1] int32 cumulative counts
        idx_mapping_ptr,                    # [num_reqs] int32 req_idx -> req_state_idx
        temp_ptr,                           # [max_num_reqs] fp32 temperatures
        seed_ptr,                           # [max_num_reqs] int64 seeds
        pos_ptr,                            # [num_logits] int64 positions
        vocab_num_blocks,                   # scalar: num blocks
        PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
        HAS_DRAFT_LOGITS: tl.constexpr,
    )

Iterates over each request's draft tokens, computing acceptance:
- Greedy (temp=0): accept iff draft token equals target argmax
- Non-greedy (draft logits available): accept with u=0 (always accept on NPU)
- Non-greedy (one-hot draft): accept (draft_log_prob=0, u=0)
Stores accepted draft tokens into sampled_ptr and the number accepted.
Stores target and draft logsumexp from the rejection step for resampling.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

import pytest


# ---------------------------------------------------------------------------
# CPU reference helpers
# ---------------------------------------------------------------------------


def _greedy_accept_cpu_ref(
    target_logits: torch.Tensor,
    target_local_argmax: torch.Tensor,
    target_local_max: torch.Tensor,
    draft_sampled: torch.Tensor,
    start_idx: int,
    num_tokens: int,
    vocab_num_blocks: int,
) -> int:
    """CPU reference: greedy acceptance. Returns number of accepted steps."""
    rejected_step = 0
    for i in range(num_tokens - 1):
        logit_idx = start_idx + i
        draft_token = draft_sampled[logit_idx + 1].item()

        max_val = float("-inf")
        max_block = 0
        for b in range(vocab_num_blocks):
            bv = target_local_max[logit_idx, b].item()
            if bv > max_val:
                max_val = bv
                max_block = b
        target_argmax = target_local_argmax[logit_idx, max_block].item()
        if target_argmax == draft_token:
            rejected_step += 1
    return rejected_step


def _non_greedy_accept_cpu_ref(
    target_logits: torch.Tensor,
    target_local_max: torch.Tensor,
    target_local_sumexp: torch.Tensor,
    draft_sampled: torch.Tensor,
    draft_logits: torch.Tensor | None,
    start_idx: int,
    num_tokens: int,
    req_state_idx: int,
    vocab_num_blocks: int,
    has_draft_logits: bool,
) -> tuple:
    """CPU reference: non-greedy acceptance with NPU u=0 behavior (always accept)."""
    import math

    rejected_step = 0
    target_lse = 0.0
    draft_lse_val = 0.0

    for i in range(num_tokens - 1):
        logit_idx = start_idx + i
        draft_token = draft_sampled[logit_idx + 1].item()

        # Compute target LSE and log prob
        local_max_vals = target_local_max[logit_idx, :vocab_num_blocks]
        local_sumexp_vals = target_local_sumexp[logit_idx, :vocab_num_blocks]
        max_val = float(local_max_vals.max())
        target_lse = max_val + math.log(float(local_sumexp_vals.sum()))
        target_logit = target_logits[logit_idx, draft_token].item()
        target_log_prob = target_logit - target_lse

        if has_draft_logits:
            dl_max = draft_logits.new_empty if isinstance(draft_logits, torch.Tensor) else None
            draft_lse_val = 0.0  # simplified; u=0 means always accept regardless
        else:
            draft_lse_val = 0.0

        # NPU: u=0.0 => log(u) = -inf => target_log_prob > -inf + draft_log_prob always True
        rejected_step += 1

    return rejected_step, target_lse, draft_lse_val


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestProbabilisticRejectionKernelPatch:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    def _run_kernel(
        self,
        num_reqs: int,
        num_spec_steps: int,
        num_logits: int,
        vocab_size: int,
        target_logits: torch.Tensor,
        draft_sampled: torch.Tensor,
        draft_logits: torch.Tensor | None,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        has_draft_logits: bool,
        VOCAB_BLOCK_SIZE: int = 8192,
    ) -> tuple:
        """Run _probabilistic_rejection_kernel and return (sampled, rejected_steps)."""
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
            _compute_local_logits_stats_kernel as _compute_block_stats_kernel,
        )
        from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import (
            _probabilistic_rejection_kernel,
        )

        device = self.device

        # Build expanded_idx_mapping and pos
        expanded_idx_mapping = torch.zeros(num_logits, dtype=torch.int64, device=device)
        expanded_local_pos = torch.arange(num_logits, dtype=torch.int64, device=device)

        # Compute block stats via _compute_block_stats_kernel
        vocab_num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
        padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)

        target_local_argmax = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.int64)
        target_local_max = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
        target_local_sumexp = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
        draft_local_max = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
        draft_local_sumexp = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)

        dl = draft_logits if has_draft_logits else target_logits.new_empty(1, 1, 1)

        _compute_block_stats_kernel[(num_logits, vocab_num_blocks)](
            target_local_argmax, target_local_argmax.stride(0),
            target_local_max, target_local_max.stride(0),
            target_local_sumexp, target_local_sumexp.stride(0),
            draft_local_max, draft_local_max.stride(0),
            draft_local_sumexp, draft_local_sumexp.stride(0),
            target_logits, target_logits.stride(0),
            dl, dl.stride(0), dl.stride(1),
            expanded_idx_mapping, expanded_local_pos,
            temperature, vocab_size, num_spec_steps,
            BLOCK_SIZE=VOCAB_BLOCK_SIZE,
            HAS_DRAFT_LOGITS=has_draft_logits,
        )
        torch.npu.synchronize()

        # Prepare kernel inputs
        cu_num_logits = torch.tensor([0, num_logits], dtype=torch.int32, device=device)
        idx_mapping = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        pos = torch.arange(num_logits, dtype=torch.int64, device=device)

        sampled = torch.zeros(num_reqs, num_spec_steps + 1, dtype=torch.int64, device=device)
        rejected_steps = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        target_rejected_lse = torch.zeros(num_reqs, dtype=torch.float32, device=device)
        draft_rejected_lse = torch.zeros(num_reqs, dtype=torch.float32, device=device)

        _probabilistic_rejection_kernel[(num_reqs,)](
            sampled, sampled.stride(0),
            rejected_steps, target_rejected_lse, draft_rejected_lse,
            target_logits, target_logits.stride(0),
            target_local_argmax, target_local_argmax.stride(0),
            target_local_max, target_local_max.stride(0),
            target_local_sumexp, target_local_sumexp.stride(0),
            draft_sampled,
            dl, dl.stride(0), dl.stride(1),
            draft_local_max, draft_local_max.stride(0),
            draft_local_sumexp, draft_local_sumexp.stride(0),
            cu_num_logits, idx_mapping, temperature, seeds, pos,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
            HAS_DRAFT_LOGITS=has_draft_logits,
            num_warps=1,
        )
        torch.npu.synchronize()

        return sampled, rejected_steps, target_local_argmax, target_local_max

    # ----------------------------------------------------------------
    # Greedy tests (temp = 0)
    # ----------------------------------------------------------------

    def test_greedy_all_accepted(self):
        """Greedy: all draft tokens match target argmax -> all accepted."""
        num_reqs = 1
        num_spec_steps = 3
        num_logits = num_spec_steps + 1
        vocab_size = 128

        target_logits = torch.full((num_logits, vocab_size), -10.0, dtype=torch.float32, device=self.device)
        for i in range(num_logits):
            target_logits[i, 42] = 0.0  # argmax always 42

        draft_sampled = torch.full((num_logits,), 42, dtype=torch.int32, device=self.device)
        temperature = torch.tensor([0.0], dtype=torch.float32, device=self.device)
        seeds = torch.tensor([42], dtype=torch.int64, device=self.device)

        sampled, rejected_steps, local_argmax, local_max = self._run_kernel(
            num_reqs, num_spec_steps, num_logits, vocab_size,
            target_logits, draft_sampled, None, temperature, seeds,
            has_draft_logits=False,
        )

        assert rejected_steps[0].item() == num_spec_steps, (
            f"Expected {num_spec_steps} accepted, got {rejected_steps[0].item()}"
        )

    def test_greedy_all_rejected(self):
        """Greedy: no draft tokens match target argmax -> none accepted."""
        num_reqs = 1
        num_spec_steps = 3
        num_logits = num_spec_steps + 1
        vocab_size = 128

        target_logits = torch.full((num_logits, vocab_size), -10.0, dtype=torch.float32, device=self.device)
        for i in range(num_logits):
            target_logits[i, 100] = 0.0  # argmax always 100

        draft_sampled = torch.full((num_logits,), 42, dtype=torch.int32, device=self.device)
        temperature = torch.tensor([0.0], dtype=torch.float32, device=self.device)
        seeds = torch.tensor([42], dtype=torch.int64, device=self.device)

        sampled, rejected_steps, local_argmax, local_max = self._run_kernel(
            num_reqs, num_spec_steps, num_logits, vocab_size,
            target_logits, draft_sampled, None, temperature, seeds,
            has_draft_logits=False,
        )

        assert rejected_steps[0].item() == 0, (
            f"Expected 0 accepted, got {rejected_steps[0].item()}"
        )

    @pytest.mark.parametrize("num_draft_tokens", [1, 3, 5])
    def test_greedy_varying_lengths(self, num_draft_tokens):
        """Greedy: parametrized over draft lengths, compared with CPU reference."""
        num_reqs = 1
        num_spec_steps = num_draft_tokens
        num_logits = num_draft_tokens + 1
        vocab_size = 128

        torch.manual_seed(42)
        target_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=self.device)
        draft_sampled = torch.randint(0, vocab_size, (num_logits,), dtype=torch.int32, device=self.device)
        temperature = torch.tensor([0.0], dtype=torch.float32, device=self.device)
        seeds = torch.tensor([42], dtype=torch.int64, device=self.device)

        sampled, rejected_steps, local_argmax, local_max = self._run_kernel(
            num_reqs, num_spec_steps, num_logits, vocab_size,
            target_logits, draft_sampled, None, temperature, seeds,
            has_draft_logits=False,
        )

        # CPU reference
        vocab_num_blocks = local_argmax.shape[1]
        expected = _greedy_accept_cpu_ref(
            target_logits.cpu(), local_argmax.cpu(), local_max.cpu(),
            draft_sampled.cpu(), 0, num_logits, vocab_num_blocks,
        )
        torch.testing.assert_close(
            rejected_steps.cpu(),
            torch.tensor([expected], dtype=torch.int32),
            rtol=0, atol=0,
        )

    def test_greedy_sampled_output(self):
        """Greedy: sampled output contains target argmax (not draft token)."""
        num_reqs = 1
        num_spec_steps = 2
        num_logits = num_spec_steps + 1
        vocab_size = 64

        target_logits = torch.full((num_logits, vocab_size), -10.0, dtype=torch.float32, device=self.device)
        for i in range(num_logits):
            target_logits[i, 7] = 0.0  # argmax always 7

        draft_sampled = torch.full((num_logits,), 7, dtype=torch.int32, device=self.device)
        temperature = torch.tensor([0.0], dtype=torch.float32, device=self.device)
        seeds = torch.tensor([42], dtype=torch.int64, device=self.device)

        sampled, rejected_steps, local_argmax, local_max = self._run_kernel(
            num_reqs, num_spec_steps, num_logits, vocab_size,
            target_logits, draft_sampled, None, temperature, seeds,
            has_draft_logits=False,
        )

        assert rejected_steps[0].item() == num_spec_steps
        for i in range(num_spec_steps):
            assert sampled[0, i].item() == 7, (
                f"Expected sampled[0,{i}] = 7, got {sampled[0, i].item()}"
            )

    def test_greedy_multi_req(self):
        """Greedy: multiple requests with different argmax patterns."""
        num_reqs = 2
        num_spec_steps = 2
        num_logits = (num_spec_steps + 1) * num_reqs  # 6
        vocab_size = 64

        torch.manual_seed(1)
        target_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=self.device)
        draft_sampled = torch.randint(0, vocab_size, (num_logits,), dtype=torch.int32, device=self.device)

        temperature = torch.tensor([0.0, 0.0], dtype=torch.float32, device=self.device)
        seeds = torch.tensor([42, 99], dtype=torch.int64, device=self.device)

        sampled, rejected_steps, local_argmax, local_max = self._run_kernel(
            num_reqs, num_spec_steps, num_logits, vocab_size,
            target_logits, draft_sampled, None, temperature, seeds,
            has_draft_logits=False,
        )

        # Build multi-req stats manually for CPU reference
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
            _compute_local_logits_stats_kernel as _compute_block_stats_kernel,
        )
        from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import (
            _probabilistic_rejection_kernel,
        )

        # Re-run with proper cu_num_logits and idx_mapping for multi-req
        device = self.device
        VOCAB_BLOCK_SIZE = 8192
        vocab_num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
        padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)

        expanded_idx_mapping = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64, device=device)
        expanded_local_pos = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64, device=device)

        target_local_argmax = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.int64)
        target_local_max = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
        target_local_sumexp = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
        draft_local_max = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
        draft_local_sumexp = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
        dl = target_logits.new_empty(1, 1, 1)

        _compute_block_stats_kernel[(num_logits, vocab_num_blocks)](
            target_local_argmax, target_local_argmax.stride(0),
            target_local_max, target_local_max.stride(0),
            target_local_sumexp, target_local_sumexp.stride(0),
            draft_local_max, draft_local_max.stride(0),
            draft_local_sumexp, draft_local_sumexp.stride(0),
            target_logits, target_logits.stride(0),
            dl, dl.stride(0), dl.stride(1),
            expanded_idx_mapping, expanded_local_pos,
            temperature, vocab_size, num_spec_steps,
            BLOCK_SIZE=VOCAB_BLOCK_SIZE,
            HAS_DRAFT_LOGITS=False,
        )
        torch.npu.synchronize()

        cu_num_logits = torch.tensor([0, 3, 6], dtype=torch.int32, device=device)
        idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=device)
        pos = torch.arange(num_logits, dtype=torch.int64, device=device)

        sampled2 = torch.zeros(num_reqs, num_spec_steps + 1, dtype=torch.int64, device=device)
        rejected_steps2 = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        target_rejected_lse = torch.zeros(num_reqs, dtype=torch.float32, device=device)
        draft_rejected_lse = torch.zeros(num_reqs, dtype=torch.float32, device=device)

        _probabilistic_rejection_kernel[(num_reqs,)](
            sampled2, sampled2.stride(0),
            rejected_steps2, target_rejected_lse, draft_rejected_lse,
            target_logits, target_logits.stride(0),
            target_local_argmax, target_local_argmax.stride(0),
            target_local_max, target_local_max.stride(0),
            target_local_sumexp, target_local_sumexp.stride(0),
            draft_sampled,
            dl, dl.stride(0), dl.stride(1),
            draft_local_max, draft_local_max.stride(0),
            draft_local_sumexp, draft_local_sumexp.stride(0),
            cu_num_logits, idx_mapping, temperature, seeds, pos,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
            HAS_DRAFT_LOGITS=False,
            num_warps=1,
        )
        torch.npu.synchronize()

        for r in range(num_reqs):
            start = r * 3
            expected = _greedy_accept_cpu_ref(
                target_logits.cpu(), target_local_argmax.cpu(),
                target_local_max.cpu(), draft_sampled.cpu(),
                start, 3, vocab_num_blocks,
            )
            torch.testing.assert_close(
                rejected_steps2[r].cpu(),
                torch.tensor(expected, dtype=torch.int32),
                rtol=0, atol=0,
            )

    # ----------------------------------------------------------------
    # Non-greedy tests (temp > 0)
    # ----------------------------------------------------------------

    def test_non_greedy_always_accept(self):
        """Non-greedy with draft logits: u=0 means always accept."""
        num_reqs = 1
        num_spec_steps = 2
        num_logits = num_spec_steps + 1
        vocab_size = 128

        torch.manual_seed(42)
        target_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=self.device)
        draft_sampled = torch.randint(0, vocab_size, (num_logits,), dtype=torch.int32, device=self.device)
        temperature = torch.tensor([1.0], dtype=torch.float32, device=self.device)
        seeds = torch.tensor([42], dtype=torch.int64, device=self.device)
        draft_logits = torch.randn(1, num_spec_steps, vocab_size, dtype=torch.float32, device=self.device)

        sampled, rejected_steps, local_argmax, local_max = self._run_kernel(
            num_reqs, num_spec_steps, num_logits, vocab_size,
            target_logits, draft_sampled, draft_logits, temperature, seeds,
            has_draft_logits=True,
        )

        assert rejected_steps[0].item() == num_spec_steps, (
            f"Expected {num_spec_steps} accepted, got {rejected_steps[0].item()}"
        )

    def test_non_greedy_no_draft_logits(self):
        """Non-greedy without draft logits (one-hot): u=0 means always accept."""
        num_reqs = 1
        num_spec_steps = 3
        num_logits = num_spec_steps + 1
        vocab_size = 128

        torch.manual_seed(42)
        target_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=self.device)
        draft_sampled = torch.randint(0, vocab_size, (num_logits,), dtype=torch.int32, device=self.device)
        temperature = torch.tensor([1.0], dtype=torch.float32, device=self.device)
        seeds = torch.tensor([42], dtype=torch.int64, device=self.device)

        sampled, rejected_steps, local_argmax, local_max = self._run_kernel(
            num_reqs, num_spec_steps, num_logits, vocab_size,
            target_logits, draft_sampled, None, temperature, seeds,
            has_draft_logits=False,
        )

        assert rejected_steps[0].item() == num_spec_steps, (
            f"Expected {num_spec_steps} accepted, got {rejected_steps[0].item()}"
        )

    @pytest.mark.parametrize("temp", [0.5, 1.0, 2.0])
    def test_non_greedy_varying_temps(self, temp):
        """Non-greedy: various temperatures all always-accept with u=0."""
        num_reqs = 1
        num_spec_steps = 2
        num_logits = num_spec_steps + 1
        vocab_size = 128

        torch.manual_seed(42)
        target_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=self.device)
        draft_sampled = torch.randint(0, vocab_size, (num_logits,), dtype=torch.int32, device=self.device)
        temperature = torch.tensor([temp], dtype=torch.float32, device=self.device)
        seeds = torch.tensor([42], dtype=torch.int64, device=self.device)
        draft_logits = torch.randn(1, num_spec_steps, vocab_size, dtype=torch.float32, device=self.device)

        sampled, rejected_steps, local_argmax, local_max = self._run_kernel(
            num_reqs, num_spec_steps, num_logits, vocab_size,
            target_logits, draft_sampled, draft_logits, temperature, seeds,
            has_draft_logits=True,
        )

        assert rejected_steps[0].item() == num_spec_steps, (
            f"temp={temp}: expected {num_spec_steps} accepted, got {rejected_steps[0].item()}"
        )

    def test_non_greedy_sampled_output(self):
        """Non-greedy: sampled output should contain the draft token values."""
        num_reqs = 1
        num_spec_steps = 2
        num_logits = num_spec_steps + 1
        vocab_size = 64

        torch.manual_seed(42)
        target_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=self.device)
        draft_values = torch.tensor([0, 10, 20, 30], dtype=torch.int32, device=self.device)
        draft_sampled = draft_values
        temperature = torch.tensor([1.0], dtype=torch.float32, device=self.device)
        seeds = torch.tensor([42], dtype=torch.int64, device=self.device)

        sampled, rejected_steps, local_argmax, local_max = self._run_kernel(
            num_reqs, num_spec_steps, num_logits, vocab_size,
            target_logits, draft_sampled, None, temperature, seeds,
            has_draft_logits=False,
        )

        assert rejected_steps[0].item() == num_spec_steps
        for i in range(num_spec_steps):
            expected = draft_values[i + 1].item()
            assert sampled[0, i].item() == expected, (
                f"Expected sampled[0,{i}] = {expected}, got {sampled[0, i].item()}"
            )

    # ----------------------------------------------------------------
    # Mixed request types
    # ----------------------------------------------------------------

    def test_mixed_greedy_and_non_greedy(self):
        """One greedy and one non-greedy request handled correctly."""
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
            _compute_local_logits_stats_kernel as _compute_block_stats_kernel,
        )
        from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import (
            _probabilistic_rejection_kernel,
        )

        num_reqs = 2
        num_spec_steps = 2
        num_logits = (num_spec_steps + 1) * num_reqs
        vocab_size = 64
        VOCAB_BLOCK_SIZE = 8192
        vocab_num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
        padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)

        device = self.device
        torch.manual_seed(42)
        target_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=self.device)
        draft_sampled = torch.randint(0, vocab_size, (num_logits,), dtype=torch.int32, device=self.device)

        # Req 0: greedy (temp=0), Req 1: non-greedy (temp=1)
        temperature = torch.tensor([0.0, 1.0], dtype=torch.float32, device=device)
        seeds = torch.tensor([42, 99], dtype=torch.int64, device=device)
        draft_logits = torch.randn(1, num_spec_steps, vocab_size, dtype=torch.float32, device=device)

        expanded_idx_mapping = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64, device=device)
        expanded_local_pos = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64, device=device)

        target_local_argmax = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.int64)
        target_local_max = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
        target_local_sumexp = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
        draft_local_max = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
        draft_local_sumexp = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
        dl = draft_logits

        _compute_block_stats_kernel[(num_logits, vocab_num_blocks)](
            target_local_argmax, target_local_argmax.stride(0),
            target_local_max, target_local_max.stride(0),
            target_local_sumexp, target_local_sumexp.stride(0),
            draft_local_max, draft_local_max.stride(0),
            draft_local_sumexp, draft_local_sumexp.stride(0),
            target_logits, target_logits.stride(0),
            dl, dl.stride(0), dl.stride(1),
            expanded_idx_mapping, expanded_local_pos,
            temperature, vocab_size, num_spec_steps,
            BLOCK_SIZE=VOCAB_BLOCK_SIZE,
            HAS_DRAFT_LOGITS=True,
        )
        torch.npu.synchronize()

        cu_num_logits = torch.tensor([0, 3, 6], dtype=torch.int32, device=device)
        idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=device)
        pos = torch.arange(num_logits, dtype=torch.int64, device=device)

        sampled = torch.zeros(num_reqs, num_spec_steps + 1, dtype=torch.int64, device=device)
        rejected_steps = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        target_rejected_lse = torch.zeros(num_reqs, dtype=torch.float32, device=device)
        draft_rejected_lse = torch.zeros(num_reqs, dtype=torch.float32, device=device)

        _probabilistic_rejection_kernel[(num_reqs,)](
            sampled, sampled.stride(0),
            rejected_steps, target_rejected_lse, draft_rejected_lse,
            target_logits, target_logits.stride(0),
            target_local_argmax, target_local_argmax.stride(0),
            target_local_max, target_local_max.stride(0),
            target_local_sumexp, target_local_sumexp.stride(0),
            draft_sampled,
            dl, dl.stride(0), dl.stride(1),
            draft_local_max, draft_local_max.stride(0),
            draft_local_sumexp, draft_local_sumexp.stride(0),
            cu_num_logits, idx_mapping, temperature, seeds, pos,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
            HAS_DRAFT_LOGITS=True,
            num_warps=1,
        )
        torch.npu.synchronize()

        # Req 0 (greedy): acceptance depends on argmax match
        expected_greedy = _greedy_accept_cpu_ref(
            target_logits.cpu(), target_local_argmax.cpu(), target_local_max.cpu(),
            draft_sampled.cpu(), 0, 3, vocab_num_blocks,
        )
        torch.testing.assert_close(
            rejected_steps[0].cpu(),
            torch.tensor(expected_greedy, dtype=torch.int32),
            rtol=0, atol=0,
        )

        # Req 1 (non-greedy): always accept
        assert rejected_steps[1].item() == num_spec_steps, (
            f"Non-greedy req: expected {num_spec_steps} accepted, got {rejected_steps[1].item()}"
        )

    # ----------------------------------------------------------------
    # Edge cases
    # ----------------------------------------------------------------

    def test_bonus_token_only(self):
        """Single bonus token (num_tokens=1) -> no iterations, rejected_steps=0."""
        num_reqs = 1
        num_spec_steps = 0
        num_logits = 1
        vocab_size = 64

        target_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=self.device)
        draft_sampled = torch.zeros(1, dtype=torch.int32, device=self.device)
        temperature = torch.tensor([1.0], dtype=torch.float32, device=self.device)
        seeds = torch.tensor([42], dtype=torch.int64, device=self.device)

        sampled, rejected_steps, local_argmax, local_max = self._run_kernel(
            num_reqs, num_spec_steps, num_logits, vocab_size,
            target_logits, draft_sampled, None, temperature, seeds,
            has_draft_logits=False,
        )

        torch.testing.assert_close(
            rejected_steps.cpu(),
            torch.tensor([0], dtype=torch.int32),
            rtol=0, atol=0,
        )

    @pytest.mark.parametrize("vocab_size", [32, 64, 128])
    def test_varying_vocab_sizes(self, vocab_size):
        """Parametrized: correct acceptance across different vocab sizes."""
        num_reqs = 1
        num_spec_steps = 3
        num_logits = num_spec_steps + 1

        torch.manual_seed(42)
        target_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=self.device)
        draft_sampled = torch.randint(0, vocab_size, (num_logits,), dtype=torch.int32, device=self.device)
        temperature = torch.tensor([0.0], dtype=torch.float32, device=self.device)
        seeds = torch.tensor([42], dtype=torch.int64, device=self.device)

        sampled, rejected_steps, local_argmax, local_max = self._run_kernel(
            num_reqs, num_spec_steps, num_logits, vocab_size,
            target_logits, draft_sampled, None, temperature, seeds,
            has_draft_logits=False,
        )

        vocab_num_blocks = local_argmax.shape[1]
        expected = _greedy_accept_cpu_ref(
            target_logits.cpu(), local_argmax.cpu(), local_max.cpu(),
            draft_sampled.cpu(), 0, num_logits, vocab_num_blocks,
        )
        torch.testing.assert_close(
            rejected_steps.cpu(),
            torch.tensor([expected], dtype=torch.int32),
            rtol=0, atol=0,
        )
