# GENERATED STRICT UT. Source: accuracy_test/codex/existing_accuracy_tests/from_vllm/test_compute_block_max_and_sumexp.py
# Do not edit mechanically; update the reviewed Codex source or strict generator.
from accuracy_test.strict_ut.runtime_npu import STRICT_DEVICE as _STRICT_DEVICE
# Standalone Ascend A3 adaptation of an upstream vLLM accuracy path.
# Accuracy UT source: vllm/tests/v1/spec_decode/test_rejection_sampler_utils.py
# Kernel source: vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
# Coverage: _compute_local_logits_stats_kernel

# vLLM vanilla kernel: _compute_local_logits_stats_kernel from
# vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py

"""
Precision test for _compute_local_logits_stats_kernel.

Kernel signature (_compute_local_logits_stats_kernel):
    _compute_local_logits_stats_kernel(
        target_local_argmax_ptr,        # int64 [num_logits, num_blocks]
        target_local_argmax_stride,     # stride(0)
        target_local_max_ptr,           # fp32 [num_logits, num_blocks]
        target_local_max_stride,        # stride(0)
        target_local_sumexp_ptr,        # fp32 [num_logits, num_blocks]
        target_local_sumexp_stride,     # stride(0)
        draft_local_max_ptr,            # fp32 [num_logits, num_blocks]
        draft_local_max_stride,         # stride(0)
        draft_local_sumexp_ptr,         # fp32 [num_logits, num_blocks]
        draft_local_sumexp_stride,      # stride(0)
        target_logits_ptr,              # fp32 [num_logits, V]
        target_logits_stride,           # stride(0)
        draft_logits_ptr,               # fp32 [max_num_reqs, num_spec_steps, V]
        draft_logits_stride_0,          # stride(0)
        draft_logits_stride_1,          # stride(1)
        expanded_idx_mapping_ptr,       # int64 [num_logits]
        expanded_local_pos_ptr,         # int64 [num_logits]
        temp_ptr,                       # fp32 [max_num_reqs]
        vocab_size,
        num_speculative_steps,
        BLOCK_SIZE: tl.constexpr,
        HAS_DRAFT_LOGITS: tl.constexpr,
    )

Kernel control flow per (logit_idx, block_idx) program:
  1. bonus branch: if expanded_local_pos[logit_idx] >= num_speculative_steps,
     the program returns early and writes NOTHING. (bonus token)
  2. greedy branch: if temp[req]==0.0, writes target_local_argmax and
     target_local_max only. target_local_sumexp / draft_* are NOT written.
  3. non-greedy branch: writes target_local_max and target_local_sumexp.
     If HAS_DRAFT_LOGITS, also writes draft_local_max and draft_local_sumexp.
     draft_logits is stored pre-temperature, so the kernel divides it by
     temp before computing max/sumexp (matches the target sampling
     distribution). target_local_argmax is NOT written in this branch.

Real-world shape (from rejection_sampler.py -> expand_idx_mapping):
  - num_logits = num_reqs * (num_speculative_steps + 1)
  - expanded_idx_mapping = [r,r,...,r (num_spec_steps+1 times), r+1, ...]
    i.e. repeat_interleave(num_spec_steps+1)
  - expanded_local_pos = [0,1,...,num_spec_steps-1, num_spec_steps(bonus)] x num_reqs
    i.e. tile(0..num_spec_steps) repeated num_reqs times
  - draft_logits is indexed as draft_logits[req_state_idx, draft_step_idx, :]
    where draft_step_idx comes from expanded_local_pos (NOT fixed 0).
"""

import math
import torch
import pytest

from vllm.triton_utils import tl, triton
try:
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        _compute_local_logits_stats_kernel,
    )
except ImportError as exc:
    pytest.skip(
        "installed vLLM does not provide _compute_local_logits_stats_kernel; "
        f"precision was not tested: {exc}",
        allow_module_level=True,
    )
from accuracy_test.strict_ut.runtime_npu import init_device_properties_triton


def _compute_max_and_sumexp_ref(logits: torch.Tensor):
    """CPU reference for _compute_block_max_and_sumexp (matches kernel helper)."""
    max_val = float(logits.max().item())
    if max_val > float("-inf"):
        sumexp = float(torch.sum(torch.exp(logits - max_val)).item())
    else:
        sumexp = 0.0
    return max_val, sumexp


def _build_real_expansion(num_reqs, num_speculative_steps, device):
    """Construct expanded_idx_mapping / expanded_local_pos like expand_idx_mapping.

    Each request contributes (num_speculative_steps + 1) logits:
      positions 0 .. num_speculative_steps-1  -> draft steps (kernel computes stats)
      position  num_speculative_steps         -> bonus token (kernel early-returns)
    """
    num_logits_per_req = num_speculative_steps + 1
    total_num_logits = num_reqs * num_logits_per_req

    # expanded_idx_mapping: each req id repeated num_logits_per_req times consecutively
    expanded_idx_mapping = torch.arange(num_reqs, device=device, dtype=torch.int64) \
        .repeat_interleave(num_logits_per_req)

    # expanded_local_pos: [0,1,...,num_spec_steps-1, num_spec_steps] repeated num_reqs times
    expanded_local_pos = torch.arange(num_logits_per_req, device=device, dtype=torch.int64) \
        .repeat(num_reqs)

    assert expanded_idx_mapping.shape[0] == total_num_logits
    assert expanded_local_pos.shape[0] == total_num_logits
    return total_num_logits, expanded_idx_mapping, expanded_local_pos


class TestComputeBlockMaxAndSumexp:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_reqs", [1, 4, 8])
    @pytest.mark.parametrize("vocab_size", [128, 8192, 129280, 248320])
    @pytest.mark.parametrize("num_speculative_steps", [1, 2, 3])
    @pytest.mark.parametrize("has_draft_logits", [True, False])
    def test_compute_block_max_and_sumexp(
        self,
        num_reqs,
        vocab_size,
        num_speculative_steps,
        has_draft_logits,
    ):
        """Compare block-level max/sumexp/argmax with CPU reference.

        Covers:
          - real expansion layout (draft steps + bonus token)
          - bonus early-return branch (outputs untouched)
          - greedy (temp=0) and non-greedy branches per request
          - HAS_DRAFT_LOGITS True/False
          - multi-block vocab (vocab_size > VOCAB_BLOCK_SIZE)
        """
        VOCAB_BLOCK_SIZE = 8192
        vocab_num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)

        total_num_logits, expanded_idx_mapping, expanded_local_pos = \
            _build_real_expansion(num_reqs, num_speculative_steps, self.device)

        # Per-request temperature: mix greedy (0.0) and non-greedy.
        # Use a deterministic pattern that exercises both branches when num_reqs>=2.
        temp_values = [0.0, 1.0, 0.8, 0.0, 1.0, 0.8, 0.0, 1.0]
        temperature = torch.tensor(
            temp_values[:num_reqs], dtype=torch.float32, device=self.device
        )

        target_logits = torch.randn(
            total_num_logits, vocab_size, dtype=torch.float32, device=self.device
        )
        draft_logits = torch.randn(
            num_reqs, num_speculative_steps, vocab_size,
            dtype=torch.float32, device=self.device,
        )

        # Outputs: fill with a sentinel so untouched positions (bonus / greedy
        # sumexp / non-greedy argmax) are distinguishable from real writes.
        sentinel = float("nan")
        target_local_argmax = torch.full(
            (total_num_logits, vocab_num_blocks), -1,
            dtype=torch.int64, device=self.device,
        )
        target_local_max = torch.full(
            (total_num_logits, vocab_num_blocks), sentinel,
            dtype=torch.float32, device=self.device,
        )
        target_local_sumexp = torch.full(
            (total_num_logits, vocab_num_blocks), sentinel,
            dtype=torch.float32, device=self.device,
        )
        draft_local_max = torch.full(
            (total_num_logits, vocab_num_blocks), sentinel,
            dtype=torch.float32, device=self.device,
        )
        draft_local_sumexp = torch.full(
            (total_num_logits, vocab_num_blocks), sentinel,
            dtype=torch.float32, device=self.device,
        )

        _compute_local_logits_stats_kernel[(total_num_logits, vocab_num_blocks)](
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
            HAS_DRAFT_LOGITS=has_draft_logits,
        )
        torch.npu.synchronize()

        # CPU reference (mirror kernel control flow exactly)
        tgt_cpu = target_logits.cpu()
        drf_cpu = draft_logits.cpu()
        idx_map_cpu = expanded_idx_mapping.cpu()
        local_pos_cpu = expanded_local_pos.cpu()
        temp_cpu = temperature.cpu()

        out_argmax = target_local_argmax.cpu()
        out_tmax = target_local_max.cpu()
        out_tsumexp = target_local_sumexp.cpu()
        out_dmax = draft_local_max.cpu()
        out_dsumexp = draft_local_sumexp.cpu()

        for li in range(total_num_logits):
            draft_step = local_pos_cpu[li].item()

            # --- bonus branch: kernel early-returns, all outputs untouched ---
            if draft_step >= num_speculative_steps:
                for bi in range(vocab_num_blocks):
                    assert out_argmax[li, bi].item() == -1, (
                        f"bonus li={li} argmax should be untouched (-1), "
                        f"got {out_argmax[li, bi].item()}"
                    )
                    assert math.isnan(out_tmax[li, bi].item()), (
                        f"bonus li={li} target_max should be untouched (nan), "
                        f"got {out_tmax[li, bi].item()}"
                    )
                    assert math.isnan(out_tsumexp[li, bi].item()), (
                        f"bonus li={li} target_sumexp should be untouched (nan), "
                        f"got {out_tsumexp[li, bi].item()}"
                    )
                    assert math.isnan(out_dmax[li, bi].item()), (
                        f"bonus li={li} draft_max should be untouched (nan), "
                        f"got {out_dmax[li, bi].item()}"
                    )
                    assert math.isnan(out_dsumexp[li, bi].item()), (
                        f"bonus li={li} draft_sumexp should be untouched (nan), "
                        f"got {out_dsumexp[li, bi].item()}"
                    )
                continue

            rs = idx_map_cpu[li].item()
            temp = temp_cpu[rs].item()

            for bi in range(vocab_num_blocks):
                start = bi * VOCAB_BLOCK_SIZE
                end = min(start + VOCAB_BLOCK_SIZE, vocab_size)
                block_logits = tgt_cpu[li, start:end]

                if temp == 0.0:
                    # --- greedy branch: only argmax + target_max written ---
                    expected_max = float(block_logits.max().item())
                    expected_argmax = start + int(block_logits.argmax().item())
                    torch.testing.assert_close(
                        out_tmax[li, bi].item(), expected_max,
                        rtol=1e-5, atol=1e-5,
                    )
                    torch.testing.assert_close(
                        out_argmax[li, bi].item(), expected_argmax,
                        rtol=0, atol=0,
                    )
                    # target_sumexp untouched
                    assert math.isnan(out_tsumexp[li, bi].item()), (
                        f"greedy li={li} bi={bi} target_sumexp should be "
                        f"untouched (nan), got {out_tsumexp[li, bi].item()}"
                    )
                    # draft_* untouched regardless of HAS_DRAFT_LOGITS
                    assert math.isnan(out_dmax[li, bi].item()), (
                        f"greedy li={li} bi={bi} draft_max should be "
                        f"untouched (nan), got {out_dmax[li, bi].item()}"
                    )
                    assert math.isnan(out_dsumexp[li, bi].item()), (
                        f"greedy li={li} bi={bi} draft_sumexp should be "
                        f"untouched (nan), got {out_dsumexp[li, bi].item()}"
                    )
                else:
                    # --- non-greedy branch: target_max + target_sumexp ---
                    expected_max, expected_sumexp = _compute_max_and_sumexp_ref(
                        block_logits
                    )
                    torch.testing.assert_close(
                        out_tmax[li, bi].item(), expected_max,
                        rtol=1e-5, atol=1e-5,
                    )
                    torch.testing.assert_close(
                        out_tsumexp[li, bi].item(), expected_sumexp,
                        rtol=1e-5, atol=1e-5,
                    )
                    # argmax untouched in non-greedy branch
                    assert out_argmax[li, bi].item() == -1, (
                        f"non-greedy li={li} bi={bi} argmax should be "
                        f"untouched (-1), got {out_argmax[li, bi].item()}"
                    )

                    # draft stats: only when HAS_DRAFT_LOGITS. draft_logits is
                    # stored pre-temperature, so the kernel applies the scale
                    # first (draft_block / temp) before max/sumexp, matching
                    # the target sampling distribution.
                    # Indexing: draft_logits[rs, draft_step, :]
                    if has_draft_logits:
                        drf_block = drf_cpu[rs, draft_step, start:end] / temp
                        dmax, dsumexp = _compute_max_and_sumexp_ref(drf_block)
                        torch.testing.assert_close(
                            out_dmax[li, bi].item(), dmax,
                            rtol=1e-5, atol=1e-5,
                        )
                        torch.testing.assert_close(
                            out_dsumexp[li, bi].item(), dsumexp,
                            rtol=1e-5, atol=1e-5,
                        )
                    else:
                        assert math.isnan(out_dmax[li, bi].item()), (
                            f"HAS_DRAFT_LOGITS=False li={li} bi={bi} "
                            f"draft_max should be untouched (nan), "
                            f"got {out_dmax[li, bi].item()}"
                        )
                        assert math.isnan(out_dsumexp[li, bi].item()), (
                            f"HAS_DRAFT_LOGITS=False li={li} bi={bi} "
                            f"draft_sumexp should be untouched (nan), "
                            f"got {out_dsumexp[li, bi].item()}"
                        )

    def test_all_neg_inf(self):
        """Vocab block where all target/draft logits are -inf.

        Greedy branch: max=-inf, argmax=block_start (argmax of all-equal -inf
        returns index 0). Non-greedy branch: max=-inf, sumexp=0.
        """
        num_reqs = 2
        num_speculative_steps = 2
        vocab_size = 8192
        VOCAB_BLOCK_SIZE = 8192
        vocab_num_blocks = 1

        total_num_logits, expanded_idx_mapping, expanded_local_pos = \
            _build_real_expansion(num_reqs, num_speculative_steps, self.device)

        target_logits = torch.full(
            (total_num_logits, vocab_size), float("-inf"),
            dtype=torch.float32, device=self.device,
        )
        draft_logits = torch.full(
            (num_reqs, num_speculative_steps, vocab_size), float("-inf"),
            dtype=torch.float32, device=self.device,
        )
        # req0 greedy, req1 non-greedy -> covers both -inf paths
        temperature = torch.tensor(
            [0.0, 1.0], dtype=torch.float32, device=self.device
        )

        sentinel = float("nan")
        target_local_argmax = torch.full(
            (total_num_logits, vocab_num_blocks), -1,
            dtype=torch.int64, device=self.device,
        )
        target_local_max = torch.full(
            (total_num_logits, vocab_num_blocks), sentinel,
            dtype=torch.float32, device=self.device,
        )
        target_local_sumexp = torch.full(
            (total_num_logits, vocab_num_blocks), sentinel,
            dtype=torch.float32, device=self.device,
        )
        draft_local_max = torch.full(
            (total_num_logits, vocab_num_blocks), sentinel,
            dtype=torch.float32, device=self.device,
        )
        draft_local_sumexp = torch.full(
            (total_num_logits, vocab_num_blocks), sentinel,
            dtype=torch.float32, device=self.device,
        )

        _compute_local_logits_stats_kernel[(total_num_logits, vocab_num_blocks)](
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
        torch.npu.synchronize()

        local_pos_cpu = expanded_local_pos.cpu()
        idx_map_cpu = expanded_idx_mapping.cpu()
        temp_cpu = temperature.cpu()

        for li in range(total_num_logits):
            draft_step = local_pos_cpu[li].item()
            if draft_step >= num_speculative_steps:
                # bonus: untouched
                assert math.isnan(target_local_max[li, 0].item())
                continue

            rs = idx_map_cpu[li].item()
            temp = temp_cpu[rs].item()
            if temp == 0.0:
                # greedy -inf: max=-inf, argmax=0 (first index of all-equal)
                assert target_local_max[li, 0].item() == float("-inf"), \
                    "greedy -inf max should be -inf"
                assert target_local_argmax[li, 0].item() == 0, \
                    "greedy -inf argmax should be 0 (first of all-equal)"
            else:
                # non-greedy -inf: max=-inf, sumexp=0
                assert target_local_max[li, 0].item() == float("-inf"), \
                    "non-greedy -inf max should be -inf"
                assert target_local_sumexp[li, 0].item() == 0.0, \
                    "non-greedy -inf sumexp should be 0"
                assert draft_local_max[li, 0].item() == float("-inf"), \
                    "draft -inf max should be -inf"
                assert draft_local_sumexp[li, 0].item() == 0.0, \
                    "draft -inf sumexp should be 0"
