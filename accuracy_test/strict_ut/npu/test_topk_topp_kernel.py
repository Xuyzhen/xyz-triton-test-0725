# GENERATED STRICT UT. Source: accuracy_test/codex/existing_accuracy_tests/from_vllm/test_topk_topp_kernel.py
# Do not edit mechanically; update the reviewed Codex source or strict generator.
# Standalone Ascend A3 adaptation of an upstream vLLM accuracy path.
# Accuracy UT source: vllm/tests/v1/sample/test_topk_topp_sampler.py
# Kernel source: vllm/vllm/v1/sample/ops/topk_topp_triton.py
# Coverage: _topk_topp_kernel

# vLLM vanilla kernel: _topk_topp_kernel from
# vllm/vllm/v1/sample/ops/topk_topp_triton.py

"""
Precision test for _topk_topp_kernel.

Combined top-k / top-p masking kernel. For each row:
1. Optionally applies top-k (keeps only k largest logits per row).
2. Then optionally applies top-p on the survivors (keeps smallest set whose
   cumulative softmax probability exceeds p).

Kernel signature:
    _topk_topp_kernel(
        LOGITS,                      # [batch_size, vocab_size] float32 (in-place)
        LOGITS_STRIDE_0,             # stride(0) of LOGITS
        BUFFER,                      # [num_programs, vocab_size] float32 workspace
        PERCENTILE_TO_STD_TABLE,     # [200] float32 lookup table
        NORMAL_CDF_TO_SIGMA_TABLE,   # [200] float32 lookup table
        K,                           # [batch_size] int32 top-k values
        P,                           # [batch_size] float32 top-p values
        BATCH_SIZE,
        VOCAB_SIZE: tl.constexpr,
        MASK_VALUE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_SIZE_TRUNC: tl.constexpr,
        TOPK_ENABLED: tl.constexpr,
        TOPP_ENABLED: tl.constexpr,
    )

Realistic shapes:
  - vocab_size: 32000 (Llama2), 128256 (Llama3), 129280 (DeepSeek V4),
    163840 (Kimi K3), 248320 (Qwen3 2.4T)
  - batch_size: 1 (single-stream decode), 4, 16, 64 (concurrent requests)
  - k: 1..20 (typical top-k sampling), or vocab_size (disabled)
  - p: 0.8..0.99 (typical top-p), or 1.0 (disabled)
"""
from __future__ import annotations

import pytest
import torch

from accuracy_test.strict_ut.runtime_npu import DEVICE, init_device_properties_triton, synchronize
from vllm.triton_utils import tl, triton
from vllm.v1.sample.ops.topk_topp_triton import (
    _NORMAL_CDF_TO_SIGMA_TABLE,
    _PERCENTILE_TO_STD_TABLE,
    _topk_topp_kernel,
)

pytestmark = [pytest.mark.npu]

BLOCK_SIZE = 32
BLOCK_SIZE_TRUNC = 16


def _apply_topk_topp_cpu(
    logits: torch.Tensor,
    k: torch.Tensor | None = None,
    p: torch.Tensor | None = None,
    mask_value: float = float("-inf"),
) -> torch.Tensor:
    """Pure PyTorch CPU reference for combined top-k / top-p masking.

    For each row:
      1. If top-k is enabled: find the k largest logits, derive a pivot
         threshold, keep values >= pivot, and handle duplicate boundary values.
      2. If top-p is enabled (after top-k): softmax survivors, find smallest
         set with cumulative probability > p, mask the rest.
    """
    logits = logits.clone()
    batch_size, vocab_size = logits.shape

    for row in range(batch_size):
        row_logits = logits[row]

        if k is not None and k[row] < vocab_size:
            kval = int(k[row])
            topk_vals, _ = torch.topk(row_logits, kval, sorted=True)
            pivot = topk_vals[-1].item()
            num_above_pivot = (row_logits > pivot).sum().item()
            num_equal_pivot = (row_logits == pivot).sum().item()
            num_keep = kval - num_above_pivot

            mask = row_logits < pivot
            equal_mask = row_logits == pivot
            if num_keep < num_equal_pivot:
                keep_indices = torch.where(equal_mask)[0]
                discard = keep_indices[num_keep:]
                mask[discard] = True
            row_logits[mask] = mask_value

        if p is not None:
            pval = float(p[row])
            if pval < 1.0:
                max_logit = row_logits.max()
                if max_logit == float("-inf"):
                    continue
                probs = torch.softmax(row_logits, dim=-1)
                sorted_probs, _ = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=0)
                cutoff_idx = torch.searchsorted(cumsum, pval).item()
                if cutoff_idx < len(cumsum):
                    threshold_prob = sorted_probs[cutoff_idx].item()
                    above_p = (probs > threshold_prob).sum().item()
                    equal_p_count = (probs == threshold_prob).sum().item()
                    num_keep = cutoff_idx + 1 - above_p

                    mask_p = probs < threshold_prob
                    equal_mask_p = probs == threshold_prob
                    if 0 < num_keep < equal_p_count:
                        keep_indices = torch.where(equal_mask_p)[0]
                        discard = keep_indices[int(num_keep):]
                        mask_p[discard] = True
                    row_logits[mask_p] = mask_value

    return logits


def _launch(logits, k, p, batch_size, vocab_size, topk_enabled, topp_enabled):
    buffer = torch.empty(1, vocab_size, dtype=torch.float32, device=DEVICE)
    percentile_table = logits.new_tensor(_PERCENTILE_TO_STD_TABLE)
    normal_cdf_table = logits.new_tensor(_NORMAL_CDF_TO_SIGMA_TABLE)

    _topk_topp_kernel[(1,)](
        logits,
        logits.stride(0),
        buffer,
        percentile_table,
        normal_cdf_table,
        k,
        p,
        BATCH_SIZE=batch_size,
        MASK_VALUE=float("-inf"),
        VOCAB_SIZE=vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_SIZE_TRUNC=BLOCK_SIZE_TRUNC,
        TOPK_ENABLED=topk_enabled,
        TOPP_ENABLED=topp_enabled,
    )
    synchronize()


@pytest.mark.parametrize(
    "batch_size,vocab_size",
    [
        (1, 32000),    # Llama2 single-stream decode
        (4, 32000),    # Llama2 small batch
        (16, 32000),   # Llama2 concurrent decode
        (1, 129280),   # DeepSeek V4 single-stream
        (4, 129280),   # DeepSeek V4 small batch
        (1, 248320),   # Qwen3 2.4T single-stream
    ],
)
@pytest.mark.parametrize("mode", ["topk_only", "topp_only", "topk_topp"])
def test_topk_topp_realistic(batch_size, vocab_size, mode):
    """Top-k/top-p masking with realistic vocab sizes and batch sizes."""
    init_device_properties_triton()
    torch.manual_seed(42)

    topk_enabled = mode in ("topk_only", "topk_topp")
    topp_enabled = mode in ("topp_only", "topk_topp")

    logits_cpu = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    if topk_enabled:
        k = torch.randint(5, 21, (batch_size,), dtype=torch.int32)
    else:
        # When top-k is disabled, k >= vocab_size fully disables the top-k path
        k = torch.full((batch_size,), vocab_size, dtype=torch.int32)
    if topp_enabled:
        p = torch.rand(batch_size, dtype=torch.float32) * 0.19 + 0.8  # 0.8..0.99
    else:
        p = torch.full((batch_size,), 1.0, dtype=torch.float32)

    logits_npu = logits_cpu.to(DEVICE)
    k_npu = k.to(DEVICE)
    p_npu = p.to(DEVICE)

    _launch(logits_npu, k_npu, p_npu, batch_size, vocab_size, topk_enabled, topp_enabled)

    k_ref = None if (topk_enabled and (k >= vocab_size).any()) or not topk_enabled else k
    p_ref = None if not topp_enabled else p
    expected = _apply_topk_topp_cpu(
        logits_cpu, k_ref if topk_enabled else None, p_ref if topp_enabled else None
    )
    torch.testing.assert_close(logits_npu.cpu(), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("vocab_size", [32000, 129280])
def test_topk_exact_count(batch_size, vocab_size):
    """Top-k only: verify exactly k tokens survive per row."""
    init_device_properties_triton()
    torch.manual_seed(7)

    logits_cpu = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    k = torch.randint(5, 21, (batch_size,), dtype=torch.int32)

    logits_npu = logits_cpu.to(DEVICE)
    k_npu = k.to(DEVICE)
    p_npu = torch.full((batch_size,), 1.0, dtype=torch.float32, device=DEVICE)

    _launch(logits_npu, k_npu, p_npu, batch_size, vocab_size, True, False)

    for row in range(batch_size):
        num_surviving = (logits_npu[row] > float("-inf")).sum().item()
        assert num_surviving == min(int(k[row]), vocab_size), (
            f"Row {row}: expected {min(int(k[row]), vocab_size)} survivors, "
            f"got {num_surviving}"
        )


def test_topk_duplicate_boundary():
    """Top-k with duplicate values at the k boundary.

    When many tokens share the same logit value at the k-th position,
    the kernel should keep exactly k tokens, handling duplicates correctly.
    """
    init_device_properties_triton()
    torch.manual_seed(8)

    batch_size = 1
    vocab_size = 32000
    logits_cpu = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    # Force duplicate values: set 20 entries to the same value at the boundary
    logits_cpu[0, 10:30] = logits_cpu[0, 9].item()
    k = torch.tensor([12], dtype=torch.int32)

    logits_npu = logits_cpu.to(DEVICE)
    k_npu = k.to(DEVICE)
    p_npu = torch.full((batch_size,), 1.0, dtype=torch.float32, device=DEVICE)

    _launch(logits_npu, k_npu, p_npu, batch_size, vocab_size, True, False)

    num_surviving = (logits_npu[0] > float("-inf")).sum().item()
    assert num_surviving == 12, (
        f"Expected 12 survivors with duplicate boundary, got {num_surviving}"
    )
