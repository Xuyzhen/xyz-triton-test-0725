# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.sample.ops.topk_topp_triton import (
    _topk_topp_kernel,
    apply_top_k_top_p_triton,
)

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _apply_topk_topp_cpu(
    logits: torch.Tensor,
    k: torch.Tensor | None = None,
    p: torch.Tensor | None = None,
    mask_value: float = float("-inf"),
) -> torch.Tensor:
    """Pure PyTorch CPU reference implementation of top-k and top-p masking.

    For each row:
      1. If top-k is enabled: find the top-k largest logit values, derive a
         pivot threshold, and mask everything below that pivot to mask_value.
         Duplicate logit values at the boundary are handled such that exactly k
         tokens are kept.
      2. If top-p is enabled (after top-k): softmax the surviving logits,
         find the smallest set of tokens whose cumulative probability exceeds p,
         and mask the rest to mask_value.
    """
    logits = logits.clone()
    batch_size, vocab_size = logits.shape

    for row in range(batch_size):
        row_logits = logits[row]

        if k is not None and k[row] < vocab_size:
            kval = int(k[row])
            # Find top-k values
            topk_vals, topk_idxs = torch.topk(row_logits, kval, sorted=True)
            pivot = topk_vals[-1].item()
            # Handle duplicates at boundary: if pivot appears more than once
            # beyond what we already counted, keep only enough to reach k.
            num_above_pivot = (row_logits > pivot).sum().item()
            num_equal_pivot = (row_logits == pivot).sum().item()
            num_keep = kval - num_above_pivot

            mask = row_logits < pivot
            equal_mask = row_logits == pivot
            if num_keep < num_equal_pivot:
                # Keep only num_keep of the equal values.
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
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=0)
                # Find cutoff index where cumulative sum >= p
                cutoff_idx = torch.searchsorted(cumsum, pval).item()
                if cutoff_idx < len(cumsum):
                    threshold_prob = sorted_probs[cutoff_idx].item()
                    # Handle duplicates at boundary
                    above_p = (probs > threshold_prob).sum().item()
                    equal_p_count = (probs == threshold_prob).sum().item()
                    num_keep = cutoff_idx + 1 - above_p

                    mask = probs < threshold_prob
                    equal_mask = probs == threshold_prob
                    if 0 < num_keep < equal_p_count:
                        keep_indices = torch.where(equal_mask)[0]
                        discard = keep_indices[int(num_keep):]
                        mask[discard] = True
                    row_logits[mask] = mask_value

    return logits


def test_topk_topp_via_api() -> None:
    """Combined top-k / top-p via the public ``apply_top_k_top_p_triton`` API.

    Verifies that the end-to-end function matches a PyTorch reference for
    small batch / vocab size.
    """
    init_device_properties_triton()
    torch.manual_seed(42)

    batch_size = 4
    vocab_size = 128

    device = torch.device("npu")

    logits_cpu = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    k = torch.randint(5, 50, (batch_size,), dtype=torch.int32)
    p = torch.rand(batch_size, dtype=torch.float32) * 0.8 + 0.1  # [0.1, 0.9]

    expected = _apply_topk_topp_cpu(logits_cpu, k, p)

    logits_npu = logits_cpu.to(device)
    k_npu = k.to(device)
    p_npu = p.to(device)

    result = apply_top_k_top_p_triton(logits_npu, k_npu, p_npu)
    torch.npu.synchronize()

    torch.testing.assert_close(result.cpu(), expected, rtol=1e-5, atol=1e-5)


def test_topk_only() -> None:
    """Top-k only (p=None): verify only k tokens survive per row."""
    init_device_properties_triton()
    torch.manual_seed(1)

    batch_size = 2
    vocab_size = 64

    device = torch.device("npu")

    logits_cpu = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    k = torch.tensor([5, 10], dtype=torch.int32)

    expected = _apply_topk_topp_cpu(logits_cpu, k, None)

    logits_npu = logits_cpu.to(device)
    k_npu = k.to(device)

    result = apply_top_k_top_p_triton(logits_npu, k_npu, None)
    torch.npu.synchronize()

    torch.testing.assert_close(result.cpu(), expected, rtol=1e-5, atol=1e-5)

    # Verify exactly k tokens survive per row (or all if k >= vocab_size).
    for row in range(batch_size):
        num_surviving = (result[row] > float("-inf")).sum().item()
        assert num_surviving == min(int(k[row]), vocab_size), (
            f"Row {row}: expected {min(int(k[row]), vocab_size)} survivors, "
            f"got {num_surviving}"
        )


def test_topp_only() -> None:
    """Top-p only (k=None): verify cumulative probability constraint."""
    init_device_properties_triton()
    torch.manual_seed(2)

    batch_size = 2
    vocab_size = 64

    device = torch.device("npu")

    logits_cpu = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    p = torch.tensor([0.5, 0.9], dtype=torch.float32)

    expected = _apply_topk_topp_cpu(logits_cpu, None, p)

    logits_npu = logits_cpu.to(device)
    p_npu = p.to(device)

    result = apply_top_k_top_p_triton(logits_npu, None, p_npu)
    torch.npu.synchronize()

    torch.testing.assert_close(result.cpu(), expected, rtol=1e-5, atol=1e-5)

    # Verify cumulative probability of survivors meets p threshold.
    for row in range(batch_size):
        row_result = result[row]
        surviving_mask = row_result > float("-inf")
        num_surviving = surviving_mask.sum().item()
        if num_surviving == 0:
            continue
        probs = torch.softmax(row_result.clone().clamp(min=float("-inf")), dim=-1)
        # Masked positions get near-zero prob.
        cum_prob = probs[surviving_mask].sum().item()
        assert cum_prob >= float(p[row]) - 0.05, (
            f"Row {row}: cumulative prob {cum_prob:.4f} < p={float(p[row]):.4f}"
        )


def test_topk_topp_direct_kernel_small() -> None:
    """Direct ``_topk_topp_kernel`` launch with small parameters.

    Verifies the kernel masks values correctly when launched directly.
    """
    init_device_properties_triton()
    torch.manual_seed(3)

    batch_size = 1
    vocab_size = 64
    BLOCK_SIZE = 32
    BLOCK_SIZE_TRUNC = 16

    device = torch.device("npu")

    logits_cpu = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    k = torch.tensor([5], dtype=torch.int32)
    p = torch.tensor([0.5], dtype=torch.float32)

    logits_npu = logits_cpu.to(device)
    k_npu = k.to(device)
    p_npu = p.to(device)

    buffer = torch.empty(1, vocab_size, dtype=torch.float32, device=device)
    percentile_table = logits_npu.new_tensor(
        [2.576, 2.319, 2.178, 2.064, 1.968, 1.892, 1.819, 1.757, 1.708, 1.659,
         1.616, 1.568, 1.526, 1.492, 1.456, 1.420, 1.382, 1.342, 1.309, 1.280,
         1.249, 1.221, 1.193, 1.169, 1.145, 1.121, 1.095, 1.073, 1.050, 1.030,
         1.008, 0.987, 0.966, 0.945, 0.926, 0.910, 0.891, 0.871, 0.854, 0.837,
         0.819, 0.803, 0.784, 0.767, 0.753, 0.734, 0.719, 0.702, 0.690, 0.675,
         0.658, 0.640, 0.625, 0.609, 0.595, 0.578, 0.564, 0.550, 0.537, 0.521,
         0.509, 0.495, 0.481, 0.466, 0.453, 0.439, 0.424, 0.410, 0.397, 0.383,
         0.370, 0.356, 0.343, 0.330, 0.316, 0.302, 0.289, 0.274, 0.261, 0.247,
         0.235, 0.223, 0.209, 0.196, 0.184, 0.172, 0.159, 0.149, 0.137, 0.124,
         0.112, 0.100, 0.086, 0.074, 0.062, 0.050, 0.035, 0.023, 0.009, -0.003,
         -0.015, -0.027, -0.039, -0.052, -0.063, -0.074, -0.085, -0.097, -0.109,
         -0.122, -0.134, -0.147, -0.158, -0.171, -0.184, -0.196, -0.210, -0.223,
         -0.235, -0.248, -0.261, -0.275, -0.289, -0.302, -0.317, -0.328, -0.341,
         -0.353, -0.368, -0.382, -0.396, -0.410, -0.426, -0.439, -0.452, -0.465,
         -0.480, -0.493, -0.507, -0.521, -0.537, -0.551, -0.568, -0.582, -0.597,
         -0.614, -0.628, -0.643, -0.658, -0.673, -0.691, -0.706, -0.721, -0.738,
         -0.754, -0.769, -0.789, -0.808, -0.824, -0.838, -0.857, -0.877, -0.893,
         -0.912, -0.929, -0.947, -0.965, -0.983, -1.003, -1.027, -1.050, -1.070,
         -1.092, -1.117, -1.139, -1.162, -1.189, -1.216, -1.241, -1.272, -1.300,
         -1.330, -1.367, -1.404, -1.441, -1.485, -1.523, -1.564, -1.607, -1.658,
         -1.710, -1.778, -1.832, -1.901, -1.978, -2.068, -2.174, -2.325, -2.577,
         -3.813],
    )
    normal_cdf_table = logits_npu.new_tensor(
        [3.656, 3.650, 3.650, 3.650, 3.626, 3.626, 3.626, 3.514, 3.514, 3.503,
         3.503, 3.434, 3.434, 3.428, 3.428, 3.387, 3.380, 3.380, 3.376, 3.373,
         3.373, 3.356, 3.354, 3.354, 3.291, 3.249, 3.234, 3.214, 3.198, 3.198,
         3.185, 3.177, 3.177, 3.165, 3.164, 3.161, 3.138, 3.120, 3.115, 3.113,
         3.093, 3.066, 3.054, 3.043, 3.037, 3.023, 2.993, 2.991, 2.976, 2.970,
         2.952, 2.946, 2.932, 2.908, 2.902, 2.895, 2.886, 2.874, 2.861, 2.844,
         2.836, 2.810, 2.801, 2.790, 2.784, 2.779, 2.767, 2.757, 2.745, 2.733,
         2.723, 2.716, 2.693, 2.678, 2.671, 2.656, 2.649, 2.629, 2.611, 2.595,
         2.592, 2.585, 2.574, 2.550, 2.543, 2.534, 2.521, 2.518, 2.497, 2.485,
         2.468, 2.450, 2.441, 2.430, 2.412, 2.402, 2.389, 2.383, 2.377, 2.364,
         2.349, 2.338, 2.332, 2.319, 2.310, 2.301, 2.282, 2.274, 2.266, 2.250,
         2.242, 2.236, 2.226, 2.215, 2.207, 2.196, 2.179, 2.171, 2.162, 2.147,
         2.135, 2.121, 2.109, 2.095, 2.085, 2.073, 2.063, 2.045, 2.030, 2.016,
         2.003, 1.992, 1.983, 1.972, 1.960, 1.949, 1.940, 1.928, 1.912, 1.897,
         1.881, 1.869, 1.854, 1.838, 1.824, 1.807, 1.792, 1.779, 1.764, 1.751,
         1.739, 1.726, 1.711, 1.697, 1.685, 1.668, 1.652, 1.636, 1.622, 1.603,
         1.585, 1.568, 1.551, 1.534, 1.513, 1.499, 1.480, 1.464, 1.441, 1.422,
         1.394, 1.373, 1.347, 1.320, 1.296, 1.270, 1.246, 1.219, 1.190, 1.163,
         1.135, 1.104, 1.073, 1.041, 1.006, 0.969, 0.931, 0.894, 0.851, 0.806,
         0.757, 0.702, 0.643, 0.574, 0.498, 0.405, 0.288, 0.134, -0.110, -3.813],
    )

    _topk_topp_kernel[(1,)](
        logits_npu,
        logits_npu.stride(0),
        buffer,
        percentile_table,
        normal_cdf_table,
        k_npu,
        p_npu,
        BATCH_SIZE=batch_size,
        MASK_VALUE=float("-inf"),
        VOCAB_SIZE=vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_SIZE_TRUNC=BLOCK_SIZE_TRUNC,
        TOPK_ENABLED=True,
        TOPP_ENABLED=True,
    )
    torch.npu.synchronize()

    expected = _apply_topk_topp_cpu(logits_cpu, k, p)
    torch.testing.assert_close(logits_npu.cpu(), expected, rtol=1e-5, atol=1e-5)


def test_topk_topp_noop_params() -> None:
    """Top-k with k >= vocab_size and top-p with p >= 1.0 should be no-ops."""
    init_device_properties_triton()
    torch.manual_seed(5)

    batch_size = 2
    vocab_size = 32

    device = torch.device("npu")

    logits_cpu = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    logits_npu = logits_cpu.clone().to(device)

    # k >= vocab_size, p >= 1.0 => no masking.
    k = torch.full((batch_size,), vocab_size, dtype=torch.int32)
    p = torch.full((batch_size,), 1.0, dtype=torch.float32)
    k_npu = k.to(device)
    p_npu = p.to(device)

    result = apply_top_k_top_p_triton(logits_npu, k_npu, p_npu)
    torch.npu.synchronize()

    torch.testing.assert_close(result.cpu(), logits_cpu, rtol=0, atol=0)


def test_topk_topp_all_masked() -> None:
    """Both k and p are None should return logits unchanged."""
    init_device_properties_triton()
    torch.manual_seed(6)

    batch_size = 2
    vocab_size = 32

    device = torch.device("npu")

    logits_cpu = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    logits_npu = logits_cpu.clone().to(device)

    result = apply_top_k_top_p_triton(logits_npu, None, None)
    torch.npu.synchronize()

    torch.testing.assert_close(result.cpu(), logits_cpu, rtol=0, atol=0)
