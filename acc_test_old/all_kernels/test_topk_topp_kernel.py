# vLLM vanilla kernel: _topk_topp_kernel from
# vllm/vllm/v1/sample/ops/topk_topp_triton.py

"""
Precision test for _topk_topp_kernel (vanilla vLLM version).

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
"""

import torch
import pytest

from vllm.triton_utils import tl, triton
from vllm.v1.sample.ops.topk_topp_triton import (
    _topk_topp_kernel,
    _PERCENTILE_TO_STD_TABLE,
    _NORMAL_CDF_TO_SIGMA_TABLE,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


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


class TestTopkToppKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("vocab_size", [64, 128])
    @pytest.mark.parametrize("topk_enabled", [True, False])
    @pytest.mark.parametrize("topp_enabled", [True, False])
    def test_topk_topp_combined(
        self, batch_size, vocab_size, topk_enabled, topp_enabled
    ):
        """Direct kernel launch with parametrized batch,vocab,topk,topp."""
        if not topk_enabled and not topp_enabled:
            pytest.skip("At least one of topk/topp must be enabled")

        torch.manual_seed(42)
        BLOCK_SIZE = 32
        BLOCK_SIZE_TRUNC = 16

        logits_cpu = torch.randn(batch_size, vocab_size, dtype=torch.float32)
        if topk_enabled:
            k = torch.randint(5, max(10, vocab_size // 2), (batch_size,), dtype=torch.int32)
        else:
            # When top-k is disabled, the kernel still uses top-k path for
            # outlier gathering if k < vocab_size. To fully disable top-k,
            # use k >= vocab_size.
            k = torch.full((batch_size,), vocab_size, dtype=torch.int32)
        if topp_enabled:
            p = torch.rand(batch_size, dtype=torch.float32) * 0.8 + 0.1
        else:
            p = torch.full((batch_size,), 1.0, dtype=torch.float32)

        logits_npu = logits_cpu.to(self.device)
        k_npu = k.to(self.device)
        p_npu = p.to(self.device)

        buffer = torch.empty(1, vocab_size, dtype=torch.float32, device=self.device)
        percentile_table = logits_npu.new_tensor(_PERCENTILE_TO_STD_TABLE)
        normal_cdf_table = logits_npu.new_tensor(_NORMAL_CDF_TO_SIGMA_TABLE)

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
            TOPK_ENABLED=topk_enabled,
            TOPP_ENABLED=topp_enabled,
        )
        torch.npu.synchronize()

        k_ref = None if (topk_enabled and (k >= vocab_size).any()) or not topk_enabled else k
        p_ref = None if not topp_enabled else p
        expected = _apply_topk_topp_cpu(logits_cpu, k_ref if topk_enabled else None,
                                        p_ref if topp_enabled else None)
        torch.testing.assert_close(logits_npu.cpu(), expected, rtol=1e-5, atol=1e-5)

    def test_topk_only_exact_count(self):
        """Top-k only: verify exactly k tokens survive per row."""
        torch.manual_seed(7)
        batch_size = 3
        vocab_size = 128
        BLOCK_SIZE = 32
        BLOCK_SIZE_TRUNC = 16

        logits_cpu = torch.randn(batch_size, vocab_size, dtype=torch.float32)
        k = torch.tensor([5, 10, 20], dtype=torch.int32)

        logits_npu = logits_cpu.to(self.device)
        k_npu = k.to(self.device)
        p_npu = torch.full((batch_size,), 1.0, dtype=torch.float32, device=self.device)

        buffer = torch.empty(1, vocab_size, dtype=torch.float32, device=self.device)
        percentile_table = logits_npu.new_tensor(_PERCENTILE_TO_STD_TABLE)
        normal_cdf_table = logits_npu.new_tensor(_NORMAL_CDF_TO_SIGMA_TABLE)

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
            TOPP_ENABLED=False,
        )
        torch.npu.synchronize()

        for row in range(batch_size):
            num_surviving = (logits_npu[row] > float("-inf")).sum().item()
            assert num_surviving == min(int(k[row]), vocab_size), (
                f"Row {row}: expected {min(int(k[row]), vocab_size)} survivors, "
                f"got {num_surviving}"
            )

    def test_topk_duplicate_boundary(self):
        """Top-k with duplicate values at the k boundary.

        When many tokens share the same logit value at the k-th position,
        the kernel should keep exactly k tokens, handling duplicates correctly.
        """
        torch.manual_seed(8)
        batch_size = 1
        vocab_size = 64
        BLOCK_SIZE = 32
        BLOCK_SIZE_TRUNC = 16

        logits_cpu = torch.randn(batch_size, vocab_size, dtype=torch.float32)
        # Force duplicate values: set many entries to the same value
        logits_cpu[0, 10:30] = logits_cpu[0, 9].item()  # 20 duplicates of the 10th value
        k = torch.tensor([12], dtype=torch.int32)

        logits_npu = logits_cpu.to(self.device)
        k_npu = k.to(self.device)
        p_npu = torch.full((batch_size,), 1.0, dtype=torch.float32, device=self.device)

        buffer = torch.empty(1, vocab_size, dtype=torch.float32, device=self.device)
        percentile_table = logits_npu.new_tensor(_PERCENTILE_TO_STD_TABLE)
        normal_cdf_table = logits_npu.new_tensor(_NORMAL_CDF_TO_SIGMA_TABLE)

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
            TOPP_ENABLED=False,
        )
        torch.npu.synchronize()

        num_surviving = (logits_npu[0] > float("-inf")).sum().item()
        assert num_surviving == int(k[0]), (
            f"Expected {int(k[0])} survivors, got {num_surviving}"
        )

    def test_topp_cumulative_probability(self):
        """Top-p only: verify cumulative probability constraint is satisfied."""
        torch.manual_seed(9)
        batch_size = 2
        vocab_size = 128
        BLOCK_SIZE = 32
        BLOCK_SIZE_TRUNC = 16

        logits_cpu = torch.randn(batch_size, vocab_size, dtype=torch.float32)
        p = torch.tensor([0.7, 0.9], dtype=torch.float32)
        # k >= vocab_size to effectively disable top-k
        k = torch.full((batch_size,), vocab_size, dtype=torch.int32)

        logits_npu = logits_cpu.to(self.device)
        k_npu = k.to(self.device)
        p_npu = p.to(self.device)

        buffer = torch.empty(1, vocab_size, dtype=torch.float32, device=self.device)
        percentile_table = logits_npu.new_tensor(_PERCENTILE_TO_STD_TABLE)
        normal_cdf_table = logits_npu.new_tensor(_NORMAL_CDF_TO_SIGMA_TABLE)

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
            TOPK_ENABLED=False,
            TOPP_ENABLED=True,
        )
        torch.npu.synchronize()

        for row in range(batch_size):
            row_result = logits_npu[row]
            surviving_mask = row_result > float("-inf")
            num_surviving = surviving_mask.sum().item()
            if num_surviving == 0:
                continue
            probs = torch.softmax(row_result.clone().clamp(min=float("-inf")), dim=-1)
            cum_prob = probs[surviving_mask].sum().item()
            assert cum_prob >= float(p[row]) - 0.05, (
                f"Row {row}: cumulative prob {cum_prob:.4f} < p={float(p[row]):.4f}"
            )

    def test_topk_topp_noop(self):
        """When k >= vocab_size and p >= 1.0, logits should be unchanged."""
        torch.manual_seed(10)
        batch_size = 2
        vocab_size = 32
        BLOCK_SIZE = 32
        BLOCK_SIZE_TRUNC = 16

        logits_cpu = torch.randn(batch_size, vocab_size, dtype=torch.float32)
        logits_npu = logits_cpu.clone().to(self.device)
        k_npu = torch.full((batch_size,), vocab_size, dtype=torch.int32, device=self.device)
        p_npu = torch.full((batch_size,), 1.0, dtype=torch.float32, device=self.device)

        buffer = torch.empty(1, vocab_size, dtype=torch.float32, device=self.device)
        percentile_table = logits_npu.new_tensor(_PERCENTILE_TO_STD_TABLE)
        normal_cdf_table = logits_npu.new_tensor(_NORMAL_CDF_TO_SIGMA_TABLE)

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

        torch.testing.assert_close(logits_npu.cpu(), logits_cpu, rtol=0, atol=0)
