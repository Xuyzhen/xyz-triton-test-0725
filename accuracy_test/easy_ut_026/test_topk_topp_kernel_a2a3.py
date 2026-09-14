# Standalone Ascend A2/A3 precision UT for the fused top-k/top-p replacement.
# Accuracy UT source: vllm/tests/v1/sample/test_topk_topp_sampler.py
# Replaced implementation: vllm/vllm/v1/sample/ops/topk_topp_triton.py (_topk_topp_kernel)
# Tested implementation: torch_npu.npu_top_k_top_p
#   == vllm_ascend/sample/sampler.py::_apply_top_k_top_p_torch_npu (the non-TP
#   branch), which is the production sampling path ONLY on A2/A3 devices:
#
#       apply_top_k_top_p = (
#           _apply_top_k_top_p_torch_npu       # A2 / A3
#           if get_ascend_device_type() in [AscendDeviceType.A2, AscendDeviceType.A3]
#           else _apply_top_k_top_p_pytorch    # A5 -> see test_topk_topp_kernel_a5.py
#       )
#
# The A5 counterpart UT (test_topk_topp_kernel_a5.py) tests the PyTorch
# fallback implementation. Both UTs intentionally coexist so each device
# family validates the implementation its production path actually calls.

"""
Precision test for the fused NPU top-k/top-p operator (A2/A3 sampling path).

Why the Triton kernel is not launched directly:
    The upstream ``_topk_topp_kernel`` calls ``tl.cumsum`` on a bool tensor
    (``outlier_mask``). The Ascend Triton backend does not ship a bool cumsum
    library function, so compilation fails with ``undefined symbol:
    _mlir_ciface_cumsum_1d_bool_dim0``. vllm-ascend does not fix the kernel;
    it replaces the whole top-k/top-p step per device family. On A2/A3 the
    replacement is the fused NPU operator ``torch_npu.npu_top_k_top_p``
    (vllm_ascend/sample/sampler.py, ``_apply_top_k_top_p_torch_npu``).
    This UT validates the precision of that NPU replacement operator
    against a CPU reference.

Operator under test (torch_npu.npu_top_k_top_p(logits, k=k, p=p)):
    logits : [batch_size, vocab_size] float32
    k      : [batch_size] int32, or None (None disables top-k filtering)
    p      : [batch_size] float32, or None (None disables top-p filtering)
    returns: filtered logits, filtered positions set to -inf

Semantics (mirrors vllm-ascend's own reference implementation from the
removed AscendC UT tests/e2e/.../test_apply_top_k_top_p_custom.py):
    1. Sort each row ascending (stable).
    2. top-k (k is clamped to vocab_size): the k-th largest value is the
       threshold; values strictly below it become -inf. Tied values at the
       threshold are ALL kept (threshold-based), which differs from the
       upstream Triton kernel that truncates ties to exactly k.
    3. top-p (applied on the top-k survivors): softmax the row, accumulate
       probabilities in ascending order, mask the smallest-probability tail
       whose cumulative probability is <= 1 - p; the last element is always
       kept so at least one entry survives.
    4. Scatter back to the original order.

Judgement criteria (mirrors vllm-ascend's UT):
    1. mask (-inf positions) mismatch ratio <= 0.1%, tolerating top-p
       floating-point boundary differences,
    2. surviving finite values compared with rtol=atol=1e-4.

Device policy (faithful problem reporting):
    - On A2/A3 hosts the operator must run and pass; any failure is a real
      precision/functional bug and fails the UT.
    - On other hosts (e.g. A5) the operator is attempted anyway: if it runs,
      its precision is still validated (valuable signal); if the call itself
      raises (operator not built/registered for that SoC), the test skips
      with the device name and the reason instead of erroring, because
      "operator unavailable" is a functional fact already known to
      vllm-ascend's sampler (which is exactly why A5 routes to the PyTorch
      fallback tested in test_topk_topp_kernel_a5.py), not a precision
      regression of this UT.

Realistic shapes:
  - vocab_size: 32000 (Llama2), 128256 (Llama3), 129280 (DeepSeek V4),
    163840 (Kimi K3), 248320 (Qwen3 2.4T)
  - batch_size: 1 (single-stream decode), 4, 16, 64 (concurrent requests)
  - k: 5..20 (typical top-k sampling)
  - p: 0.8..0.99 (typical top-p)
"""
from __future__ import annotations

import pytest
import torch
import torch_npu

from accuracy_test.easy_ut_026.runtime_npu import DEVICE, synchronize

pytestmark = [pytest.mark.npu]

# Best-effort device identification for the skip guard below. Import of
# vllm_ascend.device.device_config is side-effect free (build-info based).
try:
    from vllm_ascend.device.device_config import get_ascend_device_type

    _DEVICE_NAME = get_ascend_device_type().name  # "A2" / "A3" / "_310P" / "A5"
except Exception:  # noqa: BLE001 - device probe must never break collection
    _DEVICE_NAME = "UNKNOWN"
_IS_A2A3_HOST = _DEVICE_NAME in ("A2", "A3")


def _apply_topk_topp_cpu_ref(
    logits: torch.Tensor,
    k: torch.Tensor | None = None,
    p: torch.Tensor | None = None,
) -> torch.Tensor:
    """CPU reference for the fused NPU top-k/top-p replacement operator.

    Mirrors vllm-ascend's reference implementation (cpu_op_exec) exactly:
    ascending stable sort, threshold-based top-k, ascending-cumsum top-p,
    filtered positions set to -inf, scattered back to the original order.
    """
    # Sort logits in ascending order
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False, stable=True)

    # 1. Apply top-k filtering
    if k is not None:
        # Ensure k does not exceed vocab_size
        k = torch.minimum(k, torch.tensor(logits.size(-1), dtype=k.dtype))
        top_k_mask_idx = logits_sort.size(1) - k.to(torch.long)
        top_k_threshold = logits_sort.gather(1, top_k_mask_idx.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_threshold
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    # 2. Apply top-p (nucleus) filtering
    if p is not None:
        probs_sort = logits_sort.to(torch.float32).softmax(dim=-1)
        probs_sum = probs_sort.cumsum(dim=-1)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # 3. Restore original order
    return torch.empty_like(logits_sort).scatter_(dim=-1, index=logits_idx, src=logits_sort)


def _launch_npu(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """Run the fused NPU replacement operator (A2/A3 vllm-ascend sampling path)."""
    try:
        out = torch_npu.npu_top_k_top_p(logits, k=k, p=p)
    except Exception as exc:  # noqa: BLE001
        if _IS_A2A3_HOST:
            # On the operator's home devices any failure is a real bug.
            raise
        pytest.skip(
            f"torch_npu.npu_top_k_top_p is unavailable on device "
            f"{_DEVICE_NAME}: {exc!r}. vllm-ascend only wires this fused "
            f"operator for A2/A3 (vllm_ascend/sample/sampler.py::"
            f"apply_top_k_top_p); the A5 production path uses "
            f"_apply_top_k_top_p_pytorch, covered by "
            f"test_topk_topp_kernel_a5.py."
        )
    synchronize()
    return out


def _assert_output_close(
    name: str,
    out_cpu: torch.Tensor,
    out_npu: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    max_mask_mismatch_ratio: float = 0.001,
) -> None:
    """Compare NPU output with the CPU reference.

    Two-stage judgement, mirroring vllm-ascend's UT:
      1. mask (-inf positions) consistency, allowing up to 0.1% mismatch for
         top-p floating-point boundary differences,
      2. surviving finite values compared element-wise.
    """
    # 1. Check mask consistency (inf vs finite)
    mask_cpu = torch.isinf(out_cpu) & (out_cpu < 0)
    mask_npu = torch.isinf(out_npu) & (out_npu < 0)

    mismatch_mask = mask_cpu ^ mask_npu
    mismatch_count = mismatch_mask.sum().item()
    total_elements = out_cpu.numel()
    mismatch_ratio = mismatch_count / total_elements
    assert mismatch_ratio <= max_mask_mismatch_ratio, (
        f"[{name}] mask mismatch ratio too high: {mismatch_ratio:.6f} "
        f"({mismatch_count}/{total_elements})"
    )

    # 2. Check value consistency for valid elements
    valid_mask = (~mask_cpu) & (~mask_npu)
    if valid_mask.any():
        torch.testing.assert_close(
            out_cpu[valid_mask], out_npu[valid_mask], rtol=rtol, atol=atol
        )


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
    """Fused NPU top-k/top-p filtering with realistic vocab and batch sizes."""
    torch.manual_seed(42)

    topk_enabled = mode in ("topk_only", "topk_topp")
    topp_enabled = mode in ("topp_only", "topk_topp")

    logits_cpu = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    # The NPU operator disables a filter by passing None (unlike the Triton
    # kernel, which used k >= vocab_size / p == 1.0 sentinels).
    k_cpu = torch.randint(5, 21, (batch_size,), dtype=torch.int32) if topk_enabled else None
    p_cpu = torch.rand(batch_size, dtype=torch.float32) * 0.19 + 0.8 if topp_enabled else None  # 0.8..0.99

    logits_npu = logits_cpu.to(DEVICE)
    k_npu = k_cpu.to(DEVICE) if k_cpu is not None else None
    p_npu = p_cpu.to(DEVICE) if p_cpu is not None else None

    out_npu = _launch_npu(logits_npu, k_npu, p_npu).cpu()
    expected = _apply_topk_topp_cpu_ref(logits_cpu, k_cpu, p_cpu)

    _assert_output_close("logits", expected, out_npu)


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("vocab_size", [32000, 129280])
def test_topk_exact_count(batch_size, vocab_size):
    """Top-k only: exactly k tokens survive per row (duplicate-free input)."""
    torch.manual_seed(7)

    logits_cpu = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    k_cpu = torch.randint(5, 21, (batch_size,), dtype=torch.int32)

    out_npu = _launch_npu(
        logits_cpu.to(DEVICE), k_cpu.to(DEVICE), None
    ).cpu()
    expected = _apply_topk_topp_cpu_ref(logits_cpu, k_cpu, None)
    _assert_output_close("logits", expected, out_npu)

    for row in range(batch_size):
        num_surviving = (out_npu[row] > float("-inf")).sum().item()
        assert num_surviving == int(k_cpu[row]), (
            f"Row {row}: expected {int(k_cpu[row])} survivors, "
            f"got {num_surviving}"
        )


def test_topk_duplicate_boundary():
    """Top-k with duplicate values at the k-th boundary.

    The fused NPU operator is threshold-based: every logit >= the k-th
    largest value survives, so tied boundary values are ALL kept and the
    survivor count exceeds k. This is the documented semantic difference
    from the upstream Triton kernel (which truncates ties to exactly k) and
    must match the CPU reference mask exactly.
    """
    torch.manual_seed(8)

    batch_size = 1
    vocab_size = 32000
    logits_cpu = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    # Force a tie group at the GLOBAL maximum: entries at indices 9..29 all
    # share the row max, so the k-th largest value (k=12) is the tie value
    # itself and the boundary falls inside the tie group. (A tie group placed
    # at a random rank would sit far below the k-th threshold and the
    # survivor-count assertion below would be meaningless.)
    logits_cpu[0, 9:30] = logits_cpu[0].max().item()
    k_cpu = torch.tensor([12], dtype=torch.int32)

    out_npu = _launch_npu(
        logits_cpu.to(DEVICE), k_cpu.to(DEVICE), None
    ).cpu()
    expected = _apply_topk_topp_cpu_ref(logits_cpu, k_cpu, None)
    _assert_output_close("logits", expected, out_npu)

    num_surviving_npu = (out_npu[0] > float("-inf")).sum().item()
    num_surviving_ref = (expected[0] > float("-inf")).sum().item()
    assert num_surviving_npu == num_surviving_ref, (
        f"Survivor count mismatch: NPU kept {num_surviving_npu}, "
        f"reference kept {num_surviving_ref}"
    )
    # All 21 tied boundary values are kept, so strictly more than k survive.
    assert num_surviving_npu > 12, (
        f"Threshold-based semantics should keep all tied boundary values, "
        f"got {num_surviving_npu} survivors for k=12"
    )


def test_topp_at_least_one_survivor():
    """Top-p with an extreme p: at least one token must survive per row.

    Even when p is very small (tiny nucleus), the operator guarantees that
    the last (largest) element of the ascending order is never masked.
    """
    torch.manual_seed(11)

    batch_size, vocab_size = 4, 32000
    logits_cpu = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    p_cpu = torch.full((batch_size,), 0.001, dtype=torch.float32)

    out_npu = _launch_npu(
        logits_cpu.to(DEVICE), None, p_cpu.to(DEVICE)
    ).cpu()
    expected = _apply_topk_topp_cpu_ref(logits_cpu, None, p_cpu)
    _assert_output_close("logits", expected, out_npu)

    for row in range(batch_size):
        num_surviving = (out_npu[row] > float("-inf")).sum().item()
        assert num_surviving >= 1, (
            f"Row {row}: top-p must keep at least one token, got {num_surviving}"
        )
        # The global maximum of the row must survive the nucleus filter.
        max_idx = logits_cpu[row].argmax().item()
        assert out_npu[row, max_idx] > float("-inf"), (
            f"Row {row}: row maximum at index {max_idx} was masked by top-p"
        )
