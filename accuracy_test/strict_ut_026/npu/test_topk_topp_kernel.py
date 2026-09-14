# Standalone Ascend A5 precision UT for the PyTorch top-k/top-p fallback.
# Accuracy UT source: vllm/tests/v1/sample/test_topk_topp_sampler.py
# Replaced implementation: vllm/vllm/v1/sample/ops/topk_topp_triton.py (_topk_topp_kernel)
# Tested implementation: vllm_ascend/sample/sampler.py::_apply_top_k_top_p_pytorch
#   (the non-reduce-sample branch), which is the production sampling path on
#   A5 devices:
#
#       apply_top_k_top_p = (
#           _apply_top_k_top_p_torch_npu       # A2 / A3 -> test_topk_topp_kernel_a2a3.py
#           if get_ascend_device_type() in [AscendDeviceType.A2, AscendDeviceType.A3]
#           else _apply_top_k_top_p_pytorch    # A5 -> this UT
#       )
#
# The A2/A3 counterpart UT (test_topk_topp_kernel_a2a3.py) tests the fused
# ``torch_npu.npu_top_k_top_p`` operator. Both UTs intentionally coexist so
# each device family validates the implementation its production path
# actually calls, faithfully reporting each one's precision behaviour.

"""
Precision test for the PyTorch top-k/top-p fallback (A5 sampling path).

Why the Triton kernel is not launched directly:
    The upstream ``_topk_topp_kernel`` calls ``tl.cumsum`` on a bool tensor
    (``outlier_mask``). The Ascend Triton backend does not ship a bool cumsum
    library function, so compilation fails with ``undefined symbol:
    _mlir_ciface_cumsum_1d_bool_dim0``. vllm-ascend does not fix the kernel;
    it replaces the whole top-k/top-p step per device family. On A5 (and any
    non-A2/A3 SoC) the replacement is the pure-PyTorch composition
    ``_apply_top_k_top_p_pytorch`` (vllm_ascend/sample/sampler.py), because
    the fused ``torch_npu.npu_top_k_top_p`` operator is only wired for
    A2/A3. This UT validates the precision of that PyTorch implementation
    running on the NPU (softmax/sort/cumsum/gather/masked_fill execute as
    NPU kernels) against a CPU reference running the identical algorithm.

Implementation under test (called directly from vllm-ascend):
    _apply_top_k_top_p_pytorch(logits, k, p)   # non-distributed branch
    logits : [batch_size, vocab_size] float32 (modified in-place by the
             implementation; this UT always passes a clone)
    k      : [batch_size] int32, or None (None disables top-k filtering)
    p      : [batch_size] float32, or None (None disables top-p filtering)
    returns: filtered logits, filtered positions set to -inf

Semantics (transcribed from vllm_ascend/sample/sampler.py; differs from the
A2/A3 fused operator's reference in tie handling):
    1. probs = softmax(logits); probs_sort = probs sorted ascending.
    2. top-k: cutoff = the k-th largest prob; discard elements with
       ``probs < cutoff`` (STRICT less-than). Ties at the cutoff are ALL
       kept, so the survivor count can exceed k.
    3. top-p: cumprob = cumsum(probs_sort); count how many positions have
       ``cumprob <= 1 - p`` (the last position is exempted so at least one
       survives); cutoff = the prob at that count index; discard elements
       with ``probs < cutoff``. IMPORTANT: because the discard test is a
       single threshold on probs, a group of EQUAL probabilities that
       straddles the cumulative boundary is kept ENTIRELY - unlike the
       A2/A3 fused operator's per-position masking, which would mask only
       the group members whose cumulative sum is still <= 1 - p.
    4. The output preserves the original (unsorted) order.

Judgement criteria (same two-stage scheme as the A2/A3 UT):
    1. mask (-inf positions) mismatch ratio <= 0.1%, tolerating top-p
       floating-point boundary differences between NPU and CPU softmax /
       cumsum,
    2. surviving finite values compared with rtol=atol=1e-4.

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

# runtime_npu MUST be imported before vllm_ascend.sample.sampler: it installs
# the vllm.triton_utils shim (triton 3.2.0 lacks triton.experimental) and the
# vllm_ascend.ops package stubs that keep the sampler import chain light.
from accuracy_test.easy_ut_026.runtime_npu import (
    DEVICE,
    ensure_default_ascend_config,
    synchronize,
)

try:
    # The actual A5 production implementation, imported from vllm-ascend so
    # the UT tracks the real code path (not a local copy).
    from vllm_ascend.sample.sampler import (
        _apply_top_k_top_p_pytorch as _apply_top_k_top_p_a5_impl,
    )
except Exception as exc:  # noqa: BLE001 - any import-chain break is a skip
    pytest.skip(
        f"cannot import the A5 production implementation "
        f"vllm_ascend.sample.sampler._apply_top_k_top_p_pytorch: {exc!r}",
        allow_module_level=True,
    )

pytestmark = [pytest.mark.npu]


def _apply_topk_topp_pytorch_cpu_ref(
    logits: torch.Tensor,
    k: torch.Tensor | None = None,
    p: torch.Tensor | None = None,
) -> torch.Tensor:
    """CPU reference: faithful transcription of ``_apply_top_k_top_p_pytorch``.

    Line-for-line the same algorithm as the non-reduce-sample branch in
    vllm_ascend/sample/sampler.py (probs-space thresholds, strict
    less-than discard tests, count-then-gather top-p cutoff), executed on
    CPU so the NPU run is compared against the identical semantics.
    """
    if p is None and k is None:
        return logits.clone()

    probs = logits.softmax(dim=-1)
    probs_sort, _ = probs.sort(dim=-1, descending=False)

    out = logits.clone()
    if k is not None:
        top_k_count = probs_sort.size(1) - k.to(torch.long)  # shape: (batch,)
        top_k_count = top_k_count.unsqueeze(dim=1)
        top_k_cutoff = probs_sort.gather(-1, top_k_count)

        # Make sure the no top-k rows are no-op.
        no_top_k_mask = (k == logits.shape[1]).unsqueeze(dim=1)
        top_k_cutoff.masked_fill_(no_top_k_mask, -float("inf"))

        elements_to_discard = probs < top_k_cutoff
        out.masked_fill_(elements_to_discard, -float("inf"))

    if p is not None:
        cumprob = torch.cumsum(probs_sort, dim=-1)
        top_p_mask = cumprob <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False  # at least one

        top_p_count = top_p_mask.sum(dim=-1).unsqueeze(1)
        top_p_cutoff = probs_sort.gather(-1, top_p_count)
        elements_to_discard = probs < top_p_cutoff
        out.masked_fill_(elements_to_discard, -float("inf"))

    return out


def _launch_npu(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """Run the A5 production implementation (vllm-ascend sampling path).

    ``ensure_default_ascend_config`` pins ``enable_reduce_sample=False`` so
    the implementation takes the single-card branch (a UT process never
    initializes the real engine config). The implementation fills -inf
    in-place, so a clone is passed to keep the caller's input intact.
    """
    ensure_default_ascend_config()
    out = _apply_top_k_top_p_a5_impl(logits.clone(), k, p)
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

    Two-stage judgement (same scheme as the A2/A3 UT):
      1. mask (-inf positions) consistency, allowing up to 0.1% mismatch for
         top-p floating-point boundary differences (NPU softmax/cumsum vs
         CPU),
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
    """A5 PyTorch top-k/top-p filtering with realistic vocab and batch sizes."""
    torch.manual_seed(42)

    topk_enabled = mode in ("topk_only", "topk_topp")
    topp_enabled = mode in ("topp_only", "topk_topp")

    logits_cpu = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    # Same None convention as the fused NPU operator: None disables the
    # corresponding filter.
    k_cpu = torch.randint(5, 21, (batch_size,), dtype=torch.int32) if topk_enabled else None
    p_cpu = torch.rand(batch_size, dtype=torch.float32) * 0.19 + 0.8 if topp_enabled else None  # 0.8..0.99

    logits_npu = logits_cpu.to(DEVICE)
    k_npu = k_cpu.to(DEVICE) if k_cpu is not None else None
    p_npu = p_cpu.to(DEVICE) if p_cpu is not None else None

    out_npu = _launch_npu(logits_npu, k_npu, p_npu).cpu()
    expected = _apply_topk_topp_pytorch_cpu_ref(logits_cpu, k_cpu, p_cpu)

    _assert_output_close("logits", expected, out_npu)


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("vocab_size", [32000, 129280])
def test_topk_exact_count(batch_size, vocab_size):
    """Top-k only: exactly k tokens survive per row (duplicate-free input).

    The implementation discards with a strict ``probs < cutoff`` test on a
    duplicate-free row, so exactly k elements satisfy ``probs >= cutoff``.
    """
    torch.manual_seed(7)

    logits_cpu = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    k_cpu = torch.randint(5, 21, (batch_size,), dtype=torch.int32)

    out_npu = _launch_npu(
        logits_cpu.to(DEVICE), k_cpu.to(DEVICE), None
    ).cpu()
    expected = _apply_topk_topp_pytorch_cpu_ref(logits_cpu, k_cpu, None)
    _assert_output_close("logits", expected, out_npu)

    for row in range(batch_size):
        num_surviving = (out_npu[row] > float("-inf")).sum().item()
        assert num_surviving == int(k_cpu[row]), (
            f"Row {row}: expected {int(k_cpu[row])} survivors, "
            f"got {num_surviving}"
        )


def test_topk_duplicate_boundary():
    """Top-k with duplicate values at the k-th boundary.

    The implementation is threshold-based in probs space: every prob >= the
    k-th largest prob survives, so tied boundary values are ALL kept and the
    survivor count exceeds k. This matches the A2/A3 fused operator's
    threshold semantics (see test_topk_topp_kernel_a2a3.py) and must match
    the CPU reference mask exactly.
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
    expected = _apply_topk_topp_pytorch_cpu_ref(logits_cpu, k_cpu, None)
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

    Even when p is very small (tiny nucleus), the last (largest) position of
    the ascending order is exempted from the mask, so the cutoff equals the
    maximum probability and at least the argmax token survives.
    """
    torch.manual_seed(11)

    batch_size, vocab_size = 4, 32000
    logits_cpu = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    p_cpu = torch.full((batch_size,), 0.001, dtype=torch.float32)

    out_npu = _launch_npu(
        logits_cpu.to(DEVICE), None, p_cpu.to(DEVICE)
    ).cpu()
    expected = _apply_topk_topp_pytorch_cpu_ref(logits_cpu, None, p_cpu)
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


def test_topp_tie_boundary():
    """Top-p with a tie group straddling the cumulative boundary (A5-specific).

    The A5 implementation reduces top-p to ONE threshold on probs
    (``probs < cutoff``), unlike the A2/A3 fused operator's per-position
    cumulative masking. When several EQUAL probabilities straddle the
    ``cumsum <= 1 - p`` boundary, the per-position semantics would mask the
    leading part of the group while this implementation keeps the WHOLE
    group. This UT pins that documented A5 behaviour.

    Construction (vocab=16): logits come in 4 equal-valued groups
    [0, 1, 2, 3] x 4, so probs form 4 distinct equal-prob groups. With
    p = 0.75 the cumulative boundary (1 - p = 0.25) falls INSIDE the third
    group (ascending): its first two members have cumsum <= 0.25 while the
    group value equals the cutoff, so all four members survive together
    with the whole top group -> 8 survivors.
    """
    torch.manual_seed(13)

    logits_cpu = torch.tensor(
        [[0.0] * 4 + [1.0] * 4 + [2.0] * 4 + [3.0] * 4],
        dtype=torch.float32,
    )
    p_cpu = torch.tensor([0.75], dtype=torch.float32)

    out_npu = _launch_npu(
        logits_cpu.to(DEVICE), None, p_cpu.to(DEVICE)
    ).cpu()
    expected = _apply_topk_topp_pytorch_cpu_ref(logits_cpu, None, p_cpu)
    _assert_output_close("logits", expected, out_npu)

    num_surviving_npu = (out_npu[0] > float("-inf")).sum().item()
    num_surviving_ref = (expected[0] > float("-inf")).sum().item()
    assert num_surviving_npu == num_surviving_ref == 8, (
        f"Expected the two highest equal-prob groups (8 tokens) to survive "
        f"entirely; NPU kept {num_surviving_npu}, reference kept "
        f"{num_surviving_ref}"
    )
    # The straddling group itself (logit value 2.0, indices 8..11) must be
    # kept ENTIRELY even though its first two members have
    # cumsum <= 1 - p (per-position masking would keep only 6 tokens).
    assert bool((out_npu[0, 8:12] > float("-inf")).all()), (
        "The tie group straddling the top-p boundary must survive entirely "
        f"under the threshold-based A5 semantics; got {out_npu[0, 8:12].tolist()}"
    )
