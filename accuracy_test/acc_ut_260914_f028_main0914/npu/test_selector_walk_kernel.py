# SPDX-License-Identifier: Apache-2.0
"""NPU-side accuracy UT for the downstream (vllm-ascend) selector walk kernel.

Tests the **downstream** (vllm-ascend) implementation directly:

    from vllm_ascend.ops.triton.spec_decode.utils import (
        dflash2_greedy_selector_walk_kernel,
    )

The downstream kernel is the Ascend port of upstream ``_selector_walk_kernel``
in ``vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py``.  Differences vs.
upstream:

  * Greedy-only: the Gumbel probabilistic branch is removed; the kernel is a
    pure argmax walk over the candidate tree.
  * No req_state / sample_pos / temperature / seeds inputs; no
    realized_scores output.
  * Signature: (scores_ptr, candidate_ids_ptr, output_ptr, num_reqs,
    num_steps: tl.constexpr, top_k: tl.constexpr).
  * Grid-stride loop over requests (``req = pid; while req < num_reqs: ...``
    with ``req += num_programs``); the kernel handles any grid size up to
    num_reqs.
  * Tie-break rule: when multiple candidates share the max score, the
    *smallest* index wins
    (``next_idx = tl.min(tl.where(row == max_value, offsets, top_k), axis=0)``).

A new CPU reference is provided here that mirrors the downstream kernel
exactly (same flat layout, same tie-break rule).  Pass criteria is bitwise
equality on the output token tensor -- the kernel is deterministic by
construction.
"""
from __future__ import annotations

import traceback

import pytest
import torch

# runtime_npu must be imported before any vllm_ascend import: it installs
# the vllm.triton_utils shim (required for vllm_ascend.ops.triton.* to
# resolve `from vllm.triton_utils import tl, triton`) and device helpers.
from accuracy_test.acc_ut_260914_f028_main0914.runtime_npu import (  # noqa: F401
    STRICT_DEVICE,
    get_vectorcore_num,
    init_device_properties_triton,
    synchronize,
)


# --- Import the downstream (vllm-ascend) kernel -----------------------
_npu_import_error: Exception | None = None
_npu_import_traceback: str | None = None
try:
    from vllm_ascend.ops.triton.spec_decode.utils import (
        dflash2_greedy_selector_walk_kernel as npu_kernel,
    )
except Exception as exc:  # pragma: no cover
    npu_kernel = None
    _npu_import_error = exc
    _npu_import_traceback = traceback.format_exc()


@pytest.fixture
def rt():
    """Fixture consumed by shared helpers -- exposes runtime_npu."""
    import accuracy_test.acc_ut_260914_f028_main0914.runtime_npu as _rt
    return _rt


# =====================================================================
# CPU reference -- mirrors downstream kernel exactly
# =====================================================================
def _ref_greedy_walk(
    scores: torch.Tensor,       # [num_reqs * num_steps, top_k, top_k] float32
    candidates: torch.Tensor,   # [num_reqs * num_steps, top_k] int64
    num_reqs: int,
    num_steps: int,
    top_k: int,
) -> torch.Tensor:
    """CPU reference for ``dflash2_greedy_selector_walk_kernel``.

    Walks the candidate tree per request, starting from ``prev_idx = 0`` at
    step 0.  At each step it picks the argmax of ``scores[flat, prev_idx, :]``
    with the *smallest* index winning ties (matching the kernel's
    ``tl.min(tl.where(row == max_value, offsets, top_k), axis=0)``).

    Returns:
        tokens: [num_reqs * num_steps] int64
    """
    total = num_reqs * num_steps
    tokens = torch.zeros(total, dtype=torch.int64)

    scores_f32 = scores.to(torch.float32).cpu()
    candidates_cpu = candidates.cpu()

    for req in range(num_reqs):
        prev_idx = 0
        for step in range(num_steps):
            flat = req * num_steps + step
            row = scores_f32[flat, prev_idx, :]
            max_value = row.max().item()
            # Tie-break: smallest index where row == max_value
            matches = (row == max_value).nonzero(as_tuple=False).flatten()
            next_idx = int(matches[0].item()) if matches.numel() > 0 else 0
            tokens[flat] = candidates_cpu[flat, next_idx]
            prev_idx = next_idx

    return tokens


def _gen_inputs(
    num_reqs: int,
    num_steps: int,
    top_k: int,
    *,
    seed: int = 0,
    device,
) -> dict:
    """Build deterministic inputs with a clear winner at each step.

    ``scores`` is laid out as ``[num_reqs * num_steps, top_k, top_k]`` which,
    when flattened in row-major order, matches the kernel's flat layout
    ``scores_ptr + (req*num_steps+step)*top_k*top_k + prev_idx*top_k + i``.

    Scores use small integer values (exactly representable in float32) to
    avoid FP precision issues in the downstream kernel's
    ``row == max_value`` comparison.  The downstream kernel uses
    ``tl.min(tl.where(row == max_value, offsets, top_k), axis=0)`` for
    argmax+tie-break; if ``tl.max`` returns a value that's not
    bitwise-identical to any input element, the equality fails, ``next_idx``
    defaults to ``top_k`` (out-of-bounds), and the kernel reads garbage.
    Integer-valued float32 scores guarantee exact comparison.
    """
    g = torch.Generator(device="cpu").manual_seed(
        42 + seed + num_reqs * 7 + num_steps * 3 + top_k * 11
    )
    total = num_reqs * num_steps

    # Use small integer-valued scores in [0, 9] (exactly representable in
    # float32).  Then boost one candidate to 100 so argmax is unambiguous.
    scores = torch.randint(0, 10, (total, top_k, top_k), generator=g).to(
        torch.float32
    )
    for flat in range(total):
        for prev in range(top_k):
            winner = torch.randint(0, top_k, (1,), generator=g).item()
            scores[flat, prev, winner] = 100.0

    candidates = torch.randint(0, 1000, (total, top_k), dtype=torch.int64, generator=g)

    return {
        "scores": scores.to(device),
        "candidates": candidates.to(device),
        "num_reqs": num_reqs,
        "num_steps": num_steps,
        "top_k": top_k,
    }


def _gen_inputs_tied(
    num_reqs: int,
    num_steps: int,
    top_k: int,
    *,
    device,
) -> dict:
    """Build inputs where multiple candidates tie for the max score.

    The kernel's tie-break rule (smallest index wins) is observable only
    when there is a tie.  This generator forces ties by zeroing the score
    matrix at every position; with all scores equal, the smallest index
    should always win.
    """
    total = num_reqs * num_steps
    scores = torch.zeros(total, top_k, top_k, dtype=torch.float32, device=device)
    candidates = torch.arange(total * top_k, dtype=torch.int64, device=device).view(
        total, top_k
    )
    return {
        "scores": scores,
        "candidates": candidates,
        "num_reqs": num_reqs,
        "num_steps": num_steps,
        "top_k": top_k,
    }


def _launch(k, inputs: dict) -> torch.Tensor:
    """Launch the downstream kernel and return the output tokens tensor."""
    num_reqs = inputs["num_reqs"]
    num_steps = inputs["num_steps"]
    top_k = inputs["top_k"]
    device = inputs["scores"].device

    total = num_reqs * num_steps
    # Output dtype follows the kernel: it stores the loaded candidate id
    # verbatim.  Upstream speculator uses int64 for tokens; the downstream
    # kernel stores whatever dtype candidate_ids_ptr points to, so we mirror
    # the candidates dtype.
    output = torch.zeros(total, dtype=inputs["candidates"].dtype, device=device)

    # Flat 1D views; downstream kernel indexes pointer-arithmetic style.
    scores_flat = inputs["scores"].contiguous().reshape(-1)
    candidates_flat = inputs["candidates"].contiguous().reshape(-1)

    # Match the production caller (greedy_select_path) EXACTLY:
    #   grid = min(num_reqs, get_vectorcore_num())
    #   num_warps left at the default (do NOT pass num_warps=1; the
    #     single-warp launch changes the Ascend code generation for the
    #     grid-stride while-loop with carried scalars and produces garbage
    #     token reads at certain shapes, e.g. num_reqs=16/num_steps=4/
    #     top_k=4).
    num_programs = min(num_reqs, get_vectorcore_num())
    k[(num_programs,)](
        scores_flat,
        candidates_flat,
        output,
        num_reqs,
        num_steps=num_steps,
        top_k=top_k,
    )
    return output


# Realistic DFlash2 shapes:
#   - top_k (selector_top_k): {4, 8, 16, 32}
#   - num_steps (num_speculative_steps): {1, 2, 4, 8}
#   - num_reqs: 1, 4, 8, 16, 32, 64, 128, 256
SHAPE_PARAMS = [
    # (num_reqs, num_steps, top_k)
    (1, 1, 4),
    (1, 2, 4),
    (1, 4, 8),
    (4, 1, 4),
    (4, 2, 8),
    (4, 4, 8),
    (8, 2, 4),
    (8, 4, 8),
    (8, 4, 16),
    (16, 2, 8),
    (16, 4, 4),
    (32, 2, 8),
    (32, 4, 16),
    (64, 1, 4),
    (64, 2, 8),
    (128, 2, 4),
    (256, 1, 4),
    # Edge: single step, large top_k
    (4, 1, 32),
    # Edge: many steps
    (4, 8, 4),
    # Edge: top_k=1 (degenerate: only one candidate, always wins)
    (4, 2, 1),
]


# =====================================================================
# Test 1: Greedy walk accuracy -- bitwise equality vs CPU reference
# =====================================================================
@pytest.mark.parametrize("num_reqs,num_steps,top_k", SHAPE_PARAMS)
def test_dflash2_greedy_walk_accuracy(
    num_reqs: int,
    num_steps: int,
    top_k: int,
    rt,
):
    """Verify downstream kernel's greedy walk matches CPU reference bitwise.

    The downstream kernel is fully deterministic (no Gumbel sampling), so
    pass requires bitwise equality on every output token.
    """
    if npu_kernel is None:
        pytest.fail(
            "downstream dflash2_greedy_selector_walk_kernel import failed.\n"
            f"error={_npu_import_error}\ntraceback:\n{_npu_import_traceback}",
            pytrace=False,
        )

    rt.init_device_properties_triton()
    device = rt.STRICT_DEVICE

    inputs = _gen_inputs(
        num_reqs=num_reqs,
        num_steps=num_steps,
        top_k=top_k,
        device=device,
    )

    # CPU reference (run on the CPU copies)
    expected = _ref_greedy_walk(
        scores=inputs["scores"],
        candidates=inputs["candidates"],
        num_reqs=num_reqs,
        num_steps=num_steps,
        top_k=top_k,
    )

    # Device run with downstream kernel
    dev_tokens = _launch(npu_kernel, inputs)
    rt.synchronize()
    dev_tokens_cpu = dev_tokens.cpu()

    # Compare tokens bitwise
    assert dev_tokens_cpu.dtype == expected.dtype, (
        f"dtype mismatch: device={dev_tokens_cpu.dtype} ref={expected.dtype}"
    )
    assert dev_tokens_cpu.shape == expected.shape, (
        f"shape mismatch: device={tuple(dev_tokens_cpu.shape)} "
        f"ref={tuple(expected.shape)}"
    )
    if not torch.equal(dev_tokens_cpu, expected):
        mismatched = torch.ne(dev_tokens_cpu, expected)
        count = int(mismatched.sum().item())
        total = int(mismatched.numel())
        first_idx = torch.nonzero(mismatched, as_tuple=False)
        first_info = ""
        if first_idx.numel() > 0:
            loc = int(first_idx[0].item())
            first_info = (
                f"; first mismatch at [{loc}]: "
                f"device={dev_tokens_cpu[loc].item()} ref={expected[loc].item()}"
            )
        pytest.fail(
            f"[downstream] dflash2_greedy_selector_walk_kernel tokens mismatch: "
            f"{count}/{total} elements differ{first_info}"
        )


# =====================================================================
# Test 2: Tie-break rule -- smallest index wins
# =====================================================================
@pytest.mark.parametrize(
    "num_reqs,num_steps,top_k",
    [
        (1, 1, 4),
        (1, 2, 8),
        (4, 2, 4),
        (8, 1, 16),
    ],
)
def test_dflash2_greedy_walk_tie_break(
    num_reqs: int,
    num_steps: int,
    top_k: int,
    rt,
):
    """Verify that ties resolve to the smallest candidate index.

    With all scores zeroed, every candidate ties for the max; the kernel's
    tie-break rule (``tl.min(tl.where(row == max_value, offsets, top_k))``)
    must pick index 0 at every step.  The CPU reference uses the same rule.
    """
    if npu_kernel is None:
        pytest.fail("downstream kernel import failed", pytrace=False)

    rt.init_device_properties_triton()
    device = rt.STRICT_DEVICE

    inputs = _gen_inputs_tied(
        num_reqs=num_reqs,
        num_steps=num_steps,
        top_k=top_k,
        device=device,
    )

    expected = _ref_greedy_walk(
        scores=inputs["scores"],
        candidates=inputs["candidates"],
        num_reqs=num_reqs,
        num_steps=num_steps,
        top_k=top_k,
    )

    # Every selected index should be 0 -> tokens must equal candidates[:, 0]
    candidates_col0 = inputs["candidates"].cpu()[:, 0].contiguous()
    assert torch.equal(expected, candidates_col0), (
        "CPU reference did not pick index 0 for all-tied scores"
    )

    dev_tokens = _launch(npu_kernel, inputs)
    rt.synchronize()
    dev_tokens_cpu = dev_tokens.cpu()

    if not torch.equal(dev_tokens_cpu, expected):
        mismatched = torch.ne(dev_tokens_cpu, expected)
        count = int(mismatched.sum().item())
        pytest.fail(
            f"[downstream] Tie-break rule broken: {count} elements differ. "
            f"Expected all-zero indices (smallest wins); got mismatches."
        )


# =====================================================================
# Test 3: Determinism -- same input twice must produce identical output
# =====================================================================
def test_dflash2_greedy_walk_determinism(rt):
    """Re-running the kernel with identical inputs must produce identical output."""
    if npu_kernel is None:
        pytest.fail("downstream kernel import failed", pytrace=False)

    rt.init_device_properties_triton()
    device = rt.STRICT_DEVICE

    inputs = _gen_inputs(
        num_reqs=8,
        num_steps=2,
        top_k=8,
        seed=123,
        device=device,
    )

    out1 = _launch(npu_kernel, inputs)
    rt.synchronize()
    out2 = _launch(npu_kernel, inputs)
    rt.synchronize()

    assert torch.equal(out1, out2), (
        "[downstream] Greedy walk should be deterministic across runs"
    )


# =====================================================================
# Test 4: Edge case -- top_k=1 (degenerate, only one candidate per step)
# =====================================================================
def test_dflash2_greedy_walk_topk1(rt):
    """With top_k=1, the only candidate always wins; walk is trivially correct."""
    if npu_kernel is None:
        pytest.fail("downstream kernel import failed", pytrace=False)

    rt.init_device_properties_triton()
    device = rt.STRICT_DEVICE

    num_reqs = 4
    num_steps = 2
    top_k = 1
    total = num_reqs * num_steps

    # With top_k=1, scores shape is [total, 1, 1]; candidate id is just the
    # single value at flat index.  Use integer-valued score for FP-exact
    # comparison in the kernel.
    scores = torch.full((total, 1, 1), 1.0, dtype=torch.float32, device=device)
    candidates = torch.arange(total, dtype=torch.int64, device=device).view(total, 1)

    inputs = {
        "scores": scores,
        "candidates": candidates,
        "num_reqs": num_reqs,
        "num_steps": num_steps,
        "top_k": top_k,
    }

    expected = _ref_greedy_walk(
        scores=scores, candidates=candidates,
        num_reqs=num_reqs, num_steps=num_steps, top_k=top_k,
    )

    dev_tokens = _launch(npu_kernel, inputs)
    rt.synchronize()

    if not torch.equal(dev_tokens.cpu(), expected):
        pytest.fail(
            f"[downstream] top_k=1 mismatch: "
            f"device={dev_tokens.cpu().tolist()} ref={expected.tolist()}"
        )


# =====================================================================
# Test 5: Verify that output tokens are drawn from the candidate set
# =====================================================================
def test_dflash2_greedy_walk_tokens_in_candidates(rt):
    """Each output token must appear in its row's candidate set."""
    if npu_kernel is None:
        pytest.fail("downstream kernel import failed", pytrace=False)

    rt.init_device_properties_triton()
    device = rt.STRICT_DEVICE

    inputs = _gen_inputs(
        num_reqs=8,
        num_steps=4,
        top_k=8,
        device=device,
    )

    dev_tokens = _launch(npu_kernel, inputs)
    rt.synchronize()
    tokens_cpu = dev_tokens.cpu()
    candidates_cpu = inputs["candidates"].cpu()

    for flat in range(tokens_cpu.shape[0]):
        token = int(tokens_cpu[flat].item())
        row = candidates_cpu[flat].tolist()
        assert token in row, (
            f"flat={flat}: token {token} not in candidates {row}"
        )


def test_import_error() -> None:
    if _npu_import_error is not None:
        pytest.fail(
            f"Failed to import downstream dflash2_greedy_selector_walk_kernel:\n"
            f"{_npu_import_traceback}"
        )
