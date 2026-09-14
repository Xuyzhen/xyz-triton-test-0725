"""Comparison metrics for the GPU-NPU precision harness.

Distinct from the suite-level ``metrics.py`` (per-side accuracy standard used
by the pytest tests): this module grades GPU-vs-NPU output divergence with
the PASS/WARN/FAIL ladder used by compare_results.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from capture_runtime import MODE_INT_EXACT, TOLERANCES

WARN_MISMATCH_RATIO = 1e-3     # <=0.1% mismatched elements may downgrade to WARN
WARN_TOL_FACTOR = 10.0         # max_err <= 10x tolerance may downgrade to WARN


@dataclass
class CompareResult:
    status: str                 # PASS / WARN / FAIL / SKIP / ERROR
    detail: str
    max_abs_err: float = 0.0
    mismatch_ratio: float = 0.0


def compare_int(a: torch.Tensor, b: torch.Tensor) -> CompareResult:
    if a.shape != b.shape or a.dtype != b.dtype:
        return CompareResult("ERROR", f"shape/dtype mismatch {a.shape}{a.dtype} vs {b.shape}{b.dtype}")
    a64, b64 = a.to(torch.int64), b.to(torch.int64)
    bad = a64 != b64
    n_bad = int(bad.sum().item())
    if n_bad == 0:
        return CompareResult("PASS", "bitwise equal")
    ratio = n_bad / max(a64.numel(), 1)
    status = "WARN" if ratio <= WARN_MISMATCH_RATIO else "FAIL"
    first = int(bad.reshape(-1).nonzero()[0].item()) if n_bad else -1
    return CompareResult(status, f"{n_bad}/{a64.numel()} elements differ (first flat idx {first})", mismatch_ratio=ratio)


def compare_float(a: torch.Tensor, b: torch.Tensor, mode: str) -> CompareResult:
    atol, rtol = TOLERANCES[mode]
    if a.shape != b.shape:
        return CompareResult("ERROR", f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    a64, b64 = a.to(torch.float64), b.to(torch.float64)
    finite = torch.isfinite(a64) & torch.isfinite(b64)
    nan_mismatch = bool((torch.isnan(a64) != torch.isnan(b64)).any().item())
    if nan_mismatch:
        return CompareResult("FAIL", "NaN placement differs")
    if not finite.any():
        return CompareResult("PASS", "no finite elements to compare")
    diff = (a64[finite] - b64[finite]).abs()
    tol = atol + rtol * b64[finite].abs()
    bad = diff > tol
    n_bad = int(bad.sum().item())
    max_err = float(diff.max().item())
    if n_bad == 0:
        return CompareResult("PASS", f"within tol (max_err={max_err:.3e})", max_abs_err=max_err)
    ratio = n_bad / max(a64.numel(), 1)
    status = "WARN" if (ratio <= WARN_MISMATCH_RATIO and max_err <= WARN_TOL_FACTOR * atol) else "FAIL"
    return CompareResult(status, f"{n_bad}/{a64.numel()} exceed tol (max_err={max_err:.3e})",
                         max_abs_err=max_err, mismatch_ratio=ratio)


def compare_output(mode: str, a: torch.Tensor, b: torch.Tensor) -> CompareResult:
    if mode == MODE_INT_EXACT:
        return compare_int(a, b)
    return compare_float(a, b, mode)
