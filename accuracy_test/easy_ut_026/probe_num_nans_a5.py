#!/usr/bin/env python
r"""Root-cause probe for the _num_nans_kernel undercount observed on Ascend A5.

Symptom (accuracy_test/easy_ut_026/test_num_nans_kernel.py, case
test_num_nans[0.1-128-1]): vocab_size=128, 12 NaNs injected at the head of
the row -> the kernel reports 1 while the CPU reference reports 12.

The upstream kernel (vllm/vllm/v1/worker/gpu/metrics/logits.py) is::

    num_nans = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(logits_ptr + req_idx * logits_stride + block,
                         mask=mask, other=0)
        logits = logits.to(tl.float32)
        is_nan = libdevice.isnan(logits).to(tl.int1)
        num_nans += tl.sum(is_nan).to(tl.int32)
    tl.store(num_nans_ptr + req_idx, num_nans)

Three layers can each be responsible:

    layer 1  tl.load(mask=...)      masked block load (BLOCK 8192 > vocab 128)
    layer 2  libdevice.isnan(...)   CANN libdevice, element-wise NaN test
    layer 3  tl.sum(int1 tensor)    block reduction over boolean flags

                 +---------------------------------------------+
   logits --load-> | 12 x NaN | 116 x finite | 8180 x 0 (masked) |
                 +---------------------------------------------+
                      | libdevice.isnan (per element)
                      v
                 [ 1 1 1 ... 1 | 0 0 ... 0 | 0 0 ... 0 ]
                      \------------- tl.sum -------------/
                                    v
                       expected 12, observed 1

Hypotheses:
    H1  libdevice.isnan is not element-wise on CANN (e.g. only lane 0).
    H2  tl.sum over an int1 tensor is lowered as a boolean OR reduction on
        the Ascend Triton backend, so any non-zero count collapses to 1.
    H3  the masked tl.load drops elements when BLOCK_SIZE > vocab_size.

Discriminators used by this probe:
    D1  NaN position (head/middle/tail of the row) on the upstream kernel.
        head->1 & tail->0   => H1 (only head lanes seen)
        head->1 & tail->1   => H2 (the count collapsed to a boolean)
    D2  per-element flags: store libdevice.isnan results and diff them
        against torch.isnan  => direct verdict on H1.
    D3  flag dtype: tl.sum(flags.to(int1)) vs tl.sum(flags.to(int32)) on the
        same input, with flags from libdevice.isnan and from ``x != x``
        => direct verdict on H2 and on libdevice itself.
    D4  BLOCK_SIZE sweep on the untouched upstream kernel
        => direct verdict on H3.

The probe never modifies the upstream kernel; it only launches it with
different inputs and compares against torch.isnan on the host.

Usage (on the A5 host, from the repo root):
    python accuracy_test/easy_ut_026/probe_num_nans_a5.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow "python accuracy_test/easy_ut_026/probe_num_nans_a5.py" from any cwd:
# put the repo root (two levels up from this file) on sys.path so that
# ``accuracy_test.easy_ut_026.runtime_npu`` resolves exactly like under pytest.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import the group-local runtime FIRST (before any vllm.* import): it installs
# the vllm.triton_utils package shim for triton==3.2.0.
from accuracy_test.easy_ut_026.runtime_npu import init_device_properties_triton

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.metrics import logits as _metrics_logits

# vllm-ascend PR #13159 adaptation (same as the UT): rebind the module-level
# libdevice of the upstream kernel module to the CANN libdevice BEFORE the
# kernel is compiled, so libdevice.isnan resolves to an Ascend symbol.
try:
    import triton.language.extra.cann as _cann_mod

    _CANN_LIBDEVICE = _cann_mod.libdevice
except (ImportError, AttributeError) as exc:
    sys.exit(
        "probe aborted: triton.language.extra.cann.libdevice is unavailable "
        f"on this host ({exc!r}); the num_nans UT would skip here as well."
    )

_metrics_logits.libdevice = _CANN_LIBDEVICE

# The probe kernels below are defined in THIS module, so they resolve their
# ``libdevice`` global from this module's namespace at compile time. Give them
# the same CANN libdevice the upstream kernel was rebound to.
libdevice = _CANN_LIBDEVICE

# Only import the kernel after the rebinding above (mirrors the UT ordering).
from vllm.v1.worker.gpu.metrics.logits import _num_nans_kernel  # noqa: E402


# --------------------------------------------------------------------------
# Probe kernels (diagnostics only; the UT and the upstream kernel stay as-is)
# --------------------------------------------------------------------------
@triton.jit
def _probe_isnan_flags_kernel(x_ptr, flags_ptr, size, BLOCK_SIZE: tl.constexpr):
    """D2: store per-element libdevice.isnan flags (int32, 0/1) to memory."""
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < size
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    flag = libdevice.isnan(x).to(tl.int32)
    tl.store(flags_ptr + offs, flag, mask=mask)


@triton.jit
def _probe_reduce_kernel(
    x_ptr,
    out_ptr,
    size,
    USE_LIBDEVICE: tl.constexpr,
    FLAG_INT1: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """D3: block-reduce NaN flags over one row.

    ``USE_LIBDEVICE`` selects the flag source (libdevice.isnan vs ``x != x``);
    ``FLAG_INT1`` selects the flag dtype fed into tl.sum (int1 = the exact
    upstream path, int32 = flags widened BEFORE the reduction). Everything
    else (load, mask, single block) mirrors the upstream kernel.
    """
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < size
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if USE_LIBDEVICE:
        is_nan = libdevice.isnan(x)
    else:
        is_nan = x != x
    if FLAG_INT1:
        total = tl.sum(is_nan.to(tl.int1)).to(tl.int32)
    else:
        total = tl.sum(is_nan.to(tl.int32))
    tl.store(out_ptr + tl.program_id(0), total)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _expected(logits: torch.Tensor) -> torch.Tensor:
    """Trustworthy host reference: torch.isnan + row-wise sum."""
    return torch.isnan(logits).sum(dim=-1).to(torch.int32)


def _run_upstream(logits: torch.Tensor, block_size: int) -> torch.Tensor:
    """Launch the untouched upstream _num_nans_kernel and return its counts."""
    num_reqs, vocab_size = logits.shape
    out = torch.empty(num_reqs, dtype=torch.int32, device=logits.device)
    _num_nans_kernel[(num_reqs,)](
        logits,
        logits.stride(0),
        out,
        vocab_size,
        BLOCK_SIZE=block_size,
    )
    torch.npu.synchronize()
    return out.cpu()


def _make_logits(vocab_size: int, num_nan: int, where: str) -> torch.Tensor:
    """One row with ``num_nan`` NaNs placed at head / middle / tail."""
    row = torch.randn(vocab_size, dtype=torch.float32)
    if num_nan > 0:
        if where == "head":
            row[:num_nan] = float("nan")
        elif where == "tail":
            row[vocab_size - num_nan:] = float("nan")
        elif where == "middle":
            start = (vocab_size - num_nan) // 2
            row[start:start + num_nan] = float("nan")
        else:
            raise ValueError(where)
    return row.unsqueeze(0).to("npu")


def _verdict(ok: bool) -> str:
    return "OK " if ok else "BAD"


# --------------------------------------------------------------------------
# Probe sections
# --------------------------------------------------------------------------
def section_matrix() -> list[str]:
    """[1] Full vocab x frac signature of the upstream kernel (num_reqs=1)."""
    print("[1] upstream kernel failure signature (num_reqs=1, BLOCK_SIZE=8192)")
    print("    vocab  frac   expected  kernel  verdict")
    failing: list[tuple[int, float]] = []
    for vocab in (128, 1024, 8192, 16384):
        for frac in (0.0, 0.1, 0.5, 1.0):
            num_nan = int(vocab * frac)
            logits = _make_logits(vocab, num_nan, "head")
            got = _run_upstream(logits, 8192)
            want = _expected(logits.cpu())
            ok = bool(torch.equal(got, want))
            print(
                f"    {vocab:>5}  {frac:<4}  {int(want[0]):>8}  "
                f"{int(got[0]):>6}  {_verdict(ok)}"
            )
            if not ok:
                failing.append((vocab, frac))
    if not failing:
        print("    -> all cases pass; the UT failure did not reproduce here.")
    else:
        only_partial_block = all(v < 8192 for v, _ in failing)
        all_nonzero_frac = all(f > 0.0 for _, f in failing)
        print(f"    -> {len(failing)} failing case(s).")
        if all_nonzero_frac:
            print("    -> every failing case has NaNs present: consistent "
                  "with H1/H2 (isnan or int1 reduction), not with masking.")
        if only_partial_block:
            print("    -> every failing case has vocab < BLOCK_SIZE (masked "
                  "lanes present): consistent with H3 (masked load).")
    return [f"matrix: {v}@{f}" for v, f in failing]


def section_position() -> dict[str, int]:
    """[1b] D1: head / middle / tail NaN placement, upstream kernel."""
    print()
    print("[1b] D1 NaN position (vocab=128, 12 NaNs, BLOCK_SIZE=8192)")
    results: dict[str, int] = {}
    for where in ("head", "middle", "tail"):
        logits = _make_logits(128, 12, where)
        got = int(_run_upstream(logits, 8192)[0])
        results[where] = got
        print(f"    {where:<6} -> kernel count {got:>3}  (expected 12)")
    head, tail = results["head"], results["tail"]
    if head == 1 and tail == 0:
        print("    -> head->1 & tail->0: only the head lanes are seen => H1 "
              "(libdevice.isnan not element-wise).")
    elif head == 1 and tail == 1:
        print("    -> head->1 & tail->1: all NaNs are seen but the count "
              "collapsed to a boolean => H2 (int1 reduction).")
    elif head == 12 and tail == 12:
        print("    -> position-independent correct counts here; check [1].")
    else:
        print(f"    -> mixed pattern head={head}, tail={tail}; see [2]-[4].")
    return results


def section_flags() -> bool:
    """[2] D2: per-element libdevice.isnan flags vs torch.isnan."""
    print()
    print("[2] D2 element-wise libdevice.isnan flags (vocab=128, 12 head NaNs)")
    logits = _make_logits(128, 12, "head")
    flags = torch.zeros(128, dtype=torch.int32, device="npu")
    _probe_isnan_flags_kernel[(1,)](logits, flags, 128, BLOCK_SIZE=8192)
    torch.npu.synchronize()
    host_flags = torch.isnan(logits.cpu()).to(torch.int32)
    mismatches = int((flags.cpu() != host_flags).sum())
    flagged = int(flags.cpu().sum())
    ok = mismatches == 0
    print(f"    kernel-flagged NaNs: {flagged} (expected 12), "
          f"element mismatches vs torch.isnan: {mismatches}")
    print(f"    -> libdevice.isnan element-wise: {_verdict(ok).strip()}")
    if not ok:
        print("       first 16 kernel flags:",
              flags.cpu()[:16].tolist())
        print("       first 16 torch  flags:",
              host_flags[:16].tolist())
    return ok


def section_reduce() -> dict[str, int]:
    """[3] D3: int1 vs int32 reduction, libdevice vs ``x != x`` flags."""
    print()
    print("[3] D3 reduction dtype on the failing shape "
          "(vocab=128, 12 head NaNs, BLOCK_SIZE=8192)")
    logits = _make_logits(128, 12, "head")
    results: dict[str, int] = {}
    for use_libdevice in (True, False):
        for flag_int1 in (True, False):
            out = torch.empty(1, dtype=torch.int32, device="npu")
            _probe_reduce_kernel[(1,)](
                logits, out, 128,
                USE_LIBDEVICE=use_libdevice,
                FLAG_INT1=flag_int1,
                BLOCK_SIZE=8192,
            )
            torch.npu.synchronize()
            src = "libdevice.isnan" if use_libdevice else "x != x       "
            dtype = "int1 " if flag_int1 else "int32"
            key = f"{'lib' if use_libdevice else 'neq'}_" \
                  f"{'int1' if flag_int1 else 'int32'}"
            results[key] = int(out.cpu()[0])
            print(f"    flags from {src}  sum over {dtype}: "
                  f"{results[key]:>3}  (expected 12)")
    return results


def section_block_sweep() -> dict[int, int]:
    """[4] D4: BLOCK_SIZE sweep on the upstream kernel (vocab=128)."""
    print()
    print("[4] D4 BLOCK_SIZE sweep on upstream kernel "
          "(vocab=128, 12 head NaNs)")
    logits = _make_logits(128, 12, "head")
    results: dict[int, int] = {}
    for block in (128, 256, 8192):
        results[block] = int(_run_upstream(logits, block)[0])
        note = "full block, no masked lanes" if block <= 128 else \
            ("partial mask" if block < 8192 else "upstream config")
        print(f"    BLOCK_SIZE={block:<5} -> {results[block]:>3}  "
              f"(expected 12)   [{note}]")
    if results[128] == 12 and results[8192] != 12:
        print("    -> correct without masked lanes, wrong with them => H3 "
              "(masked tl.load path).")
    elif results[128] != 12:
        print("    -> wrong even without masked lanes => not a masking "
              "issue (H1/H2).")
    return results


def section_multirow() -> list[int]:
    """[4b] spot check: num_reqs=4 rows, per-row counts."""
    print()
    print("[4b] multi-row spot check (num_reqs=4, vocab=128, 12 NaNs/row)")
    base = torch.randn(4, 128, dtype=torch.float32)
    base[:, :12] = float("nan")
    logits = base.to("npu")
    got = _run_upstream(logits, 8192).tolist()
    print(f"    kernel counts per row: {got}  (expected [12, 12, 12, 12])")
    return got


# --------------------------------------------------------------------------
# Synthesis
# --------------------------------------------------------------------------
def synthesize(
    failing: list[str],
    pos: dict[str, int],
    flags_ok: bool | None,
    red: dict[str, int],
    blocks: dict[int, int],
) -> None:
    print()
    print("[5] synthesis")
    if not failing:
        print("    The probe could not reproduce the UT failure. Rerun the "
              "UT; if it still fails, compare process environments (the UT "
              "runs under pytest with the same shim).")
        return

    if flags_ok is False:
        # The per-element flags themselves are wrong => nothing downstream
        # can be trusted; no need to interpret the reduction numbers.
        print("    ROOT CAUSE (H1): libdevice.isnan is NOT element-wise on "
              "this backend -- the per-element flags themselves are wrong.")
        if pos.get("tail") == 0 and pos.get("head") == 1:
            print("    Corroborated by D1: head->1 & tail->0 (only the head "
                  "lanes are ever flagged).")
        print("    Kernel-level fix direction: replace libdevice.isnan with "
              "an element-wise NaN test the backend supports (e.g. x != x).")
        return

    if red and red.get("lib_int1") != 12 and red.get("lib_int32") == 12:
        print("    ROOT CAUSE (H2): tl.sum over an int1 tensor is lowered "
              "incorrectly on the Ascend Triton backend -- any non-zero "
              "count collapses to 1 (boolean-style reduction).")
        if pos.get("head") == 1 and pos.get("tail") == 1:
            print("    Corroborated by D1: head->1 & tail->1 (all NaNs are "
                  "seen, the count itself collapses).")
        if red.get("neq_int1") != 12 and red.get("neq_int32") == 12:
            print("    Confirmed independent of libdevice: the x != x flag "
                  "source shows the same int1-vs-int32 split.")
        print("    Kernel-level fix direction (upstream / vllm-ascend, NOT "
              "the UT): widen the flags before the reduction, e.g.")
        print("        num_nans += tl.sum(is_nan.to(tl.int32))")
    elif flags_ok and red and red.get("lib_int32") != 12:
        # Element-wise flags are fine yet even the int32 reduction is wrong:
        # the reduction is broken beyond the flag dtype.
        print("    ROOT CAUSE (reduction): per-element flags are correct but "
              "even tl.sum over int32 flags miscounts -- the block "
              "reduction itself is broken on this backend.")
    elif blocks.get(128) == 12 and blocks.get(8192) != 12:
        print("    ROOT CAUSE (H3): the masked tl.load path drops elements "
              "when BLOCK_SIZE > vocab_size.")
        print("    Kernel-level fix direction: launch with "
              "BLOCK_SIZE=min(BLOCK_SIZE, next_pow2(vocab_size)) or rework "
              "the masked load.")
    else:
        print("    Inconclusive: combination not covered by the "
              "discriminators above. Report the tables to the maintainers.")

    print()
    print("    UT disposition: keep test_num_nans failing (no xfail/skip).")
    print("    The UT compares the untouched upstream kernel against a CPU ")
    print("    reference; the mismatch is a genuine backend correctness ")
    print("    issue in the Ascend-compiled kernel, which the current ")
    print("    vllm-ascend patch set (PR #13159, libdevice rebinding only) ")
    print("    does not address. Failing faithfully is the correct signal.")


def main() -> None:
    init_device_properties_triton()
    torch.manual_seed(20260821)
    print("=" * 68)
    print(" probe_num_nans_a5 -- _num_nans_kernel undercount root-cause probe")
    print(f" triton {triton.__version__}, torch {torch.__version__}, "
          f"device npu:{torch.npu.current_device()}")
    print("=" * 68)

    failing = section_matrix()
    pos = section_position()
    flags_ok: bool | None = None
    try:
        flags_ok = section_flags()
    except Exception as exc:  # noqa: BLE001 - a compile failure is itself data
        print(f"    [2] aborted with {exc!r}")
    try:
        red = section_reduce()
    except Exception as exc:  # noqa: BLE001
        print(f"    [3] aborted with {exc!r}")
        red = {}
    blocks = section_block_sweep()
    section_multirow()

    synthesize(failing, pos, flags_ok, red, blocks)
    print()
    print("probe complete.")


if __name__ == "__main__":
    main()
