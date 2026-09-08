# SPDX-License-Identifier: Apache-2.0
"""
NaN-semantics verification for _compute_global_logsumexp on Ascend NPU (Triton).

Upload this file anywhere on the a5 box (same python env as the UT suite) and run:

    python verify_nan_semantics.py

It prints one self-contained report. No pytest needed.

Why this exists
---------------
accuracy_test/acc_ut_260907/npu/test_compute_global_logsumexp_downstream.py
::test_all_neg_inf_blocks fails after upgrading vllm / vllm-ascend to latest
main (kernel source is byte-identical to v0.27.0):

    assert output[0].item() == float("-inf")
    E   AssertionError: assert nan == -inf

Upstream kernel (vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py,
introduced by PR #46781 on 2026-06-30, unchanged since) computes:

    global_max = tl.max(maxes, axis=0)
    global_lse = global_max + tl.log(tl.sum(sumexps * tl.exp(maxes - global_max)))

With all-(-inf) maxes, the subtraction maxes - global_max is (-inf) - (-inf),
which is NaN under IEEE 754. The kernel has NO -inf guard, so under an
IEEE-conformant runtime the whole chain collapses to NaN. If the runtime
instead flushes NaN to 0 (fast-math / FTZ), the chain collapses to -inf and
the test happens to pass -- the suspected reason v0.27 passed.

This script measures, on the CURRENT runtime:
  [1] primitive NaN semantics: exp(nan), 0*nan, (-inf)-(-inf), log(0), log(nan)
  [2] the exact LSE chain stage by stage with all-(-inf) inputs
  [3] the real upstream kernel: all -inf (failure repro) / mixed / fully finite
"""

import math

import torch

from vllm.triton_utils import tl, triton

NEG_INF = float("-inf")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
print("=" * 72)
print("[env] collecting environment info ...")

try:
    import vllm as _vllm

    print(f"[env] vllm         = {_vllm.__version__}")
except Exception as e:  # noqa: BLE001
    print(f"[env] vllm         = <import failed: {e}>")

try:
    import vllm_ascend as _va

    print(f"[env] vllm_ascend  = {getattr(_va, '__version__', '<unknown>')}")
except Exception as e:  # noqa: BLE001
    print(f"[env] vllm_ascend  = <import failed: {e}>")

print(f"[env] torch        = {torch.__version__}")
print(f"[env] triton       = {getattr(triton, '__version__', '<unknown>')}")

try:
    import torch_npu  # noqa: F401

    print(f"[env] torch_npu    = {getattr(torch_npu, '__version__', '<unknown>')}")
except Exception as e:  # noqa: BLE001
    print(f"[env] torch_npu    = <import failed: {e}>")

try:
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    init_device_properties_triton()
    print("[env] init_device_properties_triton() = ok")
except Exception as e:  # noqa: BLE001
    print(f"[env] init_device_properties_triton() = <failed: {e}>")

DEVICE = torch.device("npu")

# Resolve the helper exactly like the UT does (vllm_ascend alias first).
try:
    from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import (
        _compute_global_lse as _compute_global_lse_helper,
    )

    HELPER_SRC = "vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils._compute_global_lse"
except Exception:  # noqa: BLE001
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        _compute_global_logsumexp as _compute_global_lse_helper,
    )

    HELPER_SRC = "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils._compute_global_logsumexp"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def fmt(x) -> str:
    if isinstance(x, torch.Tensor):
        x = x.item()
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "-inf" if x < 0 else "+inf"
        return f"{x:.6g}"
    return str(x)


def sync() -> None:
    torch.npu.synchronize()


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
@triton.jit
def _k_exp(x_ptr, out_ptr):
    tl.store(out_ptr, tl.exp(tl.load(x_ptr)))


@triton.jit
def _k_mul(a_ptr, b_ptr, out_ptr):
    tl.store(out_ptr, tl.load(a_ptr) * tl.load(b_ptr))


@triton.jit
def _k_sub(a_ptr, b_ptr, out_ptr):
    tl.store(out_ptr, tl.load(a_ptr) - tl.load(b_ptr))


@triton.jit
def _k_add(a_ptr, b_ptr, out_ptr):
    tl.store(out_ptr, tl.load(a_ptr) + tl.load(b_ptr))


@triton.jit
def _k_log(x_ptr, out_ptr):
    tl.store(out_ptr, tl.log(tl.load(x_ptr)))


@triton.jit
def _k_lse_chain(
    local_max_ptr,
    local_sumexp_ptr,
    out_global_max_ptr,
    out_diff_ptr,
    out_exp_ptr,
    out_prod_ptr,
    out_sum_ptr,
    out_log_ptr,
    out_result_ptr,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    # Byte-for-byte the same ops as upstream _compute_global_logsumexp,
    # with every intermediate stage stored for inspection.
    blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
    mask = blocks < vocab_num_blocks
    maxes = tl.load(local_max_ptr + blocks, mask=mask, other=float("-inf"))
    sumexps = tl.load(local_sumexp_ptr + blocks, mask=mask, other=0.0)
    global_max = tl.max(maxes, axis=0)
    diff = maxes - global_max
    exps = tl.exp(diff)
    prods = sumexps * exps
    total = tl.sum(prods, axis=0)
    log_total = tl.log(total)
    result = global_max + log_total
    tl.store(out_global_max_ptr, global_max)
    tl.store(out_diff_ptr + blocks, diff)
    tl.store(out_exp_ptr + blocks, exps)
    tl.store(out_prod_ptr + blocks, prods)
    tl.store(out_sum_ptr, total)
    tl.store(out_log_ptr, log_total)
    tl.store(out_result_ptr, result)


# Same wrapper the UT uses to call the installed upstream helper.
@triton.jit
def _global_logsumexp_wrapper(
    local_max_ptr,
    local_max_stride,
    local_sumexp_ptr,
    local_sumexp_stride,
    output_ptr,
    logit_idx,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    result = _compute_global_lse_helper(
        local_max_ptr,
        local_max_stride,
        local_sumexp_ptr,
        local_sumexp_stride,
        logit_idx,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS,
    )
    tl.store(output_ptr, result)


# ---------------------------------------------------------------------------
# [1] Primitive NaN semantics
# ---------------------------------------------------------------------------
print()
print("[1] primitive NaN semantics (scalar fp32 ops)")
print("-" * 72)


def _run_scalar_kernel(kernel, *in_tensors):
    out = torch.zeros(1, dtype=torch.float32, device=DEVICE)
    kernel[(1,)](*in_tensors, out)
    sync()
    return out.item()


NAN_T = torch.tensor([float("nan")], dtype=torch.float32, device=DEVICE)
ZERO_T = torch.tensor([0.0], dtype=torch.float32, device=DEVICE)
NINF_T = torch.tensor([NEG_INF], dtype=torch.float32, device=DEVICE)

r_exp_nan = _run_scalar_kernel(_k_exp, NAN_T)
r_mul = _run_scalar_kernel(_k_mul, ZERO_T, NAN_T)
r_sub = _run_scalar_kernel(_k_sub, NINF_T, NINF_T)
r_add = _run_scalar_kernel(_k_add, NINF_T, NAN_T)
r_log0 = _run_scalar_kernel(_k_log, ZERO_T)
r_lognan = _run_scalar_kernel(_k_log, NAN_T)

print(f"  exp(nan)      = {fmt(r_exp_nan):>6}    IEEE: nan | fast-math/FTZ: 0")
print(f"  0 * nan       = {fmt(r_mul):>6}    IEEE: nan | fast-math/FTZ: 0")
print(f"  (-inf)-(-inf) = {fmt(r_sub):>6}    IEEE: nan")
print(f"  (-inf)+nan    = {fmt(r_add):>6}    IEEE: nan")
print(f"  log(0)        = {fmt(r_log0):>6}    IEEE: -inf")
print(f"  log(nan)      = {fmt(r_lognan):>6}    IEEE: nan")

runtime_flushes_nan = (not math.isnan(r_exp_nan)) or (not math.isnan(r_mul))
if runtime_flushes_nan:
    print("  -> runtime FLUSHES nan (fast-math/FTZ) on scalar exp/mul")
else:
    print("  -> runtime propagates nan (IEEE-conformant) on scalar exp/mul")

# ---------------------------------------------------------------------------
# [2] LSE chain, stage by stage, all-(-inf) maxes
# ---------------------------------------------------------------------------
print()
print("[2] LSE chain stage-by-stage, all-(-inf) maxes (vector fp32, exact upstream ops)")
print("-" * 72)

NUM_BLOCKS = 3
PADDED = triton.next_power_of_2(NUM_BLOCKS)  # 4

local_max = torch.full((PADDED,), NEG_INF, dtype=torch.float32, device=DEVICE)
local_sumexp = torch.zeros(PADDED, dtype=torch.float32, device=DEVICE)

out_global_max = torch.zeros(1, dtype=torch.float32, device=DEVICE)
out_diff = torch.zeros(PADDED, dtype=torch.float32, device=DEVICE)
out_exp = torch.zeros(PADDED, dtype=torch.float32, device=DEVICE)
out_prod = torch.zeros(PADDED, dtype=torch.float32, device=DEVICE)
out_sum = torch.zeros(1, dtype=torch.float32, device=DEVICE)
out_log = torch.zeros(1, dtype=torch.float32, device=DEVICE)
out_result = torch.zeros(1, dtype=torch.float32, device=DEVICE)

_k_lse_chain[(1,)](
    local_max,
    local_sumexp,
    out_global_max,
    out_diff,
    out_exp,
    out_prod,
    out_sum,
    out_log,
    out_result,
    NUM_BLOCKS,
    PADDED_VOCAB_NUM_BLOCKS=PADDED,
)
sync()

print(f"  inputs: maxes = [-inf]*{NUM_BLOCKS} (padded to {PADDED}), sumexps = [0]*{PADDED}")
print(f"  stage 1  global_max = tl.max(maxes)       = {fmt(out_global_max.item())}")
print(f"  stage 2  diff    = maxes - global_max     = {[fmt(v) for v in out_diff.tolist()]}")
print(f"  stage 3  exps    = tl.exp(diff)           = {[fmt(v) for v in out_exp.tolist()]}")
print(f"  stage 4  prods   = sumexps * exps         = {[fmt(v) for v in out_prod.tolist()]}")
print(f"  stage 5  total   = tl.sum(prods)          = {fmt(out_sum.item())}")
print(f"  stage 6  log_t   = tl.log(total)          = {fmt(out_log.item())}")
print(f"  stage 7  result  = global_max + log_t     = {fmt(out_result.item())}")

stage_names = [
    ("stage2 diff  (maxes - global_max)", out_diff.tolist()),
    ("stage3 exps  (tl.exp(diff))", out_exp.tolist()),
    ("stage4 prods (sumexps * exps)", out_prod.tolist()),
    ("stage5 total (tl.sum(prods))", [out_sum.item()]),
    ("stage6 log   (tl.log(total))", [out_log.item()]),
    ("stage7 result(global_max + log)", [out_result.item()]),
]
first_nan = next((name for name, vals in stage_names if any(math.isnan(v) for v in vals)), None)
print(f"  first NaN appears at: {first_nan or 'nowhere'}")

# ---------------------------------------------------------------------------
# [3] Real upstream kernel via the UT's wrapper
# ---------------------------------------------------------------------------
print()
print("[3] real upstream kernel _compute_global_logsumexp (via UT's wrapper)")
print("-" * 72)
print(f"  helper resolved from: {HELPER_SRC}")


def _run_upstream(maxes, sumexps):
    num_blocks = len(maxes)
    padded = triton.next_power_of_2(num_blocks)
    lm = torch.full((1, padded), NEG_INF, dtype=torch.float32, device=DEVICE)
    ls = torch.zeros(1, padded, dtype=torch.float32, device=DEVICE)
    lm[0, :num_blocks] = torch.tensor(maxes, dtype=torch.float32, device=DEVICE)
    ls[0, :num_blocks] = torch.tensor(sumexps, dtype=torch.float32, device=DEVICE)
    out = torch.zeros(1, dtype=torch.float32, device=DEVICE)
    _global_logsumexp_wrapper[(1,)](
        lm,
        lm.stride(0),
        ls,
        ls.stride(0),
        out,
        0,
        num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded,
    )
    sync()
    return out.item()


def _lse_ref(maxes, sumexps):
    gm = max(maxes)
    if gm == NEG_INF:
        return NEG_INF
    return gm + math.log(sum(s * math.exp(m - gm) for m, s in zip(maxes, sumexps)))


# Case A: exact UT failure repro (test_all_neg_inf_blocks inputs)
case_a = _run_upstream([NEG_INF, NEG_INF, NEG_INF], [0.0, 0.0, 0.0])
# Case B: fully finite (sanity)
ref_b = _lse_ref([1.0, 2.0, 3.0], [0.5, 1.0, 2.0])
case_b = _run_upstream([1.0, 2.0, 3.0], [0.5, 1.0, 2.0])
# Case C: mixed finite / -inf (sanity)
ref_c = _lse_ref([1.0, NEG_INF], [1.0, 0.0])
case_c = _run_upstream([1.0, NEG_INF], [1.0, 0.0])

print(f"  case A all -inf   (UT failure repro): kernel = {fmt(case_a):>6} | UT expects -inf")
print(f"  case B all finite  (sanity)         : kernel = {fmt(case_b):>6} | CPU ref = {fmt(ref_b)}")
print(f"  case C finite + -inf (sanity)      : kernel = {fmt(case_c):>6} | CPU ref = {fmt(ref_c)}")

finite_ok = abs(case_b - ref_b) < 1e-5 and abs(case_c - ref_c) < 1e-5
print(f"  -> finite/mixed paths: {'OK' if finite_ok else 'BROKEN'} "
      "(kernel correct outside the all-(-inf) corner)")

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("[verdict]")
if math.isnan(case_a):
    print("  1. The upstream kernel returns NaN for all-(-inf) maxes on THIS runtime.")
    print(f"  2. In the stage-by-stage chain, the first NaN appears at: {first_nan}.")
    print("     Expected IEEE chain: diff = (-inf)-(-inf) = nan -> exp(nan) = nan")
    print("     -> sum = nan -> log(nan) = nan -> result = nan.")
    print("  3. The kernel formula has NO -inf guard (unchanged since PR #46781),")
    print("     so NaN is the inherent IEEE result -- the UT failure is explained:")
    print("     kernel gives nan, UT asserts -inf.")
    print("  4. v0.27 passing means the OLD runtime flushed nan -> 0 somewhere in the")
    print("     chain (fast-math/FTZ), collapsing the result to -inf 'by luck'.")
    print("  5. Recommended UT fix: relax the assertion to accept the kernel's real")
    print("     semantics (nan OR -inf), or add a -inf guard upstream (kernel change).")
elif case_a == NEG_INF:
    print("  1. The upstream kernel returns -inf for all-(-inf) maxes on THIS runtime")
    print("     (NaN is flushed somewhere in the chain -- see [2]).")
    print("  2. The UT failure is NOT reproduced by this script. The discrepancy must")
    print("     come from the UT harness / environment: compare the [env] versions")
    print("     printed above against the failing pytest run's environment.")
else:
    print(f"  Unexpected upstream result for all-(-inf): {fmt(case_a)} -- investigate manually.")

print()
print("DONE")
