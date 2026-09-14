"""Stage 4: dual-benchmark ratio report (精度标准 2.1 §4.5, CPU golden anchor).

    python precision/ratio_report.py            # results/{cpu,gpu,npu} -> report/

双标杆比对: the cpu side (fp64 ref()) is the golden truth; GPU and NPU are
both measured against it, and NPU's error metrics are normalized by GPU's
(标准 §4.5.1):

    Ratio = metric_npu / max(metric_gpu, err)      err = 小值域error指标 (§4.5.3)

Grades (all three ratios must hold):
    L2: MARE比 ≤ 2,   MERE比 ≤ 1.2, RMSE比 ≤ 1.2
    L1: MARE比 ≤ 5,   MERE比 ≤ 1.5, RMSE比 ≤ 1.5
    L0: MARE比 ≤ 10,  MERE比 ≤ 2.0, RMSE比 ≤ 2.0

Error metrics per §4.1 (actual vs golden):
    MARE = max(|a-g| / (|g| + 1e-7))
    MERE = avg(|a-g| / (|g| + 1e-7))
    RMSE = sqrt(mean((a-g)^2))

小值域 (§4.5.3): ErrorCount = Σ I(|g| < threshold ∧ |a-g| > error);
pass iff ErrorCount_npu / max(ErrorCount_gpu, 1) ≤ 2.

inf/nan (§4.5.1 note): positions where golden is inf/nan are excluded from
the statistical metrics but must agree bitwise (same inf sign / same nan)
between actual and golden; any disagreement fails the output.

Integer outputs (§4.2/4.3): NPU must match the golden bitwise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PREC_ROOT = Path(__file__).resolve().parent
SUITE_ROOT = PREC_ROOT.parent
sys.path.insert(0, str(PREC_ROOT))

import capture_runtime as cr
import kernel_cases

# §4.5.3 小值域阈值表: dtype -> (small value threshold, error metric)
SMALL_VALUE = {
    "torch.float16": (2.0**-11, 2.0**-16),
    "torch.bfloat16": (2.0**-8, 2.0**-16),
    "torch.float32": (2.0**-14, 2.0**-30),
    "torch.float64": (2.0**-14, 2.0**-30),
}
DEFAULT_SMALL = (2.0**-14, 2.0**-30)  # fp32 when dtype unknown

REL_EPS = 1e-7  # §4.1: relative-error denominator guard

# (grade, mare_ratio, mere_ratio, rmse_ratio) - strictest first
GRADES = [("L2", 2.0, 1.2, 1.2), ("L1", 5.0, 1.5, 1.5), ("L0", 10.0, 2.0, 2.0)]

_ORDER = {"ERROR": 0, "FAIL": 1, "MISSING": 2, "NO-GOLDEN": 3, "PASS": 4, "SKIP": 5}


def error_metrics(actual: torch.Tensor, golden: torch.Tensor) -> tuple[float, float, float]:
    """MARE / MERE / RMSE of actual vs golden over positions where golden is
    finite (§4.1 formulas; inf/nan golden positions handled separately)."""
    a = actual.to(torch.float64)
    g = golden.to(torch.float64)
    finite = torch.isfinite(g)
    if not finite.any():
        return 0.0, 0.0, 0.0
    diff = (a[finite] - g[finite]).abs()
    rel = diff / (g[finite].abs() + REL_EPS)
    return float(rel.max()), float(rel.mean()), float(torch.sqrt((diff ** 2).mean()))


def inf_nan_mismatch(actual: torch.Tensor, golden: torch.Tensor) -> int:
    """Count positions where inf/nan-ness (or inf sign) differs from golden."""
    a, g = actual, golden
    bad = int((torch.isinf(a) != torch.isinf(g)).sum())
    bad += int((torch.isnan(a) != torch.isnan(g)).sum())
    both = torch.isinf(a) & torch.isinf(g)
    if both.any():
        bad += int((torch.sign(a[both]) != torch.sign(g[both])).sum())
    return bad


def error_count(actual: torch.Tensor, golden: torch.Tensor,
                 threshold: float, error: float) -> int:
    """§4.5.3 ErrorCount = Σ I(|golden| < threshold ∧ |actual-golden| > error)."""
    a = actual.to(torch.float64)
    g = golden.to(torch.float64)
    small = g.abs() < threshold
    return int((small & ((a - g).abs() > error)).sum())


def grade_of(r_mare: float, r_mere: float, r_rmse: float) -> str | None:
    for name, m, e, r in GRADES:
        if r_mare <= m and r_mere <= e and r_rmse <= r:
            return name
    return None


def _load_inputs(kernel: str, cid: str) -> dict[str, torch.Tensor]:
    blob = torch.load(cr.inputs_path(SUITE_ROOT / "inputs", kernel, cid),
                      map_location="cpu", weights_only=False)
    return blob["tensors"]


def report_all(results_root: Path) -> dict:
    cpu_ids = set(cr.list_cases(results_root, "cpu"))
    gpu_ids = set(cr.list_cases(results_root, "gpu"))
    npu_ids = set(cr.list_cases(results_root, "npu"))
    triple = sorted((cpu_ids & gpu_ids) & npu_ids)

    rows: list[dict] = []
    for kernel, cid in triple:
        cpu_t, _ = cr.load_case_result(results_root, "cpu", kernel, cid)
        gpu_t, gpu_meta = cr.load_case_result(results_root, "gpu", kernel, cid)
        npu_t, _ = cr.load_case_result(results_root, "npu", kernel, cid)

        mod = kernel_cases.load_module(kernel)
        spec = next((s for s in mod.CASES if cr.case_id(s) == cid), None)
        inputs = {}
        if spec is not None and spec.normalize is not None:
            inputs = _load_inputs(kernel, cid)

        for out_name, mode in (spec.output_modes if spec else gpu_meta["output_modes"]).items():
            base = {"kernel": kernel, "case_id": cid, "case": gpu_meta["case"],
                    "output": out_name, "mode": mode}
            if mode == cr.MODE_SKIP:
                rows.append({**base, "status": "SKIP",
                             "detail": "stochastic output, not compared"})
                continue
            if out_name not in cpu_t or out_name not in gpu_t or out_name not in npu_t:
                rows.append({**base, "status": "ERROR", "detail": "missing on one side"})
                continue

            golden = cpu_t[out_name]
            a_gpu, a_npu = gpu_t[out_name], npu_t[out_name]
            if spec is not None and spec.normalize is not None:
                a_gpu = spec.normalize(out_name, "gpu", a_gpu, inputs, spec.params)
                a_npu = spec.normalize(out_name, "npu", a_npu, inputs, spec.params)

            if not golden.is_floating_point():
                # §4.2/4.3 整数/非计算类: bitwise vs golden.
                npu_ok = bool(torch.equal(a_npu, golden))
                gpu_ok = bool(torch.equal(a_gpu, golden))
                rows.append({**base, "status": "PASS" if npu_ok else "FAIL",
                             "detail": "bitwise vs golden" if npu_ok else
                                       f"{int((a_npu != golden).sum())} mismatched elements vs golden"
                                       + ("" if gpu_ok else " (gpu also mismatched)")})
                continue

            # float output: dual-benchmark ratio metrics (§4.5.1)
            out_dtype = gpu_meta["outputs"].get(out_name, {}).get("dtype", "torch.float32")
            threshold, err = SMALL_VALUE.get(out_dtype, DEFAULT_SMALL)

            inf_bad = inf_nan_mismatch(a_gpu, golden) + inf_nan_mismatch(a_npu, golden)
            g_mare, g_mere, g_rmse = error_metrics(a_gpu, golden)
            n_mare, n_mere, n_rmse = error_metrics(a_npu, golden)

            r_mare = n_mare / max(g_mare, err)
            r_mere = n_mere / max(g_mere, err)
            r_rmse = n_rmse / max(g_rmse, err)

            ec_gpu = error_count(a_gpu, golden, threshold, err)
            ec_npu = error_count(a_npu, golden, threshold, err)
            ec_ratio = ec_npu / max(ec_gpu, 1)

            grade = grade_of(r_mare, r_mere, r_rmse)
            small_ok = ec_ratio <= 2.0
            if inf_bad:
                status, detail = "FAIL", f"{inf_bad} inf/nan mismatch(es) vs golden"
            elif grade is None:
                status, detail = "FAIL", "ratios exceed L0 thresholds"
            elif not small_ok:
                status, detail = "FAIL", f"small-value ErrorCount ratio {ec_ratio:.1f} > 2"
            else:
                status, detail = "PASS", f"{grade} (小值域 PASS)"

            rows.append({**base, "dtype": out_dtype,
                         "gpu_mare": g_mare, "gpu_mere": g_mere, "gpu_rmse": g_rmse,
                         "npu_mare": n_mare, "npu_mere": n_mere, "npu_rmse": n_rmse,
                         "ratio_mare": r_mare, "ratio_mere": r_mere, "ratio_rmse": r_rmse,
                         "ec_gpu": ec_gpu, "ec_npu": ec_npu, "ec_ratio": ec_ratio,
                         "grade": grade or "-", "status": status, "detail": detail})

    for kernel, cid in sorted((gpu_ids & npu_ids) - cpu_ids):
        rows.append({"kernel": kernel, "case_id": cid, "status": "NO-GOLDEN",
                     "detail": "captured on gpu+npu but no cpu golden - run run_capture.py --side cpu"})
    for kernel, cid in sorted((cpu_ids & gpu_ids) - npu_ids):
        rows.append({"kernel": kernel, "case_id": cid, "status": "MISSING",
                     "detail": "no npu result - run run_capture.py --side npu"})
    for kernel, cid in sorted((cpu_ids & npu_ids) - gpu_ids):
        rows.append({"kernel": kernel, "case_id": cid, "status": "MISSING",
                     "detail": "no gpu baseline - run run_capture.py --side gpu"})
    return {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows}


def _fmt(v) -> str:
    if not isinstance(v, float):
        return "-"
    if v != v or v in (float("inf"), float("-inf")):
        return f"{v}"
    return f"{v:.3g}"


def write_report(report: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"ratio_{stamp}.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    rows = report["rows"]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    lines = [
        "# 双标杆精度 Ratio 报告 (精度标准 2.1 §4.5, CPU fp64 真值锚点)",
        "",
        f"- generated: {report['generated_at']}",
        f"- total rows: {len(rows)}  " + "  ".join(f"{k}: {v}" for k, v in sorted(counts.items())),
        "- Ratio = metric_npu / max(metric_gpu, 小值域error); err floor per §4.5.3",
        "- 等级判定: L2 ≤(2,1.2,1.2)  L1 ≤(5,1.5,1.5)  L0 ≤(10,2,2)  [MARE比, MERE比, RMSE比]",
        "",
        "| kernel | case | output | grade | status | MARE比 | MERE比 | RMSE比 | ec_gpu | ec_npu | detail |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (_ORDER.get(r["status"], 9), r.get("kernel", ""), r.get("case", ""))):
        lines.append(
            f"| {r.get('kernel','')} | {r.get('case', r.get('case_id',''))} | {r.get('output','-')} "
            f"| {r.get('grade','-')} | {r.get('status','?')} "
            f"| {_fmt(r.get('ratio_mare'))} | {_fmt(r.get('ratio_mere'))} | {_fmt(r.get('ratio_rmse'))} "
            f"| {r.get('ec_gpu','-')} | {r.get('ec_npu','-')} | {r.get('detail','')} |"
        )
    md_path = out_dir / f"ratio_{stamp}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(SUITE_ROOT / "results"))
    parser.add_argument("--out", default=str(SUITE_ROOT / "report"))
    args = parser.parse_args()

    report = report_all(Path(args.results))
    md_path = write_report(report, Path(args.out))
    bad = sum(1 for r in report["rows"] if r["status"] in ("FAIL", "ERROR", "MISSING", "NO-GOLDEN"))
    print(f"report: {md_path}")
    print(f"summary: {bad} FAIL/ERROR/MISSING row(s) out of {len(report['rows'])}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
