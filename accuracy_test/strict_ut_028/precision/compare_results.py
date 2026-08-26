"""Stage 3: compare gpu baseline vs npu results and emit a report.

    python precision/compare_results.py            # uses results/gpu + results/npu
    python precision/compare_results.py --gpu A --npu B --out report/

Join key: (kernel, case_id). For every declared output the case's compare
mode applies (int exact / dtype tolerance / skip), after the case's optional
normalize hook brings both sides onto a common semantic basis (e.g. gumbel
logits-cache pre/post-temperature divergence).
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
from compare_metrics import compare_output


def compare_all(gpu_root: Path, npu_root: Path, out_dir: Path) -> dict:
    gpu_ids = set(cr.list_cases(gpu_root, "gpu"))
    npu_ids = set(cr.list_cases(npu_root, "npu"))

    rows: list[dict] = []
    for kernel, cid in sorted(gpu_ids & npu_ids):
        gpu_t, gpu_meta = cr.load_case_result(gpu_root, "gpu", kernel, cid)
        npu_t, npu_meta = cr.load_case_result(npu_root, "npu", kernel, cid)

        input_mismatch = [
            name for name in gpu_meta["inputs"]
            if name in npu_meta["inputs"]
            and gpu_meta["inputs"][name]["digest"] != npu_meta["inputs"][name]["digest"]
        ]
        if input_mismatch:
            rows.append({"kernel": kernel, "case_id": cid, "case": gpu_meta["case"],
                         "status": "ERROR", "detail": f"input digests differ: {input_mismatch}"})
            continue

        mod = kernel_cases.load_module(kernel)
        spec = next((s for s in mod.CASES if cr.case_id(s) == cid), None)
        inputs = {}
        if spec is not None and spec.normalize is not None:
            blob = torch.load(cr.inputs_path(SUITE_ROOT / "inputs", kernel, cid),
                              map_location="cpu", weights_only=False)
            inputs = blob["tensors"]

        for out_name, mode in (spec.output_modes if spec else gpu_meta["output_modes"]).items():
            if out_name not in gpu_t or out_name not in npu_t:
                rows.append({"kernel": kernel, "case_id": cid, "case": gpu_meta["case"],
                             "output": out_name, "status": "ERROR", "detail": "missing on one side"})
                continue
            a, b = gpu_t[out_name], npu_t[out_name]
            if spec is not None and spec.normalize is not None:
                a = spec.normalize(out_name, "gpu", a, inputs, spec.params)
                b = spec.normalize(out_name, "npu", b, inputs, spec.params)
            res = compare_output(mode, a, b)
            rows.append({"kernel": kernel, "case_id": cid, "case": gpu_meta["case"],
                         "output": out_name, "mode": mode, "status": res.status,
                         "detail": res.detail, "max_abs_err": res.max_abs_err,
                         "mismatch_ratio": res.mismatch_ratio})

    for kernel, cid in sorted(gpu_ids - npu_ids):
        rows.append({"kernel": kernel, "case_id": cid, "status": "MISSING",
                     "detail": "captured on gpu only - run --side npu"})
    for kernel, cid in sorted(npu_ids - gpu_ids):
        rows.append({"kernel": kernel, "case_id": cid, "status": "MISSING",
                     "detail": "captured on npu only - gpu baseline absent"})

    return {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows}


def write_report(report: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"compare_{stamp}.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    rows = report["rows"]
    order = {"ERROR": 0, "FAIL": 1, "MISSING": 2, "WARN": 3, "PASS": 4, "SKIP": 4}
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    lines = [
        "# GPU-NPU precision compare report",
        "",
        f"- generated: {report['generated_at']}",
        f"- total rows: {len(rows)}  " + "  ".join(f"{k}: {v}" for k, v in sorted(counts.items())),
        "",
        "| kernel | case | output | mode | status | max_abs_err | detail |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (order.get(r["status"], 9), r.get("kernel", ""), r.get("case", ""))):
        lines.append(
            f"| {r.get('kernel','')} | {r.get('case', r.get('case_id',''))} | {r.get('output','-')} "
            f"| {r.get('mode','-')} | {r.get('status', '?')} "
            f"| {r.get('max_abs_err', '-') if isinstance(r.get('max_abs_err'), float) else '-'} "
            f"| {r.get('detail', '')} |"
        )
    md_path = out_dir / f"compare_{stamp}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=str(SUITE_ROOT / "results"))
    parser.add_argument("--npu", default=str(SUITE_ROOT / "results"))
    parser.add_argument("--out", default=str(SUITE_ROOT / "report"))
    args = parser.parse_args()

    report = compare_all(Path(args.gpu), Path(args.npu), Path(args.out))
    md_path = write_report(report, Path(args.out))
    bad = sum(1 for r in report["rows"] if r["status"] in ("FAIL", "ERROR", "MISSING"))
    print(f"report: {md_path}")
    print(f"summary: {bad} FAIL/ERROR/MISSING row(s) out of {len(report['rows'])}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
