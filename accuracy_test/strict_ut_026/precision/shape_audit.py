"""Shape audit: verify both sides ran the SAME cases on the SAME shapes.

The workflow is NPU-first for shapes: once NPU case params are frozen, the
GPU side must capture with identical params (identical case_id). This script
cross-checks, per (kernel, case_id):

  1. presence   - captured on both sides (missing side reported)
  2. inputs     - shape and dtype identical in both side manifests
  3. outputs    - shape and dtype identical (digest may differ - that is the
                  precision question compare_results.py answers, not this one)

    python precision/shape_audit.py         # audit results/gpu vs results/npu
    python precision/shape_audit.py --list  # just dump the registry + case ids

Exit code 0 = shapes aligned, 1 = any mismatch/missing case.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PREC_ROOT = Path(__file__).resolve().parent
SUITE_ROOT = PREC_ROOT.parent
sys.path.insert(0, str(PREC_ROOT))

import capture_runtime as cr
import kernel_cases


def _sig(entry: dict) -> str:
    """shape+dtype signature of one tensor metadata entry."""
    return f"{entry['shape']}:{entry['dtype']}"


def _diff_tensors(label: str, gpu: dict, npu: dict) -> list[str]:
    issues = []
    for name in sorted(set(gpu) | set(npu)):
        if name not in gpu:
            issues.append(f"{label} '{name}' npu-only {npu[name]['shape']}")
        elif name not in npu:
            issues.append(f"{label} '{name}' gpu-only {gpu[name]['shape']}")
        elif _sig(gpu[name]) != _sig(npu[name]):
            issues.append(f"{label} '{name}' {label} sig {gpu[name]['shape']}:{gpu[name]['dtype']}"
                          f" vs {npu[name]['shape']}:{npu[name]['dtype']}")
    return issues


def audit(results_root: Path) -> int:
    gpu_ids = set(cr.list_cases(results_root, "gpu"))
    npu_ids = set(cr.list_cases(results_root, "npu"))

    if not gpu_ids and not npu_ids:
        print("nothing captured yet under results/ - run run_capture.py first")
        return 1

    problems: list[str] = []
    print(f"{'kernel':<22} {'case_id':<14} case")
    for kernel, cid in sorted(gpu_ids & npu_ids):
        gpu_meta = json.loads((results_root / "gpu" / "cases" / kernel / f"{cid}.json").read_text())
        npu_meta = json.loads((results_root / "npu" / "cases" / kernel / f"{cid}.json").read_text())
        print(f"{kernel:<22} {cid:<14} {gpu_meta['case']}")
        problems += _diff_tensors("input", gpu_meta["inputs"], npu_meta["inputs"])
        problems += _diff_tensors("output", gpu_meta["outputs"], npu_meta["outputs"])

    for kernel, cid in sorted(gpu_ids - npu_ids):
        print(f"{kernel:<22} {cid:<14} MISSING on npu")
        problems.append(f"{kernel}/{cid} not captured on npu")
    for kernel, cid in sorted(npu_ids - gpu_ids):
        print(f"{kernel:<22} {cid:<14} MISSING on gpu")
        problems.append(f"{kernel}/{cid} not captured on gpu (baseline absent)")

    if problems:
        print(f"\n{len(problems)} shape/presence issue(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nOK: both sides captured identical cases on identical shapes")
    return 0


def list_registry() -> int:
    print(f"{'kernel':<22} {'case_id':<14} {'stoch':<6} case")
    for kernel, spec in kernel_cases.all_specs():
        print(f"{kernel:<22} {cr.case_id(spec):<14} {str(spec.stochastic):<6} {spec.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list registered cases instead of auditing results")
    parser.add_argument("--results", default=str(SUITE_ROOT / "results"))
    args = parser.parse_args()
    return list_registry() if args.list else audit(Path(args.results))


if __name__ == "__main__":
    raise SystemExit(main())
