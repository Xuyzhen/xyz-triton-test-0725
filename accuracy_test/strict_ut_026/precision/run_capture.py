"""Stage entry: capture one side (gpu baseline or npu run).

    python precision/run_capture.py --side gpu                 # stage 1: baseline
    python precision/run_capture.py --side npu                 # stage 2: per-case subprocess
    python precision/run_capture.py --side gpu --kernels penalties,gumbel_sample

GPU runs in-process (fast). NPU always spawns run_one_case.py per case so a
kernel crash or poisoned device context cannot abort the whole sweep. Shared
inputs/ and results/ live at the suite root.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PREC_ROOT = Path(__file__).resolve().parent
SUITE_ROOT = PREC_ROOT.parent
sys.path.insert(0, str(PREC_ROOT))


def run_side(side: str, kernels: list[str] | None, inproc: bool) -> int:
    import capture_runtime as cr
    import kernel_cases

    specs = list(kernel_cases.all_specs(kernels))
    print(f"[{side}] {len(specs)} cases: " + ", ".join(f"{k}/{s.name}" for k, s in specs))

    if inproc:
        import run_one_case

    failures: list[str] = []
    for index, (kernel, spec) in enumerate(specs, 1):
        cid = cr.case_id(spec)
        tag = f"[{side} {index}/{len(specs)}] {kernel}/{cid} ({spec.name})"
        try:
            if inproc:
                rc = run_one_case.execute_case(side, kernel, cid)
            else:
                cmd = [sys.executable, str(PREC_ROOT / "run_one_case.py"),
                       "--side", side, "--kernel", kernel, "--case-id", cid]
                rc = subprocess.run(cmd, cwd=SUITE_ROOT).returncode
        except Exception as exc:  # noqa: BLE001
            rc = 1
            print(f"{tag} EXCEPTION {type(exc).__name__}: {exc}")
        if rc == 0:
            print(tag, "OK")
        else:
            failures.append(f"{kernel}/{cid} (exit {rc})")
            print(tag, f"FAILED (exit {rc})")

    if failures:
        print(f"\n[{side}] {len(failures)} case(s) failed:")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"\n[{side}] all cases captured -> {SUITE_ROOT / 'results' / side}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=["gpu", "npu"], required=True)
    parser.add_argument("--kernels", default=None, help="comma-separated subset, default all")
    parser.add_argument("--inproc", action="store_true",
                        help="run in-process even on npu (debug only; a crash takes down the sweep)")
    args = parser.parse_args()

    kernels = args.kernels.split(",") if args.kernels else None
    inproc = args.inproc or args.side == "gpu"
    return run_side(args.side, kernels, inproc)


if __name__ == "__main__":
    raise SystemExit(main())
