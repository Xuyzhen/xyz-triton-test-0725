"""Execute ONE capture case in isolation (works for both sides).

Used directly by run_capture.py for NPU (one subprocess per case, keeping a
crashed kernel or poisoned device context from taking down the whole run),
and usable standalone for debugging a single case:

    python precision/run_one_case.py --side gpu --kernel penalties --case-id <cid>

Self-contained: NPU setup reuses this suite's own ``runtime_npu`` module
(full triton-ascend shim incl. insert_slice/extract_slice/get_element), NOT
a trimmed copy. Shared inputs/results live at the suite root (inputs/,
results/) so the whole strict_ut_028 directory travels as one unit.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

PREC_ROOT = Path(__file__).resolve().parent          # .../strict_ut_028/precision
SUITE_ROOT = PREC_ROOT.parent                        # .../strict_ut_028
sys.path.insert(0, str(PREC_ROOT))
sys.path.insert(0, str(SUITE_ROOT))


def _install_npu_runtime() -> None:
    """Import the suite's own runtime_npu (installs the triton_utils shim)."""
    try:
        import runtime_npu  # noqa: F401  (import executes the shim install)
    except Exception as exc:  # e.g. pytest Skipped on a non-NPU box
        print(f"ERROR: NPU runtime unavailable in this process: {exc}", file=sys.stderr)
        raise


def execute_case(side: str, kernel: str, cid: str) -> int:
    """Run a single case in the current process. Returns exit code."""
    import torch

    device = "cuda" if side == "gpu" else "npu"
    if side == "npu":
        _install_npu_runtime()

    import capture_runtime as cr
    import kernel_cases

    mod = kernel_cases.load_module(kernel)
    spec = next((s for s in mod.CASES if cr.case_id(s) == cid), None)
    if spec is None:
        print(f"ERROR: case id {cid} not found under kernel {kernel}", file=sys.stderr)
        return 2

    inputs_root, results_root = SUITE_ROOT / "inputs", SUITE_ROOT / "results"
    if not cr.inputs_path(inputs_root, kernel, cid).exists():
        # First-ever run: generate and persist the shared inputs (GPU-first
        # workflow normally does this, but allow either side to bootstrap).
        cr.save_inputs(inputs_root, spec, mod.build_inputs(spec.params, spec.seed))
    tensors = cr.load_inputs(inputs_root, spec, device)
    outputs = mod.run(side, tensors, spec.params)
    cr.save_case_result(results_root, side, spec, outputs, tensors)
    print(f"OK {kernel}/{cid} ({spec.name}) -> results/{side}/")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=["gpu", "npu"], required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    return execute_case(args.side, args.kernel, args.case_id)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
