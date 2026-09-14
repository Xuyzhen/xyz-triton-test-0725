"""Run every strict_ut_027 NPU-side test module in a fresh process.

Reuses the isolation model of ``easy_ut_026``: an Ascend vector-core
exception in one kernel poisons the current process device context, which
would otherwise make unrelated tests fail when they create tensors. Starting
a fresh pytest subprocess per test file prevents this cross-contamination.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


# strict_ut_027 directory; the repo root (two levels up) is where pytest must
# run so that ``from accuracy_test.strict_ut_027...`` resolves.
SUITE_DIR = Path(__file__).resolve().parent
ROOT_DIR = SUITE_DIR.parent.parent


def main() -> int:
    extra_args = sys.argv[1:]
    # NPU-side entries only (gpu/ entries run via run_gpu.sh on CUDA hosts).
    files = sorted(SUITE_DIR.glob("npu/test_*.py"))
    if not extra_args:
        # Default: run every NPU test module isolated in its own process.
        selected = files
    else:
        # Forward extra args to a single fresh pytest invocation (they may be a
        # filename, a -K filter, or arbitrary pytest flags).
        selected = ()
    failures: list[tuple[str, int]] = []

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{ROOT_DIR}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(ROOT_DIR)
    )

    if selected:
        for index, path in enumerate(selected, 1):
            print(f"\n[{index}/{len(selected)}] {path.name}", flush=True)
            command = [
                sys.executable,
                "-m",
                "pytest",
                str(path),
                "-v",
                "--tb=short",
                "-ra",
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT_DIR,
                env=env,
                check=False,
            )
            # exit 5 = no tests collected (e.g. whole-module skip) - acceptable.
            if completed.returncode not in (0, 5):
                failures.append((path.name, completed.returncode))
    else:
        # Extra args given: run them once in a fresh process (keeps the
        # "one thing per fresh process" guarantee for partial runs).
        print(f"\nRunning: python -m pytest {extra_args}", flush=True)
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", *extra_args, "-v", "--tb=short", "-ra"],
            cwd=ROOT_DIR,
            env=env,
            check=False,
        )
        if completed.returncode not in (0, 5):
            failures.append((" ".join(extra_args), completed.returncode))

    if failures:
        print("\nNPU strict_ut_027 modules with failures:")
        for name, returncode in failures:
            print(f"  {name}: pytest exit {returncode}")
        return 1

    print(f"\nAll {len(files)} NPU strict_ut_027 modules passed or explicitly skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
