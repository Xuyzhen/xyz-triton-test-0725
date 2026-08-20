"""Run every easy_ut_026 NPU precision test module in a fresh process.

Reuses the isolation model of ``strict_ut_026``: an Ascend vector-core
exception in one kernel poisons the current process device context, which
would otherwise make unrelated tests fail when they create tensors. Starting a
fresh pytest subprocess per test file prevents this cross-contamination.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    extra_args = sys.argv[1:]
    files = sorted(ROOT.glob("test_*.py"))
    if not extra_args:
        # Default: run every test module isolated in its own process.
        selected = files
    else:
        # Forward extra args to a single fresh pytest invocation (they may be a
        # filename, a -k filter, or arbitrary pytest flags).
        selected = ()
    failures: list[tuple[str, int]] = []

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
                "-x",
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=os.environ.copy(),
                check=False,
            )
            if completed.returncode not in (0, 5):
                failures.append((path.name, completed.returncode))
    else:
        # Extra args given: run them once in a fresh process (keeps the
        # "one thing per fresh process" guarantee for partial runs).
        print(f"\nRunning: python -m pytest {extra_args}", flush=True)
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", *extra_args, "-v", "--tb=short", "-ra", "-x"],
            cwd=ROOT,
            env=os.environ.copy(),
            check=False,
        )
        if completed.returncode not in (0, 5):
            failures.append((" ".join(extra_args), completed.returncode))

    if failures:
        print("\nNPU easy modules with failures:")
        for name, returncode in failures:
            print(f"  {name}: pytest exit {returncode}")
        return 1

    print(f"\nAll {len(files)} NPU easy modules passed or explicitly skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())