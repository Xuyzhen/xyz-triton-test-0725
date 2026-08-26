"""Run every NPU test module in a fresh process."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NPU_DIR = ROOT / "npu"


def main() -> int:
    extra_args = sys.argv[1:]
    files = sorted(NPU_DIR.glob("test_*.py"))
    failures: list[tuple[str, int]] = []

    for index, path in enumerate(files, 1):
        print(f"\n[{index}/{len(files)}] {path.name}", flush=True)
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(ROOT / "pytest.ini"),
            str(path),
            "-m",
            "npu",
            "-v",
            "--tb=short",
            "-x",
            *extra_args,
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            check=False,
        )
        if completed.returncode not in (0, 5):
            failures.append((path.name, completed.returncode))

    if failures:
        print("\nNPU strict modules with failures:")
        for name, returncode in failures:
            print(f"  {name}: pytest exit {returncode}")
        return 1

    print(f"\nAll {len(files)} NPU strict modules passed or explicitly skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
