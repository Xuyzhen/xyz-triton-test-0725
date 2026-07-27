# SPDX-License-Identifier: Apache-2.0
"""Run upstream vLLM Triton kernels as isolated NPU smoke tests."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any


SCRIPT_PREFIX = "benchmark_mrv2_upstream"
VERSION_PACKAGES = ("torch", "torch-npu", "triton", "vllm", "vllm-ascend")


def package_versions() -> dict[str, str]:
    versions = {}
    for package in VERSION_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def classify_failure(output: str, returncode: int) -> str:
    text = output.lower()
    traceback_start = text.rfind("traceback (most recent call last)")
    diagnostic = text[traceback_start:] if traceback_start >= 0 else text
    if "no module named" in diagnostic or "cannot import name" in diagnostic:
        return "IMPORT_ERROR"
    if any(token in text for token in (
        "not implemented for dt_",
        "dtype support list",
        "unsupported dtype",
        "invalid operands of type",
    )):
        return "UNSUPPORTED_DTYPE"
    if any(token in text for token in (
        "compilationerror",
        "compile error",
        "failed to compile",
        "triton compilation",
    )):
        return "COMPILE_ERROR"
    if returncode < 0:
        return "PROCESS_SIGNAL"
    return "RUNTIME_ERROR"


def discover_scripts(ops_dir: Path, filters: list[str]) -> list[Path]:
    scripts = sorted(ops_dir.glob(f"{SCRIPT_PREFIX}*.py"))
    if not filters:
        return scripts
    return [
        script for script in scripts
        if any(value in script.stem for value in filters)
    ]


def run_script(script: Path, args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        str(script),
        "--device",
        args.device,
        "--warmup",
        "0",
        "--repeat",
        "1",
    ]
    env = os.environ.copy()
    env["VLLM_PLUGINS"] = ""
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=script.parent,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=args.timeout,
            check=False,
        )
        output = completed.stdout
        status = (
            "PASS" if completed.returncode == 0
            else classify_failure(output, completed.returncode)
        )
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        status = "TIMEOUT"
        returncode = None

    return {
        "script": script.name,
        "status": status,
        "returncode": returncode,
        "duration_s": round(time.perf_counter() - started, 3),
        "command": command,
        "output": output,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile and launch upstream vLLM Triton kernels once on NPU. "
            "Each script runs in a separate process."
        )
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--match",
        action="append",
        default=[],
        help="Run scripts whose filename contains this value (repeatable).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON result path (default: benchmark/results/upstream_smoke_*.json).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ops_dir = Path(__file__).resolve().parent
    scripts = discover_scripts(ops_dir, args.match)
    if not scripts:
        print("No matching upstream benchmark scripts found.", file=sys.stderr)
        return 2

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = args.output or (
        ops_dir.parent / "results" / f"upstream_smoke_{timestamp}.json"
    )
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for index, script in enumerate(scripts, start=1):
        print(f"[{index}/{len(scripts)}] {script.name}", flush=True)
        result = run_script(script, args)
        results.append(result)
        print(f"  {result['status']} ({result['duration_s']:.3f}s)", flush=True)

    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1

    report = {
        "scope": "upstream vLLM Triton kernels on Triton Ascend",
        "device": args.device,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "summary": counts,
        "results": results,
    }
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(f"Summary: {counts}")
    print(f"Report: {output_path}")
    return 0 if counts == {"PASS": len(results)} else 1


if __name__ == "__main__":
    raise SystemExit(main())
