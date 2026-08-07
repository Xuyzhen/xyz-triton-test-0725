"""Standalone A3 diagnostic for the _bincount_kernel atomic_or hang.

This is intentionally not a pytest test. Each probe runs in a fresh child
process so a compiler or device hang can be bounded by a parent-side timeout.
It imports the installed vLLM-Ascend kernel and never modifies project source.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import traceback


PROBES = (
    "store",
    "atomic_add_unique",
    "atomic_add_contended",
    "atomic_or_single",
    "atomic_or_unique",
    "atomic_or_contended_same",
    "atomic_or_contended_bits",
    "bincount",
)


def _load_child_runtime():
    import torch
    from vllm.triton_utils import tl, triton
    from vllm_ascend.ops.triton.triton_utils import (
        init_device_properties_triton,
    )
    from vllm_ascend.worker.v2.sample.penalties import _bincount_kernel

    return torch, tl, triton, init_device_properties_triton, _bincount_kernel


def _run_child(probe: str) -> int:
    torch, tl, triton, init_triton, bincount_kernel = _load_child_runtime()

    # Triton reparses JIT function source against the function's module globals.
    # The runtime is imported lazily for parent/child isolation, so expose tl
    # there before defining the nested diagnostic kernels.
    globals()["tl"] = tl

    @triton.jit
    def store_kernel(out_ptr):
        tl.store(out_ptr, 1)

    @triton.jit
    def atomic_add_unique_kernel(out_ptr, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        tl.atomic_add(out_ptr + offsets, 1)

    @triton.jit
    def atomic_add_contended_kernel(out_ptr, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        tl.atomic_add(out_ptr + offsets * 0, 1)

    @triton.jit
    def atomic_or_single_kernel(out_ptr):
        tl.atomic_or(out_ptr, 1)

    @triton.jit
    def atomic_or_unique_kernel(out_ptr, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        tl.atomic_or(out_ptr + offsets, 1)

    @triton.jit
    def atomic_or_contended_same_kernel(out_ptr, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        tl.atomic_or(out_ptr + offsets * 0, 1)

    @triton.jit
    def atomic_or_contended_bits_kernel(out_ptr, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        bits = tl.full((BLOCK,), 1, tl.int32) << offsets
        tl.atomic_or(out_ptr + offsets * 0, bits)

    init_triton()
    device = torch.device("npu")
    block = 8

    print(f"MARK-1 before launch: {probe}", flush=True)
    if probe == "store":
        output = torch.zeros(1, dtype=torch.int32, device=device)
        store_kernel[(1,)](output)
        expected = torch.tensor([1], dtype=torch.int32)
    elif probe == "atomic_add_unique":
        output = torch.zeros(block, dtype=torch.int32, device=device)
        atomic_add_unique_kernel[(1,)](output, BLOCK=block)
        expected = torch.ones(block, dtype=torch.int32)
    elif probe == "atomic_add_contended":
        output = torch.zeros(1, dtype=torch.int32, device=device)
        atomic_add_contended_kernel[(1,)](output, BLOCK=block)
        expected = torch.tensor([block], dtype=torch.int32)
    elif probe == "atomic_or_single":
        output = torch.zeros(1, dtype=torch.int32, device=device)
        atomic_or_single_kernel[(1,)](output)
        expected = torch.tensor([1], dtype=torch.int32)
    elif probe == "atomic_or_unique":
        output = torch.zeros(block, dtype=torch.int32, device=device)
        atomic_or_unique_kernel[(1,)](output, BLOCK=block)
        expected = torch.ones(block, dtype=torch.int32)
    elif probe == "atomic_or_contended_same":
        output = torch.zeros(1, dtype=torch.int32, device=device)
        atomic_or_contended_same_kernel[(1,)](output, BLOCK=block)
        expected = torch.tensor([1], dtype=torch.int32)
    elif probe == "atomic_or_contended_bits":
        output = torch.zeros(1, dtype=torch.int32, device=device)
        atomic_or_contended_bits_kernel[(1,)](output, BLOCK=block)
        expected = torch.tensor([(1 << block) - 1], dtype=torch.int32)
    elif probe == "bincount":
        expanded_idx_mapping = torch.tensor([0], dtype=torch.int32, device=device)
        all_token_ids = torch.tensor([[1, 1, 2, 3]], dtype=torch.int32, device=device)
        prompt_len = torch.tensor([3], dtype=torch.int32, device=device)
        prefill_len = torch.tensor([4], dtype=torch.int32, device=device)
        prompt_bin_mask = torch.zeros((1, 1), dtype=torch.int32, device=device)
        output_bin_counts = torch.zeros((1, 8), dtype=torch.int32, device=device)
        bincount_kernel[(1, 1)](
            expanded_idx_mapping,
            all_token_ids,
            all_token_ids.stride(0),
            prompt_len,
            prefill_len,
            prompt_bin_mask,
            prompt_bin_mask.stride(0),
            output_bin_counts,
            output_bin_counts.stride(0),
            BLOCK_SIZE=8,
        )
        print("MARK-2 launch returned: bincount", flush=True)
        torch.npu.synchronize()
        print("MARK-3 synchronize returned: bincount", flush=True)
        actual_mask = prompt_bin_mask.cpu()
        actual_counts = output_bin_counts.cpu()
        expected_mask = torch.tensor([[0b110]], dtype=torch.int32)
        expected_counts = torch.zeros((1, 8), dtype=torch.int32)
        expected_counts[0, 3] = 1
        torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=0)
        torch.testing.assert_close(actual_counts, expected_counts, rtol=0, atol=0)
        print(f"RESULT PASS: {probe}", flush=True)
        return 0
    else:
        raise ValueError(f"unknown probe: {probe}")

    print(f"MARK-2 launch returned: {probe}", flush=True)
    torch.npu.synchronize()
    print(f"MARK-3 synchronize returned: {probe}", flush=True)
    actual = output.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(f"RESULT PASS: {probe}; output={actual.tolist()}", flush=True)
    return 0


def _run_parent(args: argparse.Namespace) -> int:
    selected = (args.only,) if args.only else PROBES
    script = str(Path(__file__).resolve())
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TRITON_ALWAYS_COMPILE"] = "1"
    if args.debug:
        env["TRITON_DEBUG"] = "1"

    failures = 0
    print("Standalone _bincount_kernel atomic diagnostic")
    print(f"timeout per probe: {args.timeout}s")
    for probe in selected:
        print(f"\n===== {probe} =====", flush=True)
        command = [sys.executable, script, "--child", probe]
        try:
            completed = subprocess.run(
                command,
                env=env,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr
            if stdout:
                print(stdout, end="")
            if stderr:
                print(stderr, end="", file=sys.stderr)
            print(f"RESULT TIMEOUT: {probe}", flush=True)
            print(
                "Stop here: the child was killed and the NPU context may need "
                "reinitialization before more tests.",
                flush=True,
            )
            return 124

        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        if completed.returncode != 0:
            failures += 1
            print(f"RESULT ERROR: {probe}; exit={completed.returncode}")

    print("\n===== summary =====")
    if failures:
        print(f"completed with {failures} error probe(s); inspect the first failure")
        return 1
    print("all selected probes passed; the installed atomic_or did not reproduce the hang")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose the Triton-Ascend atomic_or hang without pytest."
    )
    parser.add_argument("--timeout", type=int, default=60, help="seconds per probe")
    parser.add_argument("--only", choices=PROBES, help="run one parent-side probe")
    parser.add_argument("--debug", action="store_true", help="enable TRITON_DEBUG=1")
    parser.add_argument("--child", choices=PROBES, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.child:
        try:
            return _run_child(args.child)
        except Exception:
            traceback.print_exc()
            return 1
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
