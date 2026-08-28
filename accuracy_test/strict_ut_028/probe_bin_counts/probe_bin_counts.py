"""CPU probe: is penalties output_bin_counts generation deterministic across platforms?

Run on EACH server (no GPU/NPU device needed, CPU only):
    cd .../strict_ut_028/probe_YYYYMMDD
    python probe_bin_counts.py            # prints result, appends to probe_result_<side>.txt
    python probe_bin_counts.py --side gpu # explicit side tag (default: guess from env)

The probe rebuilds penalties inputs with the exact same code path as
run_capture (build_inputs + fixed seed 42) and digests every tensor.
If output_bin_counts digests differ between servers while all other tensors
match, the scatter with duplicated indices (build_inputs) is non-deterministic
across torch builds -> fix build_inputs, not the kernels.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import platform
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "precision"))                 # for capture_runtime
sys.path.insert(0, os.path.join(_HERE, "..", "precision", "kernel_cases")) # for penalties_cases
import penalties_cases as pc  # noqa: E402

LARGE = {"num_tokens": 4, "vocab_size": 129280, "num_status": 4,
         "num_speculative_tokens": 3, "dtype": "bfloat16"}


def digest(t: torch.Tensor) -> str:
    if t.dtype == torch.bfloat16:
        t = t.view(torch.uint16)
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()[:16]


def guess_side() -> str:
    v = sys.executable.lower()
    if "va027_a5" in v or "npu" in v:
        return "npu"
    return "gpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default=guess_side(), choices=["gpu", "npu"])
    args = ap.parse_args()

    t = pc.build_inputs(LARGE, 42)
    lines = [
        f"# probe side={args.side} time={datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        f"torch={torch.__version__} python={platform.python_version()} "
        f"machine={platform.machine()}",
    ]
    for k in sorted(t):
        lines.append(f"{k}|{t[k].shape}|{t[k].dtype}|{digest(t[k])}")

    # extra diagnostics, derived from the result tensor itself (no RNG replay):
    # nonzero bins == unique token indices written; duplicates = n_out - nonzero.
    n_status, vocab = t["output_bin_counts"].shape
    n_out = max(1, vocab // 20)
    nonzero = int(torch.count_nonzero(t["output_bin_counts"]).item())
    lines.append(f"diag|n_out_per_status={n_out} nonzero_bins={nonzero} "
                 f"dup_writes={n_out * n_status - nonzero}")

    out = "\n".join(lines)
    print(out)
    fname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f"probe_result_{args.side}_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt")
    with open(fname, "w", encoding="utf-8") as f:
        f.write(out + "\n")
    print(f"saved -> {os.path.basename(fname)}")


if __name__ == "__main__":
    main()
