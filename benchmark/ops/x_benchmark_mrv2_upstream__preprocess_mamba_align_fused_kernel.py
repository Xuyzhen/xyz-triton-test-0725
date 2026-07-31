# SPDX-License-Identifier: Apache-2.0
"""Benchmark preprocess_mamba_align_fused_kernel on NPU."""
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.v1.worker.mamba_utils import preprocess_mamba_align_fused_kernel


def case_preprocess_mamba_align_fused(device):
    for num_reqs_val in (4, 16):
        BLOCK_SIZE = 256
        MAMBA_BLOCK_SIZE = 16

        idx_mapping = torch.arange(num_reqs_val, dtype=torch.int32, device=device)
        state_idx = torch.zeros(num_reqs_val, dtype=torch.int32, device=device)
        num_computed_tokens = torch.full((num_reqs_val,), 10, dtype=torch.int32, device=device)
        query_start_loc = torch.arange(num_reqs_val + 1, dtype=torch.int32, device=device)
        num_accepted_tokens = torch.ones(num_reqs_val, dtype=torch.int32, device=device)
        src_col = torch.zeros(num_reqs_val, dtype=torch.int32, device=device)
        src_off = torch.zeros(num_reqs_val, dtype=torch.int32, device=device)

        n_programs = (num_reqs_val + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid = (n_programs,)
        holder = {}

        def fn(_grid=grid, _im=idx_mapping, _si=state_idx, _nct=num_computed_tokens, _qsl=query_start_loc, _nat=num_accepted_tokens, _sc=src_col, _so=src_off, _nr=num_reqs_val, _mbs=MAMBA_BLOCK_SIZE, _bs=BLOCK_SIZE):
            preprocess_mamba_align_fused_kernel[_grid](_im, _si, _nct, _qsl, _nat, _sc, _so, _nr, BLOCK_SIZE=_bs, MAMBA_BLOCK_SIZE=_mbs)
            holder['out'] = _si
            return holder['out']

        yield (f"num_reqs={num_reqs_val}", fn, lambda: int(holder['out'].sum().item()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_preprocess_mamba_align_fused(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=preprocess_mamba_align_fused_kernel {spec} latency_us={latency_us:.2f} checksum={checksum()}")


if __name__ == "__main__":
    main()
