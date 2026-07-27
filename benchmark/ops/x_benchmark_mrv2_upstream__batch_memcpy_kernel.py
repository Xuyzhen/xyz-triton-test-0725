# SPDX-License-Identifier: Apache-2.0
"""Benchmark batch_memcpy_kernel on NPU."""
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.v1.worker.mamba_utils import batch_memcpy


def case_batch_memcpy(device):
    for n_copies, copy_size in ((4, 4096), (16, 4096), (64, 4096)):
        src_data = torch.randn(n_copies, copy_size // 4, device=device, dtype=torch.float32)
        dst_data = torch.zeros(n_copies, copy_size // 4, device=device, dtype=torch.float32)
        src_ptrs = torch.tensor([src_data[i].data_ptr() for i in range(n_copies)], dtype=torch.uint64, device=device)
        dst_ptrs = torch.tensor([dst_data[i].data_ptr() for i in range(n_copies)], dtype=torch.uint64, device=device)
        sizes = torch.full((n_copies,), copy_size, dtype=torch.int32, device=device)
        holder = {}

        def fn(_sp=src_ptrs, _dp=dst_ptrs, _sz=sizes, _dst=dst_data):
            batch_memcpy(_sp, _dp, _sz)
            holder['out'] = _dst
            return holder['out']

        yield (f"n_copies={n_copies} size={copy_size}", fn, lambda: float(holder['out'].sum().item()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_batch_memcpy(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=batch_memcpy_kernel {spec} latency_us={latency_us:.2f} checksum={checksum():.3f}")


if __name__ == "__main__":
    main()
