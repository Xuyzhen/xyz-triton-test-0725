# SPDX-License-Identifier: Apache-2.0
"""Benchmark expand_kernel on NPU."""
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.v1.sample.rejection_sampler import expand_kernel


def case_expand_kernel(device):
    for batch, max_num_tokens in ((4, 128), (16, 128)):
        cu_num_tokens = torch.arange(1, batch + 1, dtype=torch.int32, device=device) * 3
        num_tokens = int(cu_num_tokens[-1].item())
        input_tensor = torch.arange(batch, dtype=torch.int32, device=device)
        output_tensor = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        holder = {}

        def fn(_out=output_tensor, _inp=input_tensor, _cnt=cu_num_tokens, _b=batch, _mnt=max_num_tokens):
            expand_kernel[(_b,)](_out, _inp, _cnt, 0, 0, MAX_NUM_TOKENS=_mnt)
            holder['out'] = _out
            return holder['out']

        yield (f"batch={batch} max_num_tokens={max_num_tokens}", fn, lambda: int(holder['out'].sum().item()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_expand_kernel(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=expand_kernel {spec} latency_us={latency_us:.2f} checksum={checksum()}")


if __name__ == "__main__":
    main()
