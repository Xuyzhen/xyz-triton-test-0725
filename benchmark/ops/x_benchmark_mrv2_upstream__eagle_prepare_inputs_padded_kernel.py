# SPDX-License-Identifier: Apache-2.0
"""Benchmark eagle_prepare_inputs_padded_kernel on NPU."""
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.v1.spec_decode.utils import eagle_prepare_inputs_padded_kernel


def case_eagle_prepare_inputs_padded(device):
    for num_reqs_val in (4, 16):
        cu_num_draft_tokens_vals = [2, 5, 7, 10, 13, 16, 18, 21, 23, 26, 28, 31, 33, 36, 38, 41]
        cu_num_draft_tokens = torch.tensor(cu_num_draft_tokens_vals[:num_reqs_val], dtype=torch.int32, device=device)
        valid_sampled_tokens_count = torch.ones(num_reqs_val, dtype=torch.int32, device=device)
        query_start_loc = torch.arange(num_reqs_val + 1, dtype=torch.int32, device=device)
        token_indices_to_sample = torch.zeros(num_reqs_val, dtype=torch.int32, device=device)
        num_rejected_tokens = torch.zeros(num_reqs_val, dtype=torch.int32, device=device)
        holder = {}

        def fn(_cndt=cu_num_draft_tokens, _vstc=valid_sampled_tokens_count, _qsl=query_start_loc, _tits=token_indices_to_sample, _nrt=num_rejected_tokens, _nr=num_reqs_val):
            eagle_prepare_inputs_padded_kernel[(_nr,)](_cndt, _vstc, _qsl, _tits, _nrt, _nr)
            holder['out'] = _tits
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

    for spec, fn, checksum in case_eagle_prepare_inputs_padded(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=eagle_prepare_inputs_padded_kernel {spec} latency_us={latency_us:.2f} checksum={checksum()}")


if __name__ == "__main__":
    main()
