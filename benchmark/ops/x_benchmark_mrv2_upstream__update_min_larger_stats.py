# SPDX-License-Identifier: Apache-2.0
"""Benchmark _update_min_larger_stats on NPU.

NOTE: _update_min_larger_stats is a @triton.jit device function that cannot
be launched as a standalone kernel. It is tested indirectly through
_topk_topp_kernel (apply_top_k_top_p_triton). This benchmark covers the
top-k only path which exercises _update_min_larger_stats in the ternary
search loop.
"""
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton


def case_topk_only(device):
    for batch, vocab, k_val in ((16, 32000, 5), (64, 32000, 5)):
        logits = torch.randn((batch, vocab), device=device)
        k = torch.full((batch,), k_val, device=device, dtype=torch.int32)
        holder = {}
        def fn(_logits=logits, _k=k):
            holder['out'] = apply_top_k_top_p_triton(_logits, _k, None)
            return holder['out']
        yield (f"batch_size={batch} vocab_size={vocab} k={k_val}",
               fn, lambda: float(holder['out'].sum().item()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_topk_only(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_update_min_larger_stats(via_topk) {spec} latency_us={latency_us:.2f} "
              f"checksum={checksum():.3f}")


if __name__ == "__main__":
    main()
