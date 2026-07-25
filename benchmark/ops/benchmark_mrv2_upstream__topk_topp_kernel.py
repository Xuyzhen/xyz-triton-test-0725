# SPDX-License-Identifier: Apache-2.0
"""Benchmark _topk_topp_kernel on NPU.

Covers three modes: top-k only, top-p only, and combined top-k+top-p.
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
        yield (f"topk_only batch_size={batch} vocab_size={vocab} k={k_val}",
               fn, lambda: float(holder['out'].sum().item()))


def case_topp_only(device):
    for batch, vocab, p_val in ((16, 32000, 0.9), (64, 32000, 0.9)):
        logits = torch.randn((batch, vocab), device=device)
        p = torch.full((batch,), p_val, device=device)
        holder = {}
        def fn(_logits=logits, _p=p):
            holder['out'] = apply_top_k_top_p_triton(_logits, None, _p)
            return holder['out']
        yield (f"topp_only batch_size={batch} vocab_size={vocab} p={p_val}",
               fn, lambda: float(holder['out'].sum().item()))


def case_topk_topp(device):
    for batch, vocab, k_val, p_val in ((16, 32000, 5, 0.9), (64, 32000, 5, 0.9)):
        logits = torch.randn((batch, vocab), device=device)
        k = torch.full((batch,), k_val, device=device, dtype=torch.int32)
        p = torch.full((batch,), p_val, device=device)
        holder = {}
        def fn(_logits=logits, _k=k, _p=p):
            holder['out'] = apply_top_k_top_p_triton(_logits, _k, _p)
            return holder['out']
        yield (f"topk_topp batch_size={batch} vocab_size={vocab} k={k_val} p={p_val}",
               fn, lambda: float(holder['out'].sum().item()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for case_fn in (case_topk_only, case_topp_only, case_topk_topp):
        for spec, fn, checksum in case_fn(args.device):
            latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
            print(f"op=_topk_topp_kernel {spec} latency_us={latency_us:.2f} "
                  f"checksum={checksum():.3f}")


if __name__ == "__main__":
    main()
