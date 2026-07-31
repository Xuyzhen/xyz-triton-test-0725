# SPDX-License-Identifier: Apache-2.0
"""Benchmark _zero_kv_blocks_kernel on NPU.

This kernel zeros KV cache blocks via absolute byte addresses.
We simulate the KVBlockZeroer.zero_block_ids path by constructing
segment address tensors and calling the kernel directly.
"""
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.v1.worker.utils import _zero_kv_blocks_kernel


def case_zero_kv_blocks(device):
    for num_blocks_val, n_segs_val in ((1, 2), (8, 2), (32, 2)):
        num_kv_heads = 8
        head_size = 128
        block_size = 16
        page_size_el = block_size * num_kv_heads * head_size  # 16384
        blk_size = 1024

        total_blocks = num_blocks_val + 1
        kv_cache = torch.zeros(
            (total_blocks, num_kv_heads, head_size, block_size),
            dtype=torch.float16, device=device,
        )
        seg_addrs = torch.tensor(
            [kv_cache.data_ptr(), kv_cache.data_ptr()],
            dtype=torch.uint64, device=device,
        )
        block_ids = torch.arange(num_blocks_val, dtype=torch.int64, device=device)

        grid = (num_blocks_val * n_segs_val * (page_size_el // blk_size),)
        holder = {}
        def fn(
            _grid=grid, _sa=seg_addrs, _bi=block_ids,
            _nb=num_blocks_val, _pse=page_size_el, _bs=blk_size,
            _ns=n_segs_val, _kv=kv_cache,
        ):
            _zero_kv_blocks_kernel[_grid](
                _sa, _bi, _nb,
                N_SEGS=_ns, PAGE_SIZE_EL=_pse, BLOCK_SIZE=_bs,
            )
            holder['out'] = _kv
            return holder['out']

        yield (f"n_blocks={num_blocks_val} n_segs={n_segs_val} "
               f"page_size_el={page_size_el} block_size={blk_size}",
               fn, lambda: float(holder['out'].sum().item()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_zero_kv_blocks(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_zero_kv_blocks_kernel {spec} latency_us={latency_us:.2f} "
              f"checksum={checksum():.3f}")


if __name__ == "__main__":
    main()
