# SPDX-License-Identifier: Apache-2.0
"""Benchmark eagle_step_slot_mapping_metadata_kernel on NPU."""
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.v1.spec_decode.utils import eagle_step_slot_mapping_metadata_kernel, PADDING_SLOT_ID


def case_eagle_step_slot_mapping(device):
    for batch, n_blocks, block_size_val in ((4, 16, 16), (16, 16, 16)):
        positions = torch.full((batch,), 10, dtype=torch.int32, device=device)
        block_table = torch.zeros(batch, n_blocks, dtype=torch.int32, device=device)
        for i in range(batch):
            for j in range(n_blocks):
                block_table[i, j] = i * n_blocks + j
        seq_lens = torch.full((batch,), 11, dtype=torch.int32, device=device)
        out_clamped_pos = torch.zeros(batch, dtype=torch.int32, device=device)
        out_slot_mapping = torch.full((batch,), PADDING_SLOT_ID, dtype=torch.int32, device=device)
        max_model_len = 8192
        holder = {}

        def fn(_pos=positions, _bt=block_table, _sl=seq_lens, _ocp=out_clamped_pos, _osm=out_slot_mapping, _bs=block_size_val, _mml=max_model_len, _nb=n_blocks, _b=batch):
            eagle_step_slot_mapping_metadata_kernel[(_b,)](_pos, _bt, _bt.stride(0), _sl, _ocp, _osm, block_size=_bs, max_model_len=_mml, n_blocks_per_req=_nb, PAD_ID=PADDING_SLOT_ID, batch_size=_b)
            holder['out'] = _osm
            return holder['out']

        yield (f"batch={batch} n_blocks={n_blocks} block_size={block_size_val}", fn, lambda: int(holder['out'].sum().item()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_eagle_step_slot_mapping(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=eagle_step_slot_mapping_metadata_kernel {spec} latency_us={latency_us:.2f} checksum={checksum()}")


if __name__ == "__main__":
    main()
