# SPDX-License-Identifier: Apache-2.0
"""Benchmark copy_and_expand_dflash_inputs_kernel on NPU."""
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.v1.spec_decode.utils import copy_and_expand_dflash_inputs_kernel


def case_copy_and_expand_dflash_inputs(device):
    for num_reqs_val, num_spec_tokens in ((4, 5), (16, 5)):
        vocab = 32000
        block_size_val = 16
        num_query_per_req = num_spec_tokens + 1
        total_input_tokens = num_reqs_val * 3

        next_token_ids = torch.randint(0, vocab, (num_reqs_val,), dtype=torch.int32, device=device)
        target_positions = torch.arange(total_input_tokens, dtype=torch.int32, device=device)
        num_query_total = num_reqs_val * num_query_per_req
        out_input_ids = torch.zeros(num_query_total, dtype=torch.int32, device=device)
        out_ctx_pos = torch.zeros(total_input_tokens, dtype=torch.int32, device=device)
        out_query_pos = torch.zeros(num_query_total, dtype=torch.int32, device=device)
        out_ctx_slot = torch.zeros(total_input_tokens, dtype=torch.int32, device=device)
        out_query_slot = torch.zeros(num_query_total, dtype=torch.int32, device=device)
        out_token_indices = torch.zeros(num_reqs_val * num_spec_tokens, dtype=torch.int32, device=device)

        n_blocks = 4
        block_table = torch.zeros(num_reqs_val, n_blocks, dtype=torch.int32, device=device)
        for i in range(num_reqs_val):
            for j in range(n_blocks):
                block_table[i, j] = i * n_blocks + j

        query_start_loc = torch.arange(num_reqs_val + 1, dtype=torch.int32, device=device) * 3

        BLOCK_SIZE = 4
        num_blocks_per_token = (3 + num_query_per_req + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid = (num_reqs_val, num_blocks_per_token)
        holder = {}

        def fn(_grid=grid, _nti=next_token_ids, _tp=target_positions, _oii=out_input_ids, _ocp=out_ctx_pos, _oqp=out_query_pos, _ocs=out_ctx_slot, _oqs=out_query_slot, _oti=out_token_indices, _bt=block_table, _bts=block_table.stride(0), _qsl=query_start_loc, _nr=num_reqs_val, _bs=block_size_val, _nqpr=num_query_per_req, _nst=num_spec_tokens, _tit=total_input_tokens):
            copy_and_expand_dflash_inputs_kernel[_grid](_nti, _tp, _oii, _ocp, _oqp, _ocs, _oqs, _oti, _bt, _bts, _qsl, None, -2, _bs, _nqpr, _nst, _tit, BLOCK_SIZE=BLOCK_SIZE, HAS_NUM_REJECTED=False)
            holder['out'] = _oii
            return holder['out']

        yield (f"num_reqs={num_reqs_val} num_spec={num_spec_tokens}", fn, lambda: int(holder['out'].sum().item()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_copy_and_expand_dflash_inputs(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=copy_and_expand_dflash_inputs_kernel {spec} latency_us={latency_us:.2f} checksum={checksum()}")


if __name__ == "__main__":
    main()
