# SPDX-License-Identifier: Apache-2.0
"""Benchmark copy_and_expand_eagle_inputs_kernel on NPU."""
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.v1.spec_decode.utils import copy_and_expand_eagle_inputs_kernel


def case_copy_and_expand_eagle_inputs(device):
    for num_reqs_val, num_spec_tokens in ((4, 5), (16, 5)):
        vocab = 32000
        num_padding_slots = num_spec_tokens
        total_input_tokens = num_reqs_val * 3
        BLOCK_SIZE_TOKENS = 4

        target_token_ids = torch.randint(0, vocab, (total_input_tokens,), dtype=torch.int32, device=device)
        target_positions = torch.arange(total_input_tokens, dtype=torch.int32, device=device)
        next_token_ids = torch.randint(0, vocab, (num_reqs_val,), dtype=torch.int32, device=device)

        total_draft_tokens = num_reqs_val * (3 + num_padding_slots)
        out_input_ids = torch.zeros(total_draft_tokens, dtype=torch.int32, device=device)
        out_positions = torch.zeros(total_draft_tokens, dtype=torch.int32, device=device)
        out_is_rejected = torch.zeros(total_draft_tokens, dtype=torch.int32, device=device)
        out_is_masked = torch.zeros(total_draft_tokens, dtype=torch.int32, device=device)
        out_new_token_indices = torch.zeros(num_padding_slots * num_reqs_val, dtype=torch.int32, device=device)
        out_hidden_state_mapping = torch.zeros(total_input_tokens, dtype=torch.int32, device=device)

        query_start_loc = torch.arange(num_reqs_val + 1, dtype=torch.int32, device=device) * 3
        query_end_loc = query_start_loc[1:] - 1

        grid = (num_reqs_val, (3 + num_padding_slots + BLOCK_SIZE_TOKENS - 1) // BLOCK_SIZE_TOKENS)
        holder = {}

        def fn(_grid=grid, _tti=target_token_ids, _tp=target_positions, _nti=next_token_ids, _oii=out_input_ids, _op=out_positions, _oir=out_is_rejected, _oim=out_is_masked, _onti=out_new_token_indices, _ohsm=out_hidden_state_mapping, _qsl=query_start_loc, _qel=query_end_loc, _nr=num_reqs_val, _nps=num_padding_slots, _tit=total_input_tokens, _bst=BLOCK_SIZE_TOKENS):
            copy_and_expand_eagle_inputs_kernel[_grid](_tti, _tp, _nti, _oii, _op, _oir, _oim, _onti, _ohsm, _qsl, _qel, 0, -2, _tit, _nps, False, BLOCK_SIZE_TOKENS=_bst)
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

    for spec, fn, checksum in case_copy_and_expand_eagle_inputs(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=copy_and_expand_eagle_inputs_kernel {spec} latency_us={latency_us:.2f} checksum={checksum()}")


if __name__ == "__main__":
    main()
