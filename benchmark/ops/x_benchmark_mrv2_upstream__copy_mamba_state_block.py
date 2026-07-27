# SPDX-License-Identifier: Apache-2.0
"""Benchmark _copy_mamba_state_block on NPU.

NOTE: _copy_mamba_state_block is a @triton.jit device function that cannot
be launched as a standalone kernel. It is tested indirectly through
postprocess_mamba_fused_kernel. This benchmark exercises the SD conv state
copy path (conv_width > 0, CONV_STATE_DIM_FIRST=False).
"""
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.v1.worker.mamba_utils import postprocess_mamba_fused_kernel


def case_copy_mamba_state_block(device):
    for num_reqs_val, num_layers_val in ((4, 1), (16, 1)):
        num_state_types = 2
        block_size_val = 16
        total_states = num_layers_val * num_state_types

        num_accepted_tokens = torch.ones(num_reqs_val, dtype=torch.int32, device=device)
        mamba_state_idx = torch.zeros(num_reqs_val, dtype=torch.int32, device=device)
        num_scheduled_tokens = torch.ones(num_reqs_val, dtype=torch.int32, device=device)
        num_computed_tokens = torch.zeros(num_reqs_val, dtype=torch.int32, device=device)
        num_draft_tokens = torch.ones(num_reqs_val, dtype=torch.int32, device=device)

        num_blocks = 4
        state_tensor = torch.zeros((num_blocks, 128), dtype=torch.float16, device=device)
        state_base_addrs = torch.full((total_states,), state_tensor.data_ptr(),
                                       dtype=torch.int64, device=device)
        state_block_strides = torch.full((total_states,),
                                          state_tensor.stride(0) * state_tensor.element_size(),
                                          dtype=torch.int64, device=device)
        state_elem_sizes = torch.full((total_states,), 2, dtype=torch.int32, device=device)
        state_inner_sizes = torch.full((total_states,), 128, dtype=torch.int64, device=device)
        state_conv_widths = torch.tensor([4, 0] * num_layers_val, dtype=torch.int32, device=device)
        state_group_indices = torch.zeros(total_states, dtype=torch.int32, device=device)
        state_dim_row_count = torch.zeros(total_states, dtype=torch.int32, device=device)
        state_dim_row_stride = torch.zeros(total_states, dtype=torch.int64, device=device)

        block_table = torch.zeros(1, num_reqs_val, num_blocks, dtype=torch.int32, device=device)
        block_table_ptrs = torch.tensor([block_table.data_ptr()], dtype=torch.int64, device=device)
        block_table_stride_req = block_table.stride(1)

        num_accepted_tokens_out = torch.zeros(num_reqs_val, dtype=torch.int32, device=device)
        num_accepted_tokens_out.copy_(num_accepted_tokens)

        grid = (num_reqs_val, total_states)
        holder = {}
        def fn(
            _grid=grid,
            _nat=num_accepted_tokens, _msi=mamba_state_idx,
            _nst=num_scheduled_tokens, _nct=num_computed_tokens,
            _ndt=num_draft_tokens,
            _btp=block_table_ptrs, _bts=block_table_stride_req,
            _sba=state_base_addrs, _sbs=state_block_strides,
            _ses=state_elem_sizes, _sis=state_inner_sizes,
            _scw=state_conv_widths, _sgi=state_group_indices,
            _sdrc=state_dim_row_count, _sdrs=state_dim_row_stride,
            _nao=num_accepted_tokens_out, _nr=num_reqs_val,
            _bs=block_size_val,
        ):
            postprocess_mamba_fused_kernel[_grid](
                _nat, _msi, _nst, _nct, _ndt,
                _btp, _bts,
                _sba, _sbs, _ses, _sis, _scw, _sgi,
                _sdrc, _sdrs,
                _nao,
                None,
                _nr,
                block_size=_bs,
                COPY_BLOCK_SIZE=1024,
                CONV_STATE_DIM_FIRST=False,
                HAS_IDX_MAPPING=False,
                PRECOMPUTED_NEW_COMPUTED=False,
            )
            holder['out'] = _nao
            return holder['out']

        yield (f"num_reqs={num_reqs_val} num_layers={num_layers_val} "
               f"num_states={total_states}",
               fn, lambda: int(holder['out'].sum().item()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_copy_mamba_state_block(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_copy_mamba_state_block(via_postprocess) {spec} latency_us={latency_us:.2f} "
              f"checksum={checksum()}")


if __name__ == "__main__":
    main()
