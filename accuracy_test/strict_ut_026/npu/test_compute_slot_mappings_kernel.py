# Strict NPU accuracy test for _compute_slot_mappings_kernel.
from __future__ import annotations

import importlib

import pytest
import torch

from accuracy_test.strict_ut.metrics import assert_exact
from accuracy_test.strict_ut.runtime_npu import DEVICE, synchronize
from vllm.v1.worker.gpu.block_table import (
    _compute_slot_mappings_kernel as _upstream_kernel,
)


def _resolve_kernel():
    """Prefer the Ascend adaptation; reuse upstream on older releases."""
    try:
        module = importlib.import_module("vllm_ascend.worker.v2.block_table")
    except (ImportError, ModuleNotFoundError):
        return _upstream_kernel, "upstream_reuse"
    kernel = getattr(module, "_compute_slot_mappings_kernel", None)
    if kernel is None:
        return _upstream_kernel, "upstream_reuse"
    return kernel, "ascend_adapted"


KERNEL, IMPLEMENTATION_KIND = _resolve_kernel()
pytestmark = [pytest.mark.npu]
if IMPLEMENTATION_KIND == "ascend_adapted":
    pytestmark.append(pytest.mark.npu_ascend_adapted)
else:
    pytestmark.append(pytest.mark.npu_upstream_reuse)


def _reference(
    block_table,
    idx_mapping,
    query_start_loc,
    positions,
    *,
    max_num_tokens,
    block_size,
    cp_size,
    cp_rank,
    cp_interleave,
):
    output = torch.full((1, max_num_tokens), -1, dtype=torch.int32)
    idx_cpu = idx_mapping.cpu()
    starts = query_start_loc.cpu()
    pos_cpu = positions.cpu()
    table_cpu = block_table.cpu()
    for req_idx in range(idx_cpu.numel()):
        state_idx = int(idx_cpu[req_idx])
        start = int(starts[req_idx])
        end = int(starts[req_idx + 1])
        for token_idx in range(start, end):
            position = int(pos_cpu[token_idx])
            global_block_size = block_size * cp_size
            block_index = position // global_block_size
            block_offset = position % global_block_size
            block_number = int(table_cpu[state_idx, block_index])
            if cp_size == 1:
                slot = block_number * block_size + block_offset
            else:
                is_local = (block_offset // cp_interleave) % cp_size == cp_rank
                if not is_local:
                    slot = -1
                else:
                    rounds = block_offset // (cp_interleave * cp_size)
                    remainder = block_offset % cp_interleave
                    slot = block_number * block_size + rounds * cp_interleave + remainder
            output[0, token_idx] = slot
    return output


@pytest.mark.parametrize(
    "block_size,positions,cp_size,cp_rank,cp_interleave",
    [
        (16, [15, 16, 17, 31, 32], 1, 0, 1),
        (32, [31, 32, 33, 63, 64], 1, 0, 1),
        (16, [0, 1, 2, 3, 16, 17, 18, 19], 2, 0, 2),
        (16, [0, 1, 2, 3, 16, 17, 18, 19], 2, 1, 2),
    ],
)
def test_compute_slot_mappings(
    block_size, positions, cp_size, cp_rank, cp_interleave
):
    torch.manual_seed(42)
    max_num_tokens = 64
    max_num_reqs = 8
    max_num_blocks = 4096 if IMPLEMENTATION_KIND == "ascend_adapted" else 64
    num_reqs = 2
    first_len = len(positions) // 2
    query_start_loc = torch.tensor(
        [0, first_len, len(positions)], dtype=torch.int32, device=DEVICE
    )
    idx_mapping = torch.tensor([5, 2], dtype=torch.int32, device=DEVICE)
    positions_tensor = torch.tensor(positions, dtype=torch.int64, device=DEVICE)
    block_table = torch.arange(
        max_num_reqs * max_num_blocks, dtype=torch.int32, device=DEVICE
    ).reshape(max_num_reqs, max_num_blocks)
    block_table_ptrs = torch.tensor(
        [block_table.data_ptr()], dtype=torch.uint64, device=DEVICE
    )
    block_table_strides = torch.tensor(
        [block_table.stride(0)], dtype=torch.int32, device=DEVICE
    )
    block_sizes = torch.tensor([block_size], dtype=torch.int32, device=DEVICE)
    output = torch.full(
        (1, max_num_tokens), 123456789, dtype=torch.int32, device=DEVICE
    )

    kwargs = dict(
        CP_SIZE=cp_size,
        CP_INTERLEAVE=cp_interleave,
        PAD_ID=-1,
        TRITON_BLOCK_SIZE=1024,
    )
    if "TOTAL_BLOCK_SIZE" in tuple(KERNEL.arg_names):
        kwargs["TOTAL_BLOCK_SIZE"] = 4096

    KERNEL[(1, num_reqs + 1)](
        max_num_tokens,
        idx_mapping,
        query_start_loc,
        positions_tensor,
        block_table_ptrs,
        block_table_strides,
        block_sizes,
        output,
        output.stride(0),
        cp_rank,
        **kwargs,
    )
    synchronize()

    expected = _reference(
        block_table,
        idx_mapping,
        query_start_loc,
        positions_tensor,
        max_num_tokens=max_num_tokens,
        block_size=block_size,
        cp_size=cp_size,
        cp_rank=cp_rank,
        cp_interleave=cp_interleave,
    )
    assert_exact(output, expected)
