import pytest
import torch

from vllm.v1.worker.gpu.block_table import (
    _gather_block_tables_kernel,
    _compute_slot_mappings_kernel,
)
from vllm.v1.worker.gpu.buffer_utils import _load_ptr


def _gather_block_tables_cpu(
    idx_mapping: torch.Tensor,
    src_block_table: torch.Tensor,
    num_blocks_per_req: torch.Tensor,
    num_reqs: int,
    max_num_blocks: int,
    num_reqs_padded: int,
):
    dst = torch.zeros(num_reqs_padded, max_num_blocks, dtype=torch.int32)

    for batch_idx in range(num_reqs_padded):
        if batch_idx >= num_reqs:
            continue
        req_idx = int(idx_mapping[batch_idx])
        n_blocks = int(num_blocks_per_req[req_idx])
        for i in range(n_blocks):
            dst[batch_idx, i] = int(src_block_table[req_idx, i])

    return dst


def test_gather_block_tables_kernel():
    torch.manual_seed(42)
    num_kv_cache_groups = 1
    num_reqs = 3
    num_reqs_padded = 5
    max_num_blocks = 8

    idx_mapping = torch.tensor([2, 0, 1], dtype=torch.int32)

    src_block_table = torch.zeros(num_reqs, max_num_blocks, dtype=torch.int32)
    src_block_table[0, :3] = torch.tensor([10, 20, 30])
    src_block_table[1, :2] = torch.tensor([40, 50])
    src_block_table[2, :4] = torch.tensor([60, 70, 80, 90])

    num_blocks_per_req = torch.tensor([3, 2, 4], dtype=torch.int32)

    expected = _gather_block_tables_cpu(
        idx_mapping,
        src_block_table,
        num_blocks_per_req,
        num_reqs,
        max_num_blocks,
        num_reqs_padded,
    )

    device = torch.device("npu")

    src_block_table_ptrs = torch.tensor(
        [src_block_table.data_ptr()], dtype=torch.uint64, device=device
    )
    dst_block_table = torch.zeros(
        num_reqs_padded, max_num_blocks, dtype=torch.int32, device=device
    )
    dst_block_table_ptrs = torch.tensor(
        [dst_block_table.data_ptr()], dtype=torch.uint64, device=device
    )
    block_table_strides = torch.tensor(
        [max_num_blocks], dtype=torch.int64, device=device
    )

    _gather_block_tables_kernel[(num_kv_cache_groups, num_reqs_padded)](
        idx_mapping.to(device),
        src_block_table_ptrs,
        dst_block_table_ptrs,
        block_table_strides,
        num_blocks_per_req.unsqueeze(0).to(device),
        num_blocks_per_req.unsqueeze(0).stride(0),
        num_reqs,
        BLOCK_SIZE=1024,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(dst_block_table.cpu(), expected, rtol=0, atol=0)


def test_compute_slot_mappings_kernel_simple():
    torch.manual_seed(42)
    num_kv_cache_groups = 1
    num_reqs = 2
    max_num_tokens = 10

    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    query_lens = torch.tensor([3, 2], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32)
    positions = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int64)

    block_size = 4
    block_table = torch.tensor([[5, 10], [20, 30]], dtype=torch.int32)

    block_table_ptrs = torch.tensor(
        [block_table.data_ptr()], dtype=torch.uint64, device="cpu"
    )
    block_table_strides = torch.tensor([2], dtype=torch.int64)
    block_sizes = torch.tensor([block_size], dtype=torch.int32)

    expected = torch.full((num_kv_cache_groups, max_num_tokens), -1, dtype=torch.int64)
    for batch_idx in range(num_reqs):
        req_state_idx = int(idx_mapping[batch_idx])
        start = int(query_start_loc[batch_idx])
        end = int(query_start_loc[batch_idx + 1])
        for i in range(start, end):
            pos = int(positions[i])
            block_idx = pos // block_size
            block_offset = pos % block_size
            block_num = int(block_table[req_state_idx, block_idx])
            expected[0, i] = block_num * block_size + block_offset

    device = torch.device("npu")
    slot_mappings = torch.full(
        (num_kv_cache_groups, max_num_tokens), -1, dtype=torch.int64, device=device
    )

    from vllm.v1.attention.backends.utils import PAD_SLOT_ID

    _compute_slot_mappings_kernel[(num_kv_cache_groups, num_reqs + 1)](
        max_num_tokens,
        idx_mapping.to(device),
        query_start_loc.to(device),
        positions.to(device),
        block_table_ptrs.to(device),
        block_table_strides.to(device),
        block_sizes.to(device),
        slot_mappings,
        slot_mappings.stride(0),
        0,
        CP_SIZE=1,
        CP_INTERLEAVE=1,
        PAD_ID=PAD_SLOT_ID,
        TRITON_BLOCK_SIZE=1024,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        slot_mappings.cpu(), expected, rtol=0, atol=0
    )
