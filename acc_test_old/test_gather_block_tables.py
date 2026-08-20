# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.gpu.block_table import _gather_block_tables_kernel
from vllm.v1.worker.gpu.buffer_utils import _load_ptr

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _gather_block_tables_cpu(
    batch_idx_to_req_idx: torch.Tensor,
    src_block_tables: list[torch.Tensor],
    dst_block_tables: list[torch.Tensor],
    num_blocks: torch.Tensor,
    num_reqs: int,
) -> None:
    """Pure PyTorch CPU reference implementation of gather block tables.

    For each group and batch index, copies block IDs from src to dst
    using the request index mapping. Padded rows (batch_idx >= num_reqs)
    are zeroed out.
    """
    num_groups = len(src_block_tables)
    max_num_blocks = src_block_tables[0].shape[1]

    for group_id in range(num_groups):
        src = src_block_tables[group_id].clone()
        dst = dst_block_tables[group_id].clone()

        for batch_idx in range(dst.shape[0]):
            if batch_idx >= num_reqs:
                dst[batch_idx].zero_()
                continue

            req_idx = int(batch_idx_to_req_idx[batch_idx])
            n_blocks = int(num_blocks[group_id, req_idx])
            dst[batch_idx, :n_blocks] = src[req_idx, :n_blocks]

        dst_block_tables[group_id].copy_(dst)


@pytest.mark.parametrize("num_groups", [1, 3])
def test_gather_block_tables_basic(num_groups: int) -> None:
    """Gather block tables kernel: basic functionality.

    Verifies that block IDs are correctly gathered from source block
    tables to destination block tables using batch-to-request index
    mapping, and that padded rows are zeroed out.
    """
    init_device_properties_triton()
    torch.manual_seed(42)

    batch_size = 4
    max_num_reqs = 6
    max_num_blocks = 8
    BLOCK_SIZE = 4

    device = torch.device("npu")

    batch_idx_to_req_idx = torch.tensor([0, 3, 1, 5], dtype=torch.int32)

    src_block_tables = []
    dst_block_tables = []
    src_ptrs = []
    dst_ptrs = []
    strides = []

    for g in range(num_groups):
        src = torch.randint(0, 100, (max_num_reqs, max_num_blocks), dtype=torch.int32)
        dst = torch.zeros(batch_size, max_num_blocks, dtype=torch.int32, device=device)
        src_block_tables.append(src.to(device))
        dst_block_tables.append(dst)

        src_ptrs.append(src_block_tables[g].data_ptr())
        dst_ptrs.append(dst_block_tables[g].data_ptr())
        strides.append(src_block_tables[g].stride(0))

    src_block_table_ptrs = torch.tensor(src_ptrs, dtype=torch.uint64, device=device)
    dst_block_table_ptrs = torch.tensor(dst_ptrs, dtype=torch.uint64, device=device)
    block_table_strides = torch.tensor(strides, dtype=torch.int64, device=device)

    num_blocks = torch.randint(1, max_num_blocks + 1, (num_groups, max_num_reqs), dtype=torch.int32)
    num_blocks_stride = num_blocks.stride(0)

    num_reqs = 4

    grid = (num_groups, batch_size)
    _gather_block_tables_kernel[grid](
        batch_idx_to_req_idx.to(device),
        src_block_table_ptrs,
        dst_block_table_ptrs,
        block_table_strides,
        num_blocks.to(device),
        num_blocks_stride,
        num_reqs,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    # CPU reference
    expected_src = [t.cpu() for t in src_block_tables]
    expected_dst = [t.clone() for t in torch.zeros_like(dst_block_tables[0].cpu()).unsqueeze(0).expand(num_groups, -1, -1)]
    expected_dst_list = [expected_dst[g].clone() for g in range(num_groups)]

    for group_id in range(num_groups):
        src_cpu = src_block_tables[group_id].cpu()
        for batch_idx in range(batch_size):
            if batch_idx >= num_reqs:
                expected_dst_list[group_id][batch_idx].zero_()
            else:
                req_idx = int(batch_idx_to_req_idx[batch_idx])
                n_blocks = int(num_blocks[group_id, req_idx])
                expected_dst_list[group_id][batch_idx, :n_blocks] = src_cpu[req_idx, :n_blocks]

    for g in range(num_groups):
        torch.testing.assert_close(
            dst_block_tables[g].cpu(), expected_dst_list[g], rtol=0, atol=0
        )


def test_gather_block_tables_padding() -> None:
    """Gather block tables: all padded rows should be zeroed.

    When num_reqs == 0, every row is padding.
    """
    init_device_properties_triton()

    num_groups = 2
    batch_size = 3
    max_num_reqs = 5
    max_num_blocks = 4
    BLOCK_SIZE = 4

    device = torch.device("npu")

    batch_idx_to_req_idx = torch.zeros(batch_size, dtype=torch.int32)
    src_block_tables = torch.randint(1, 100, (max_num_reqs, max_num_blocks), dtype=torch.int32, device=device)
    dst_block_tables = torch.full((batch_size, max_num_blocks), -1, dtype=torch.int32, device=device)
    num_blocks = torch.full((num_groups, max_num_reqs), 2, dtype=torch.int32, device=device)

    src_ptrs = torch.tensor([src_block_tables.data_ptr()] * num_groups, dtype=torch.uint64, device=device)
    dst_ptrs = torch.tensor([dst_block_tables.data_ptr()] * num_groups, dtype=torch.uint64, device=device)
    strides = torch.tensor([src_block_tables.stride(0)] * num_groups, dtype=torch.int64, device=device)

    grid = (num_groups, batch_size)
    _gather_block_tables_kernel[grid](
        batch_idx_to_req_idx.to(device),
        src_ptrs,
        dst_ptrs,
        strides,
        num_blocks,
        num_blocks.stride(0),
        0,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    assert (dst_block_tables.cpu() == 0).all(), "All rows should be zeroed when num_reqs == 0"


@pytest.mark.parametrize("max_num_blocks", [2, 16, 64])
def test_gather_block_tables_block_size_variants(max_num_blocks: int) -> None:
    """Gather block tables with varying block counts.

    Tests different max_num_blocks values to verify BLOCK_SIZE handling.
    """
    init_device_properties_triton()
    torch.manual_seed(99)

    num_groups = 1
    batch_size = 2
    max_num_reqs = 4
    BLOCK_SIZE = 4

    device = torch.device("npu")

    batch_idx_to_req_idx = torch.tensor([1, 0], dtype=torch.int32, device=device)
    src = torch.randint(0, 1000, (max_num_reqs, max_num_blocks), dtype=torch.int32, device=device)
    dst = torch.zeros(batch_size, max_num_blocks, dtype=torch.int32, device=device)
    num_blocks = torch.tensor([[max_num_blocks, max_num_blocks // 2, max_num_blocks, max_num_blocks // 3]],
                              dtype=torch.int32, device=device)

    src_ptr = torch.tensor([src.data_ptr()], dtype=torch.uint64, device=device)
    dst_ptr = torch.tensor([dst.data_ptr()], dtype=torch.uint64, device=device)
    strides = torch.tensor([src.stride(0)], dtype=torch.int64, device=device)

    grid = (num_groups, batch_size)
    _gather_block_tables_kernel[grid](
        batch_idx_to_req_idx,
        src_ptr,
        dst_ptr,
        strides,
        num_blocks,
        num_blocks.stride(0),
        batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    dst_cpu = dst.cpu()
    src_cpu = src.cpu()

    # req_idx=1 for batch 0
    n0 = int(num_blocks[0, 1])
    assert (dst_cpu[0, :n0] == src_cpu[1, :n0]).all()
    assert (dst_cpu[0, n0:] == 0).all()

    # req_idx=0 for batch 1
    n1 = int(num_blocks[0, 0])
    assert (dst_cpu[1, :n1] == src_cpu[0, :n1]).all()
    assert (dst_cpu[1, n1:] == 0).all()
