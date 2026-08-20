# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.utils import _zero_kv_blocks_kernel

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _zero_kv_blocks_cpu(
    scratch: torch.Tensor,
    block_ids: torch.Tensor,
    n_blocks: int,
    n_segs: int,
    page_size_el: int,
    block_size: int,
) -> torch.Tensor:
    """Pure PyTorch CPU reference for zero_kv_blocks.

    The kernel flattens a 3D program space (block_index, seg_index, chunk_index).
    For each work item:
      - block_id from block_ids[block_index]
      - seg_addr from seg_addrs[seg_index]
      - zero BLOCK_SIZE int32 elements at offset
        block_id * PAGE_SIZE_EL + chunk_index * BLOCK_SIZE

    Returns modified scratch (all segments concatenated conceptually).
    """
    result = scratch.clone()
    chunks = page_size_el // block_size

    # Simulate the 3D grid mapping
    work_per_block = n_segs * chunks
    total_work = n_blocks * work_per_block

    for pid in range(total_work):
        block_index = pid // work_per_block
        if block_index >= n_blocks:
            break
        remainder = pid % work_per_block
        seg_index = remainder // chunks
        chunk_index = remainder % chunks

        block_id = int(block_ids[block_index])
        seg_offset = block_id * page_size_el + chunk_index * block_size
        for i in range(block_size):
            idx = seg_offset + i
            if idx < len(result):
                result[idx] = 0

    return result


@pytest.mark.parametrize("n_segs", [1, 2])
def test_zero_kv_blocks_basic(n_segs: int) -> None:
    """Zero KV blocks kernel: basic functionality.

    Verifies that the kernel zeroes specific blocks across one or more
    segments.
    """
    init_device_properties_triton()
    torch.manual_seed(42)

    page_size_el = 64
    block_size = 16
    max_n_blocks = 4
    n_blocks = 3

    device = torch.device("npu")

    scratch = torch.randint(
        1, 100, (max_n_blocks * page_size_el,), dtype=torch.int32, device=device
    )
    block_ids = torch.tensor([0, 2, 3], dtype=torch.int64, device=device)

    seg_addrs = torch.tensor(
        [scratch.data_ptr()] * n_segs, dtype=torch.uint64, device=device
    )

    grid = (n_blocks * n_segs * (page_size_el // block_size),)
    _zero_kv_blocks_kernel[grid](
        seg_addrs,
        block_ids,
        n_blocks,
        N_SEGS=n_segs,
        PAGE_SIZE_EL=page_size_el,
        BLOCK_SIZE=block_size,
    )
    torch.npu.synchronize()

    # CPU reference: zero blocks 0, 2, 3 in scratch.
    expected = _zero_kv_blocks_cpu(
        scratch.cpu(), block_ids.cpu(), n_blocks, n_segs,
        page_size_el, block_size,
    )

    torch.testing.assert_close(scratch.cpu(), expected, rtol=0, atol=0)


def test_zero_kv_blocks_single_segment() -> None:
    """Zero KV blocks: single segment, verify specific blocks zeroed.

    Only blocks listed in block_ids should be zeroed (not ALL blocks).
    """
    init_device_properties_triton()

    page_size_el = 32
    block_size = 8
    n_segs = 1
    n_blocks = 2

    device = torch.device("npu")

    scratch = torch.full(
        (4 * page_size_el,), 42, dtype=torch.int32, device=device
    )
    block_ids = torch.tensor([1, 3], dtype=torch.int64, device=device)
    seg_addrs = torch.tensor(
        [scratch.data_ptr()], dtype=torch.uint64, device=device
    )

    grid = (n_blocks * n_segs * (page_size_el // block_size),)
    _zero_kv_blocks_kernel[grid](
        seg_addrs,
        block_ids,
        n_blocks,
        N_SEGS=n_segs,
        PAGE_SIZE_EL=page_size_el,
        BLOCK_SIZE=block_size,
    )
    torch.npu.synchronize()

    scratch_cpu = scratch.cpu()

    # Block 1: elements [32, 64) should be zero.
    assert (scratch_cpu[32:64] == 0).all(), "Block 1 should be zeroed"
    # Block 3: elements [96, 128) should be zero.
    assert (scratch_cpu[96:128] == 0).all(), "Block 3 should be zeroed"
    # Block 0 and 2 should remain 42.
    assert (scratch_cpu[0:32] == 42).all(), "Block 0 should be unchanged"
    assert (scratch_cpu[64:96] == 42).all(), "Block 2 should be unchanged"


def test_zero_kv_blocks_empty_block_ids() -> None:
    """Zero KV blocks: no block IDs (n_blocks=0) is a no-op."""
    init_device_properties_triton()

    page_size_el = 32
    block_size = 8
    n_segs = 1
    n_blocks = 0

    device = torch.device("npu")

    scratch = torch.full((32,), 42, dtype=torch.int32, device=device)
    block_ids = torch.zeros(0, dtype=torch.int64, device=device)
    seg_addrs = torch.tensor(
        [scratch.data_ptr()], dtype=torch.uint64, device=device
    )

    grid = (n_blocks * n_segs * (page_size_el // block_size),)
    _zero_kv_blocks_kernel[grid](
        seg_addrs,
        block_ids,
        n_blocks,
        N_SEGS=n_segs,
        PAGE_SIZE_EL=page_size_el,
        BLOCK_SIZE=block_size,
    )
    torch.npu.synchronize()

    assert (scratch.cpu() == 42).all(), "No blocks should be zeroed"
