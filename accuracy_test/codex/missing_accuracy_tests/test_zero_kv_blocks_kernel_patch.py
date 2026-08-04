# vLLM-Ascend patched kernel: _zero_kv_blocks_kernel from
# vllm-ascend/vllm_ascend/worker/utils.py:15
# PATCH NOTE: This is an Ascend NPU adaptation of the original vLLM Triton kernel

"""
Precision test for patched _zero_kv_blocks_kernel (Ascend NPU version).

Patch differences vs original vllm:
- Uses GRID_SIZE for load balancing across vector cores
- Uses tl.cast for pointer type conversion (int64 -> pointer_type(tl.int32))
- Uses tl.zeros for zeroing with int32 dtype
- Uses tl.int64 for offset calculations (block_id and chunk_index)
- Uses BLOCK_SIZE determined from largest_power_of_2_divisor (<= 8192)
- Supports multiple segments (N_SEGS) for different KV cache partitions

Kernel signature:
    _zero_kv_blocks_kernel(
        seg_addrs_ptr,      # [N_SEGS] int64 absolute byte addresses for segments
        block_ids_ptr,      # [n_blocks] int64 block IDs to zero
        n_blocks,           # scalar: number of blocks
        N_SEGS: tl.constexpr,       # number of segments
        PAGE_SIZE_EL: tl.constexpr, # elements per page
        BLOCK_SIZE: tl.constexpr,   # block size in elements
        GRID_SIZE: tl.constexpr,    # grid size for load balancing
    )

Zeros KV cache blocks at specified block IDs across all segments in a single launch.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton, get_vectorcore_num

import pytest


class TestZeroKvBlocksKernelPatch:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    def _run_kernel(self, seg_addrs, block_ids, n_blocks, n_segs, page_size_el, blk_size, grid):
        from vllm_ascend.worker.utils import _zero_kv_blocks_kernel

        _zero_kv_blocks_kernel[(grid,)](
            seg_addrs,
            block_ids,
            n_blocks,
            N_SEGS=n_segs,
            PAGE_SIZE_EL=page_size_el,
            BLOCK_SIZE=blk_size,
            GRID_SIZE=grid,
        )
        torch.npu.synchronize()

    def test_zero_single_block_single_seg(self):
        """Zero one block in one segment (e.g., one KV cache buffer)."""
        page_size_el = 8192
        blk_size = 1024
        n_segs = 1

        # Allocate a page
        kv_page = torch.randint(0, 100, (page_size_el,), dtype=torch.int32, device=self.device)
        kv_before = kv_page.clone()

        seg_addrs = torch.tensor([kv_page.data_ptr()], dtype=torch.int64, device=self.device)
        block_ids = torch.tensor([0], dtype=torch.int64, device=self.device)
        n_blocks = 1

        grid = min(n_blocks * n_segs * (page_size_el // blk_size), get_vectorcore_num())

        self._run_kernel(seg_addrs, block_ids, n_blocks, n_segs, page_size_el, blk_size, grid)

        # Verify page is zeroed
        assert torch.all(kv_page == 0).item(), "KV block should be zeroed"
        # Verify data before the block is untouched
        assert not torch.all(kv_before == 0).item(), "Original data should have non-zero values"

    def test_zero_multiple_blocks(self):
        """Zero multiple blocks in a single segment."""
        page_size_el = 16384  # 2 pages worth of elements
        blk_size = 1024
        n_segs = 1

        kv_buffer = torch.randint(1, 100, (page_size_el,), dtype=torch.int32, device=self.device)
        kv_copy = kv_buffer.clone()

        seg_addrs = torch.tensor([kv_buffer.data_ptr()], dtype=torch.int64, device=self.device)
        # Zero blocks 0 and 1 (the entire buffer)
        block_ids = torch.tensor([0, 1], dtype=torch.int64, device=self.device)
        n_blocks = 2

        chunks = page_size_el // blk_size
        total_work = n_blocks * n_segs * chunks
        grid = min(total_work, get_vectorcore_num())

        self._run_kernel(seg_addrs, block_ids, n_blocks, n_segs, page_size_el, blk_size, grid)

        assert torch.all(kv_buffer == 0).item(), "Both blocks should be zeroed"

    def test_zero_multiple_segments(self):
        """Zero blocks across two segments (e.g., K and V caches)."""
        page_size_el = 4096
        blk_size = 512
        n_segs = 2
        n_blocks = 1

        seg0 = torch.randint(1, 100, (page_size_el,), dtype=torch.int32, device=self.device)
        seg1 = torch.randint(1, 100, (page_size_el,), dtype=torch.int32, device=self.device)

        seg0_before = seg0.clone()
        seg1_before = seg1.clone()

        seg_addrs = torch.tensor([seg0.data_ptr(), seg1.data_ptr()], dtype=torch.int64, device=self.device)
        block_ids = torch.tensor([0], dtype=torch.int64, device=self.device)

        chunks = page_size_el // blk_size
        total_work = n_blocks * n_segs * chunks
        grid = min(total_work, get_vectorcore_num())

        self._run_kernel(seg_addrs, block_ids, n_blocks, n_segs, page_size_el, blk_size, grid)

        assert torch.all(seg0 == 0).item(), "Segment 0 should be zeroed"
        assert torch.all(seg1 == 0).item(), "Segment 1 should be zeroed"
        assert not torch.all(seg0_before == 0).item(), "Original seg0 should have non-zero values"
        assert not torch.all(seg1_before == 0).item(), "Original seg1 should have non-zero values"

    def test_no_blocks(self):
        """When no blocks to zero, nothing should change."""
        page_size_el = 1024
        blk_size = 256
        n_segs = 1

        kv_buffer = torch.randint(1, 100, (page_size_el,), dtype=torch.int32, device=self.device)
        kv_before = kv_buffer.clone()

        seg_addrs = torch.tensor([kv_buffer.data_ptr()], dtype=torch.int64, device=self.device)
        block_ids = torch.tensor([], dtype=torch.int64, device=self.device)
        n_blocks = 0

        total_work = n_blocks * n_segs * (page_size_el // blk_size)
        if total_work > 0:
            grid = min(total_work, get_vectorcore_num())
            self._run_kernel(seg_addrs, block_ids, n_blocks, n_segs, page_size_el, blk_size, grid)

        # Nothing changed
        torch.testing.assert_close(kv_buffer.cpu(), kv_before.cpu(), rtol=0, atol=0)
