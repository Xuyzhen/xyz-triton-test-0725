# vLLM vanilla kernel: _zero_kv_blocks_kernel from
# vllm/vllm/v1/worker/utils.py

"""
Precision test for _zero_kv_blocks_kernel (vanilla vLLM version).

Zeros KV cache blocks at specified block IDs across all segments in a single
launch. Programs are mapped as (block_index, seg_index, chunk_index).

Kernel signature:
    _zero_kv_blocks_kernel(
        seg_addrs_ptr,      # [N_SEGS] int64 absolute byte addresses
        block_ids_ptr,      # [n_blocks] int64 block IDs to zero
        n_blocks,
        N_SEGS: tl.constexpr,
        PAGE_SIZE_EL: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    )
"""

import torch
import pytest

from vllm.triton_utils import tl, triton
from vllm.v1.worker.utils import _zero_kv_blocks_kernel
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


class TestZeroKvBlocksKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    def test_zero_single_block_single_seg(self):
        """Zero one block in one segment (e.g., one KV cache buffer)."""
        page_size_el = 8192
        blk_size = 1024
        n_segs = 1

        kv_page = torch.randint(1, 100, (page_size_el,), dtype=torch.int32, device=self.device)
        kv_before = kv_page.clone()

        seg_addrs = torch.tensor([kv_page.data_ptr()], dtype=torch.int64, device=self.device)
        block_ids = torch.tensor([0], dtype=torch.int64, device=self.device)
        n_blocks = 1

        chunks = page_size_el // blk_size
        grid = (n_blocks * n_segs * chunks,)

        _zero_kv_blocks_kernel[(grid,)](
            seg_addrs,
            block_ids,
            n_blocks,
            N_SEGS=n_segs,
            PAGE_SIZE_EL=page_size_el,
            BLOCK_SIZE=blk_size,
        )
        torch.npu.synchronize()

        assert torch.all(kv_page == 0).item(), "KV block should be zeroed"
        assert not torch.all(kv_before == 0).item(), "Original data should have non-zero values"

    def test_zero_multiple_blocks(self):
        """Zero multiple blocks in a single segment."""
        page_size_el = 16384
        blk_size = 1024
        n_segs = 1

        kv_buffer = torch.randint(1, 100, (page_size_el,), dtype=torch.int32, device=self.device)
        kv_copy = kv_buffer.clone()

        seg_addrs = torch.tensor([kv_buffer.data_ptr()], dtype=torch.int64, device=self.device)
        block_ids = torch.tensor([0, 1], dtype=torch.int64, device=self.device)
        n_blocks = 2

        chunks = page_size_el // blk_size
        grid = (n_blocks * n_segs * chunks,)

        _zero_kv_blocks_kernel[(grid,)](
            seg_addrs,
            block_ids,
            n_blocks,
            N_SEGS=n_segs,
            PAGE_SIZE_EL=page_size_el,
            BLOCK_SIZE=blk_size,
        )
        torch.npu.synchronize()

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
        grid = (n_blocks * n_segs * chunks,)

        _zero_kv_blocks_kernel[(grid,)](
            seg_addrs,
            block_ids,
            n_blocks,
            N_SEGS=n_segs,
            PAGE_SIZE_EL=page_size_el,
            BLOCK_SIZE=blk_size,
        )
        torch.npu.synchronize()

        assert torch.all(seg0 == 0).item(), "Segment 0 should be zeroed"
        assert torch.all(seg1 == 0).item(), "Segment 1 should be zeroed"
        assert not torch.all(seg0_before == 0).item(), "Original seg0 should have non-zero values"
        assert not torch.all(seg1_before == 0).item(), "Original seg1 should have non-zero values"

    def test_no_blocks(self):
        """When no blocks to zero, nothing should change."""
        page_size_el = 1024
        blk_size = 256
        n_segs = 1
        n_blocks = 0

        kv_buffer = torch.randint(1, 100, (page_size_el,), dtype=torch.int32, device=self.device)
        kv_before = kv_buffer.clone()

        seg_addrs = torch.tensor([kv_buffer.data_ptr()], dtype=torch.int64, device=self.device)
        block_ids = torch.tensor([], dtype=torch.int64, device=self.device)

        chunks = page_size_el // blk_size
        grid = (n_blocks * n_segs * chunks,)

        _zero_kv_blocks_kernel[(grid,)](
            seg_addrs,
            block_ids,
            n_blocks,
            N_SEGS=n_segs,
            PAGE_SIZE_EL=page_size_el,
            BLOCK_SIZE=blk_size,
        )
        torch.npu.synchronize()

        torch.testing.assert_close(kv_buffer.cpu(), kv_before.cpu(), rtol=0, atol=0)

    def test_block_id_offset(self):
        """Verify that block ID 1 zeros offset PAGE_SIZE_EL, not block 0."""
        page_size_el = 2048
        blk_size = 256
        n_segs = 1

        kv_buffer = torch.randint(1, 100, (page_size_el * 2,), dtype=torch.int32, device=self.device)
        first_half_before = kv_buffer[:page_size_el].clone()

        seg_addrs = torch.tensor([kv_buffer.data_ptr()], dtype=torch.int64, device=self.device)
        block_ids = torch.tensor([1], dtype=torch.int64, device=self.device)
        n_blocks = 1

        chunks = page_size_el // blk_size
        grid = (n_blocks * n_segs * chunks,)

        _zero_kv_blocks_kernel[(grid,)](
            seg_addrs,
            block_ids,
            n_blocks,
            N_SEGS=n_segs,
            PAGE_SIZE_EL=page_size_el,
            BLOCK_SIZE=blk_size,
        )
        torch.npu.synchronize()

        # Second block should be zeroed
        assert torch.all(kv_buffer[page_size_el:] == 0).item(), "Block 1 should be zeroed"
        # First block should be unchanged
        torch.testing.assert_close(kv_buffer[:page_size_el].cpu(), first_half_before.cpu(), rtol=0, atol=0)
