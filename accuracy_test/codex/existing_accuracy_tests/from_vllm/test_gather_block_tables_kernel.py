# Standalone Ascend A3 adaptation of an upstream vLLM accuracy path.
# Accuracy UT source: vllm/tests/v1/worker/test_gpu_block_table.py
# Kernel source: vllm/vllm/v1/worker/gpu/block_table.py
# Coverage: _gather_block_tables_kernel

# vLLM vanilla kernel: _gather_block_tables_kernel from
# vllm/vllm/v1/worker/gpu/block_table.py

"""
Precision test for _gather_block_tables_kernel.

Gathers block tables from source (request-indexed) to destination
(batch-indexed) layout, with zero-padding for out-of-range rows.

Kernel signature:
    _gather_block_tables_kernel(
        batch_idx_to_req_idx,  # [batch_size] int32
        src_block_table_ptrs,  # [num_kv_cache_groups] ptr-to-ptrs (uint64)
        dst_block_table_ptrs,  # [num_kv_cache_groups]
        block_table_strides,   # [num_kv_cache_groups] int64
        num_blocks_ptr,        # [num_kv_cache_groups, max_num_reqs] int32
        num_blocks_stride,
        num_reqs,              # actual request count
        BLOCK_SIZE: tl.constexpr,
    )

Uses _load_ptr for indirection.
"""

import torch
import pytest

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.block_table import _gather_block_tables_kernel
from vllm.v1.worker.gpu.buffer_utils import _load_ptr
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _gather_block_tables_ref(
    batch_idx_to_req_idx,  # [batch_size]
    src_block_tables,      # [num_groups, max_num_reqs, max_num_blocks]
    block_table_strides,   # [num_groups]
    num_blocks,            # [num_groups, max_num_reqs]
    num_reqs,
):
    """CPU reference for gather block tables."""
    num_groups = src_block_tables.shape[0]
    batch_size = batch_idx_to_req_idx.shape[0]
    max_num_blocks = src_block_tables.shape[-1]

    out = torch.zeros(num_groups, batch_size, max_num_blocks, dtype=torch.int32)
    for g in range(num_groups):
        for b in range(batch_size):
            if b >= num_reqs:
                continue  # stays zero
            req_idx = int(batch_idx_to_req_idx[b].item())
            n_blocks = int(num_blocks[g, req_idx].item())
            out[g, b, :n_blocks] = src_block_tables[g, req_idx, :n_blocks]
    return out


class TestGatherBlockTablesKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_groups", [1, 2, 4])
    @pytest.mark.parametrize("max_num_reqs", [4, 8])
    @pytest.mark.parametrize("max_num_blocks", [64, 128])
    def test_gather_basic(self, num_groups, max_num_reqs, max_num_blocks):
        """Test basic gathering of block tables."""
        num_reqs = max_num_reqs  # no padding

        batch_idx_to_req_idx = torch.arange(
            num_reqs, dtype=torch.int32, device=self.device
        )

        # Use direct tensors for src/dst (no ptr indirection for simplicity).
        # The kernel uses _load_ptr to dereference src/dst pointers.
        # We create separate tensors for each group.
        src_block_tables_cpu = torch.randint(
            0, 1000, (num_groups, max_num_reqs, max_num_blocks), dtype=torch.int32
        )
        dst_block_tables_cpu = torch.zeros(
            num_groups, max_num_reqs, max_num_blocks, dtype=torch.int32
        )

        src_block_tables = src_block_tables_cpu.to(self.device)
        dst_block_tables = dst_block_tables_cpu.to(self.device)

        block_table_strides = torch.full(
            (num_groups,), max_num_blocks, dtype=torch.int64, device=self.device
        )

        num_blocks = torch.full(
            (num_groups, max_num_reqs), max_num_blocks, dtype=torch.int32, device=self.device
        )

        # Build ptr tensors. The kernel expects ptr-to-pointer indirection
        # via _load_ptr. We prepare separate ptr tensors.
        src_ptrs = torch.tensor(
            [t.data_ptr() for t in src_block_tables],
            dtype=torch.uint64, device=self.device
        )
        dst_ptrs = torch.tensor(
            [t.data_ptr() for t in dst_block_tables],
            dtype=torch.uint64, device=self.device
        )

        BLOCK_SIZE = 16

        _gather_block_tables_kernel[(num_groups, num_reqs)](
            batch_idx_to_req_idx,
            src_ptrs,
            dst_ptrs,
            block_table_strides,
            num_blocks,
            num_blocks.stride(0),
            num_reqs,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.npu.synchronize()

        ref = _gather_block_tables_ref(
            batch_idx_to_req_idx.cpu(),
            src_block_tables_cpu,
            block_table_strides.cpu(),
            num_blocks.cpu(),
            num_reqs,
        )
        torch.testing.assert_close(
            dst_block_tables.cpu(),
            ref,
            rtol=0, atol=0,
        )

    @pytest.mark.parametrize("num_groups", [1, 2])
    def test_padding_zeros(self, num_groups):
        """Rows beyond num_reqs should be zeroed out."""
        max_num_reqs = 8
        batch_size = max_num_reqs
        num_reqs = 4  # actual requests
        max_num_blocks = 32

        batch_idx_to_req_idx = torch.arange(
            batch_size, dtype=torch.int32, device=self.device
        )

        src_block_tables_cpu = torch.randint(
            0, 1000, (num_groups, max_num_reqs, max_num_blocks), dtype=torch.int32
        )
        dst_block_tables_cpu = torch.full(
            (num_groups, batch_size, max_num_blocks), -1, dtype=torch.int32
        )

        src_block_tables = src_block_tables_cpu.to(self.device)
        dst_block_tables = dst_block_tables_cpu.to(self.device)

        block_table_strides = torch.full(
            (num_groups,), max_num_blocks, dtype=torch.int64, device=self.device
        )
        num_blocks = torch.full(
            (num_groups, max_num_reqs), max_num_blocks, dtype=torch.int32, device=self.device
        )

        src_ptrs = torch.tensor(
            [t.data_ptr() for t in src_block_tables],
            dtype=torch.uint64, device=self.device
        )
        dst_ptrs = torch.tensor(
            [t.data_ptr() for t in dst_block_tables],
            dtype=torch.uint64, device=self.device
        )

        _gather_block_tables_kernel[(num_groups, batch_size)](
            batch_idx_to_req_idx,
            src_ptrs,
            dst_ptrs,
            block_table_strides,
            num_blocks,
            num_blocks.stride(0),
            num_reqs,
            BLOCK_SIZE=16,
        )
        torch.npu.synchronize()

        # Rows 0..num_reqs-1 should be gathered from src
        ref = _gather_block_tables_ref(
            batch_idx_to_req_idx[:num_reqs].cpu(),
            src_block_tables_cpu,
            block_table_strides.cpu(),
            num_blocks.cpu(),
            num_reqs,
        )
        # Pad rows: check they are zero
        padded = dst_block_tables.cpu()
        torch.testing.assert_close(
            padded[:, :num_reqs, :],
            ref,
            rtol=0, atol=0,
        )
        # Padded rows should be zeros
        for g in range(num_groups):
            for b in range(num_reqs, batch_size):
                assert torch.all(padded[g, b] == 0), f"Row {b} in group {g} should be zero"
