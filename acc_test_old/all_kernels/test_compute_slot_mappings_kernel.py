# vLLM vanilla kernel: _compute_slot_mappings_kernel from
# vllm/vllm/v1/worker/gpu/block_table.py

"""
Precision test for _compute_slot_mappings_kernel.

Computes slot mappings (block_number * block_size + block_offset) per token.
Supports context parallelism (CP_SIZE > 1).

Kernel signature:
    _compute_slot_mappings_kernel(
        max_num_tokens,
        idx_mapping,          # [num_reqs] int32
        query_start_loc,      # [num_reqs + 1] int32
        pos,                  # [num_tokens] int64
        block_table_ptrs,     # [num_kv_cache_groups] uint64 (ptr-to-ptr)
        block_table_strides,  # [num_kv_cache_groups] int64
        block_sizes,          # [num_kv_cache_groups] int32
        slot_mappings_ptr,    # [num_kv_cache_groups, max_num_tokens] int64
        slot_mappings_stride,
        cp_rank,
        CP_SIZE: tl.constexpr,
        CP_INTERLEAVE: tl.constexpr,
        PAD_ID: tl.constexpr,
        TRITON_BLOCK_SIZE: tl.constexpr,
    )
"""

import torch
import pytest

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.block_table import _compute_slot_mappings_kernel
from vllm.v1.worker.gpu.buffer_utils import _load_ptr
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _compute_slot_mappings_ref(
    idx_mapping,         # [num_reqs]
    query_start_loc,     # [num_reqs + 1]
    pos,                 # [num_tokens]
    block_tables,        # list of [max_num_reqs, max_num_blocks] per group
    block_sizes,
    max_num_tokens,
    cp_rank=0,
    cp_size=1,
    cp_interleave=1,
    pad_id=-1,
):
    """CPU reference for slot mappings computation."""
    num_groups = len(block_tables)
    num_reqs = idx_mapping.shape[0]
    out = torch.full((num_groups, max_num_tokens), pad_id, dtype=torch.int64)

    for g in range(num_groups):
        bt = block_tables[g]
        block_size = int(block_sizes[g].item()) if isinstance(block_sizes[g], torch.Tensor) else block_sizes[g]
        for b in range(num_reqs):
            req_state_idx = int(idx_mapping[b].item())
            start = int(query_start_loc[b].item())
            end = int(query_start_loc[b + 1].item())
            for i in range(start, end):
                p = int(pos[i].item())
                block_idx = p // (block_size * cp_size)
                block_offset = p % (block_size * cp_size)
                block_num = int(bt[req_state_idx, block_idx].item())
                if cp_size == 1:
                    slot_id = block_num * block_size + block_offset
                else:
                    is_local = (block_offset // cp_interleave) % cp_size == cp_rank
                    rounds = block_offset // (cp_interleave * cp_size)
                    remainder = block_offset % cp_interleave
                    local_offset = rounds * cp_interleave + remainder
                    slot_id = block_num * block_size + local_offset
                    if not is_local:
                        slot_id = pad_id
                out[g, i] = slot_id

        # Pad remaining slots for the last batch (CUDA graph behavior)
        last_batch = num_reqs
        actual_num_tokens = int(query_start_loc[last_batch].item())
        for i in range(actual_num_tokens, max_num_tokens):
            out[g, i] = pad_id

    return out


class TestComputeSlotMappingsKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_reqs", [1, 2, 4])
    @pytest.mark.parametrize("num_tokens", [8, 16])
    @pytest.mark.parametrize("cp_size", [1, 2])
    def test_slot_mappings_basic(self, num_reqs, num_tokens, cp_size):
        """Test slot mappings computation without context parallelism."""
        num_groups = 1
        max_num_reqs = num_reqs
        max_num_blocks = 8
        block_size = 16
        max_num_tokens = num_tokens + 4  # some padding
        cp_rank = 0
        cp_interleave = 1

        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=self.device)
        tokens_per_req = num_tokens // num_reqs
        for i in range(num_reqs):
            query_start_loc[i + 1] = query_start_loc[i] + tokens_per_req
        query_start_loc[-1] = num_tokens  # ensure exact

        pos = torch.arange(num_tokens, dtype=torch.int64, device=self.device)

        block_table_cpu = torch.randint(0, 100, (max_num_reqs, max_num_blocks), dtype=torch.int32)
        block_table = block_table_cpu.to(self.device)

        block_table_ptrs = torch.tensor(
            [block_table.data_ptr()], dtype=torch.uint64, device=self.device
        )
        block_table_strides = torch.full(
            (num_groups,), max_num_blocks, dtype=torch.int64, device=self.device
        )
        block_sizes = torch.tensor([block_size], dtype=torch.int32, device=self.device)

        slot_mappings = torch.full(
            (num_groups, max_num_tokens), -1, dtype=torch.int64, device=self.device
        )

        _compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](
            max_num_tokens,
            idx_mapping,
            query_start_loc,
            pos,
            block_table_ptrs,
            block_table_strides,
            block_sizes,
            slot_mappings,
            slot_mappings.stride(0),
            cp_rank,
            CP_SIZE=cp_size,
            CP_INTERLEAVE=cp_interleave,
            PAD_ID=-1,
            TRITON_BLOCK_SIZE=4,
        )
        torch.npu.synchronize()

        ref = _compute_slot_mappings_ref(
            idx_mapping.cpu(),
            query_start_loc.cpu(),
            pos.cpu(),
            [block_table_cpu],
            block_sizes.cpu(),
            max_num_tokens,
            cp_rank=cp_rank,
            cp_size=cp_size,
            cp_interleave=cp_interleave,
            pad_id=-1,
        )
        torch.testing.assert_close(slot_mappings.cpu(), ref, rtol=0, atol=0)

    @pytest.mark.parametrize("num_reqs", [2, 4])
    def test_with_context_parallelism(self, num_reqs):
        """Test with context parallelism (cp_size > 1)."""
        num_groups = 1
        max_num_reqs = num_reqs
        max_num_blocks = 8
        block_size = 8
        num_tokens = 12
        max_num_tokens = num_tokens + 2
        cp_size = 2
        cp_rank = 0
        cp_interleave = 2

        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=self.device)
        for i in range(num_reqs):
            query_start_loc[i + 1] = query_start_loc[i] + (num_tokens // num_reqs)
        query_start_loc[-1] = num_tokens

        pos = torch.arange(num_tokens, dtype=torch.int64, device=self.device)

        block_table_cpu = torch.randint(0, 100, (max_num_reqs, max_num_blocks), dtype=torch.int32)
        block_table = block_table_cpu.to(self.device)

        block_table_ptrs = torch.tensor(
            [block_table.data_ptr()], dtype=torch.uint64, device=self.device
        )
        block_table_strides = torch.full(
            (num_groups,), max_num_blocks, dtype=torch.int64, device=self.device
        )
        block_sizes = torch.tensor([block_size], dtype=torch.int32, device=self.device)

        slot_mappings = torch.full(
            (num_groups, max_num_tokens), -1, dtype=torch.int64, device=self.device
        )

        _compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](
            max_num_tokens,
            idx_mapping,
            query_start_loc,
            pos,
            block_table_ptrs,
            block_table_strides,
            block_sizes,
            slot_mappings,
            slot_mappings.stride(0),
            cp_rank,
            CP_SIZE=cp_size,
            CP_INTERLEAVE=cp_interleave,
            PAD_ID=-1,
            TRITON_BLOCK_SIZE=4,
        )
        torch.npu.synchronize()

        ref = _compute_slot_mappings_ref(
            idx_mapping.cpu(),
            query_start_loc.cpu(),
            pos.cpu(),
            [block_table_cpu],
            block_sizes.cpu(),
            max_num_tokens,
            cp_rank=cp_rank,
            cp_size=cp_size,
            cp_interleave=cp_interleave,
            pad_id=-1,
        )
        torch.testing.assert_close(slot_mappings.cpu(), ref, rtol=0, atol=0)

    def test_multiple_kv_cache_groups(self):
        """Test with multiple KV cache groups."""
        num_reqs = 2
        num_groups = 3
        max_num_reqs = num_reqs
        max_num_blocks = 8
        block_size = 16
        num_tokens = 6
        max_num_tokens = num_tokens + 2
        cp_size = 1

        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=self.device)
        for i in range(num_reqs):
            query_start_loc[i + 1] = query_start_loc[i] + (num_tokens // num_reqs)
        query_start_loc[-1] = num_tokens

        pos = torch.arange(num_tokens, dtype=torch.int64, device=self.device)

        block_tables_cpu = [
            torch.randint(0, 100, (max_num_reqs, max_num_blocks), dtype=torch.int32)
            for _ in range(num_groups)
        ]
        block_tables = [bt.to(self.device) for bt in block_tables_cpu]

        block_table_ptrs = torch.tensor(
            [bt.data_ptr() for bt in block_tables],
            dtype=torch.uint64, device=self.device
        )
        block_table_strides = torch.full(
            (num_groups,), max_num_blocks, dtype=torch.int64, device=self.device
        )
        block_sizes = torch.full((num_groups,), block_size, dtype=torch.int32, device=self.device)

        slot_mappings = torch.full(
            (num_groups, max_num_tokens), -1, dtype=torch.int64, device=self.device
        )

        _compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](
            max_num_tokens,
            idx_mapping,
            query_start_loc,
            pos,
            block_table_ptrs,
            block_table_strides,
            block_sizes,
            slot_mappings,
            slot_mappings.stride(0),
            0,
            CP_SIZE=1,
            CP_INTERLEAVE=1,
            PAD_ID=-1,
            TRITON_BLOCK_SIZE=4,
        )
        torch.npu.synchronize()

        ref = _compute_slot_mappings_ref(
            idx_mapping.cpu(),
            query_start_loc.cpu(),
            pos.cpu(),
            block_tables_cpu,
            block_sizes.cpu(),
            max_num_tokens,
            cp_rank=0, cp_size=1, cp_interleave=1, pad_id=-1,
        )
        torch.testing.assert_close(slot_mappings.cpu(), ref, rtol=0, atol=0)
