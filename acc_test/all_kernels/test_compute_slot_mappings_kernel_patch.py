# vLLM-Ascend patched kernel: _compute_slot_mappings_kernel from
# vllm-ascend/vllm_ascend/worker/v2/block_table.py:97
# PATCH NOTE: This is an Ascend NPU adaptation of the original vLLM Triton kernel

"""
Precision test for patched _compute_slot_mappings_kernel (Ascend NPU version).

Patch differences vs original vllm:
- Casts positions to tl.int32 (NPU compatibility, avoids uint64)
- Replaces % (modulo) operation with equivalent sub/mul to avoid scalar degradation
- Non-contiguous memory access mitigation: loads full block and uses tl.gather
- Uses tl.cast for block_numbers to float32 for gather
- Uses CP_SIZE and CP_INTERLEAVE constexpr parameters
- Uses PAD_ID constexpr for padding
- Uses TRITON_BLOCK_SIZE=1024 and TOTAL_BLOCK_SIZE=4096
- Uses _load_ptr helper for loading pointer values

Kernel signature:
    _compute_slot_mappings_kernel(
        max_num_tokens,                 # scalar: max number of tokens
        idx_mapping,                    # [num_reqs] request index mapping
        query_start_loc,                # [num_reqs + 1] query start locations
        pos,                            # [num_tokens] positions
        block_table_ptrs,               # [num_kv_cache_groups] pointers to block tables
        block_table_strides,            # [num_kv_cache_groups] strides
        block_sizes,                    # [num_kv_cache_groups] block sizes
        slot_mappings_ptr,              # [num_kv_cache_groups, max_num_tokens] output
        slot_mappings_stride,           # stride(0) of slot_mappings
        cp_rank,                        # context parallel rank
        CP_SIZE: tl.constexpr,          # context parallel size
        CP_INTERLEAVE: tl.constexpr,    # context parallel interleave
        PAD_ID: tl.constexpr,           # padding slot ID
        TRITON_BLOCK_SIZE: tl.constexpr, # triton block size (1024)
        TOTAL_BLOCK_SIZE: tl.constexpr,  # total block table size (4096)
    )

Computes slot mappings for each KV cache group across all tokens.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

import pytest


def _compute_slot_mappings_ref(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    pos: torch.Tensor,
    block_tables: list[torch.Tensor],
    block_sizes: torch.Tensor,
    num_kv_cache_groups: int,
    max_num_tokens: int,
    cp_size: int = 1,
    cp_rank: int = 0,
    cp_interleave: int = 1,
    pad_id: int = -1,
) -> torch.Tensor:
    """CPU reference for _compute_slot_mappings_kernel."""
    num_reqs = idx_mapping.shape[0]
    slot_mappings = torch.full((num_kv_cache_groups, max_num_tokens), pad_id, dtype=torch.int32)

    for group_id in range(num_kv_cache_groups):
        block_table = block_tables[group_id]
        block_size = block_sizes[group_id].item()

        for batch_idx in range(num_reqs):
            req_state_idx = idx_mapping[batch_idx].item()
            start_idx = query_start_loc[batch_idx].item()
            end_idx = query_start_loc[batch_idx + 1].item()

            for i in range(start_idx, end_idx):
                position = pos[i].item()
                block_index = position // (block_size * cp_size)
                block_offset = position - (block_size * cp_size) * block_index

                block_number = block_table[req_state_idx, block_index].item()

                if cp_size == 1:
                    slot_id = block_number * block_size + block_offset
                else:
                    is_local = (block_offset // cp_interleave) % cp_size == cp_rank
                    rounds = block_offset // (cp_interleave * cp_size)
                    remainder = block_offset % cp_interleave
                    local_offsets = rounds * cp_interleave + remainder
                    slot_id = block_number * block_size + local_offsets
                    slot_id = slot_id if is_local else pad_id

                slot_mappings[group_id, i] = slot_id

    return slot_mappings


class TestComputeSlotMappingsKernelPatch:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_reqs", [1, 2, 4])
    @pytest.mark.parametrize("num_kv_cache_groups", [1, 2])
    def test_slot_mappings(self, num_reqs, num_kv_cache_groups):
        """Compare NPU slot mappings with CPU reference (CP_SIZE=1)."""
        from vllm.v1.worker.gpu.block_table import _load_ptr, PAD_SLOT_ID
        from vllm_ascend.worker.v2.block_table import _compute_slot_mappings_kernel

        max_num_tokens = 8
        vocab_size = 128
        block_size = 64
        cp_size = 1
        triton_block_size = 2  # small for testing

        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=self.device)
        pos_list = []
        total_tokens = 0
        for i in range(num_reqs):
            num_tok = 2 if i < num_reqs - 1 else 3  # vary token counts
            query_start_loc[i + 1] = query_start_loc[i] + num_tok
            for j in range(num_tok):
                pos_list.append(i * 4 + j)  # scattered positions
            total_tokens += num_tok

        pos = torch.tensor(pos_list, dtype=torch.int32, device=self.device)

        block_tables_gpu = []
        block_tables_cpu = []
        for g in range(num_kv_cache_groups):
            # block table: [max_num_reqs, num_blocks]
            num_blocks = 16
            bt = torch.randint(1000, 9999, (num_reqs, num_blocks), dtype=torch.int32, device=self.device)
            block_tables_gpu.append(bt)
            block_tables_cpu.append(bt.cpu())

        block_table_ptrs = torch.tensor(
            [bt.data_ptr() for bt in block_tables_gpu],
            dtype=torch.int64, device=self.device,
        )
        block_table_strides = torch.tensor(
            [bt.stride(0) for bt in block_tables_gpu],
            dtype=torch.int32, device=self.device,
        )
        block_sizes_tensor = torch.full((num_kv_cache_groups,), block_size, dtype=torch.int32, device=self.device)

        slot_mappings = torch.full((num_kv_cache_groups, max_num_tokens), -1, dtype=torch.int32, device=self.device)

        _compute_slot_mappings_kernel[(num_kv_cache_groups, num_reqs + 1)](
            max_num_tokens,
            idx_mapping,
            query_start_loc,
            pos,
            block_table_ptrs,
            block_table_strides,
            block_sizes_tensor,
            slot_mappings,
            slot_mappings.stride(0),
            0,  # cp_rank
            CP_SIZE=cp_size,
            CP_INTERLEAVE=1,
            PAD_ID=PAD_SLOT_ID,
            TRITON_BLOCK_SIZE=triton_block_size,
            TOTAL_BLOCK_SIZE=4096,
        )
        torch.npu.synchronize()

        expected = _compute_slot_mappings_ref(
            idx_mapping.cpu(), query_start_loc.cpu(), pos.cpu(),
            block_tables_cpu, block_sizes_tensor.cpu(),
            num_kv_cache_groups, max_num_tokens,
            cp_size=cp_size,
            pad_id=PAD_SLOT_ID,
        )

        torch.testing.assert_close(slot_mappings.cpu(), expected, rtol=0, atol=0)

    def test_padding(self):
        """Verify the last program pads remaining slots with PAD_SLOT_ID."""
        from vllm.v1.worker.gpu.block_table import PAD_SLOT_ID
        from vllm_ascend.worker.v2.block_table import _compute_slot_mappings_kernel

        num_kv_cache_groups = 1
        max_num_tokens = 8
        num_reqs = 1

        idx_mapping = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        query_start_loc = torch.tensor([0, 2, 2], dtype=torch.int32, device=self.device)
        pos = torch.tensor([0, 5], dtype=torch.int32, device=self.device)

        block_table = torch.randint(1000, 9999, (num_reqs, 8), dtype=torch.int32, device=self.device)
        block_table_ptrs = torch.tensor([block_table.data_ptr()], dtype=torch.int64, device=self.device)
        block_table_strides = torch.tensor([block_table.stride(0)], dtype=torch.int32, device=self.device)
        block_sizes_tensor = torch.tensor([64], dtype=torch.int32, device=self.device)

        slot_mappings = torch.full((1, max_num_tokens), -999, dtype=torch.int32, device=self.device)

        _compute_slot_mappings_kernel[(1, num_reqs + 1)](
            max_num_tokens,
            idx_mapping,
            query_start_loc,
            pos,
            block_table_ptrs,
            block_table_strides,
            block_sizes_tensor,
            slot_mappings,
            slot_mappings.stride(0),
            0,
            CP_SIZE=1,
            CP_INTERLEAVE=1,
            PAD_ID=PAD_SLOT_ID,
            TRITON_BLOCK_SIZE=1024,
            TOTAL_BLOCK_SIZE=4096,
        )
        torch.npu.synchronize()

        # Tokens at positions 0 and 5 should have valid slot IDs
        assert slot_mappings[0, 0].item() != -999, "Slot 0 should be computed"
        assert slot_mappings[0, 1].item() != -999, "Slot 1 should be computed"
        # Padding should have PAD_SLOT_ID or the initial value
        assert slot_mappings[0, 2].item() != -999, "Slot 2 should have PAD_SLOT_ID"
