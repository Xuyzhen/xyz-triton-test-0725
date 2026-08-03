# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.gpu.block_table import _compute_slot_mappings_kernel
from vllm.v1.worker.gpu.buffer_utils import _load_ptr
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _compute_slot_mappings_cpu(
    max_num_tokens: int,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    pos: torch.Tensor,
    block_tables: list[torch.Tensor],
    block_sizes: list[int],
    cp_size: int = 1,
    cp_interleave: int = 1,
    cp_rank: int = 0,
    pad_id: int = PAD_SLOT_ID,
) -> list[torch.Tensor]:
    """Pure PyTorch CPU reference for compute_slot_mappings.

    For each group and batch, computes slot IDs for each token position.
    """
    num_groups = len(block_tables)
    num_reqs = len(idx_mapping)
    num_tokens = len(pos)

    out = []
    for g in range(num_groups):
        group_out = torch.full(
            (num_reqs, max_num_tokens), pad_id, dtype=torch.int64
        )
        bt = block_tables[g]
        block_size = block_sizes[g]

        for batch_idx in range(num_reqs):
            if batch_idx == num_reqs - 1:
                # Pad remaining slots
                actual_tokens = int(query_start_loc[batch_idx])
                group_out[batch_idx, actual_tokens:] = pad_id
                break

            req_state_idx = int(idx_mapping[batch_idx])
            start_idx = int(query_start_loc[batch_idx])
            end_idx = int(query_start_loc[batch_idx + 1])

            for i in range(start_idx, end_idx):
                position = int(pos[i])

                block_index = position // (block_size * cp_size)
                block_offset = position % (block_size * cp_size)
                block_number = int(bt[req_state_idx, block_index])

                if cp_size == 1:
                    slot_id = block_number * block_size + block_offset
                else:
                    is_local = (block_offset // cp_interleave) % cp_size == cp_rank
                    rounds = block_offset // (cp_interleave * cp_size)
                    remainder = block_offset % cp_interleave
                    local_offset = rounds * cp_interleave + remainder
                    slot_id = block_number * block_size + local_offset
                    if not is_local:
                        slot_id = pad_id

                group_out[batch_idx, i] = slot_id

        out.append(group_out)

    return out


@pytest.mark.parametrize("cp_size", [1, 2])
def test_compute_slot_mappings_basic(cp_size: int) -> None:
    """Compute slot mappings kernel: basic case without context parallelism.

    With cp_size=1, slot_id = block_number * block_size + block_offset.
    """
    init_device_properties_triton()
    torch.manual_seed(42)

    num_groups = 1
    num_reqs = 2
    max_num_tokens = 8
    block_size = 16
    cp_rank = 0
    cp_interleave = 1
    TRITON_BLOCK_SIZE = 4
    pad_id = PAD_SLOT_ID

    device = torch.device("npu")

    idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    positions = torch.tensor([0, 16, 32, 0, 8], dtype=torch.int64, device=device)

    # Block table: [num_reqs, max_num_blocks]
    num_blocks = block_size  # stride = num blocks
    block_table = torch.tensor(
        [[10, 20, 30], [40, 50, 60]], dtype=torch.int32, device=device
    )

    slot_mappings = torch.full(
        (num_groups, max_num_tokens), -2, dtype=torch.int64, device=device
    )

    num_tokens = len(positions)

    grid = (num_groups, num_reqs)
    _compute_slot_mappings_kernel[grid](
        max_num_tokens,
        idx_mapping,
        query_start_loc,
        positions,
        torch.tensor([block_table.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([block_table.stride(0)], dtype=torch.int64, device=device),
        torch.tensor([block_size], dtype=torch.int32, device=device),
        slot_mappings,
        slot_mappings.stride(0),
        cp_rank,
        CP_SIZE=cp_size,
        CP_INTERLEAVE=cp_interleave,
        PAD_ID=pad_id,
        TRITON_BLOCK_SIZE=TRITON_BLOCK_SIZE,
    )
    torch.npu.synchronize()

    expected = _compute_slot_mappings_cpu(
        max_num_tokens,
        idx_mapping.cpu(),
        query_start_loc.cpu(),
        positions.cpu(),
        [block_table.cpu()],
        [block_size],
        cp_size=cp_size,
        cp_interleave=cp_interleave,
        cp_rank=cp_rank,
        pad_id=pad_id,
    )

    torch.testing.assert_close(slot_mappings[0].cpu(), expected[0], rtol=0, atol=0)


def test_compute_slot_mappings_multi_group() -> None:
    """Compute slot mappings with multiple KV cache groups."""
    init_device_properties_triton()
    torch.manual_seed(123)

    num_groups = 3
    num_reqs = 1
    max_num_tokens = 4
    block_size = 16
    cp_size = 1
    cp_rank = 0
    cp_interleave = 1
    TRITON_BLOCK_SIZE = 4
    pad_id = PAD_SLOT_ID

    device = torch.device("npu")

    idx_mapping = torch.tensor([0], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 4], dtype=torch.int32, device=device)
    positions = torch.tensor([0, 16, 32, 48], dtype=torch.int64, device=device)

    block_tables = [
        torch.tensor([[10, 20, 30, 40]], dtype=torch.int32, device=device),
        torch.tensor([[100, 200, 300, 400]], dtype=torch.int32, device=device),
        torch.tensor([[1000, 2000, 3000, 4000]], dtype=torch.int32, device=device),
    ]

    block_table_ptrs = torch.tensor(
        [bt.data_ptr() for bt in block_tables], dtype=torch.uint64, device=device
    )
    block_table_strides = torch.tensor(
        [bt.stride(0) for bt in block_tables], dtype=torch.int64, device=device
    )
    block_sizes = torch.tensor(
        [block_size] * num_groups, dtype=torch.int32, device=device
    )

    slot_mappings = torch.full(
        (num_groups, max_num_tokens), -2, dtype=torch.int64, device=device
    )

    grid = (num_groups, num_reqs)
    _compute_slot_mappings_kernel[grid](
        max_num_tokens,
        idx_mapping,
        query_start_loc,
        positions,
        block_table_ptrs,
        block_table_strides,
        block_sizes,
        slot_mappings,
        slot_mappings.stride(0),
        cp_rank,
        CP_SIZE=cp_size,
        CP_INTERLEAVE=cp_interleave,
        PAD_ID=pad_id,
        TRITON_BLOCK_SIZE=TRITON_BLOCK_SIZE,
    )
    torch.npu.synchronize()

    expected_list = _compute_slot_mappings_cpu(
        max_num_tokens,
        idx_mapping.cpu(),
        query_start_loc.cpu(),
        positions.cpu(),
        [bt.cpu() for bt in block_tables],
        [block_size] * num_groups,
        cp_size=cp_size,
        cp_interleave=cp_interleave,
        cp_rank=cp_rank,
        pad_id=pad_id,
    )

    for g in range(num_groups):
        torch.testing.assert_close(
            slot_mappings[g].cpu(), expected_list[g], rtol=0, atol=0
        )


def test_compute_slot_mappings_padding() -> None:
    """Compute slot mappings: padding for CUDA graphs.

    The last batch program pads unused slots to PAD_ID.
    """
    init_device_properties_triton()

    num_groups = 1
    num_reqs = 3
    max_num_tokens = 8
    block_size = 16
    cp_size = 1
    cp_rank = 0
    cp_interleave = 1
    TRITON_BLOCK_SIZE = 4
    pad_id = PAD_SLOT_ID

    device = torch.device("npu")

    idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 2, 4, 5], dtype=torch.int32, device=device)
    positions = torch.tensor([0, 16, 0, 16, 0], dtype=torch.int64, device=device)

    block_table = torch.tensor(
        [[10, 20], [30, 40], [50, 60]], dtype=torch.int32, device=device
    )

    slot_mappings = torch.full(
        (num_groups, max_num_tokens), -2, dtype=torch.int64, device=device
    )

    grid = (num_groups, num_reqs)
    _compute_slot_mappings_kernel[grid](
        max_num_tokens,
        idx_mapping,
        query_start_loc,
        positions,
        torch.tensor([block_table.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([block_table.stride(0)], dtype=torch.int64, device=device),
        torch.tensor([block_size], dtype=torch.int32, device=device),
        slot_mappings,
        slot_mappings.stride(0),
        cp_rank,
        CP_SIZE=cp_size,
        CP_INTERLEAVE=cp_interleave,
        PAD_ID=pad_id,
        TRITON_BLOCK_SIZE=TRITON_BLOCK_SIZE,
    )
    torch.npu.synchronize()

    # CPU reference
    expected = _compute_slot_mappings_cpu(
        max_num_tokens,
        idx_mapping.cpu(),
        query_start_loc.cpu(),
        positions.cpu(),
        [block_table.cpu()],
        [block_size],
        cp_size=cp_size,
        cp_interleave=cp_interleave,
        cp_rank=cp_rank,
        pad_id=pad_id,
    )

    torch.testing.assert_close(slot_mappings[0].cpu(), expected[0], rtol=0, atol=0)
