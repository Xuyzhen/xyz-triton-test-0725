# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.gpu.input_batch import _prepare_pos_seq_lens_kernel

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _prepare_pos_seq_lens_cpu(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    max_num_reqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent PyTorch CPU reference for positions and seq_lens preparation.

    Returns (positions, seq_lens).
    """
    num_reqs = idx_mapping.shape[0]
    num_tokens = int(query_start_loc[-1])

    seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32)
    positions = torch.zeros(num_tokens, dtype=torch.int64)

    for req_id in range(num_reqs):
        req_state_idx = int(idx_mapping[req_id])
        num_computed = int(num_computed_tokens[req_state_idx])

        query_start = int(query_start_loc[req_id])
        query_end = int(query_start_loc[req_id + 1])
        query_len = query_end - query_start

        # Store seq_len
        seq_lens[req_id] = num_computed + query_len

        # Store positions
        for i in range(query_len):
            positions[query_start + i] = num_computed + i

    # Padding entries (beyond num_reqs) remain 0

    return positions, seq_lens


@pytest.mark.parametrize("num_reqs, max_num_reqs", [
    (3, 5),
    (5, 8),
    (1, 4),
    (4, 4),
])
def test_prepare_pos_seq_lens_matches_cpu(
    num_reqs: int, max_num_reqs: int
) -> None:
    """Compare Triton positions/seq_lens kernel with CPU reference.

    The kernel is launched with (num_reqs + 1,) grid where the last block
    handles padding of seq_lens entries beyond num_reqs up to max_num_reqs.
    """
    init_device_properties_triton()

    max_model_len = 2048
    query_lens = torch.randint(1, 20, (num_reqs,), dtype=torch.int32)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    query_start_loc[1:] = query_lens.cumsum(dim=0)
    num_tokens = int(query_start_loc[-1])

    idx_mapping = torch.arange(num_reqs, dtype=torch.int32)
    num_computed_tokens = torch.randint(
        0, 100, (max_num_reqs,), dtype=torch.int32
    )

    expected_positions, expected_seq_lens = _prepare_pos_seq_lens_cpu(
        idx_mapping,
        query_start_loc,
        num_computed_tokens,
        max_num_reqs,
    )

    device = torch.device("npu")
    idx_mapping_npu = idx_mapping.to(device)
    query_start_loc_npu = query_start_loc.to(device)
    num_computed_tokens_npu = num_computed_tokens.to(device)

    # Use padded storage to verify stride handling
    positions_storage = torch.full(
        (num_tokens + 5,), -1, dtype=torch.int64, device=device
    )
    positions = positions_storage[:num_tokens]

    seq_lens = torch.full(
        (max_num_reqs,), -1, dtype=torch.int32, device=device
    )

    _prepare_pos_seq_lens_kernel[(num_reqs + 1,)](
        positions,
        seq_lens,
        idx_mapping_npu,
        query_start_loc_npu,
        num_computed_tokens_npu,
        max_num_reqs,
        BLOCK_SIZE=1024,
    )
    torch.npu.synchronize()

    assert positions.dtype == torch.int64
    assert seq_lens.dtype == torch.int32
    torch.testing.assert_close(
        positions.cpu(), expected_positions, rtol=0, atol=0
    )
    torch.testing.assert_close(
        seq_lens.cpu(), expected_seq_lens, rtol=0, atol=0
    )

    # Verify padding storage was untouched
    torch.testing.assert_close(
        positions_storage[num_tokens:].cpu(),
        torch.full((5,), -1, dtype=torch.int64),
        rtol=0,
        atol=0,
    )


def test_prepare_pos_seq_lens_non_contiguous_mapping() -> None:
    """Test that idx_mapping that differs from sequential order works."""
    init_device_properties_triton()

    max_num_reqs = 6
    num_reqs = 3
    query_lens = torch.tensor([5, 3, 7], dtype=torch.int32)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    query_start_loc[1:] = query_lens.cumsum(dim=0)
    num_tokens = int(query_start_loc[-1])

    # Non-contiguous mapping: batch_idx -> req_state_idx
    idx_mapping = torch.tensor([3, 0, 5], dtype=torch.int32)
    num_computed_tokens = torch.tensor(
        [10, 20, 30, 5, 40, 15], dtype=torch.int32
    )

    expected_positions, expected_seq_lens = _prepare_pos_seq_lens_cpu(
        idx_mapping,
        query_start_loc,
        num_computed_tokens,
        max_num_reqs,
    )

    device = torch.device("npu")
    positions = torch.zeros(num_tokens, dtype=torch.int64, device=device)
    seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

    _prepare_pos_seq_lens_kernel[(num_reqs + 1,)](
        positions,
        seq_lens,
        idx_mapping.to(device),
        query_start_loc.to(device),
        num_computed_tokens.to(device),
        max_num_reqs,
        BLOCK_SIZE=1024,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        positions.cpu(), expected_positions, rtol=0, atol=0
    )
    torch.testing.assert_close(
        seq_lens.cpu(), expected_seq_lens, rtol=0, atol=0
    )
