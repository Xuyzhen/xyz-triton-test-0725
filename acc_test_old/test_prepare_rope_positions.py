# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.gpu.mm.rope import _prepare_rope_positions_kernel

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _prepare_rope_positions_cpu(
    prefill_positions: torch.Tensor,
    prefill_delta: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    prefill_lens: torch.Tensor,
    num_computed_tokens: torch.Tensor,
) -> torch.Tensor:
    """Independent PyTorch CPU reference implementation."""
    num_dims = prefill_positions.shape[1]
    num_tokens = int(query_start_loc[-1])
    output = torch.empty((num_dims, num_tokens), dtype=torch.int64)

    for batch_idx, req_state_idx_tensor in enumerate(idx_mapping):
        req_state_idx = int(req_state_idx_tensor)
        query_start = int(query_start_loc[batch_idx])
        query_end = int(query_start_loc[batch_idx + 1])
        num_computed = int(num_computed_tokens[req_state_idx])
        orig_pos = torch.arange(
            num_computed,
            num_computed + query_end - query_start,
            dtype=torch.int64,
        )

        if num_computed < int(prefill_lens[req_state_idx]):
            output[:, query_start:query_end] = prefill_positions[
                req_state_idx, :, orig_pos
            ]
        else:
            output[:, query_start:query_end] = (
                orig_pos + int(prefill_delta[req_state_idx])
            )

    return output


@pytest.mark.parametrize("num_dims,has_delta", [(3, True), (4, False)])
def test_prepare_rope_positions_matches_cpu(
    num_dims: int,
    has_delta: bool,
) -> None:
    init_device_properties_triton()

    max_num_reqs = 5
    max_model_len = 2048
    query_lens = torch.tensor([3, 1025, 1, 7], dtype=torch.int32)
    query_start_loc = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    query_start_loc[1:] = query_lens.cumsum(dim=0)
    num_tokens = int(query_start_loc[-1])

    # Batch order intentionally differs from request-state order.
    idx_mapping = torch.tensor([3, 0, 4, 1], dtype=torch.int32)
    prefill_lens = torch.tensor([1600, 20, 10, 100, 50], dtype=torch.int32)
    num_computed_tokens = torch.tensor([20, 30, 0, 5, 50], dtype=torch.int32)
    prefill_delta = torch.tensor([11, -3, 7, 5, 19], dtype=torch.int32)
    if not has_delta:
        prefill_delta.zero_()

    req_offsets = torch.arange(max_num_reqs, dtype=torch.int32)[:, None, None]
    dim_offsets = torch.arange(num_dims, dtype=torch.int32)[None, :, None]
    pos_offsets = torch.arange(max_model_len, dtype=torch.int32)[None, None, :]
    prefill_positions = (
        req_offsets * 100_000 + dim_offsets * 10_000 + pos_offsets
    )

    expected = _prepare_rope_positions_cpu(
        prefill_positions,
        prefill_delta,
        idx_mapping,
        query_start_loc,
        prefill_lens,
        num_computed_tokens,
    )

    device = torch.device("npu")
    prefill_positions_npu = prefill_positions.to(device)
    prefill_delta_npu = prefill_delta.to(device)
    idx_mapping_npu = idx_mapping.to(device)
    query_start_loc_npu = query_start_loc.to(device)
    prefill_lens_npu = prefill_lens.to(device)
    num_computed_tokens_npu = num_computed_tokens.to(device)

    # Keep a padded row stride to verify positions_stride handling.
    positions_storage = torch.full(
        (num_dims, num_tokens + 5), -1, dtype=torch.int64, device=device
    )
    positions = positions_storage[:, :num_tokens]
    assert not positions.is_contiguous()

    _prepare_rope_positions_kernel[(len(idx_mapping),)](
        positions,
        positions.stride(0),
        prefill_positions_npu,
        prefill_positions_npu.stride(0),
        prefill_positions_npu.stride(1),
        prefill_delta_npu,
        idx_mapping_npu,
        query_start_loc_npu,
        prefill_lens_npu,
        num_computed_tokens_npu,
        BLOCK_SIZE=1024,
        NUM_DIMS=num_dims,
    )
    torch.npu.synchronize()

    assert positions.dtype == torch.int64
    torch.testing.assert_close(positions.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        positions_storage[:, num_tokens:].cpu(),
        torch.full((num_dims, 5), -1, dtype=torch.int64),
        rtol=0,
        atol=0,
    )
