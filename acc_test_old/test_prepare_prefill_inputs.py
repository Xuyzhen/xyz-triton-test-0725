# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.gpu.input_batch import _prepare_prefill_inputs_kernel

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _prepare_prefill_inputs_cpu(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    prefill_lens: torch.Tensor,
    num_computed_tokens: torch.Tensor,
) -> torch.Tensor:
    """Independent PyTorch CPU reference for prefill input preparation.

    Returns (input_ids, next_prefill_tokens).
    """
    num_reqs = idx_mapping.shape[0]
    num_tokens = int(query_start_loc[-1])
    max_num_reqs = all_token_ids.shape[0]

    input_ids = torch.zeros(num_tokens, dtype=torch.int32)
    next_prefill_tokens = torch.full((max_num_reqs,), -1, dtype=torch.int32)

    for batch_idx, req_state_idx_tensor in enumerate(idx_mapping):
        req_state_idx = int(req_state_idx_tensor)
        prefill_len = int(prefill_lens[req_state_idx])
        num_computed = int(num_computed_tokens[req_state_idx])

        if num_computed >= prefill_len:
            # Decode request - skip
            continue

        query_start = int(query_start_loc[batch_idx])
        query_end = int(query_start_loc[batch_idx + 1])
        query_len = query_end - query_start

        # Copy tokens from all_token_ids to input_ids
        src_start = num_computed
        src_end = num_computed + query_len
        input_ids[query_start:query_end] = all_token_ids[
            req_state_idx, src_start:src_end
        ]

        # Store next prefill token if applicable
        next_pos = num_computed + query_len
        if next_pos < prefill_len:
            next_prefill_tokens[req_state_idx] = all_token_ids[
                req_state_idx, next_pos
            ]

    return input_ids, next_prefill_tokens


@pytest.mark.parametrize("max_num_reqs", [5, 8])
def test_prepare_prefill_inputs_matches_cpu(max_num_reqs: int) -> None:
    """Compare Triton prefill inputs kernel with CPU reference."""
    init_device_properties_triton()

    max_model_len = 512
    # Mix of prefill and decode requests
    query_lens = torch.tensor([3, 128, 1, 7, 256], dtype=torch.int32)
    query_start_loc = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    query_start_loc[1:] = query_lens.cumsum(dim=0)
    num_tokens = int(query_start_loc[-1])

    # Batch order may differ from request-state order
    num_reqs = len(query_lens)
    idx_mapping = torch.tensor([2, 0, 4, 1, 3], dtype=torch.int32)

    # Request-state arrays
    prefill_lens = torch.tensor([200, 50, 512, 150, 300], dtype=torch.int32)
    num_computed_tokens = torch.tensor(
        [10, 50, 5, 150, 100], dtype=torch.int32
    )

    # all_token_ids: [max_num_reqs, max_model_len]
    all_token_ids = torch.randint(
        0, 32000, (max_num_reqs, max_model_len), dtype=torch.int32
    )

    expected_input_ids, expected_next_tokens = _prepare_prefill_inputs_cpu(
        idx_mapping,
        query_start_loc,
        all_token_ids,
        prefill_lens,
        num_computed_tokens,
    )

    device = torch.device("npu")
    idx_mapping_npu = idx_mapping.to(device)
    query_start_loc_npu = query_start_loc.to(device)
    all_token_ids_npu = all_token_ids.to(device)
    prefill_lens_npu = prefill_lens.to(device)
    num_computed_tokens_npu = num_computed_tokens.to(device)

    # Use padded storage to verify stride handling
    input_ids_storage = torch.full(
        (num_tokens + 3,), -1, dtype=torch.int32, device=device
    )
    input_ids = input_ids_storage[:num_tokens]

    next_prefill_tokens = torch.full(
        (max_num_reqs,), -1, dtype=torch.int32, device=device
    )

    _prepare_prefill_inputs_kernel[(num_reqs,)](
        input_ids,
        next_prefill_tokens,
        idx_mapping_npu,
        query_start_loc_npu,
        all_token_ids_npu,
        all_token_ids_npu.stride(0),
        prefill_lens_npu,
        num_computed_tokens_npu,
        BLOCK_SIZE=1024,
    )
    torch.npu.synchronize()

    assert input_ids.dtype == torch.int32
    torch.testing.assert_close(
        input_ids.cpu(), expected_input_ids, rtol=0, atol=0
    )
    torch.testing.assert_close(
        next_prefill_tokens.cpu(), expected_next_tokens, rtol=0, atol=0
    )

    # Verify padding was never written
    torch.testing.assert_close(
        input_ids_storage[num_tokens:].cpu(),
        torch.full((3,), -1, dtype=torch.int32),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("max_num_reqs", [5])
def test_prepare_prefill_inputs_all_decode(max_num_reqs: int) -> None:
    """Test the case where all requests are decode (no prefill)."""
    init_device_properties_triton()

    max_model_len = 512
    query_lens = torch.tensor([1, 1, 1], dtype=torch.int32)
    query_start_loc = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    query_start_loc[1:] = query_lens.cumsum(dim=0)
    num_tokens = int(query_start_loc[-1])

    idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int32)

    # All requests have num_computed >= prefill_len (decode)
    prefill_lens = torch.tensor([10, 20, 30], dtype=torch.int32)
    num_computed_tokens = torch.tensor(
        [10, 25, 30], dtype=torch.int32
    )

    all_token_ids = torch.randint(
        0, 32000, (max_num_reqs, max_model_len), dtype=torch.int32
    )

    expected_input_ids, expected_next_tokens = _prepare_prefill_inputs_cpu(
        idx_mapping,
        query_start_loc,
        all_token_ids,
        prefill_lens,
        num_computed_tokens,
    )

    device = torch.device("npu")
    input_ids = torch.full(
        (num_tokens,), -1, dtype=torch.int32, device=device
    )
    next_prefill_tokens = torch.full(
        (max_num_reqs,), -1, dtype=torch.int32, device=device
    )

    _prepare_prefill_inputs_kernel[(len(idx_mapping),)](
        input_ids,
        next_prefill_tokens,
        idx_mapping.to(device),
        query_start_loc.to(device),
        all_token_ids.to(device),
        all_token_ids.stride(0),
        prefill_lens.to(device),
        num_computed_tokens.to(device),
        BLOCK_SIZE=1024,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        input_ids.cpu(), expected_input_ids, rtol=0, atol=0
    )
    torch.testing.assert_close(
        next_prefill_tokens.cpu(), expected_next_tokens, rtol=0, atol=0
    )
