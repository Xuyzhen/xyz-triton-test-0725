# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.gpu.sample.prompt_logprob import (
    _prompt_logprobs_token_ids_kernel,
)

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _prompt_logprobs_token_ids_cpu(
    query_start_loc: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    all_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch CPU reference.

    For each batch request, reads the target token IDs from all_token_ids
    at position num_computed_tokens + 1 + offset (shifted by one because
    the logprob is computed for the *next* token) and writes them into
    the output at query_start_loc[batch_idx] + offset.
    """
    num_reqs = idx_mapping.shape[0]
    total_tokens = int(query_start_loc[-1])
    output = torch.zeros(total_tokens, dtype=torch.int64)

    for batch_idx in range(num_reqs):
        req_state_idx = int(idx_mapping[batch_idx])
        start = int(query_start_loc[batch_idx])
        end = int(query_start_loc[batch_idx + 1])
        query_len = end - start

        nct = int(num_computed_tokens[req_state_idx])

        for i in range(query_len):
            # Shift by one: logprob for token *after* the current one
            target_pos = nct + 1 + i
            output[start + i] = int(all_token_ids[req_state_idx, target_pos])

    return output


@pytest.mark.parametrize("num_reqs", [1, 4, 8])
@pytest.mark.parametrize("query_len", [1, 5, 32])
def test_prompt_logprobs_token_ids_basic(
    num_reqs: int, query_len: int,
) -> None:
    """Fill prompt logprob token IDs from all_token_ids.

    Tests basic functionality with multiple requests, each having the same
    query length. Verifies that target positions are shifted by 1 and
    written at the correct offsets in the output buffer.
    """
    init_device_properties_triton()
    torch.manual_seed(42)

    max_num_reqs = 16
    max_model_len = 512

    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int64)
    for i in range(num_reqs):
        query_start_loc[i + 1] = query_start_loc[i] + query_len

    idx_mapping = torch.randint(0, max_num_reqs, (num_reqs,), dtype=torch.int64)

    num_computed_tokens = torch.randint(
        0, max_model_len // 2, (max_num_reqs,), dtype=torch.int64
    )
    all_token_ids = torch.randint(
        0, 32000, (max_num_reqs, max_model_len), dtype=torch.int64
    )

    expected = _prompt_logprobs_token_ids_cpu(
        query_start_loc, idx_mapping, num_computed_tokens, all_token_ids,
    )

    device = torch.device("npu")
    output = torch.empty(
        int(query_start_loc[-1]), dtype=torch.int64, device=device,
    )

    BLOCK_SIZE = 1024
    _prompt_logprobs_token_ids_kernel[(num_reqs,)](
        output,
        query_start_loc.to(device),
        idx_mapping.to(device),
        num_computed_tokens.to(device),
        all_token_ids.to(device),
        all_token_ids.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("num_reqs", [3, 6])
def test_prompt_logprobs_token_ids_varying_query_len(
    num_reqs: int,
) -> None:
    """Requests with different query lengths."""
    init_device_properties_triton()
    torch.manual_seed(7)

    max_num_reqs = 8
    max_model_len = 256

    # Varying query lengths: [2, 5, 3, 7, ...]
    query_lens = [2, 5, 3, 7, 4, 6][:num_reqs]
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int64)
    for i, ql in enumerate(query_lens):
        query_start_loc[i + 1] = query_start_loc[i] + ql

    idx_mapping = torch.tensor(
        [0, 1, 2, 3, 4, 5][:num_reqs], dtype=torch.int64,
    )

    num_computed_tokens = torch.tensor(
        [10, 20, 5, 15, 8, 12][:num_reqs], dtype=torch.int64,
    )
    all_token_ids = torch.randint(
        0, 32000, (max_num_reqs, max_model_len), dtype=torch.int64,
    )

    expected = _prompt_logprobs_token_ids_cpu(
        query_start_loc, idx_mapping, num_computed_tokens, all_token_ids,
    )

    device = torch.device("npu")
    output = torch.empty(
        int(query_start_loc[-1]), dtype=torch.int64, device=device,
    )

    _prompt_logprobs_token_ids_kernel[(num_reqs,)](
        output,
        query_start_loc.to(device),
        idx_mapping.to(device),
        num_computed_tokens.to(device),
        all_token_ids.to(device),
        all_token_ids.stride(0),
        BLOCK_SIZE=1024,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


def test_prompt_logprobs_token_ids_single_token() -> None:
    """Single request, single query token (edge case)."""
    init_device_properties_triton()
    torch.manual_seed(3)

    num_reqs = 1
    query_start_loc = torch.tensor([0, 1], dtype=torch.int64)
    idx_mapping = torch.tensor([0], dtype=torch.int64)
    num_computed_tokens = torch.tensor([5], dtype=torch.int64)
    all_token_ids = torch.randint(0, 32000, (1, 128), dtype=torch.int64)

    expected = _prompt_logprobs_token_ids_cpu(
        query_start_loc, idx_mapping, num_computed_tokens, all_token_ids,
    )

    device = torch.device("npu")
    output = torch.empty(1, dtype=torch.int64, device=device)

    _prompt_logprobs_token_ids_kernel[(1,)](
        output,
        query_start_loc.to(device),
        idx_mapping.to(device),
        num_computed_tokens.to(device),
        all_token_ids.to(device),
        all_token_ids.stride(0),
        BLOCK_SIZE=1024,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
