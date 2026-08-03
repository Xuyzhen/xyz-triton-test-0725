# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.gpu.input_batch import (
    _post_update_num_computed_tokens_kernel,
    post_update_num_computed_tokens,
)

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _post_update_num_computed_tokens_cpu(
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    """Independent PyTorch CPU reference implementation.

    Returns updated num_computed_tokens.
    """
    result = num_computed_tokens.clone()
    for batch_idx in range(idx_mapping.shape[0]):
        req_state_idx = int(idx_mapping[batch_idx])
        query_start = int(query_start_loc[batch_idx])
        query_end = int(query_start_loc[batch_idx + 1])
        query_len = query_end - query_start
        result[req_state_idx] += query_len
    return result


@pytest.mark.parametrize("num_reqs, max_num_reqs", [
    (3, 5),
    (5, 8),
    (1, 4),
    (4, 4),
])
def test_post_update_num_computed_tokens_matches_cpu(
    num_reqs: int, max_num_reqs: int
) -> None:
    """Compare Triton num_computed_tokens update with CPU reference."""
    init_device_properties_triton()

    idx_mapping = torch.arange(num_reqs, dtype=torch.int32)
    query_lens = torch.randint(1, 10, (num_reqs,), dtype=torch.int32)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    query_start_loc[1:] = query_lens.cumsum(dim=0)

    # Pre-populate num_computed_tokens with varied values
    num_computed_tokens = torch.randint(
        0, 100, (max_num_reqs,), dtype=torch.int32
    )

    expected = _post_update_num_computed_tokens_cpu(
        idx_mapping, num_computed_tokens, query_start_loc,
    )

    device = torch.device("npu")
    num_computed_tokens_npu = num_computed_tokens.to(device)

    _post_update_num_computed_tokens_kernel[(num_reqs,)](
        idx_mapping.to(device),
        num_computed_tokens_npu,
        query_start_loc.to(device),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        num_computed_tokens_npu.cpu(), expected, rtol=0, atol=0
    )


def test_post_update_num_computed_tokens_non_contiguous_mapping() -> None:
    """Test with non-sequential idx_mapping."""
    init_device_properties_triton()

    num_reqs = 3
    max_num_reqs = 6
    idx_mapping = torch.tensor([4, 1, 3], dtype=torch.int32)
    query_lens = torch.tensor([3, 7, 2], dtype=torch.int32)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    query_start_loc[1:] = query_lens.cumsum(dim=0)

    num_computed_tokens = torch.tensor(
        [10, 20, 30, 40, 50, 60], dtype=torch.int32
    )

    expected = _post_update_num_computed_tokens_cpu(
        idx_mapping, num_computed_tokens, query_start_loc,
    )

    device = torch.device("npu")
    num_computed_tokens_npu = num_computed_tokens.to(device)

    _post_update_num_computed_tokens_kernel[(num_reqs,)](
        idx_mapping.to(device),
        num_computed_tokens_npu,
        query_start_loc.to(device),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        num_computed_tokens_npu.cpu(), expected, rtol=0, atol=0
    )


def test_post_update_num_computed_tokens_through_wrapper() -> None:
    """Test that the wrapper function produces correct results."""
    init_device_properties_triton()

    num_reqs = 4
    max_num_reqs = 6
    idx_mapping = torch.tensor([0, 2, 5, 3], dtype=torch.int32)
    query_lens = torch.tensor([2, 4, 1, 3], dtype=torch.int32)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    query_start_loc[1:] = query_lens.cumsum(dim=0)

    num_computed_tokens = torch.tensor(
        [5, 10, 15, 20, 25, 30], dtype=torch.int32
    )

    expected = _post_update_num_computed_tokens_cpu(
        idx_mapping, num_computed_tokens, query_start_loc,
    )

    device = torch.device("npu")
    num_computed_tokens_npu = num_computed_tokens.to(device)

    post_update_num_computed_tokens(
        idx_mapping.to(device),
        num_computed_tokens_npu,
        query_start_loc.to(device),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        num_computed_tokens_npu.cpu(), expected, rtol=0, atol=0
    )
