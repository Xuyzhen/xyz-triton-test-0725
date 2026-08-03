# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.gpu.input_batch import (
    _expand_idx_mapping_kernel,
    expand_idx_mapping,
)

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _expand_idx_mapping_cpu(
    idx_mapping: torch.Tensor,
    cu_num_logits: torch.Tensor,
    total_num_logits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent PyTorch CPU reference implementation.

    Returns (expanded_idx_mapping, expanded_local_pos).
    """
    num_reqs = idx_mapping.shape[0]
    expanded_idx_mapping = torch.empty(total_num_logits, dtype=torch.int32)
    expanded_local_pos = torch.empty(total_num_logits, dtype=torch.int32)

    for batch_idx in range(num_reqs):
        req_state_idx = int(idx_mapping[batch_idx])
        start_idx = int(cu_num_logits[batch_idx])
        end_idx = int(cu_num_logits[batch_idx + 1])
        num_tokens = end_idx - start_idx

        for i in range(num_tokens):
            expanded_idx_mapping[start_idx + i] = req_state_idx
            expanded_local_pos[start_idx + i] = i

    return expanded_idx_mapping, expanded_local_pos


@pytest.mark.parametrize("dtype", [torch.int32])
def test_expand_idx_mapping_basic(dtype: torch.dtype) -> None:
    """Test basic expansion of idx_mapping with uniform logit counts."""
    init_device_properties_triton()

    num_reqs = 4
    idx_mapping = torch.tensor([0, 2, 1, 3], dtype=dtype)
    logits_per_req = torch.tensor([3, 5, 2, 4], dtype=torch.int32)
    cu_num_logits = torch.zeros(num_reqs + 1, dtype=torch.int32)
    cu_num_logits[1:] = logits_per_req.cumsum(dim=0)
    total_num_logits = int(cu_num_logits[-1])

    expected_expanded_idx, expected_local_pos = _expand_idx_mapping_cpu(
        idx_mapping, cu_num_logits, total_num_logits,
    )

    # Test via kernel directly
    device = torch.device("npu")
    idx_mapping_npu = idx_mapping.to(device)
    cu_num_logits_npu = cu_num_logits.to(device)

    expanded_idx = torch.empty(total_num_logits, dtype=dtype, device=device)
    expanded_local_pos = torch.empty(
        total_num_logits, dtype=torch.int32, device=device
    )

    max_expand_len = int(logits_per_req.max().item())
    BLOCK_SIZE = 2 ** (max_expand_len - 1).bit_length()

    _expand_idx_mapping_kernel[(num_reqs,)](
        idx_mapping_npu,
        expanded_idx,
        expanded_local_pos,
        cu_num_logits_npu,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        expanded_idx.cpu(), expected_expanded_idx, rtol=0, atol=0
    )
    torch.testing.assert_close(
        expanded_local_pos.cpu(), expected_local_pos, rtol=0, atol=0
    )


def test_expand_idx_mapping_through_wrapper() -> None:
    """Test the full pipeline through the public wrapper function."""
    init_device_properties_triton()

    num_reqs = 5
    idx_mapping = torch.tensor([4, 0, 3, 1, 2], dtype=torch.int32)
    logits_per_req = torch.tensor([1, 7, 3, 5, 2], dtype=torch.int32)
    cu_num_logits = torch.zeros(num_reqs + 1, dtype=torch.int32)
    cu_num_logits[1:] = logits_per_req.cumsum(dim=0)
    total_num_logits = int(cu_num_logits[-1])
    max_expand_len = int(logits_per_req.max().item())

    expected_expanded_idx, expected_local_pos = _expand_idx_mapping_cpu(
        idx_mapping, cu_num_logits, total_num_logits,
    )

    device = torch.device("npu")
    actual_expanded_idx, actual_local_pos = expand_idx_mapping(
        idx_mapping.to(device),
        total_num_logits,
        cu_num_logits.to(device),
        max_expand_len,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        actual_expanded_idx.cpu(), expected_expanded_idx, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_local_pos.cpu(), expected_local_pos, rtol=0, atol=0
    )


def test_expand_idx_mapping_single_request() -> None:
    """Test expansion with a single request."""
    init_device_properties_triton()

    num_reqs = 1
    idx_mapping = torch.tensor([5], dtype=torch.int32)
    cu_num_logits = torch.tensor([0, 10], dtype=torch.int32)
    total_num_logits = 10

    expected_expanded_idx, expected_local_pos = _expand_idx_mapping_cpu(
        idx_mapping, cu_num_logits, total_num_logits,
    )

    device = torch.device("npu")
    expanded_idx, expanded_local_pos = expand_idx_mapping(
        idx_mapping.to(device),
        total_num_logits,
        cu_num_logits.to(device),
        max_expand_len=10,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        expanded_idx.cpu(), expected_expanded_idx, rtol=0, atol=0
    )
    torch.testing.assert_close(
        expanded_local_pos.cpu(), expected_local_pos, rtol=0, atol=0
    )
