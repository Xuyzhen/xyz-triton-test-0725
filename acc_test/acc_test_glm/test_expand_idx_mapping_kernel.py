import pytest
import torch

from vllm.v1.worker.gpu.input_batch import _expand_idx_mapping_kernel


def _expand_idx_mapping_cpu(
    idx_mapping: torch.Tensor,
    total_num_logits: int,
    cu_num_logits: torch.Tensor,
):
    num_reqs = idx_mapping.shape[0]
    expanded_idx_mapping = torch.empty(total_num_logits, dtype=idx_mapping.dtype)
    expanded_local_pos = torch.empty(total_num_logits, dtype=torch.int32)

    for req_idx in range(num_reqs):
        start_idx = int(cu_num_logits[req_idx])
        end_idx = int(cu_num_logits[req_idx + 1])
        num_tokens = end_idx - start_idx
        req_state_idx = int(idx_mapping[req_idx])
        for i in range(num_tokens):
            expanded_idx_mapping[start_idx + i] = req_state_idx
            expanded_local_pos[start_idx + i] = i

    return expanded_idx_mapping, expanded_local_pos


def test_expand_idx_mapping_kernel():
    torch.manual_seed(42)
    idx_mapping = torch.tensor([3, 0, 1], dtype=torch.int32)
    cu_num_logits = torch.tensor([0, 2, 5, 8], dtype=torch.int32)
    total_num_logits = 8
    max_expand_len = 3

    expected_mapping, expected_pos = _expand_idx_mapping_cpu(
        idx_mapping, total_num_logits, cu_num_logits
    )

    device = torch.device("npu")
    from vllm.triton_utils import triton

    expanded_idx_mapping = idx_mapping.new_empty(total_num_logits).to(device)
    expanded_local_pos = torch.empty(
        total_num_logits, dtype=torch.int32, device=device
    )

    _expand_idx_mapping_kernel[(3,)](
        idx_mapping.to(device),
        expanded_idx_mapping,
        expanded_local_pos,
        cu_num_logits.to(device),
        BLOCK_SIZE=triton.next_power_of_2(max_expand_len),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        expanded_idx_mapping.cpu(), expected_mapping, rtol=0, atol=0
    )
    torch.testing.assert_close(
        expanded_local_pos.cpu(), expected_pos, rtol=0, atol=0
    )


def test_expand_idx_mapping_kernel_uniform_expand():
    idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int32)
    cu_num_logits = torch.tensor([0, 4, 8, 12], dtype=torch.int32)
    total_num_logits = 12
    max_expand_len = 4

    expected_mapping, expected_pos = _expand_idx_mapping_cpu(
        idx_mapping, total_num_logits, cu_num_logits
    )

    device = torch.device("npu")
    from vllm.triton_utils import triton

    expanded_idx_mapping = idx_mapping.new_empty(total_num_logits).to(device)
    expanded_local_pos = torch.empty(
        total_num_logits, dtype=torch.int32, device=device
    )

    _expand_idx_mapping_kernel[(3,)](
        idx_mapping.to(device),
        expanded_idx_mapping,
        expanded_local_pos,
        cu_num_logits.to(device),
        BLOCK_SIZE=triton.next_power_of_2(max_expand_len),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        expanded_idx_mapping.cpu(), expected_mapping, rtol=0, atol=0
    )
    torch.testing.assert_close(
        expanded_local_pos.cpu(), expected_pos, rtol=0, atol=0
    )
