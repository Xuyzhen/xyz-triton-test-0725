import pytest
import torch

from vllm.v1.worker.gpu.input_batch import _post_update_num_computed_tokens_kernel


def _post_update_num_computed_tokens_cpu(
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
):
    num_reqs = idx_mapping.shape[0]
    for batch_id in range(num_reqs):
        query_start = int(query_start_loc[batch_id])
        query_end = int(query_start_loc[batch_id + 1])
        query_len = query_end - query_start

        req_state_idx = int(idx_mapping[batch_id])
        num_computed_tokens[req_state_idx] += query_len


def test_post_update_num_computed_tokens_kernel():
    max_num_reqs = 4
    idx_mapping = torch.tensor([2, 0, 3, 1], dtype=torch.int32)
    query_lens = torch.tensor([3, 5, 1, 2], dtype=torch.int32)
    query_start_loc = torch.zeros(max_num_reqs + 1, dtype=torch.int32)
    query_start_loc[1:] = query_lens.cumsum(dim=0)

    num_computed_tokens = torch.tensor([10, 20, 5, 15], dtype=torch.int32)

    expected = num_computed_tokens.clone()
    _post_update_num_computed_tokens_cpu(
        idx_mapping, expected, query_start_loc
    )

    device = torch.device("npu")
    num_computed_tokens_npu = num_computed_tokens.to(device)

    _post_update_num_computed_tokens_kernel[(max_num_reqs,)](
        idx_mapping.to(device),
        num_computed_tokens_npu,
        query_start_loc.to(device),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        num_computed_tokens_npu.cpu(), expected, rtol=0, atol=0
    )


def test_post_update_num_computed_tokens_kernel_single():
    max_num_reqs = 1
    idx_mapping = torch.tensor([0], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 7], dtype=torch.int32)
    num_computed_tokens = torch.tensor([0], dtype=torch.int32)

    expected = num_computed_tokens.clone()
    _post_update_num_computed_tokens_cpu(
        idx_mapping, expected, query_start_loc
    )

    device = torch.device("npu")
    num_computed_tokens_npu = num_computed_tokens.to(device)

    _post_update_num_computed_tokens_kernel[(1,)](
        idx_mapping.to(device),
        num_computed_tokens_npu,
        query_start_loc.to(device),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        num_computed_tokens_npu.cpu(), expected, rtol=0, atol=0
    )
