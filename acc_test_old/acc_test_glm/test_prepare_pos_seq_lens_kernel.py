import pytest
import torch

from vllm.v1.worker.gpu.input_batch import _prepare_pos_seq_lens_kernel


def _prepare_pos_seq_lens_cpu(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    max_num_reqs: int,
):
    num_reqs = idx_mapping.shape[0]
    pos = torch.zeros(int(query_start_loc[-1]), dtype=torch.int64)
    seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32)

    for req_id in range(num_reqs):
        req_state_idx = int(idx_mapping[req_id])
        num_computed = int(num_computed_tokens[req_state_idx])
        start = int(query_start_loc[req_id])
        end = int(query_start_loc[req_id + 1])
        query_len = end - start

        seq_lens[req_id] = num_computed + query_len
        for i in range(query_len):
            pos[start + i] = num_computed + i

    return pos, seq_lens


def test_prepare_pos_seq_lens_kernel():
    torch.manual_seed(42)
    max_num_reqs = 4

    idx_mapping = torch.tensor([1, 0, 3, 2], dtype=torch.int32)
    query_lens = torch.tensor([3, 2, 5, 1], dtype=torch.int32)
    query_start_loc = torch.zeros(max_num_reqs + 1, dtype=torch.int32)
    query_start_loc[1:] = query_lens.cumsum(dim=0)
    num_tokens = int(query_start_loc[-1])

    num_computed_tokens = torch.tensor([10, 5, 20, 0], dtype=torch.int32)

    expected_pos, expected_seq_lens = _prepare_pos_seq_lens_cpu(
        idx_mapping, query_start_loc, num_computed_tokens, max_num_reqs
    )

    device = torch.device("npu")
    pos = torch.zeros(num_tokens, dtype=torch.int64, device=device)
    seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

    _prepare_pos_seq_lens_kernel[(max_num_reqs + 1,)](
        pos,
        seq_lens,
        idx_mapping.to(device),
        query_start_loc.to(device),
        num_computed_tokens.to(device),
        max_num_reqs,
        BLOCK_SIZE=1024,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(pos.cpu(), expected_pos, rtol=0, atol=0)
    torch.testing.assert_close(seq_lens.cpu(), expected_seq_lens, rtol=0, atol=0)


def test_prepare_pos_seq_lens_kernel_zero_computed():
    max_num_reqs = 2

    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    query_lens = torch.tensor([4, 3], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 4, 7], dtype=torch.int32)
    num_tokens = 7

    num_computed_tokens = torch.tensor([0, 0], dtype=torch.int32)

    expected_pos, expected_seq_lens = _prepare_pos_seq_lens_cpu(
        idx_mapping, query_start_loc, num_computed_tokens, max_num_reqs
    )

    device = torch.device("npu")
    pos = torch.zeros(num_tokens, dtype=torch.int64, device=device)
    seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

    _prepare_pos_seq_lens_kernel[(max_num_reqs + 1,)](
        pos,
        seq_lens,
        idx_mapping.to(device),
        query_start_loc.to(device),
        num_computed_tokens.to(device),
        max_num_reqs,
        BLOCK_SIZE=1024,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(pos.cpu(), expected_pos, rtol=0, atol=0)
    torch.testing.assert_close(seq_lens.cpu(), expected_seq_lens, rtol=0, atol=0)
