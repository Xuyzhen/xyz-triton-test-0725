import pytest
import torch

from vllm.v1.worker.gpu.input_batch import _prepare_prefill_inputs_kernel


def _prepare_prefill_inputs_cpu(
    input_ids: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    prefill_lens: torch.Tensor,
    num_computed_tokens: torch.Tensor,
):
    num_reqs = idx_mapping.shape[0]
    for batch_idx in range(num_reqs):
        req_state_idx = int(idx_mapping[batch_idx])
        prefill_len = int(prefill_lens[req_state_idx])
        num_computed = int(num_computed_tokens[req_state_idx])
        if num_computed >= prefill_len:
            continue

        query_start = int(query_start_loc[batch_idx])
        query_end = int(query_start_loc[batch_idx + 1])
        query_len = query_end - query_start

        for i in range(query_len):
            input_ids[query_start + i] = all_token_ids[req_state_idx, num_computed + i]

        next_pos = num_computed + query_len
        if next_pos < prefill_len:
            next_prefill_tokens[req_state_idx] = all_token_ids[req_state_idx, next_pos]


def test_prepare_prefill_inputs_kernel():
    torch.manual_seed(42)
    max_num_reqs = 3
    max_model_len = 32
    num_tokens = 8

    idx_mapping = torch.tensor([2, 0, 1], dtype=torch.int32)
    query_lens = torch.tensor([3, 2, 3], dtype=torch.int32)
    query_start_loc = torch.zeros(max_num_reqs + 1, dtype=torch.int32)
    query_start_loc[1:] = query_lens.cumsum(dim=0)

    all_token_ids = torch.arange(
        max_num_reqs * max_model_len, dtype=torch.int32
    ).reshape(max_num_reqs, max_model_len)

    prefill_lens = torch.tensor([20, 15, 10], dtype=torch.int32)
    num_computed_tokens = torch.tensor([5, 0, 3], dtype=torch.int32)

    input_ids = torch.full((num_tokens,), -1, dtype=torch.int32)
    next_prefill_tokens = torch.full((max_num_reqs,), -1, dtype=torch.int32)

    expected_input_ids = input_ids.clone()
    expected_next = next_prefill_tokens.clone()
    _prepare_prefill_inputs_cpu(
        expected_input_ids,
        expected_next,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        prefill_lens,
        num_computed_tokens,
    )

    device = torch.device("npu")
    input_ids_npu = input_ids.to(device)
    next_prefill_tokens_npu = next_prefill_tokens.to(device)

    _prepare_prefill_inputs_kernel[(max_num_reqs,)](
        input_ids_npu,
        next_prefill_tokens_npu,
        idx_mapping.to(device),
        query_start_loc.to(device),
        all_token_ids.to(device),
        all_token_ids.stride(0),
        prefill_lens.to(device),
        num_computed_tokens.to(device),
        BLOCK_SIZE=1024,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(input_ids_npu.cpu(), expected_input_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        next_prefill_tokens_npu.cpu(), expected_next, rtol=0, atol=0
    )


def test_prepare_prefill_inputs_kernel_decode_skip():
    max_num_reqs = 2
    max_model_len = 16
    num_tokens = 2

    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    query_lens = torch.tensor([1, 1], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)

    all_token_ids = torch.arange(
        max_num_reqs * max_model_len, dtype=torch.int32
    ).reshape(max_num_reqs, max_model_len)

    prefill_lens = torch.tensor([5, 3], dtype=torch.int32)
    num_computed_tokens = torch.tensor([10, 5], dtype=torch.int32)

    input_ids = torch.full((num_tokens,), -1, dtype=torch.int32)
    next_prefill_tokens = torch.full((max_num_reqs,), -1, dtype=torch.int32)

    expected_input_ids = input_ids.clone()
    expected_next = next_prefill_tokens.clone()
    _prepare_prefill_inputs_cpu(
        expected_input_ids,
        expected_next,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        prefill_lens,
        num_computed_tokens,
    )

    device = torch.device("npu")
    input_ids_npu = input_ids.to(device)
    next_prefill_tokens_npu = next_prefill_tokens.to(device)

    _prepare_prefill_inputs_kernel[(max_num_reqs,)](
        input_ids_npu,
        next_prefill_tokens_npu,
        idx_mapping.to(device),
        query_start_loc.to(device),
        all_token_ids.to(device),
        all_token_ids.stride(0),
        prefill_lens.to(device),
        num_computed_tokens.to(device),
        BLOCK_SIZE=1024,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(input_ids_npu.cpu(), expected_input_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        next_prefill_tokens_npu.cpu(), expected_next, rtol=0, atol=0
    )
