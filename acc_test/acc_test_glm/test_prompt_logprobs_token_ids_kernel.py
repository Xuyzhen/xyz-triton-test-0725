import pytest
import torch

from vllm.v1.worker.gpu.sample.prompt_logprob import _prompt_logprobs_token_ids_kernel


def _prompt_logprobs_token_ids_cpu(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    all_token_ids: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty(num_tokens, dtype=torch.int64)
    num_reqs = idx_mapping.shape[0]
    for batch_idx in range(num_reqs):
        req_state_idx = int(idx_mapping[batch_idx])
        query_start = int(query_start_loc[batch_idx])
        query_end = int(query_start_loc[batch_idx + 1])
        query_len = query_end - query_start
        num_computed = int(num_computed_tokens[req_state_idx])

        for i in range(query_len):
            target_pos = num_computed + 1 + i
            output[query_start + i] = int(all_token_ids[req_state_idx, target_pos])

    return output


def test_prompt_logprobs_token_ids_kernel():
    torch.manual_seed(42)
    max_num_reqs = 3
    max_model_len = 32

    query_lens = torch.tensor([4, 2, 3], dtype=torch.int32)
    query_start_loc = torch.zeros(max_num_reqs + 1, dtype=torch.int32)
    query_start_loc[1:] = query_lens.cumsum(dim=0)
    num_tokens = int(query_start_loc[-1])

    idx_mapping = torch.tensor([2, 0, 1], dtype=torch.int32)
    num_computed_tokens = torch.tensor([5, 3, 10], dtype=torch.int32)

    all_token_ids = torch.arange(
        max_num_reqs * max_model_len, dtype=torch.int32
    ).reshape(max_num_reqs, max_model_len)

    expected = _prompt_logprobs_token_ids_cpu(
        num_tokens,
        query_start_loc,
        idx_mapping,
        num_computed_tokens,
        all_token_ids,
    )

    device = torch.device("npu")
    token_ids = torch.empty(num_tokens, dtype=torch.int64, device=device)

    _prompt_logprobs_token_ids_kernel[(max_num_reqs,)](
        token_ids,
        query_start_loc.to(device),
        idx_mapping.to(device),
        num_computed_tokens.to(device),
        all_token_ids.to(device),
        all_token_ids.stride(0),
        BLOCK_SIZE=1024,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(token_ids.cpu(), expected, rtol=0, atol=0)


def test_prompt_logprobs_token_ids_kernel_single_request():
    torch.manual_seed(42)
    max_model_len = 16
    query_lens = torch.tensor([3], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    idx_mapping = torch.tensor([0], dtype=torch.int32)
    num_computed_tokens = torch.tensor([2], dtype=torch.int32)
    all_token_ids = torch.arange(max_model_len, dtype=torch.int32).unsqueeze(0)

    num_tokens = 3
    expected = _prompt_logprobs_token_ids_cpu(
        num_tokens, query_start_loc, idx_mapping, num_computed_tokens, all_token_ids
    )

    device = torch.device("npu")
    token_ids = torch.empty(num_tokens, dtype=torch.int64, device=device)

    _prompt_logprobs_token_ids_kernel[(1,)](
        token_ids,
        query_start_loc.to(device),
        idx_mapping.to(device),
        num_computed_tokens.to(device),
        all_token_ids.to(device),
        all_token_ids.stride(0),
        BLOCK_SIZE=1024,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(token_ids.cpu(), expected, rtol=0, atol=0)
