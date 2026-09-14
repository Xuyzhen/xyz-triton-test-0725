import pytest
import torch

from vllm.v1.worker.gpu.input_batch import _combine_sampled_and_draft_tokens_kernel


def _combine_sampled_and_draft_tokens_cpu(
    input_ids: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    prefill_len: torch.Tensor,
    draft_tokens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_logits: int,
    num_new_sampled_tokens: int,
):
    num_reqs = idx_mapping.shape[0]
    logits_indices = torch.empty(num_logits, dtype=torch.int64)

    for batch_idx in range(num_reqs):
        req_state_idx = int(idx_mapping[batch_idx])

        cu_start = int(cu_num_logits[batch_idx])
        cu_end = int(cu_num_logits[batch_idx + 1])
        n_logits = cu_end - cu_start
        n_draft = n_logits - num_new_sampled_tokens

        query_end = int(query_start_loc[batch_idx + 1])
        logits_start = query_end - n_logits

        for i in range(n_logits):
            logits_indices[cu_start + i] = logits_start + i

        seq_len = int(seq_lens[batch_idx])
        p_len = int(prefill_len[req_state_idx])
        if seq_len <= p_len:
            continue

        if num_new_sampled_tokens > 0:
            last_token = int(last_sampled_tokens[req_state_idx])
            input_ids[query_end - n_logits] = last_token

        if n_draft > 0:
            for i in range(n_draft):
                input_ids[query_end - n_draft + i] = int(
                    draft_tokens[req_state_idx, i]
                )

    return logits_indices


def test_combine_sampled_and_draft_tokens_kernel():
    torch.manual_seed(42)
    num_reqs = 2
    num_speculative_steps = 3
    vocab_size = 64
    max_model_len = 32

    num_tokens = 10
    input_ids = torch.zeros(num_tokens, dtype=torch.int32)

    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    query_lens = torch.tensor([5, 5], dtype=torch.int32)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    query_start_loc[1:] = query_lens.cumsum(dim=0)

    seq_lens = torch.tensor([10, 15], dtype=torch.int32)
    prefill_len = torch.tensor([3, 5], dtype=torch.int32)
    last_sampled_tokens = torch.tensor([42, 99], dtype=torch.int32)
    draft_tokens = torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.int32)

    cu_num_logits = torch.tensor([0, 4, 8], dtype=torch.int32)
    num_logits = 8

    expected_input_ids = input_ids.clone()
    expected_logits_indices = _combine_sampled_and_draft_tokens_cpu(
        known_input_ids := input_ids.clone(),
        idx_mapping,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        cu_num_logits,
        num_logits,
        num_new_sampled_tokens=1,
    )
    expected_input_ids = known_input_ids

    device = torch.device("npu")
    input_ids_npu = input_ids.to(device)
    from vllm.triton_utils import triton

    logits_indices = torch.empty(num_logits, dtype=torch.int64, device=device)

    _combine_sampled_and_draft_tokens_kernel[(num_reqs,)](
        input_ids_npu,
        idx_mapping.to(device),
        last_sampled_tokens.to(device),
        query_start_loc.to(device),
        seq_lens.to(device),
        prefill_len.to(device),
        draft_tokens.to(device),
        draft_tokens.stride(0),
        cu_num_logits.to(device),
        logits_indices,
        NUM_NEW_SAMPLED_TOKENS=1,
        BLOCK_SIZE=triton.next_power_of_2(num_speculative_steps + 1),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        input_ids_npu.cpu(), expected_input_ids, rtol=0, atol=0
    )
    torch.testing.assert_close(
        logits_indices.cpu(), expected_logits_indices, rtol=0, atol=0
    )
