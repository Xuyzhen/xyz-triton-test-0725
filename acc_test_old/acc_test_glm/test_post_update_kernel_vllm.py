import pytest
import torch

from vllm.v1.worker.gpu.input_batch import _post_update_kernel


def _post_update_cpu(
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    output_bin_counts: torch.Tensor | None,
    sampled_tokens: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
):
    num_reqs = idx_mapping.shape[0]
    for req_id in range(num_reqs):
        req_state_idx = int(idx_mapping[req_id])
        if req_state_idx < 0:
            continue

        tl = int(total_len[req_state_idx])
        ns = int(num_sampled[req_id])
        if ns > 0:
            token_id = int(sampled_tokens[req_id, ns - 1])
            last_sampled_tokens[req_state_idx] = token_id
            total_len[req_state_idx] = tl + ns

        for i in range(ns):
            token_id = int(sampled_tokens[req_id, i])
            all_token_ids[req_state_idx, tl + i] = token_id

            if output_bin_counts is not None:
                output_bin_counts[req_state_idx, token_id] += 1

        if query_start_loc is not None:
            qs = int(query_start_loc[req_id])
            qe = int(query_start_loc[req_id + 1])
            query_len = qe - qs
        else:
            query_len = 0
        nr = int(num_rejected[req_id])
        computed_delta = query_len - nr
        if computed_delta != 0:
            num_computed_tokens[req_state_idx] += computed_delta


def test_post_update_kernel():
    torch.manual_seed(42)
    num_reqs = 3
    max_num_reqs = 4
    max_model_len = 32
    vocab_size = 64
    num_spec_steps = 3

    idx_mapping = torch.tensor([0, 2, -1], dtype=torch.int32)
    num_computed_tokens = torch.tensor([5, 10, 3, 8], dtype=torch.int32)
    last_sampled_tokens = torch.zeros(max_num_reqs, dtype=torch.int32)
    output_bin_counts = torch.zeros(max_num_reqs, vocab_size, dtype=torch.int32)
    sampled_tokens = torch.tensor(
        [[10, 20, 30], [40, 50, 60], [0, 0, 0]], dtype=torch.int32
    )
    num_sampled = torch.tensor([2, 3, 0], dtype=torch.int32)
    num_rejected = torch.tensor([0, 1, 0], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 3, 6], dtype=torch.int32)
    all_token_ids = torch.zeros(max_num_reqs, max_model_len, dtype=torch.int32)
    total_len = torch.tensor([5, 10, 3, 8], dtype=torch.int32)

    expected_num_computed = num_computed_tokens.clone()
    expected_last_sampled = last_sampled_tokens.clone()
    expected_output_bin_counts = output_bin_counts.clone()
    expected_all_token_ids = all_token_ids.clone()
    expected_total_len = total_len.clone()

    _post_update_cpu(
        idx_mapping,
        expected_num_computed,
        expected_last_sampled,
        expected_output_bin_counts,
        sampled_tokens,
        num_sampled,
        num_rejected,
        query_start_loc,
        expected_all_token_ids,
        expected_total_len,
    )

    device = torch.device("npu")
    num_computed_tokens_npu = num_computed_tokens.to(device)
    last_sampled_tokens_npu = last_sampled_tokens.to(device)
    output_bin_counts_npu = output_bin_counts.to(device)
    all_token_ids_npu = all_token_ids.to(device)
    total_len_npu = total_len.to(device)

    _post_update_kernel[(num_reqs,)](
        idx_mapping.to(device),
        num_computed_tokens_npu,
        last_sampled_tokens_npu,
        output_bin_counts_npu,
        output_bin_counts_npu.stride(0),
        sampled_tokens.to(device),
        sampled_tokens.stride(0),
        num_sampled.to(device),
        num_rejected.to(device),
        query_start_loc.to(device),
        all_token_ids_npu,
        all_token_ids_npu.stride(0),
        total_len_npu,
        num_warps=1,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        num_computed_tokens_npu.cpu(), expected_num_computed, rtol=0, atol=0
    )
    torch.testing.assert_close(
        last_sampled_tokens_npu.cpu(), expected_last_sampled, rtol=0, atol=0
    )
    torch.testing.assert_close(
        output_bin_counts_npu.cpu(), expected_output_bin_counts, rtol=0, atol=0
    )
    torch.testing.assert_close(
        all_token_ids_npu.cpu(), expected_all_token_ids, rtol=0, atol=0
    )
    torch.testing.assert_close(
        total_len_npu.cpu(), expected_total_len, rtol=0, atol=0
    )


def test_post_update_kernel_no_output_bin_counts():
    num_reqs = 2
    max_num_reqs = 2
    max_model_len = 16

    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    num_computed_tokens = torch.tensor([3, 7], dtype=torch.int32)
    last_sampled_tokens = torch.zeros(max_num_reqs, dtype=torch.int32)
    sampled_tokens = torch.tensor([[5], [8]], dtype=torch.int32)
    num_sampled = torch.tensor([1, 1], dtype=torch.int32)
    num_rejected = torch.tensor([0, 0], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    all_token_ids = torch.zeros(max_num_reqs, max_model_len, dtype=torch.int32)
    total_len = torch.tensor([3, 7], dtype=torch.int32)

    expected_num_computed = num_computed_tokens.clone()
    expected_last_sampled = last_sampled_tokens.clone()
    expected_all_token_ids = all_token_ids.clone()
    expected_total_len = total_len.clone()

    _post_update_cpu(
        idx_mapping,
        expected_num_computed,
        expected_last_sampled,
        None,
        sampled_tokens,
        num_sampled,
        num_rejected,
        query_start_loc,
        expected_all_token_ids,
        expected_total_len,
    )

    device = torch.device("npu")
    num_computed_tokens_npu = num_computed_tokens.to(device)
    last_sampled_tokens_npu = last_sampled_tokens.to(device)
    all_token_ids_npu = all_token_ids.to(device)
    total_len_npu = total_len.to(device)

    _post_update_kernel[(num_reqs,)](
        idx_mapping.to(device),
        num_computed_tokens_npu,
        last_sampled_tokens_npu,
        None,
        0,
        sampled_tokens.to(device),
        sampled_tokens.stride(0),
        num_sampled.to(device),
        num_rejected.to(device),
        query_start_loc.to(device),
        all_token_ids_npu,
        all_token_ids_npu.stride(0),
        total_len_npu,
        num_warps=1,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        num_computed_tokens_npu.cpu(), expected_num_computed, rtol=0, atol=0
    )
    torch.testing.assert_close(
        last_sampled_tokens_npu.cpu(), expected_last_sampled, rtol=0, atol=0
    )
    torch.testing.assert_close(
        all_token_ids_npu.cpu(), expected_all_token_ids, rtol=0, atol=0
    )
    torch.testing.assert_close(
        total_len_npu.cpu(), expected_total_len, rtol=0, atol=0
    )
