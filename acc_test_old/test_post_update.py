# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.gpu.input_batch import _post_update_kernel

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """Pure PyTorch CPU reference for post_update_kernel.

    For each request where num_sampled > 0:
    1. Updates last_sampled_tokens with the last sampled token
    2. Extends total_len by num_sampled
    3. Appends sampled tokens to all_token_ids at position total_len + i
    4. Increments output_bin_counts at the sampled token positions
    Then updates num_computed_tokens by computed_delta = query_len - num_rejected.
    """
    nc = num_computed_tokens.clone()
    lst = last_sampled_tokens.clone()
    tl = total_len.clone()
    abc = output_bin_counts.clone() if output_bin_counts is not None else None
    at = all_token_ids.clone()

    num_reqs = idx_mapping.shape[0]

    for req_id in range(num_reqs):
        req_state_idx = int(idx_mapping[req_id])
        if req_state_idx < 0:
            continue

        total_len_val = int(tl[req_state_idx])
        ns = int(num_sampled[req_id])

        if ns > 0:
            token_id = int(sampled_tokens[req_id, ns - 1])
            lst[req_state_idx] = token_id
            tl[req_state_idx] = total_len_val + ns

        for i in range(ns):
            token_id = int(sampled_tokens[req_id, i])
            at[req_state_idx, total_len_val + i] = token_id

            if abc is not None:
                abc[req_state_idx, token_id] += 1

        if query_start_loc is not None:
            qs = int(query_start_loc[req_id])
            qe = int(query_start_loc[req_id + 1])
            query_len = qe - qs
        else:
            query_len = 0

        nr = int(num_rejected[req_id])
        computed_delta = query_len - nr
        if computed_delta != 0:
            nc[req_state_idx] += computed_delta

    return nc, lst, tl, at, abc


@pytest.mark.parametrize("num_reqs", [1, 4, 8])
@pytest.mark.parametrize("num_spec_steps", [0, 3, 5])
def test_post_update_basic(num_reqs: int, num_spec_steps: int) -> None:
    """Post-update kernel updates state after sampling.

    Tests with varying numbers of sampled tokens, some with rejections,
    and verifies that all state tensors are correctly updated.
    """
    init_device_properties_triton()
    torch.manual_seed(42)

    max_num_reqs = 16
    max_model_len = 256
    vocab_size = 32000

    idx_mapping = torch.randint(0, max_num_reqs, (num_reqs,), dtype=torch.int32)

    num_computed_tokens_in = torch.randint(0, 50, (max_num_reqs,), dtype=torch.int64)
    last_sampled_tokens_in = torch.randint(0, vocab_size, (max_num_reqs,),
                                           dtype=torch.int64)
    total_len_in = torch.randint(10, 100, (max_num_reqs,), dtype=torch.int64)
    all_token_ids_in = torch.randint(
        0, vocab_size, (max_num_reqs, max_model_len), dtype=torch.int64,
    )

    sampled_tokens = torch.randint(
        0, vocab_size, (num_reqs, num_spec_steps + 1), dtype=torch.int64,
    )
    num_sampled = torch.zeros(num_reqs, dtype=torch.int32)
    num_rejected = torch.zeros(num_reqs, dtype=torch.int32)

    query_lens = torch.randint(0, 10, (num_reqs,), dtype=torch.int64)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int64)
    for i in range(num_reqs):
        query_start_loc[i + 1] = query_start_loc[i] + int(query_lens[i])

    output_bin_counts_in = torch.zeros(
        (max_num_reqs, vocab_size), dtype=torch.int64,
    )

    for b in range(num_reqs):
        n_gen = 1 + b % (num_spec_steps + 1) if num_spec_steps > 0 else 1
        num_sampled[b] = n_gen
        num_rejected[b] = (num_spec_steps + 1) - n_gen

    # CPU reference
    expected_nc, expected_lst, expected_tl, expected_at, expected_abc = (
        _post_update_cpu(
            idx_mapping, num_computed_tokens_in, last_sampled_tokens_in,
            output_bin_counts_in, sampled_tokens, num_sampled, num_rejected,
            query_start_loc, all_token_ids_in, total_len_in,
        )
    )

    # NPU kernel
    device = torch.device("npu")
    nc_npu = num_computed_tokens_in.to(device)
    lst_npu = last_sampled_tokens_in.to(device)
    tl_npu = total_len_in.to(device)
    at_npu = all_token_ids_in.to(device)
    abc_npu = output_bin_counts_in.to(device)

    _post_update_kernel[(num_reqs,)](
        idx_mapping.to(device),
        nc_npu,
        lst_npu,
        abc_npu,
        abc_npu.stride(0),
        sampled_tokens.to(device),
        sampled_tokens.stride(0),
        num_sampled.to(device),
        num_rejected.to(device),
        query_start_loc.to(device),
        at_npu,
        at_npu.stride(0),
        tl_npu,
        num_warps=1,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(nc_npu.cpu(), expected_nc, rtol=0, atol=0)
    torch.testing.assert_close(lst_npu.cpu(), expected_lst, rtol=0, atol=0)
    torch.testing.assert_close(tl_npu.cpu(), expected_tl, rtol=0, atol=0)
    torch.testing.assert_close(at_npu.cpu(), expected_at, rtol=0, atol=0)
    torch.testing.assert_close(abc_npu.cpu(), expected_abc, rtol=0, atol=0)


def test_post_update_no_sampled() -> None:
    """No sampled tokens for any request -- only num_computed_tokens may change."""
    init_device_properties_triton()
    torch.manual_seed(7)

    num_reqs = 3
    max_num_reqs = 4
    max_model_len = 64
    vocab_size = 32000

    idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int32)
    nc_in = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
    lst_in = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    tl_in = torch.tensor([50, 60, 70, 80], dtype=torch.int64)
    at_in = torch.randint(0, vocab_size, (max_num_reqs, max_model_len),
                          dtype=torch.int64)
    sampled_tokens = torch.randint(0, vocab_size, (num_reqs, 4), dtype=torch.int64)
    num_sampled = torch.zeros(num_reqs, dtype=torch.int32)
    num_rejected = torch.tensor([2, 0, 3], dtype=torch.int32)
    qsl = torch.tensor([0, 3, 3, 7], dtype=torch.int64)
    abc_in = torch.zeros((max_num_reqs, vocab_size), dtype=torch.int64)

    expected_nc, expected_lst, expected_tl, expected_at, expected_abc = (
        _post_update_cpu(
            idx_mapping, nc_in, lst_in, abc_in, sampled_tokens,
            num_sampled, num_rejected, qsl, at_in, tl_in,
        )
    )

    device = torch.device("npu")
    nc_npu = nc_in.to(device)
    lst_npu = lst_in.to(device)
    tl_npu = tl_in.to(device)
    at_npu = at_in.to(device)
    abc_npu = abc_in.to(device)

    _post_update_kernel[(num_reqs,)](
        idx_mapping.to(device),
        nc_npu,
        lst_npu,
        abc_npu,
        abc_npu.stride(0),
        sampled_tokens.to(device),
        sampled_tokens.stride(0),
        num_sampled.to(device),
        num_rejected.to(device),
        qsl.to(device),
        at_npu,
        at_npu.stride(0),
        tl_npu,
        num_warps=1,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(nc_npu.cpu(), expected_nc, rtol=0, atol=0)
    torch.testing.assert_close(lst_npu.cpu(), expected_lst, rtol=0, atol=0)
    torch.testing.assert_close(tl_npu.cpu(), expected_tl, rtol=0, atol=0)
    torch.testing.assert_close(at_npu.cpu(), expected_at, rtol=0, atol=0)
    torch.testing.assert_close(abc_npu.cpu(), expected_abc, rtol=0, atol=0)


def test_post_update_negative_idx_mapping() -> None:
    """Requests with negative idx_mapping entries should be skipped."""
    init_device_properties_triton()
    torch.manual_seed(3)

    num_reqs = 4
    max_num_reqs = 3
    max_model_len = 32
    vocab_size = 32000

    idx_mapping = torch.tensor([0, -1, 1, -1], dtype=torch.int32)
    nc_in = torch.tensor([100, 200, 300], dtype=torch.int64)
    lst_in = torch.zeros(max_num_reqs, dtype=torch.int64)
    tl_in = torch.tensor([10, 20, 30], dtype=torch.int64)
    at_in = torch.zeros((max_num_reqs, max_model_len), dtype=torch.int64)
    sampled_tokens = torch.randint(0, vocab_size, (num_reqs, 3), dtype=torch.int64)
    num_sampled = torch.tensor([2, 1, 0, 3], dtype=torch.int32)
    num_rejected = torch.tensor([1, 2, 3, 0], dtype=torch.int32)
    qsl = torch.tensor([0, 3, 5, 5, 8], dtype=torch.int64)

    expected_nc, expected_lst, expected_tl, expected_at, _ = _post_update_cpu(
        idx_mapping, nc_in, lst_in, None, sampled_tokens,
        num_sampled, num_rejected, qsl, at_in, tl_in,
    )

    device = torch.device("npu")
    nc_npu = nc_in.to(device)
    lst_npu = lst_in.to(device)
    tl_npu = tl_in.to(device)
    at_npu = at_in.to(device)

    _post_update_kernel[(num_reqs,)](
        idx_mapping.to(device),
        nc_npu,
        lst_npu,
        None,
        0,
        sampled_tokens.to(device),
        sampled_tokens.stride(0),
        num_sampled.to(device),
        num_rejected.to(device),
        qsl.to(device),
        at_npu,
        at_npu.stride(0),
        tl_npu,
        num_warps=1,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(nc_npu.cpu(), expected_nc, rtol=0, atol=0)
    torch.testing.assert_close(lst_npu.cpu(), expected_lst, rtol=0, atol=0)
    torch.testing.assert_close(tl_npu.cpu(), expected_tl, rtol=0, atol=0)
    torch.testing.assert_close(at_npu.cpu(), expected_at, rtol=0, atol=0)
