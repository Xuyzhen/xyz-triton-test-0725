# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.gpu.input_batch import (
    _combine_sampled_and_draft_tokens_kernel,
)

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _combine_sampled_and_draft_tokens_cpu(
    input_ids: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    prefill_len: torch.Tensor,
    draft_tokens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    logits_indices: torch.Tensor,
    num_new_sampled_tokens: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch CPU reference.

    For each request:
    1. Computes logits_indices = query_end - num_logits + offset
    2. If seq_len > prefill_len (post-prefill), writes the last sampled
       token at logits_start if NUM_NEW_SAMPLED_TOKENS > 0.
    3. Writes draft tokens (if any) at query_end - num_draft + offset.
    """
    output_input_ids = input_ids.clone()
    output_logits_indices = torch.empty_like(logits_indices)

    num_reqs = idx_mapping.shape[0]

    for batch_idx in range(num_reqs):
        req_state_idx = int(idx_mapping[batch_idx])

        cu_start = int(cu_num_logits[batch_idx])
        cu_end = int(cu_num_logits[batch_idx + 1])
        num_logits = cu_end - cu_start
        num_draft_tokens = num_logits - num_new_sampled_tokens

        query_end = int(query_start_loc[batch_idx + 1])
        logits_start = query_end - num_logits

        # Logits indices
        for j in range(num_logits):
            output_logits_indices[cu_start + j] = logits_start + j

        seq_len = int(seq_lens[batch_idx])
        prefill_len_val = int(prefill_len[req_state_idx])
        if seq_len <= prefill_len_val:
            continue

        # Write last sampled token
        first_logit_seq_pos = seq_len - num_logits
        if num_new_sampled_tokens > 0 and first_logit_seq_pos >= prefill_len_val:
            output_input_ids[logits_start] = int(last_sampled_tokens[req_state_idx])

        # Write draft tokens
        if num_draft_tokens > 0:
            for j in range(num_draft_tokens):
                output_input_ids[query_end - num_draft_tokens + j] = int(
                    draft_tokens[req_state_idx, j]
                )

    return output_input_ids, output_logits_indices


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 0 else 1


@pytest.mark.parametrize("num_reqs", [1, 4, 8])
@pytest.mark.parametrize("num_spec_steps", [0, 3, 5])
def test_combine_sampled_and_draft_tokens_basic(
    num_reqs: int, num_spec_steps: int,
) -> None:
    """Combine sampled tokens and draft tokens into input_ids.

    Tests with a mix of prefill and decode requests, some with draft tokens
    and some without. Verifies logits_indices are computed correctly.
    """
    init_device_properties_triton()
    torch.manual_seed(42)

    max_num_reqs = 16
    max_model_len = 512
    num_new_sampled_tokens = 1
    num_tokens = 128  # total output tokens length

    input_ids = torch.zeros(num_tokens, dtype=torch.int64)
    idx_mapping = torch.randint(0, max_num_reqs, (num_reqs,), dtype=torch.int32)

    last_sampled_tokens = torch.randint(0, 32000, (max_num_reqs,), dtype=torch.int64)

    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int64)
    seq_lens = torch.zeros(num_reqs, dtype=torch.int32)
    prefill_len = torch.randint(10, 50, (max_num_reqs,), dtype=torch.int32)

    cu_num_logits = torch.zeros(num_reqs + 1, dtype=torch.int64)

    draft_tokens = torch.zeros(
        (max_num_reqs, max(num_spec_steps, 1)), dtype=torch.int64
    )

    current_seq_start = 0
    for b in range(num_reqs):
        req_state_idx = int(idx_mapping[b])
        p_len = int(prefill_len[req_state_idx])

        # Some requests are post-prefill (seq_len > prefill_len), some are prefill
        if b % 2 == 0:
            sl = p_len + 2 + b % 5  # post-prefill
        else:
            sl = p_len - 1 if p_len > 1 else p_len  # still in prefill
        seq_lens[b] = sl

        query_len = num_new_sampled_tokens + num_spec_steps
        query_start_loc[b] = current_seq_start
        current_seq_start += query_len
        cu_num_logits[b] = b * (num_new_sampled_tokens + num_spec_steps)

    query_start_loc[num_reqs] = current_seq_start
    cu_num_logits[num_reqs] = num_reqs * (num_new_sampled_tokens + num_spec_steps)

    # Set draft tokens
    for r in range(max_num_reqs):
        for s in range(num_spec_steps):
            draft_tokens[r, s] = 10000 + r * 100 + s

    logits_indices = torch.empty(
        int(cu_num_logits[-1]), dtype=torch.int64,
    )

    expected_input_ids, expected_logits_indices = (
        _combine_sampled_and_draft_tokens_cpu(
            input_ids, idx_mapping, last_sampled_tokens,
            query_start_loc, seq_lens, prefill_len,
            draft_tokens, cu_num_logits, logits_indices,
            num_new_sampled_tokens,
        )
    )

    device = torch.device("npu")
    input_ids_npu = input_ids.to(device)
    logits_indices_npu = logits_indices.to(device)

    BLOCK_SIZE = _next_power_of_2(num_spec_steps + num_new_sampled_tokens)

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
        logits_indices_npu,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_NEW_SAMPLED_TOKENS=num_new_sampled_tokens,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        input_ids_npu.cpu(), expected_input_ids, rtol=0, atol=0,
    )
    torch.testing.assert_close(
        logits_indices_npu.cpu(), expected_logits_indices, rtol=0, atol=0,
    )


def test_combine_sampled_and_draft_tokens_no_draft() -> None:
    """No draft tokens (num_spec_steps = 0)."""
    init_device_properties_triton()
    torch.manual_seed(7)

    num_reqs = 3
    num_spec_steps = 0
    num_new_sampled_tokens = 1

    input_ids = torch.zeros(6, dtype=torch.int64)
    idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int32)

    last_sampled_tokens = torch.tensor([999, 888, 777], dtype=torch.int64)

    # All requests post-prefill
    query_start_loc = torch.tensor([0, 2, 4, 6], dtype=torch.int64)
    seq_lens = torch.tensor([20, 15, 25], dtype=torch.int32)
    prefill_len = torch.tensor([10, 10, 10], dtype=torch.int32)
    draft_tokens = torch.zeros((3, 1), dtype=torch.int64)
    cu_num_logits = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    logits_indices = torch.empty(3, dtype=torch.int64)

    expected_inp, expected_li = _combine_sampled_and_draft_tokens_cpu(
        input_ids, idx_mapping, last_sampled_tokens,
        query_start_loc, seq_lens, prefill_len,
        draft_tokens, cu_num_logits, logits_indices,
    )

    device = torch.device("npu")
    input_ids_npu = input_ids.to(device)
    logits_indices_npu = logits_indices.to(device)
    BLOCK_SIZE = _next_power_of_2(1)

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
        logits_indices_npu,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_NEW_SAMPLED_TOKENS=1,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(input_ids_npu.cpu(), expected_inp, rtol=0, atol=0)
    torch.testing.assert_close(logits_indices_npu.cpu(), expected_li, rtol=0, atol=0)


def test_combine_sampled_and_draft_tokens_all_prefill() -> None:
    """All requests are still in prefill phase (seq_len <= prefill_len).

    The kernel should only compute logits_indices and leave input_ids unchanged.
    """
    init_device_properties_triton()
    torch.manual_seed(3)

    num_reqs = 2
    input_ids = torch.zeros(10, dtype=torch.int64)
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    last_sampled_tokens = torch.tensor([555, 666], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 3, 10], dtype=torch.int64)
    seq_lens = torch.tensor([5, 5], dtype=torch.int32)
    prefill_len = torch.tensor([10, 10], dtype=torch.int32)
    draft_tokens = torch.zeros((2, 2), dtype=torch.int64)
    cu_num_logits = torch.tensor([0, 2, 7], dtype=torch.int64)

    logits_indices = torch.empty(7, dtype=torch.int64)

    expected_inp, expected_li = _combine_sampled_and_draft_tokens_cpu(
        input_ids, idx_mapping, last_sampled_tokens,
        query_start_loc, seq_lens, prefill_len,
        draft_tokens, cu_num_logits, logits_indices,
    )

    device = torch.device("npu")
    input_ids_npu = input_ids.to(device)
    logits_indices_npu = logits_indices.to(device)
    BLOCK_SIZE = _next_power_of_2(2 + 1)

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
        logits_indices_npu,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_NEW_SAMPLED_TOKENS=1,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(input_ids_npu.cpu(), expected_inp, rtol=0, atol=0)
    torch.testing.assert_close(logits_indices_npu.cpu(), expected_li, rtol=0, atol=0)
