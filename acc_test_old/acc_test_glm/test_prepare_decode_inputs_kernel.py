import pytest
import torch


def _prepare_decode_inputs_cpu(
    draft_tokens: torch.Tensor,
    target_seq_lens: torch.Tensor,
    num_rejected: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_model_len: int,
    max_num_reqs: int,
    advance_draft_positions: bool,
):
    num_reqs = draft_tokens.shape[0]
    for req_idx in range(num_reqs):
        draft_token = int(draft_tokens[req_idx, 0])
        input_ids[req_idx] = draft_token

        if advance_draft_positions:
            position = int(positions[req_idx])
            position = min(position + 1, max_model_len - 1)
            positions[req_idx] = position

            target_seq_len = int(target_seq_lens[req_idx])
            nr = int(num_rejected[req_idx])
            seq_len = target_seq_len - nr
            seq_len = min(seq_len + 1, max_model_len)
            seq_lens[req_idx] = seq_len

    for i in range(num_reqs, max_num_reqs + 1):
        q = min(i, num_reqs)
        query_start_loc[i] = q

    for i in range(num_reqs, max_num_reqs):
        seq_lens[i] = 0


def test_prepare_decode_inputs_kernel():
    torch.manual_seed(42)
    num_reqs = 3
    max_num_reqs = 5
    max_model_len = 1024

    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
        _prepare_decode_inputs_kernel,
    )

    draft_tokens = torch.tensor([[100], [200], [300]], dtype=torch.int32)
    target_seq_lens = torch.tensor([10, 15, 20], dtype=torch.int32)
    num_rejected = torch.tensor([0, 1, 2], dtype=torch.int32)
    input_ids = torch.zeros(num_reqs, dtype=torch.int32)
    positions = torch.tensor([9, 14, 19], dtype=torch.int64)
    query_start_loc = torch.zeros(max_num_reqs + 1, dtype=torch.int32)
    seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32)

    expected_input_ids = input_ids.clone()
    expected_positions = positions.clone()
    expected_query_start_loc = query_start_loc.clone()
    expected_seq_lens = seq_lens.clone()

    _prepare_decode_inputs_cpu(
        draft_tokens,
        target_seq_lens,
        num_rejected,
        expected_input_ids,
        expected_positions,
        expected_query_start_loc,
        expected_seq_lens,
        max_model_len,
        max_num_reqs,
        advance_draft_positions=True,
    )

    device = torch.device("npu")
    input_ids_npu = input_ids.to(device)
    positions_npu = positions.to(device)
    query_start_loc_npu = query_start_loc.to(device)
    seq_lens_npu = seq_lens.to(device)

    _prepare_decode_inputs_kernel[(num_reqs + 1,)](
        draft_tokens.to(device),
        draft_tokens.stride(0),
        target_seq_lens.to(device),
        num_rejected.to(device),
        input_ids_npu,
        positions_npu,
        query_start_loc_npu,
        seq_lens_npu,
        max_model_len,
        max_num_reqs,
        BLOCK_SIZE=1024,
        ADVANCE_DRAFT_POSITIONS=True,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(input_ids_npu.cpu(), expected_input_ids, rtol=0, atol=0)
    torch.testing.assert_close(positions_npu.cpu(), expected_positions, rtol=0, atol=0)
    torch.testing.assert_close(
        seq_lens_npu[:num_reqs].cpu(), expected_seq_lens[:num_reqs], rtol=0, atol=0
    )


def test_prepare_decode_inputs_kernel_no_advance():
    num_reqs = 2
    max_num_reqs = 3
    max_model_len = 512

    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
        _prepare_decode_inputs_kernel,
    )

    draft_tokens = torch.tensor([[50], [60]], dtype=torch.int32)
    target_seq_lens = torch.tensor([10, 20], dtype=torch.int32)
    num_rejected = torch.tensor([0, 0], dtype=torch.int32)
    input_ids = torch.zeros(num_reqs, dtype=torch.int32)
    positions = torch.tensor([5, 10], dtype=torch.int64)
    query_start_loc = torch.zeros(max_num_reqs + 1, dtype=torch.int32)
    seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32)

    expected_input_ids = input_ids.clone()
    expected_positions = positions.clone()
    expected_seq_lens = seq_lens.clone()
    expected_query_start_loc = query_start_loc.clone()

    _prepare_decode_inputs_cpu(
        draft_tokens,
        target_seq_lens,
        num_rejected,
        expected_input_ids,
        expected_positions,
        expected_query_start_loc,
        expected_seq_lens,
        max_model_len,
        max_num_reqs,
        advance_draft_positions=False,
    )

    device = torch.device("npu")
    input_ids_npu = input_ids.to(device)
    positions_npu = positions.to(device)
    query_start_loc_npu = query_start_loc.to(device)
    seq_lens_npu = seq_lens.to(device)

    _prepare_decode_inputs_kernel[(num_reqs + 1,)](
        draft_tokens.to(device),
        draft_tokens.stride(0),
        target_seq_lens.to(device),
        num_rejected.to(device),
        input_ids_npu,
        positions_npu,
        query_start_loc_npu,
        seq_lens_npu,
        max_model_len,
        max_num_reqs,
        BLOCK_SIZE=1024,
        ADVANCE_DRAFT_POSITIONS=False,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(input_ids_npu.cpu(), expected_input_ids, rtol=0, atol=0)
    torch.testing.assert_close(positions_npu.cpu(), expected_positions, rtol=0, atol=0)
