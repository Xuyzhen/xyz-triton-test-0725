import pytest
import torch

from vllm.v1.worker.gpu.sample.logprob import _fill_logprob_token_ids_kernel


def _fill_logprob_token_ids_cpu(
    batch_size: int,
    sampled_token_ids: torch.Tensor,
    topk_indices: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    num_per_req_token_ids: torch.Tensor,
    per_req_token_ids: torch.Tensor,
    num_topk: int,
    num_cols: int,
):
    out_token_ids = torch.zeros(batch_size, 1 + num_cols, dtype=torch.int64)
    out_valid_mask = torch.zeros(batch_size, 1 + num_cols, dtype=torch.bool)

    for batch_idx in range(batch_size):
        out_token_ids[batch_idx, 0] = int(sampled_token_ids[batch_idx])
        out_valid_mask[batch_idx, 0] = True

        req_state_idx = int(expanded_idx_mapping[batch_idx])
        num_custom = int(num_per_req_token_ids[req_state_idx])

        if num_custom > 0:
            for col in range(num_custom):
                out_token_ids[batch_idx, 1 + col] = int(
                    per_req_token_ids[req_state_idx, col]
                )
                out_valid_mask[batch_idx, 1 + col] = True
        else:
            for col in range(num_topk):
                out_token_ids[batch_idx, 1 + col] = int(
                    topk_indices[batch_idx, col]
                )
                out_valid_mask[batch_idx, 1 + col] = True

    return out_token_ids, out_valid_mask


def test_fill_logprob_token_ids_kernel_topk_only():
    batch_size = 4
    vocab_size = 64
    num_topk = 3
    num_cols = num_topk
    torch.manual_seed(42)

    sampled_token_ids = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
    topk_indices = torch.randint(0, vocab_size, (batch_size, num_topk), dtype=torch.int32)
    expanded_idx_mapping = torch.arange(batch_size, dtype=torch.int32)
    num_per_req_token_ids = torch.zeros(batch_size, dtype=torch.int32)
    per_req_token_ids = torch.zeros(batch_size, 64, dtype=torch.int32)

    expected_ids, expected_mask = _fill_logprob_token_ids_cpu(
        batch_size,
        sampled_token_ids,
        topk_indices,
        expanded_idx_mapping,
        num_per_req_token_ids,
        per_req_token_ids,
        num_topk,
        num_cols,
    )

    device = torch.device("npu")
    logprob_token_ids = torch.zeros(
        batch_size, 1 + num_cols, dtype=torch.int64, device=device
    )
    valid_mask = torch.zeros(
        batch_size, 1 + num_cols, dtype=torch.bool, device=device
    )

    from vllm.triton_utils import triton

    _fill_logprob_token_ids_kernel[(batch_size,)](
        logprob_token_ids,
        logprob_token_ids.stride(0),
        valid_mask,
        valid_mask.stride(0),
        sampled_token_ids.to(device),
        topk_indices.to(device),
        topk_indices.stride(0),
        expanded_idx_mapping.to(device),
        num_per_req_token_ids.to(device),
        per_req_token_ids.to(device),
        per_req_token_ids.stride(0),
        NUM_TOPK=num_topk,
        PADDED_COLS=triton.next_power_of_2(num_cols),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(logprob_token_ids.cpu(), expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(valid_mask.cpu(), expected_mask, rtol=0, atol=0)


def test_fill_logprob_token_ids_kernel_custom_ids():
    batch_size = 3
    vocab_size = 64
    num_topk = 2
    max_custom = 4
    num_cols = max(num_topk, max_custom)
    torch.manual_seed(42)

    sampled_token_ids = torch.tensor([5, 15, 25], dtype=torch.int64)
    topk_indices = torch.randint(0, vocab_size, (batch_size, num_topk), dtype=torch.int32)
    expanded_idx_mapping = torch.arange(batch_size, dtype=torch.int32)
    num_per_req_token_ids = torch.tensor([3, 0, 1], dtype=torch.int32)
    per_req_token_ids = torch.zeros(batch_size, 64, dtype=torch.int32)
    per_req_token_ids[0, :3] = torch.tensor([1, 2, 3])
    per_req_token_ids[2, :1] = torch.tensor([44])

    expected_ids, expected_mask = _fill_logprob_token_ids_cpu(
        batch_size,
        sampled_token_ids,
        topk_indices,
        expanded_idx_mapping,
        num_per_req_token_ids,
        per_req_token_ids,
        num_topk,
        num_cols,
    )

    device = torch.device("npu")
    from vllm.triton_utils import triton

    logprob_token_ids = torch.zeros(
        batch_size, 1 + num_cols, dtype=torch.int64, device=device
    )
    valid_mask = torch.zeros(
        batch_size, 1 + num_cols, dtype=torch.bool, device=device
    )

    _fill_logprob_token_ids_kernel[(batch_size,)](
        logprob_token_ids,
        logprob_token_ids.stride(0),
        valid_mask,
        valid_mask.stride(0),
        sampled_token_ids.to(device),
        topk_indices.to(device),
        topk_indices.stride(0),
        expanded_idx_mapping.to(device),
        num_per_req_token_ids.to(device),
        per_req_token_ids.to(device),
        per_req_token_ids.stride(0),
        NUM_TOPK=num_topk,
        PADDED_COLS=triton.next_power_of_2(num_cols),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(logprob_token_ids.cpu(), expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(valid_mask.cpu(), expected_mask, rtol=0, atol=0)
