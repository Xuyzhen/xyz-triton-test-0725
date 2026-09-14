import pytest
import torch

from vllm.v1.worker.gpu.input_batch import _get_num_sampled_and_rejected_kernel


def _get_num_sampled_and_rejected_cpu(
    num_sampled: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    prefill_len: torch.Tensor,
):
    num_reqs = idx_mapping.shape[0]
    num_rejected = torch.empty_like(num_sampled)

    for batch_idx in range(num_reqs):
        req_state_idx = int(idx_mapping[batch_idx])
        seq_len = int(seq_lens[batch_idx])
        p_len = int(prefill_len[req_state_idx])
        is_chunked = seq_len < p_len

        sampled = int(num_sampled[batch_idx])
        if is_chunked:
            sampled = 0
        num_sampled[batch_idx] = sampled

        logits_start = int(cu_num_logits[batch_idx])
        logits_end = int(cu_num_logits[batch_idx + 1])
        n_logits = logits_end - logits_start

        rejected = n_logits - sampled
        if is_chunked:
            rejected = 0
        num_rejected[batch_idx] = rejected

    return num_sampled, num_rejected


def test_get_num_sampled_and_rejected_kernel():
    torch.manual_seed(42)
    num_reqs = 3

    idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int32)
    num_sampled = torch.tensor([3, 2, 1], dtype=torch.int32)
    seq_lens = torch.tensor([20, 3, 50], dtype=torch.int32)
    cu_num_logits = torch.tensor([0, 4, 6, 10], dtype=torch.int32)
    prefill_len = torch.tensor([10, 5, 30], dtype=torch.int32)

    expected_sampled = num_sampled.clone()
    expected_rejected = torch.empty_like(num_sampled)
    exp_s, exp_r = _get_num_sampled_and_rejected_cpu(
        expected_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len
    )

    device = torch.device("npu")
    num_sampled_npu = num_sampled.to(device)
    num_rejected = torch.empty_like(num_sampled_npu)

    _get_num_sampled_and_rejected_kernel[(num_reqs,)](
        num_sampled_npu,
        num_rejected,
        seq_lens.to(device),
        cu_num_logits.to(device),
        idx_mapping.to(device),
        prefill_len.to(device),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(num_sampled_npu.cpu(), exp_s, rtol=0, atol=0)
    torch.testing.assert_close(num_rejected.cpu(), exp_r, rtol=0, atol=0)


def test_get_num_sampled_and_rejected_kernel_chunked_prefill():
    num_reqs = 2

    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    num_sampled = torch.tensor([2, 3], dtype=torch.int32)
    seq_lens = torch.tensor([3, 2], dtype=torch.int32)
    cu_num_logits = torch.tensor([0, 3, 6], dtype=torch.int32)
    prefill_len = torch.tensor([10, 20], dtype=torch.int32)

    expected_sampled = num_sampled.clone()
    exp_s, exp_r = _get_num_sampled_and_rejected_cpu(
        expected_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len
    )

    device = torch.device("npu")
    num_sampled_npu = num_sampled.to(device)
    num_rejected = torch.empty_like(num_sampled_npu)

    _get_num_sampled_and_rejected_kernel[(num_reqs,)](
        num_sampled_npu,
        num_rejected,
        seq_lens.to(device),
        cu_num_logits.to(device),
        idx_mapping.to(device),
        prefill_len.to(device),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(num_sampled_npu.cpu(), exp_s, rtol=0, atol=0)
    torch.testing.assert_close(num_rejected.cpu(), exp_r, rtol=0, atol=0)
