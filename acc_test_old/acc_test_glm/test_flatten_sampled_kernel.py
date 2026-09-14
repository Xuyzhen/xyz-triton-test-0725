import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.rejection_sampler import _flatten_sampled_kernel


def _flatten_sampled_cpu(
    sampled: torch.Tensor,
    num_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
):
    num_reqs = sampled.shape[0]
    num_logits = int(cu_num_logits[-1])
    flat_sampled = torch.zeros(num_logits, dtype=sampled.dtype)

    for req_idx in range(num_reqs):
        start_idx = int(cu_num_logits[req_idx])
        n_sampled = int(num_sampled[req_idx])
        for i in range(n_sampled):
            flat_sampled[start_idx + i] = sampled[req_idx, i]

    return flat_sampled


def test_flatten_sampled_kernel():
    torch.manual_seed(42)
    num_reqs = 3
    num_speculative_steps = 2
    num_logits = 9

    sampled = torch.tensor(
        [[100, 200, 300], [400, 500, 600], [700, 800, 900]],
        dtype=torch.int64,
    )
    num_sampled = torch.tensor([3, 2, 1], dtype=torch.int32)
    cu_num_logits = torch.tensor([0, 3, 5, 9], dtype=torch.int32)

    expected = _flatten_sampled_cpu(sampled, num_sampled, cu_num_logits)

    device = torch.device("npu")
    flat_sampled = torch.zeros(num_logits, dtype=torch.int64, device=device)

    _flatten_sampled_kernel[(num_reqs,)](
        flat_sampled,
        sampled.to(device),
        sampled.stride(0),
        num_sampled.to(device),
        cu_num_logits.to(device),
        num_warps=1,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(flat_sampled.cpu(), expected, rtol=0, atol=0)


def test_flatten_sampled_kernel_empty():
    num_reqs = 2
    num_logits = 4

    sampled = torch.tensor([[10, 20], [30, 40]], dtype=torch.int64)
    num_sampled = torch.tensor([1, 3], dtype=torch.int32)
    cu_num_logits = torch.tensor([0, 1, 4], dtype=torch.int32)

    expected = _flatten_sampled_cpu(sampled, num_sampled, cu_num_logits)

    device = torch.device("npu")
    flat_sampled = torch.zeros(num_logits, dtype=torch.int64, device=device)

    _flatten_sampled_kernel[(num_reqs,)](
        flat_sampled,
        sampled.to(device),
        sampled.stride(0),
        num_sampled.to(device),
        cu_num_logits.to(device),
        num_warps=1,
    )
    torch.npu.synchronize()

    expected_padded = torch.zeros(num_logits, dtype=torch.int64)
    for i in range(1):
        expected_padded[i] = expected[i]
    for i in range(1, 4):
        expected_padded[i] = expected[i]

    torch.testing.assert_close(flat_sampled.cpu(), expected, rtol=0, atol=0)
