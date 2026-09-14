# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    _flatten_sampled_kernel,
)

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _flatten_sampled_cpu(
    flat_sampled: torch.Tensor,
    sampled: torch.Tensor,
    num_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch CPU reference for flatten_sampled_kernel.

    For each request, copies num_sampled[req_idx] tokens from
    sampled[req_idx, :] into flat_sampled at the offset given by
    cu_num_logits[req_idx].
    """
    output = flat_sampled.clone()
    num_reqs = num_sampled.shape[0]

    for req_idx in range(num_reqs):
        start_idx = int(cu_num_logits[req_idx])
        ns = int(num_sampled[req_idx])
        for i in range(ns):
            output[start_idx + i] = int(sampled[req_idx, i])

    return output


@pytest.mark.parametrize("num_reqs", [1, 4, 8])
@pytest.mark.parametrize("num_spec_steps", [0, 3, 5])
def test_flatten_sampled_basic(num_reqs: int, num_spec_steps: int) -> None:
    """Flatten sampled tokens across requests into a 1D buffer.

    Tests with varying number of sampled tokens per request.
    Sampled tokens are written at offsets determined by cu_num_logits.
    """
    init_device_properties_triton()
    torch.manual_seed(42)

    max_spec_steps = 5

    sampled = torch.randint(
        0, 32000, (num_reqs, max(num_spec_steps, 1)), dtype=torch.int64,
    )
    num_sampled = torch.zeros(num_reqs, dtype=torch.int32)

    # Cumulative logits: each request produces some logits
    cu_num_logits = torch.zeros(num_reqs + 1, dtype=torch.int64)
    total_logits = 0
    for b in range(num_reqs):
        ns = 1 + b % (num_spec_steps + 1) if num_spec_steps > 0 else 1
        num_sampled[b] = ns
        total_logits += num_spec_steps + 1
        cu_num_logits[b + 1] = total_logits

    flat_sampled = torch.zeros(int(cu_num_logits[-1]), dtype=sampled.dtype)

    expected = _flatten_sampled_cpu(
        flat_sampled, sampled, num_sampled, cu_num_logits,
    )

    device = torch.device("npu")
    flat_sampled_npu = flat_sampled.to(device)

    _flatten_sampled_kernel[(num_reqs,)](
        flat_sampled_npu,
        sampled.to(device),
        sampled.stride(0),
        num_sampled.to(device),
        cu_num_logits.to(device),
        num_warps=1,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        flat_sampled_npu.cpu(), expected, rtol=0, atol=0,
    )


def test_flatten_sampled_no_sampled() -> None:
    """No sampled tokens for any request -- flat_sampled stays all zeros."""
    init_device_properties_triton()
    torch.manual_seed(7)

    num_reqs = 3

    sampled = torch.randint(0, 32000, (num_reqs, 3), dtype=torch.int64)
    num_sampled = torch.zeros(num_reqs, dtype=torch.int32)
    cu_num_logits = torch.tensor([0, 5, 8, 12], dtype=torch.int64)

    flat_sampled = torch.zeros(int(cu_num_logits[-1]), dtype=sampled.dtype)

    expected = _flatten_sampled_cpu(
        flat_sampled, sampled, num_sampled, cu_num_logits,
    )

    device = torch.device("npu")
    flat_sampled_npu = flat_sampled.to(device)

    _flatten_sampled_kernel[(num_reqs,)](
        flat_sampled_npu,
        sampled.to(device),
        sampled.stride(0),
        num_sampled.to(device),
        cu_num_logits.to(device),
        num_warps=1,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        flat_sampled_npu.cpu(), expected, rtol=0, atol=0,
    )


def test_flatten_sampled_single_request() -> None:
    """Single request with multiple sampled tokens."""
    init_device_properties_triton()
    torch.manual_seed(3)

    num_reqs = 1

    sampled = torch.tensor([[100, 200, 300, 0]], dtype=torch.int64)
    num_sampled = torch.tensor([3], dtype=torch.int32)
    cu_num_logits = torch.tensor([0, 5], dtype=torch.int64)

    flat_sampled = torch.zeros(5, dtype=torch.int64)

    expected = _flatten_sampled_cpu(
        flat_sampled, sampled, num_sampled, cu_num_logits,
    )

    device = torch.device("npu")
    flat_sampled_npu = flat_sampled.to(device)

    _flatten_sampled_kernel[(num_reqs,)](
        flat_sampled_npu,
        sampled.to(device),
        sampled.stride(0),
        num_sampled.to(device),
        cu_num_logits.to(device),
        num_warps=1,
    )
    torch.npu.synchronize()

    expected_result = torch.tensor([100, 200, 300, 0, 0], dtype=torch.int64)
    torch.testing.assert_close(
        flat_sampled_npu.cpu(), expected_result, rtol=0, atol=0,
    )


def test_flatten_sampled_all_sampled() -> None:
    """Every token slot is sampled (no rejection)."""
    init_device_properties_triton()
    torch.manual_seed(9)

    num_reqs = 2
    num_max = 4

    sampled = torch.randint(0, 32000, (num_reqs, num_max), dtype=torch.int64)
    num_sampled = torch.tensor([4, 4], dtype=torch.int32)
    cu_num_logits = torch.tensor([0, 4, 8], dtype=torch.int64)

    flat_sampled = torch.zeros(8, dtype=torch.int64)

    expected = _flatten_sampled_cpu(
        flat_sampled, sampled, num_sampled, cu_num_logits,
    )

    device = torch.device("npu")
    flat_sampled_npu = flat_sampled.to(device)

    _flatten_sampled_kernel[(num_reqs,)](
        flat_sampled_npu,
        sampled.to(device),
        sampled.stride(0),
        num_sampled.to(device),
        cu_num_logits.to(device),
        num_warps=1,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        flat_sampled_npu.cpu(), expected, rtol=0, atol=0,
    )
