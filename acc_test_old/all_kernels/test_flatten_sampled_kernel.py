# vLLM vanilla kernel: _flatten_sampled_kernel from
# vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py

"""
Precision test for _flatten_sampled_kernel.

Flattens sampled tokens from a 2D per-request array into a 1D array
indexed by cu_num_logits.

Kernel signature:
    _flatten_sampled_kernel(
        flat_sampled_ptr,     # [num_logits] int64 output
        sampled_ptr,           # [num_reqs, num_spec_steps+1] int64
        sampled_stride,
        num_sampled_ptr,       # [num_reqs] int64
        cu_num_logits_ptr,     # [num_reqs+1] int64
    )
"""

import torch
import pytest

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    _flatten_sampled_kernel,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _flatten_sampled_ref(
    sampled,           # [num_reqs, num_spec_steps + 1]
    num_sampled,       # [num_reqs]
    cu_num_logits,     # [num_reqs + 1]
    total_num_logits,
):
    """CPU reference: flatten sampled tokens into 1D array."""
    flat = torch.zeros(total_num_logits, dtype=sampled.dtype)
    for req_idx in range(sampled.shape[0]):
        start = int(cu_num_logits[req_idx].item())
        n = int(num_sampled[req_idx].item())
        for i in range(n):
            flat[start + i] = sampled[req_idx, i]
    return flat


class TestFlattenSampledKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_reqs", [1, 2, 4, 8])
    @pytest.mark.parametrize("num_spec_steps", [1, 3, 5])
    def test_flatten_basic(self, num_reqs, num_spec_steps):
        """Test basic flattening of sampled tokens."""
        total_num_logits = num_reqs * (num_spec_steps + 1)
        sampled = torch.randint(
            0, 1000, (num_reqs, num_spec_steps + 1), dtype=torch.int64, device=self.device
        )
        cu_num_logits = torch.arange(
            num_reqs + 1, device=self.device, dtype=torch.int64
        ) * (num_spec_steps + 1)

        # Vary num_sampled per request
        num_sampled = torch.randint(
            0, num_spec_steps + 2, (num_reqs,), dtype=torch.int64, device=self.device
        )
        # Clamp to reasonable values
        num_sampled = torch.clamp(num_sampled, 0, num_spec_steps + 1)

        flat_sampled = torch.zeros(total_num_logits, dtype=torch.int64, device=self.device)

        _flatten_sampled_kernel[(num_reqs,)](
            flat_sampled,
            sampled,
            sampled.stride(0),
            num_sampled,
            cu_num_logits,
        )
        torch.npu.synchronize()

        ref = _flatten_sampled_ref(
            sampled.cpu(), num_sampled.cpu(), cu_num_logits.cpu(), total_num_logits
        )
        torch.testing.assert_close(flat_sampled.cpu(), ref, rtol=0, atol=0)

    def test_all_zeros_num_sampled(self):
        """When num_sampled is 0 for all requests, output should be zeros."""
        num_reqs = 3
        num_spec_steps = 2
        total_num_logits = num_reqs * (num_spec_steps + 1)

        sampled = torch.randint(
            0, 1000, (num_reqs, num_spec_steps + 1), dtype=torch.int64, device=self.device
        )
        cu_num_logits = torch.arange(
            num_reqs + 1, device=self.device, dtype=torch.int64
        ) * (num_spec_steps + 1)
        num_sampled = torch.zeros(num_reqs, dtype=torch.int64, device=self.device)

        flat_sampled = -torch.ones(total_num_logits, dtype=torch.int64, device=self.device)

        _flatten_sampled_kernel[(num_reqs,)](
            flat_sampled,
            sampled,
            sampled.stride(0),
            num_sampled,
            cu_num_logits,
        )
        torch.npu.synchronize()

        # Should remain -1 (not written to)
        assert torch.all(flat_sampled.cpu() == -1)

    def test_single_req_multi_logits(self):
        """Single request with many logits."""
        num_reqs = 1
        num_spec_steps = 10
        total_num_logits = num_spec_steps + 1

        sampled = torch.randint(
            0, 1000, (num_reqs, num_spec_steps + 1), dtype=torch.int64, device=self.device
        )
        cu_num_logits = torch.tensor([0, total_num_logits], device=self.device, dtype=torch.int64)
        num_sampled = torch.full((num_reqs,), num_spec_steps + 1, dtype=torch.int64, device=self.device)

        flat_sampled = torch.zeros(total_num_logits, dtype=torch.int64, device=self.device)

        _flatten_sampled_kernel[(num_reqs,)](
            flat_sampled,
            sampled,
            sampled.stride(0),
            num_sampled,
            cu_num_logits,
        )
        torch.npu.synchronize()

        ref = _flatten_sampled_ref(
            sampled.cpu(), num_sampled.cpu(), cu_num_logits.cpu(), total_num_logits
        )
        torch.testing.assert_close(flat_sampled.cpu(), ref, rtol=0, atol=0)
