# GENERATED STRICT UT. Source: accuracy_test/codex/existing_accuracy_tests/from_vllm/test_scatter_num_accepted_kernel.py
# Do not edit mechanically; update the reviewed Codex source or strict generator.
from accuracy_test.strict_ut.runtime_gpu import STRICT_DEVICE as _STRICT_DEVICE
# Standalone Ascend A3 adaptation of an upstream vLLM accuracy path.
# Accuracy UT source: vllm/tests/kernels/mamba/test_mamba_ssm.py
# Kernel source: vllm/vllm/v1/worker/gpu/model_states/mamba_hybrid.py
# Coverage: _scatter_num_accepted_kernel

# vLLM vanilla kernel: _scatter_num_accepted_kernel from
# vllm/vllm/v1/worker/gpu/model_states/mamba_hybrid.py

"""
Precision test for _scatter_num_accepted_kernel.

Kernel signature:
    _scatter_num_accepted_kernel(
        idx_mapping_ptr,     # [num_reqs] batch_idx -> req_state_idx (-1 to skip)
        num_sampled_ptr,     # [num_reqs] num sampled tokens per row
        num_accepted_ptr,    # [max_num_reqs] output: max(num_sampled, 1)
    )

Scatters num_sampled values back to num_accepted_tokens_gpu array using
idx_mapping.  Skips rows where idx_mapping < 0.  Clamps to max(num_sampled, 1).
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.model_states.mamba_hybrid import _scatter_num_accepted_kernel
from accuracy_test.strict_ut.runtime_gpu import init_device_properties_triton

import pytest


def _scatter_num_accepted_ref(
    idx_mapping: torch.Tensor,
    num_sampled: torch.Tensor,
    num_accepted: torch.Tensor,
):
    """CPU reference for _scatter_num_accepted_kernel."""
    n = idx_mapping.shape[0]
    for row in range(n):
        req_state_idx = idx_mapping[row].item()
        if req_state_idx < 0:
            continue
        num_accepted[req_state_idx] = max(num_sampled[row].item(), 1)


class TestScatterNumAcceptedKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("cuda")

    @pytest.mark.parametrize("num_reqs", [1, 4, 8, 16])
    @pytest.mark.parametrize("max_num_reqs", [16, 32])
    def test_scatter_basic(self, num_reqs, max_num_reqs):
        """Normal scatter: all rows valid, clamp to >= 1."""
        idx_mapping = torch.randperm(max_num_reqs, dtype=torch.int32, device=self.device)[:num_reqs]
        num_sampled = torch.randint(0, 5, (num_reqs,), dtype=torch.int32, device=self.device)
        num_accepted = torch.ones(max_num_reqs, dtype=torch.int32, device=self.device)

        _scatter_num_accepted_kernel[(num_reqs,)](
            idx_mapping,
            num_sampled,
            num_accepted,
        )
        torch.cuda.synchronize()

        expected = torch.ones(max_num_reqs, dtype=torch.int32, device=self.device)
        _scatter_num_accepted_ref(idx_mapping, num_sampled, expected)

        torch.testing.assert_close(num_accepted.cpu(), expected.cpu(), rtol=0, atol=0)

    def test_skip_negative(self):
        """Rows with idx_mapping < 0 should be skipped."""
        num_reqs = 6
        max_num_reqs = 8
        # Index 0,2 have -1 sentinel (skipped); others valid
        idx_mapping = torch.tensor([-1, 3, -1, 1, 5, 0], dtype=torch.int32, device=self.device)
        num_sampled = torch.tensor([0, 2, 4, 0, 1, 3], dtype=torch.int32, device=self.device)
        num_accepted = torch.zeros(max_num_reqs, dtype=torch.int32, device=self.device)

        _scatter_num_accepted_kernel[(num_reqs,)](
            idx_mapping,
            num_sampled,
            num_accepted,
        )
        torch.cuda.synchronize()

        expected = torch.zeros(max_num_reqs, dtype=torch.int32, device=self.device)
        _scatter_num_accepted_ref(idx_mapping, num_sampled, expected)

        torch.testing.assert_close(num_accepted.cpu(), expected.cpu(), rtol=0, atol=0)

    def test_clamp_to_one(self):
        """num_sampled < 1 should be clamped to 1."""
        num_reqs = 4
        max_num_reqs = 4
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        num_sampled = torch.tensor([0, -5, -1, 0], dtype=torch.int32, device=self.device)
        num_accepted = torch.zeros(max_num_reqs, dtype=torch.int32, device=self.device)

        _scatter_num_accepted_kernel[(num_reqs,)](
            idx_mapping,
            num_sampled,
            num_accepted,
        )
        torch.cuda.synchronize()

        expected = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
        torch.testing.assert_close(num_accepted.cpu(), expected, rtol=0, atol=0)
