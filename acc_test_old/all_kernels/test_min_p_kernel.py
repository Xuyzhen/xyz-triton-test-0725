# vLLM vanilla kernel: _min_p_kernel from vllm/vllm/v1/worker/gpu/sample/min_p.py

"""
Precision test for _min_p_kernel.

Kernel signature:
    _min_p_kernel(
        logits_ptr,                  # fp32 logits [num_tokens, vocab_size]
        logits_stride,               # stride(0) of logits
        expanded_idx_mapping_ptr,    # int32 mapping [num_tokens] token_idx -> req_state_idx
        min_p_ptr,                   # fp32 min_p values [max_num_reqs]
        vocab_size,                  # vocab size
        BLOCK_SIZE: tl.constexpr,    # block size for iteration
    )

Applies min-p sampling: zeros out logits below threshold = max_val + log(min_p).
When min_p == 0.0, the kernel returns early (no-op).
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.min_p import _min_p_kernel
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

import pytest


def _min_p_ref(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    min_p: torch.Tensor,
) -> torch.Tensor:
    """CPU reference: apply min-p sampling threshold."""
    out = logits.clone()
    num_tokens, vocab_size = logits.shape
    for token_idx in range(num_tokens):
        req_state_idx = expanded_idx_mapping[token_idx].item()
        mp = min_p[req_state_idx].item()
        if mp == 0.0:
            continue
        max_val = float(out[token_idx].max())
        threshold = max_val + float(torch.log(torch.tensor(mp)))
        out[token_idx][out[token_idx] < threshold] = float("-inf")
    return out


class TestMinPKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_tokens", [1, 2, 4, 8])
    @pytest.mark.parametrize("vocab_size", [128, 1024, 8192, 16384])
    @pytest.mark.parametrize("min_p_val", [0.0, 0.1, 0.5, 0.9, 1.0])
    def test_min_p(self, num_tokens, vocab_size, min_p_val):
        """Compare min-p GPU output with CPU reference."""
        max_num_reqs = 4
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        min_p = torch.full((max_num_reqs,), min_p_val, dtype=torch.float32, device=self.device)

        logits_gpu = logits.clone()
        _min_p_kernel[(num_tokens,)](
            logits_gpu,
            logits_gpu.stride(0),
            expanded_idx_mapping,
            min_p,
            vocab_size,
            BLOCK_SIZE=1024,
        )
        torch.npu.synchronize()

        expected = _min_p_ref(logits.cpu(), expanded_idx_mapping.cpu(), min_p.cpu())
        torch.testing.assert_close(logits_gpu.cpu(), expected, rtol=0, atol=0)

    def test_multiple_req_states(self):
        """Each token can map to a different request with different min_p values."""
        num_tokens, vocab_size = 4, 2048
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=self.device)
        min_p = torch.tensor([0.0, 0.1, 0.5, 1.0], dtype=torch.float32, device=self.device)

        logits_gpu = logits.clone()
        _min_p_kernel[(num_tokens,)](
            logits_gpu,
            logits_gpu.stride(0),
            expanded_idx_mapping,
            min_p,
            vocab_size,
            BLOCK_SIZE=1024,
        )
        torch.npu.synchronize()

        expected = _min_p_ref(logits.cpu(), expanded_idx_mapping.cpu(), min_p.cpu())
        torch.testing.assert_close(logits_gpu.cpu(), expected, rtol=0, atol=0)

    def test_min_p_zero_is_noop(self):
        """When min_p == 0, logits must remain unchanged."""
        num_tokens, vocab_size = 2, 512
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        min_p = torch.zeros(1, dtype=torch.float32, device=self.device)

        logits_gpu = logits.clone()
        _min_p_kernel[(num_tokens,)](
            logits_gpu,
            logits_gpu.stride(0),
            expanded_idx_mapping,
            min_p,
            vocab_size,
            BLOCK_SIZE=1024,
        )
        torch.npu.synchronize()

        torch.testing.assert_close(logits_gpu.cpu(), logits.cpu(), rtol=0, atol=0)
