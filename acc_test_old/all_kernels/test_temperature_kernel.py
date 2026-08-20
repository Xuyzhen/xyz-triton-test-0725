# vLLM vanilla kernel: _temperature_kernel from
# vllm/vllm/v1/worker/gpu/sample/gumbel.py

"""
Precision test for _temperature_kernel.

Kernel signature:
    _temperature_kernel(
        logits_ptr,                 # fp32 logits [num_tokens, vocab_size]
        logits_stride,              # stride(0) of logits
        expanded_idx_mapping_ptr,   # [num_tokens] token_idx -> req_state_idx
        temperature_ptr,            # [max_num_reqs] temperature per request
        vocab_size,                 # vocab size
        BLOCK_SIZE: tl.constexpr,   # block size for iteration
    )

Applies temperature scaling: logits /= temperature.
Early-returns if temperature is 0.0 or 1.0 (no-op).
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import _temperature_kernel
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

import pytest


def _temperature_ref(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
):
    """CPU reference for _temperature kernel."""
    num_tokens = logits.shape[0]
    for token_idx in range(num_tokens):
        req_state_idx = expanded_idx_mapping[token_idx].item()
        temp = temperature[req_state_idx].item()
        if temp == 0.0 or temp == 1.0:
            continue
        logits[token_idx, :] = logits[token_idx, :] / temp


class TestTemperatureKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_tokens", [1, 4, 8])
    @pytest.mark.parametrize("vocab_size", [128, 1024, 8192])
    @pytest.mark.parametrize("temp", [0.5, 1.0, 2.0, 0.8, 0.0])
    def test_temperature(self, num_tokens, vocab_size, temp):
        """Compare GPU temperature scaling with CPU reference."""
        num_reqs = 4
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.randint(0, num_reqs, (num_tokens,), dtype=torch.int32, device=self.device)
        temperature = torch.full((num_reqs,), 1.0, dtype=torch.float32, device=self.device)
        # Assign the requested temperature to all requests
        temperature[:] = temp

        logits_copy = logits.clone().cpu()

        vocab_size_val = vocab_size
        BLOCK_SIZE = 8192
        num_blocks = triton.cdiv(vocab_size_val, BLOCK_SIZE)

        _temperature_kernel[(num_tokens, num_blocks)](
            logits,
            logits.stride(0),
            expanded_idx_mapping,
            temperature,
            vocab_size_val,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.npu.synchronize()

        expected = logits_copy.clone()
        _temperature_ref(expected, expanded_idx_mapping.cpu(), temperature.cpu())

        torch.testing.assert_close(logits.cpu(), expected, rtol=1e-5, atol=1e-5)

    def test_varying_temperatures(self):
        """Each request can have a different temperature."""
        num_tokens = 4
        vocab_size = 256
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        num_reqs = 4
        expanded_idx_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=self.device)
        temperature = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float32, device=self.device)

        logits_copy = logits.clone().cpu()

        num_blocks = triton.cdiv(vocab_size, 8192)
        _temperature_kernel[(num_tokens, num_blocks)](
            logits,
            logits.stride(0),
            expanded_idx_mapping,
            temperature,
            vocab_size,
            BLOCK_SIZE=8192,
        )
        torch.npu.synchronize()

        expected = logits_copy.clone()
        _temperature_ref(expected, expanded_idx_mapping.cpu(), temperature.cpu())

        torch.testing.assert_close(logits.cpu(), expected, rtol=1e-5, atol=1e-5)
