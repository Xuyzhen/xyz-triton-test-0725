# vLLM-Ascend patched kernel: _temperature_kernel from
# vllm-ascend/vllm_ascend/worker/v2/sample/gumbel.py:25
# PATCH NOTE: This is an Ascend NPU adaptation of the original vLLM Triton kernel

"""
Precision test for patched _temperature_kernel (Ascend NPU version).

Patch differences vs original vllm:
- Uses do_not_specialize=["logits_stride", "vocab_size"] (original also has num_tokens)
- Uses logits.to(tl.float32) explicitly before division (NPU requirement)
- Uses tl.where with mask for store instead of conditional store
- BLOCK_SIZE is set to 44032 by the wrapper (compared to different block size in original)
- Uses multibuffer=False

Kernel signature:
    _temperature_kernel(
        logits_ptr,                 # fp32 logits [num_tokens, vocab_size]
        logits_stride,              # stride(0) of logits
        expanded_idx_mapping_ptr,   # [num_tokens] token_idx -> req_state_idx
        temperature_ptr,            # [max_num_reqs] temperature per request
        vocab_size,                 # scalar: vocab size
        BLOCK_SIZE: tl.constexpr,   # block size for iteration
    )

Applies temperature scaling: logits /= temperature.
Early-returns if temperature is 0.0 or 1.0 (no-op).
"""

import torch

from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

import pytest


def _temperature_ref(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
):
    """CPU reference for _temperature_kernel."""
    num_tokens = logits.shape[0]
    for token_idx in range(num_tokens):
        req_state_idx = expanded_idx_mapping[token_idx].item()
        temp = temperature[req_state_idx].item()
        if temp == 0.0 or temp == 1.0:
            continue
        logits[token_idx, :] = logits[token_idx, :] / temp


class TestTemperatureKernelPatch:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")
        # The patched kernel uses BLOCK_SIZE=44032
        self.BLOCK_SIZE = 44032

    @pytest.mark.parametrize("num_tokens", [1, 4, 8])
    @pytest.mark.parametrize("vocab_size", [128, 1024, 8192, 44032])
    @pytest.mark.parametrize("temp", [0.0, 0.5, 1.0, 2.0, 0.8])
    def test_temperature(self, num_tokens, vocab_size, temp):
        """Compare NPU temperature scaling with CPU reference."""
        from vllm_ascend.worker.v2.sample.gumbel import _temperature_kernel

        num_reqs = 4
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.randint(0, num_reqs, (num_tokens,), dtype=torch.int32, device=self.device)
        temperature = torch.full((num_reqs,), temp, dtype=torch.float32, device=self.device)

        logits_copy = logits.clone().cpu()

        vocab_size_val = vocab_size
        num_blocks = triton.cdiv(vocab_size_val, self.BLOCK_SIZE)

        _temperature_kernel[(num_tokens, num_blocks)](
            logits,
            logits.stride(0),
            expanded_idx_mapping,
            temperature,
            vocab_size_val,
            BLOCK_SIZE=self.BLOCK_SIZE,
            multibuffer=False,
        )
        torch.npu.synchronize()

        expected = logits_copy.clone()
        _temperature_ref(expected, expanded_idx_mapping.cpu(), temperature.cpu())

        torch.testing.assert_close(logits.cpu(), expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("temp", [0.0, 1.0])
    def test_temperature_noop(self, temp):
        """When temp is 0.0 or 1.0, the kernel early-returns and logits are unchanged."""
        from vllm_ascend.worker.v2.sample.gumbel import _temperature_kernel

        num_tokens, vocab_size = 4, 256
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        temperature = torch.full((1,), temp, dtype=torch.float32, device=self.device)

        logits_copy = logits.clone().cpu()

        num_blocks = triton.cdiv(vocab_size, self.BLOCK_SIZE)
        _temperature_kernel[(num_tokens, num_blocks)](
            logits,
            logits.stride(0),
            expanded_idx_mapping,
            temperature,
            vocab_size,
            BLOCK_SIZE=self.BLOCK_SIZE,
            multibuffer=False,
        )
        torch.npu.synchronize()

        torch.testing.assert_close(logits.cpu(), logits_copy, rtol=0, atol=0)

    def test_varying_temperatures(self):
        """Each request can have a different temperature."""
        from vllm_ascend.worker.v2.sample.gumbel import _temperature_kernel

        num_tokens = 4
        vocab_size = 256
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=self.device)
        temperature = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float32, device=self.device)

        logits_copy = logits.clone().cpu()

        num_blocks = triton.cdiv(vocab_size, self.BLOCK_SIZE)
        _temperature_kernel[(num_tokens, num_blocks)](
            logits,
            logits.stride(0),
            expanded_idx_mapping,
            temperature,
            vocab_size,
            BLOCK_SIZE=self.BLOCK_SIZE,
            multibuffer=False,
        )
        torch.npu.synchronize()

        expected = logits_copy.clone()
        _temperature_ref(expected, expanded_idx_mapping.cpu(), temperature.cpu())

        torch.testing.assert_close(logits.cpu(), expected, rtol=1e-5, atol=1e-5)
