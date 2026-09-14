# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.gpu.sample.gumbel import _temperature_kernel

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _apply_temperature_cpu(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    """Pure PyTorch CPU reference implementation of temperature scaling.

    For each token, looks up the temperature via expanded_idx_mapping and
    divides the logit row by that temperature.  Temperatures of 0.0 and 1.0
    are no-ops (matching the kernel's early-return behaviour).
    """
    logits = logits.clone()
    num_tokens, vocab_size = logits.shape
    for token_idx in range(num_tokens):
        req_state_idx = int(expanded_idx_mapping[token_idx])
        temp = float(temperature[req_state_idx])
        if temp == 0.0 or temp == 1.0:
            continue
        logits[token_idx] = logits[token_idx] / temp
    return logits


@pytest.mark.parametrize("num_tokens", [1, 4, 8])
@pytest.mark.parametrize("vocab_size", [1024, 8192, 32000])
def test_temperature_kernel_basic(num_tokens: int, vocab_size: int) -> None:
    """Temperature scaling: divide logits by temperature per request state.

    Covers basic in-place division with various token / vocab sizes, plus
    the no-op cases for temp == 0.0 and temp == 1.0.
    """
    init_device_properties_triton()
    torch.manual_seed(42)

    max_num_reqs = 8
    expanded_idx_mapping = torch.randint(0, max_num_reqs, (num_tokens,),
                                         dtype=torch.int64)
    temperature = torch.rand(max_num_reqs, dtype=torch.float32) * 2.0
    # Inject no-op cases.
    temperature[0] = 0.0
    temperature[-1] = 1.0

    logits_cpu = torch.randn(num_tokens, vocab_size, dtype=torch.float32)

    # CPU reference.
    expected = _apply_temperature_cpu(
        logits_cpu, expanded_idx_mapping, temperature,
    )

    # NPU kernel.
    device = torch.device("npu")
    logits_npu = logits_cpu.to(device)
    idx_mapping_npu = expanded_idx_mapping.to(device)
    temp_npu = temperature.to(device)

    BLOCK_SIZE = 8192
    num_blocks = (vocab_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    _temperature_kernel[(num_tokens, num_blocks)](
        logits_npu,
        logits_npu.stride(0),
        idx_mapping_npu,
        temp_npu,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(logits_npu.cpu(), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("temp", [0.0, 1.0, 2.0, 0.5])
def test_temperature_kernel_single_token(temp: float) -> None:
    """Single token edge cases: each temperature value runs independently.

    Verifies that the kernel correctly handles individual temperature values
    on a small vocabulary.
    """
    init_device_properties_triton()
    torch.manual_seed(7)

    num_tokens = 1
    vocab_size = 128
    max_num_reqs = 1

    expanded_idx_mapping = torch.zeros(num_tokens, dtype=torch.int64)
    temperature = torch.tensor([temp], dtype=torch.float32)
    logits_cpu = torch.randn(num_tokens, vocab_size, dtype=torch.float32)

    expected = _apply_temperature_cpu(
        logits_cpu, expanded_idx_mapping, temperature,
    )

    device = torch.device("npu")
    logits_npu = logits_cpu.to(device)
    idx_mapping_npu = expanded_idx_mapping.to(device)
    temp_npu = temperature.to(device)

    BLOCK_SIZE = 8192
    num_blocks = (vocab_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    _temperature_kernel[(num_tokens, num_blocks)](
        logits_npu,
        logits_npu.stride(0),
        idx_mapping_npu,
        temp_npu,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(logits_npu.cpu(), expected, rtol=1e-5, atol=1e-5)


def test_temperature_kernel_negative_logits() -> None:
    """Negative logits are divided by temperature correctly (sign preserved)."""
    init_device_properties_triton()
    torch.manual_seed(1)

    num_tokens = 2
    vocab_size = 256
    max_num_reqs = 2

    expanded_idx_mapping = torch.tensor([0, 1], dtype=torch.int64)
    temperature = torch.tensor([0.8, 1.5], dtype=torch.float32)
    logits_cpu = -torch.rand(num_tokens, vocab_size, dtype=torch.float32).abs()

    expected = _apply_temperature_cpu(
        logits_cpu, expanded_idx_mapping, temperature,
    )

    device = torch.device("npu")
    logits_npu = logits_cpu.to(device)
    idx_mapping_npu = expanded_idx_mapping.to(device)
    temp_npu = temperature.to(device)

    BLOCK_SIZE = 8192
    num_blocks = (vocab_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    _temperature_kernel[(num_tokens, num_blocks)](
        logits_npu,
        logits_npu.stride(0),
        idx_mapping_npu,
        temp_npu,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(logits_npu.cpu(), expected, rtol=1e-5, atol=1e-5)
