# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.gpu.sample.min_p import _min_p_kernel

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _apply_min_p_cpu(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    min_p: torch.Tensor,
) -> None:
    """Pure PyTorch CPU reference implementation of min-p sampling.

    For each token, computes the max logit (across the entire vocab),
    calculates threshold = max_logit + log(min_p), and masks any logit
    below that threshold to -inf.
    """
    logits = logits.clone()
    num_tokens, vocab_size = logits.shape
    for token_idx in range(num_tokens):
        req_state_idx = int(expanded_idx_mapping[token_idx])
        mp = float(min_p[req_state_idx])
        if mp == 0.0:
            continue
        max_val = float(logits[token_idx].max())
        threshold = max_val + torch.tensor(mp).log().item()
        mask = logits[token_idx] < threshold
        logits[token_idx][mask] = float("-inf")
    return logits


@pytest.mark.parametrize("num_tokens", [1, 3, 8])
@pytest.mark.parametrize("vocab_size", [1024, 8192, 32000])
def test_min_p_kernel_basic(num_tokens: int, vocab_size: int) -> None:
    """Min-p sampling: mask logits below max_logit + log(min_p) to -inf.

    Verifies the kernel matches the CPU reference across various token /
    vocab sizes with random min_p values. Also covers the min_p == 0.0
    no-op case.
    """
    init_device_properties_triton()
    torch.manual_seed(42)

    max_num_reqs = 8
    expanded_idx_mapping = torch.randint(0, max_num_reqs, (num_tokens,),
                                         dtype=torch.int64)
    # Uniform random in [0, 1]; include 0 to test no-op.
    min_p = torch.rand(max_num_reqs, dtype=torch.float32)
    min_p[0] = 0.0

    logits_cpu = torch.randn(num_tokens, vocab_size, dtype=torch.float32)

    expected = _apply_min_p_cpu(logits_cpu, expanded_idx_mapping, min_p)

    device = torch.device("npu")
    logits_npu = logits_cpu.to(device)
    idx_mapping_npu = expanded_idx_mapping.to(device)
    min_p_npu = min_p.to(device)

    BLOCK_SIZE = 1024
    _min_p_kernel[(num_tokens,)](
        logits_npu,
        logits_npu.stride(0),
        idx_mapping_npu,
        min_p_npu,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(logits_npu.cpu(), expected, rtol=1e-5, atol=1e-5)


def test_min_p_zero_is_noop() -> None:
    """When min_p is 0.0 the kernel returns immediately (no-op).

    Verifies logits remain unchanged when min_p == 0.0.
    """
    init_device_properties_triton()
    torch.manual_seed(7)

    num_tokens = 2
    vocab_size = 256
    max_num_reqs = 2

    expanded_idx_mapping = torch.tensor([0, 1], dtype=torch.int64)
    min_p = torch.tensor([0.0, 0.0], dtype=torch.float32)
    logits_cpu = torch.randn(num_tokens, vocab_size, dtype=torch.float32)

    device = torch.device("npu")
    logits_npu = logits_cpu.clone().to(device)
    idx_mapping_npu = expanded_idx_mapping.to(device)
    min_p_npu = min_p.to(device)

    BLOCK_SIZE = 1024
    _min_p_kernel[(num_tokens,)](
        logits_npu,
        logits_npu.stride(0),
        idx_mapping_npu,
        min_p_npu,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(logits_npu.cpu(), logits_cpu, rtol=0, atol=0)


def test_min_p_masks_most_logits() -> None:
    """With min_p close to 1.0, only the top few logits survive."""
    init_device_properties_triton()
    torch.manual_seed(1)

    num_tokens = 1
    vocab_size = 512
    max_num_reqs = 1

    expanded_idx_mapping = torch.zeros(num_tokens, dtype=torch.int64)
    min_p = torch.tensor([0.95], dtype=torch.float32)
    logits_cpu = torch.randn(num_tokens, vocab_size, dtype=torch.float32)

    expected = _apply_min_p_cpu(logits_cpu, expanded_idx_mapping, min_p)

    device = torch.device("npu")
    logits_npu = logits_cpu.to(device)
    idx_mapping_npu = expanded_idx_mapping.to(device)
    min_p_npu = min_p.to(device)

    BLOCK_SIZE = 1024
    _min_p_kernel[(num_tokens,)](
        logits_npu,
        logits_npu.stride(0),
        idx_mapping_npu,
        min_p_npu,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(logits_npu.cpu(), expected, rtol=1e-5, atol=1e-5)

    # Fewer than ~5% of logits should survive.
    num_surviving = (logits_npu.cpu() > float("-inf")).sum().item()
    assert num_surviving <= vocab_size * 0.1
