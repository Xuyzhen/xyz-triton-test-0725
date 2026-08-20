# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from vllm.v1.worker.gpu.sample.logit_bias import _bias_kernel

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _bias_cpu(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    num_allowed_token_ids: torch.Tensor,
    allowed_token_ids: torch.Tensor,
    num_logit_bias: torch.Tensor,
    bias_token_ids: torch.Tensor,
    bias_values: torch.Tensor,
    pos: torch.Tensor,
    min_lens: torch.Tensor,
    num_stop_token_ids: torch.Tensor,
    stop_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch CPU reference for the bias kernel.

    Applies three operations in sequence:
    1. Allowed token IDs -- sets logits for non-allowed tokens to -inf
    2. Logit bias -- adds bias values to specific token positions
    3. Min tokens -- prevents stop tokens from being sampled when pos < min_len
    """
    logits = logits.clone()
    num_tokens, vocab_size = logits.shape

    for token_idx in range(num_tokens):
        req_state_idx = int(expanded_idx_mapping[token_idx])

        # 1. Allowed token IDs
        num_allowed = int(num_allowed_token_ids[req_state_idx])
        if num_allowed > 0:
            allowed = allowed_token_ids[req_state_idx, :num_allowed]
            saved = logits[token_idx, allowed].clone()
            logits[token_idx, :] = float("-inf")
            logits[token_idx, allowed] = saved

        # 2. Logit bias
        num_bias = int(num_logit_bias[req_state_idx])
        if num_bias > 0:
            ids = bias_token_ids[req_state_idx, :num_bias]
            bias_vals = bias_values[req_state_idx, :num_bias]
            logits[token_idx, ids] += bias_vals

        # 3. Min tokens (stop token suppression)
        num_stop = int(num_stop_token_ids[req_state_idx])
        min_len = int(min_lens[req_state_idx])
        pos_val = int(pos[token_idx])
        if num_stop > 0 and (pos_val + 1) < min_len:
            stop_ids = stop_token_ids[req_state_idx, :num_stop]
            logits[token_idx, stop_ids] = float("-inf")

    return logits


def _next_power_of_2(n: int) -> int:
    """Mimic triton.next_power_of_2 in plain Python."""
    return 1 << (n - 1).bit_length() if n > 0 else 1


@pytest.mark.parametrize("num_tokens", [1, 4, 8])
@pytest.mark.parametrize("vocab_size", [1024, 8192, 32000])
def test_bias_kernel_basic(num_tokens: int, vocab_size: int) -> None:
    """Bias kernel: apply allowed tokens, logit bias, and min-token suppression.

    Tests with a moderate number of requests, some having allowed tokens,
    logit bias entries, and min-token stop suppression.
    """
    init_device_properties_triton()
    torch.manual_seed(42)

    max_num_reqs = 8
    max_allowed = 64
    max_bias = 64
    max_stop = 16

    expanded_idx_mapping = torch.randint(0, max_num_reqs, (num_tokens,),
                                         dtype=torch.int64)
    pos = torch.randint(0, 50, (num_tokens,), dtype=torch.int64)

    # Per-request state tensors
    num_allowed = torch.zeros(max_num_reqs, dtype=torch.int32)
    allowed_ids = torch.zeros((max_num_reqs, max_allowed), dtype=torch.int32)
    num_bias = torch.zeros(max_num_reqs, dtype=torch.int32)
    bias_ids = torch.zeros((max_num_reqs, max_bias), dtype=torch.int32)
    bias_vals = torch.zeros((max_num_reqs, max_bias), dtype=torch.float32)
    min_lens = torch.full((max_num_reqs,), 9999, dtype=torch.int32)
    num_stop = torch.zeros(max_num_reqs, dtype=torch.int32)
    stop_ids = torch.zeros((max_num_reqs, max_stop), dtype=torch.int32)

    for r in range(max_num_reqs):
        # Some requests get allowed tokens, bias, and min tokens
        if r % 3 == 0:
            n = 5 + r
            num_allowed[r] = n
            allowed_ids[r, :n] = torch.randint(0, vocab_size, (n,))
            n_bias = 3 + r % 5
            num_bias[r] = n_bias
            bias_ids[r, :n_bias] = torch.randint(0, vocab_size, (n_bias,))
            bias_vals[r, :n_bias] = torch.randn(n_bias) * 2.0
            min_lens[r] = 100
            n_stop = 3
            num_stop[r] = n_stop
            stop_ids[r, :n_stop] = torch.tensor([0, 1, vocab_size - 1])
        # Others have no modifications
        else:
            num_allowed[r] = 0
            num_bias[r] = 0
            num_stop[r] = 0

    logits_cpu = torch.randn(num_tokens, vocab_size, dtype=torch.float32)

    expected = _bias_cpu(
        logits_cpu, expanded_idx_mapping,
        num_allowed, allowed_ids,
        num_bias, bias_ids, bias_vals,
        pos, min_lens, num_stop, stop_ids,
    )

    device = torch.device("npu")
    logits_npu = logits_cpu.to(device)
    idx_mapping_npu = expanded_idx_mapping.to(device)
    pos_npu = pos.to(device)

    num_allowed_npu = num_allowed.to(device)
    allowed_ids_npu = allowed_ids.to(device)
    num_bias_npu = num_bias.to(device)
    bias_ids_npu = bias_ids.to(device)
    bias_vals_npu = bias_vals.to(device)
    min_lens_npu = min_lens.to(device)
    num_stop_npu = num_stop.to(device)
    stop_ids_npu = stop_ids.to(device)

    BLOCK_SIZE = _next_power_of_2(max(max_allowed, max_bias, max_stop))
    LOGITS_BLOCK_SIZE = 8192

    _bias_kernel[(num_tokens,)](
        logits_npu,
        logits_npu.stride(0),
        vocab_size,
        idx_mapping_npu,
        num_allowed_npu,
        allowed_ids_npu,
        allowed_ids_npu.stride(0),
        num_bias_npu,
        bias_ids_npu,
        bias_ids_npu.stride(0),
        bias_vals_npu,
        bias_vals_npu.stride(0),
        pos_npu,
        min_lens_npu,
        num_stop_npu,
        stop_ids_npu,
        stop_ids_npu.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        LOGITS_BLOCK_SIZE=LOGITS_BLOCK_SIZE,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(logits_npu.cpu(), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("vocab_size", [1024, 32000])
def test_bias_kernel_allowed_only(vocab_size: int) -> None:
    """Only allowed-token filtering, no bias or min-token suppression."""
    init_device_properties_triton()
    torch.manual_seed(7)

    num_tokens = 3
    max_num_reqs = 2
    max_allowed = 16
    max_bias = 8
    max_stop = 4

    expanded_idx_mapping = torch.tensor([0, 1, 0], dtype=torch.int64)
    pos = torch.tensor([50, 10, 200], dtype=torch.int64)

    num_allowed = torch.tensor([0, 4], dtype=torch.int32)
    allowed_ids = torch.zeros((max_num_reqs, max_allowed), dtype=torch.int32)
    allowed_ids[1, :4] = torch.tensor([10, 100, 500, 999])
    num_bias = torch.zeros(max_num_reqs, dtype=torch.int32)
    bias_ids = torch.zeros((max_num_reqs, max_bias), dtype=torch.int32)
    bias_vals = torch.zeros((max_num_reqs, max_bias), dtype=torch.float32)
    min_lens = torch.zeros(max_num_reqs, dtype=torch.int32)
    num_stop = torch.zeros(max_num_reqs, dtype=torch.int32)
    stop_ids = torch.zeros((max_num_reqs, max_stop), dtype=torch.int32)

    logits_cpu = torch.randn(num_tokens, vocab_size, dtype=torch.float32)

    expected = _bias_cpu(
        logits_cpu, expanded_idx_mapping,
        num_allowed, allowed_ids,
        num_bias, bias_ids, bias_vals,
        pos, min_lens, num_stop, stop_ids,
    )

    device = torch.device("npu")
    logits_npu = logits_cpu.to(device)

    BLOCK_SIZE = _next_power_of_2(max(max_allowed, max_bias, max_stop))
    LOGITS_BLOCK_SIZE = 8192

    _bias_kernel[(num_tokens,)](
        logits_npu,
        logits_npu.stride(0),
        vocab_size,
        expanded_idx_mapping.to(device),
        num_allowed.to(device),
        allowed_ids.to(device),
        allowed_ids.stride(0),
        num_bias.to(device),
        bias_ids.to(device),
        bias_ids.stride(0),
        bias_vals.to(device),
        bias_vals.stride(0),
        pos.to(device),
        min_lens.to(device),
        num_stop.to(device),
        stop_ids.to(device),
        stop_ids.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        LOGITS_BLOCK_SIZE=LOGITS_BLOCK_SIZE,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(logits_npu.cpu(), expected, rtol=0, atol=0)


def test_bias_kernel_no_ops() -> None:
    """No request has allowed tokens, bias, or min-token suppression.

    The kernel should be a no-op (logits unchanged).
    """
    init_device_properties_triton()
    torch.manual_seed(3)

    num_tokens = 2
    vocab_size = 2048
    max_num_reqs = 2
    max_allowed = 16
    max_bias = 8
    max_stop = 4

    expanded_idx_mapping = torch.tensor([0, 1], dtype=torch.int64)
    pos = torch.tensor([0, 0], dtype=torch.int64)

    num_allowed = torch.zeros(max_num_reqs, dtype=torch.int32)
    allowed_ids = torch.zeros((max_num_reqs, max_allowed), dtype=torch.int32)
    num_bias = torch.zeros(max_num_reqs, dtype=torch.int32)
    bias_ids = torch.zeros((max_num_reqs, max_bias), dtype=torch.int32)
    bias_vals = torch.zeros((max_num_reqs, max_bias), dtype=torch.float32)
    min_lens = torch.zeros(max_num_reqs, dtype=torch.int32)
    num_stop = torch.zeros(max_num_reqs, dtype=torch.int32)
    stop_ids = torch.zeros((max_num_reqs, max_stop), dtype=torch.int32)

    logits_cpu = torch.randn(num_tokens, vocab_size, dtype=torch.float32)
    expected = logits_cpu.clone()

    device = torch.device("npu")
    logits_npu = logits_cpu.to(device)

    BLOCK_SIZE = _next_power_of_2(max(max_allowed, max_bias, max_stop))
    LOGITS_BLOCK_SIZE = 8192

    _bias_kernel[(num_tokens,)](
        logits_npu,
        logits_npu.stride(0),
        vocab_size,
        expanded_idx_mapping.to(device),
        num_allowed.to(device),
        allowed_ids.to(device),
        allowed_ids.stride(0),
        num_bias.to(device),
        bias_ids.to(device),
        bias_ids.stride(0),
        bias_vals.to(device),
        bias_vals.stride(0),
        pos.to(device),
        min_lens.to(device),
        num_stop.to(device),
        stop_ids.to(device),
        stop_ids.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        LOGITS_BLOCK_SIZE=LOGITS_BLOCK_SIZE,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(logits_npu.cpu(), expected, rtol=0, atol=0)
