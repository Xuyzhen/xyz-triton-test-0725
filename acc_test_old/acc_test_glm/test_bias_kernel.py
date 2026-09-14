import pytest
import torch

from vllm.v1.worker.gpu.sample.logit_bias import _bias_kernel


def _bias_kernel_cpu(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    num_allowed_token_ids: torch.Tensor,
    allowed_token_ids: torch.Tensor,
    num_logit_bias: torch.Tensor,
    bias_token_ids: torch.Tensor,
    bias: torch.Tensor,
    pos: torch.Tensor,
    min_lens: torch.Tensor,
    num_stop_token_ids: torch.Tensor,
    stop_token_ids: torch.Tensor,
) -> torch.Tensor:
    out = logits.clone()
    num_tokens, vocab_size = out.shape
    for token_idx in range(num_tokens):
        req_state_idx = int(expanded_idx_mapping[token_idx])

        n_allowed = int(num_allowed_token_ids[req_state_idx])
        if n_allowed > 0:
            saved = []
            for i in range(n_allowed):
                tid = int(allowed_token_ids[req_state_idx, i])
                saved.append((tid, out[token_idx, tid].item()))
            out[token_idx, :] = float("-inf")
            for tid, val in saved:
                out[token_idx, tid] = val

        n_bias = int(num_logit_bias[req_state_idx])
        if n_bias > 0:
            for i in range(n_bias):
                tid = int(bias_token_ids[req_state_idx, i])
                b = float(bias[req_state_idx, i])
                out[token_idx, tid] = out[token_idx, tid] + b

        n_stop = int(num_stop_token_ids[req_state_idx])
        p = int(pos[token_idx])
        ml = int(min_lens[req_state_idx])
        if n_stop > 0 and p + 1 < ml:
            for i in range(n_stop):
                tid = int(stop_token_ids[req_state_idx, i])
                out[token_idx, tid] = float("-inf")

    return out


@pytest.mark.parametrize(
    "num_tokens,vocab_size,num_requests",
    [(4, 64, 2), (8, 128, 4)],
)
def test_bias_kernel_allowed_token_ids(num_tokens, vocab_size, num_requests):
    torch.manual_seed(42)
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32)
    expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32) % num_requests

    MAX_ALLOWED = 16
    num_allowed_token_ids = torch.zeros(num_requests, dtype=torch.int32)
    allowed_token_ids = torch.zeros(num_requests, MAX_ALLOWED, dtype=torch.int32)
    num_allowed_token_ids[0] = 3
    allowed_token_ids[0, :3] = torch.tensor([5, 10, 20])
    num_allowed_token_ids[1] = 0

    num_logit_bias = torch.zeros(num_requests, dtype=torch.int32)
    bias_token_ids = torch.zeros(num_requests, 16, dtype=torch.int32)
    bias_tensor = torch.zeros(num_requests, 16, dtype=torch.float32)
    pos = torch.zeros(num_tokens, dtype=torch.int32)
    min_lens = torch.zeros(num_requests, dtype=torch.int32)
    num_stop_token_ids = torch.zeros(num_requests, dtype=torch.int32)
    stop_token_ids = torch.zeros(num_requests, 32, dtype=torch.int32)

    expected = _bias_kernel_cpu(
        logits,
        expanded_idx_mapping,
        num_allowed_token_ids,
        allowed_token_ids,
        num_logit_bias,
        bias_token_ids,
        bias_tensor,
        pos,
        min_lens,
        num_stop_token_ids,
        stop_token_ids,
    )

    device = torch.device("npu")
    BLOCK_SIZE = 16
    LOGITS_BLOCK_SIZE = 64
    _bias_kernel[(num_tokens,)](
        logits.to(device),
        logits.stride(0),
        vocab_size,
        expanded_idx_mapping.to(device),
        num_allowed_token_ids.to(device),
        allowed_token_ids.to(device),
        allowed_token_ids.stride(0),
        num_logit_bias.to(device),
        bias_token_ids.to(device),
        bias_token_ids.stride(0),
        bias_tensor.to(device),
        bias_tensor.stride(0),
        pos.to(device),
        min_lens.to(device),
        num_stop_token_ids.to(device),
        stop_token_ids.to(device),
        stop_token_ids.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        LOGITS_BLOCK_SIZE=LOGITS_BLOCK_SIZE,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(logits.cpu(), expected, atol=1e-5, rtol=1e-5)


def test_bias_kernel_logit_bias_and_min_tokens():
    torch.manual_seed(42)
    num_tokens, vocab_size, num_requests = 2, 64, 1
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32)
    expanded_idx_mapping = torch.zeros(num_tokens, dtype=torch.int32)

    MAX_BIAS = 16
    num_allowed_token_ids = torch.zeros(num_requests, dtype=torch.int32)
    allowed_token_ids = torch.zeros(num_requests, 16, dtype=torch.int32)
    num_logit_bias = torch.tensor([2], dtype=torch.int32)
    bias_token_ids = torch.zeros(num_requests, MAX_BIAS, dtype=torch.int32)
    bias_token_ids[0, :2] = torch.tensor([3, 7])
    bias_tensor = torch.zeros(num_requests, MAX_BIAS, dtype=torch.float32)
    bias_tensor[0, :2] = torch.tensor([1.5, -2.0])

    pos = torch.tensor([2, 10], dtype=torch.int32)
    min_lens = torch.tensor([15], dtype=torch.int32)
    num_stop_token_ids = torch.tensor([2], dtype=torch.int32)
    stop_token_ids = torch.zeros(num_requests, 32, dtype=torch.int32)
    stop_token_ids[0, :2] = torch.tensor([50, 51])

    expected = _bias_kernel_cpu(
        logits,
        expanded_idx_mapping,
        num_allowed_token_ids,
        allowed_token_ids,
        num_logit_bias,
        bias_token_ids,
        bias_tensor,
        pos,
        min_lens,
        num_stop_token_ids,
        stop_token_ids,
    )

    device = torch.device("npu")
    _bias_kernel[(num_tokens,)](
        logits.to(device),
        logits.stride(0),
        vocab_size,
        expanded_idx_mapping.to(device),
        num_allowed_token_ids.to(device),
        allowed_token_ids.to(device),
        allowed_token_ids.stride(0),
        num_logit_bias.to(device),
        bias_token_ids.to(device),
        bias_token_ids.stride(0),
        bias_tensor.to(device),
        bias_tensor.stride(0),
        pos.to(device),
        min_lens.to(device),
        num_stop_token_ids.to(device),
        stop_token_ids.to(device),
        stop_token_ids.stride(0),
        BLOCK_SIZE=16,
        LOGITS_BLOCK_SIZE=64,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(logits.cpu(), expected, atol=1e-5, rtol=1e-5)
