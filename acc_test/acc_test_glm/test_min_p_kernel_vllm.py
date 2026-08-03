import pytest
import torch

from vllm.v1.worker.gpu.sample.min_p import _min_p_kernel


def _min_p_cpu(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    min_p: torch.Tensor,
) -> torch.Tensor:
    out = logits.clone()
    num_tokens, vocab_size = out.shape
    for token_idx in range(num_tokens):
        req_state_idx = int(expanded_idx_mapping[token_idx])
        p = float(min_p[req_state_idx])
        if p <= 0.0 or p >= 1.0:
            continue
        row = out[token_idx].float()
        max_val = row.max().item()
        if max_val == float("-inf"):
            continue
        probs = torch.softmax(row, dim=0)
        threshold = p * probs.max().item()
        mask = probs < threshold
        out[token_idx, mask] = float("-inf")
    return out


@pytest.mark.parametrize(
    "num_tokens,vocab_size,num_requests",
    [(4, 64, 2), (8, 128, 4)],
)
def test_min_p_kernel(num_tokens, vocab_size, num_requests):
    torch.manual_seed(42)
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32)
    expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32) % num_requests
    min_p = torch.zeros(num_requests, dtype=torch.float32)
    min_p[0] = 0.05
    min_p[1] = 0.1

    expected = _min_p_cpu(logits, expanded_idx_mapping, min_p)

    device = torch.device("npu")
    from vllm.triton_utils import triton

    BLOCK_SIZE = 8192
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)

    _min_p_kernel[(num_tokens, num_blocks)](
        logits.to(device),
        logits.stride(0),
        vocab_size,
        expanded_idx_mapping.to(device),
        min_p.to(device),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    result = logits.cpu()
    for i in range(num_tokens):
        for j in range(vocab_size):
            if expected[i, j] == float("-inf"):
                assert result[i, j] == float("-inf"), (
                    f"Mismatch at [{i},{j}]: expected -inf, got {result[i,j]}"
                )
            elif result[i, j] != float("-inf"):
                torch.testing.assert_close(
                    torch.tensor(result[i, j].item()),
                    torch.tensor(expected[i, j].item()),
                    atol=1e-4,
                    rtol=1e-4,
                )


def test_min_p_kernel_zero_min_p():
    num_tokens, vocab_size, num_requests = 2, 64, 1
    torch.manual_seed(42)
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32)
    expanded_idx_mapping = torch.zeros(num_tokens, dtype=torch.int32)
    min_p = torch.tensor([0.0], dtype=torch.float32)

    expected = _min_p_cpu(logits, expanded_idx_mapping, min_p)

    device = torch.device("npu")
    from vllm.triton_utils import triton

    _min_p_kernel[(num_tokens, triton.cdiv(vocab_size, 8192))](
        logits.to(device),
        logits.stride(0),
        vocab_size,
        expanded_idx_mapping.to(device),
        min_p.to(device),
        BLOCK_SIZE=8192,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(logits.cpu(), expected, atol=1e-4, rtol=1e-4)
