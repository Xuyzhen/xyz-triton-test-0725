import pytest
import torch

from vllm.v1.worker.gpu.sample.logprob import _ranks_kernel


def _ranks_cpu(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    batch_size = logits.shape[0]
    ranks = torch.empty(batch_size, dtype=torch.int64)

    for req_idx in range(batch_size):
        token_id = int(token_ids[req_idx])
        x = logits[req_idx, token_id].item()
        n = int((logits[req_idx] >= x).sum().item())
        ranks[req_idx] = n

    return ranks


@pytest.mark.parametrize(
    "batch_size,vocab_size",
    [(1, 64), (4, 128), (8, 256)],
)
def test_ranks_kernel(batch_size, vocab_size):
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    token_ids = torch.randint(0, vocab_size, (batch_size,), dtype=torch.int64)

    expected = _ranks_cpu(logits, token_ids, vocab_size)

    device = torch.device("npu")
    token_ranks = torch.empty(batch_size, dtype=torch.int64, device=device)

    _ranks_kernel[(batch_size,)](
        token_ranks,
        logits.to(device),
        logits.stride(0),
        token_ids.to(device),
        vocab_size,
        BLOCK_SIZE=8192,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(token_ranks.cpu(), expected, rtol=0, atol=0)


def test_ranks_kernel_identical_logits():
    batch_size, vocab_size = 2, 16
    logits = torch.zeros(batch_size, vocab_size, dtype=torch.float32)
    logits[0, 5] = 1.0
    logits[0, 7] = 1.0
    logits[0, 10] = 2.0
    token_ids = torch.tensor([5, 0], dtype=torch.int64)

    expected = _ranks_cpu(logits, token_ids, vocab_size)

    device = torch.device("npu")
    token_ranks = torch.empty(batch_size, dtype=torch.int64, device=device)

    _ranks_kernel[(batch_size,)](
        token_ranks,
        logits.to(device),
        logits.stride(0),
        token_ids.to(device),
        vocab_size,
        BLOCK_SIZE=8192,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(token_ranks.cpu(), expected, rtol=0, atol=0)
