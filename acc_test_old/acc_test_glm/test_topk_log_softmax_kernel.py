import pytest
import torch

from vllm.v1.worker.gpu.sample.logprob import _topk_log_softmax_kernel


def _topk_log_softmax_cpu(
    logits: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    vocab_size: int,
) -> torch.Tensor:
    batch_size = logits.shape[0]
    output = torch.zeros(batch_size, topk, dtype=torch.float32)

    for req_idx in range(batch_size):
        row = logits[req_idx].float()
        max_val = row.max()
        exp_vals = torch.exp(row - max_val)
        lse = torch.log(exp_vals.sum())

        for k_idx in range(topk):
            tid = int(topk_ids[req_idx, k_idx])
            output[req_idx, k_idx] = row[tid] - max_val - lse

    return output


@pytest.mark.parametrize(
    "batch_size,vocab_size,topk",
    [(2, 64, 3), (4, 128, 5), (1, 256, 1)],
)
def test_topk_log_softmax_kernel(batch_size, vocab_size, topk):
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    topk_ids = torch.randint(0, vocab_size, (batch_size, topk), dtype=torch.int64)

    expected = _topk_log_softmax_cpu(logits, topk_ids, topk, vocab_size)

    device = torch.device("npu")
    from vllm.triton_utils import triton

    logprobs = torch.empty(
        batch_size, topk, dtype=torch.float32, device=device
    )

    _topk_log_softmax_kernel[(batch_size,)](
        logprobs,
        logits.to(device),
        logits.stride(0),
        topk_ids.to(device),
        topk,
        vocab_size,
        BLOCK_SIZE=1024,
        PADDED_TOPK=triton.next_power_of_2(topk),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(logprobs.cpu(), expected, atol=1e-4, rtol=1e-4)
