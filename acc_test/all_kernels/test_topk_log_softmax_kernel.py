# vLLM vanilla kernel: _topk_log_softmax_kernel from
# vllm/vllm/v1/worker/gpu/sample/logprob.py

"""
Precision test for _topk_log_softmax_kernel.

Kernel signature:
    _topk_log_softmax_kernel(
        output_ptr,              # fp32 logprobs [batch_size, topk]
        logits_ptr,              # fp32 logits [batch_size, vocab_size]
        logits_stride,           # stride(0) of logits
        topk_ids_ptr,            # int64 token IDs [batch_size, topk]
        topk,                    # number of top-k tokens per row
        vocab_size,              # vocab size
        BLOCK_SIZE: tl.constexpr,      # block for reduction
        TOPK_BLOCK_SIZE: tl.constexpr,  # block for gather
    )

Computes log softmax for specified token IDs per row:
    max_val = max over vocab_size
    lse = log(sum(exp(logits - max_val)))
    output = logits[token_ids] - max_val - lse
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.logprob import _topk_log_softmax_kernel
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

import pytest


def _topk_log_softmax_ref(
    logits: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """CPU reference: log softmax at specified token IDs."""
    batch_size, vocab_size = logits.shape
    topk = topk_ids.shape[1]
    output = torch.empty(batch_size, topk, dtype=torch.float32)
    for b in range(batch_size):
        row = logits[b].to(torch.float32)
        max_val = row.max()
        lse = torch.log(torch.sum(torch.exp(row - max_val)))
        for k in range(topk):
            tid = topk_ids[b, k].item()
            output[b, k] = (row[tid] - max_val - lse).item()
    return output


class TestTopkLogSoftmaxKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("vocab_size", [128, 1024, 4096])
    @pytest.mark.parametrize("topk", [1, 5, 10])
    def test_topk_log_softmax(self, batch_size, vocab_size, topk):
        """Compare GPU topk log softmax with CPU reference."""
        logits = torch.randn(batch_size, vocab_size, dtype=torch.float32, device=self.device)
        # Pick random token IDs
        topk_ids = torch.randint(0, vocab_size, (batch_size, topk), dtype=torch.int64, device=self.device)

        output = torch.empty(batch_size, topk, dtype=torch.float32, device=self.device)

        BLOCK_SIZE = 1024
        TOPK_BLOCK_SIZE = triton.next_power_of_2(topk)

        _topk_log_softmax_kernel[(batch_size,)](
            output,
            logits,
            logits.stride(0),
            topk_ids,
            topk,
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
            TOPK_BLOCK_SIZE=TOPK_BLOCK_SIZE,
        )
        torch.npu.synchronize()

        expected = _topk_log_softmax_ref(logits.cpu(), topk_ids.cpu())

        torch.testing.assert_close(output.cpu(), expected, rtol=1e-5, atol=1e-5)

    def test_extreme_values(self):
        """Test with extreme logit values to check numerical stability."""
        batch_size, vocab_size, topk = 2, 256, 4
        logits = torch.empty(batch_size, vocab_size, dtype=torch.float32, device=self.device)
        logits[0] = torch.tensor(
            [1000.0] * 128 + [-1000.0] * 128, dtype=torch.float32, device=self.device
        )
        logits[1] = torch.tensor(
            [-1000.0] * 128 + [1000.0] * 128, dtype=torch.float32, device=self.device
        )
        topk_ids = torch.tensor([[0, 1, 127, 128], [200, 201, 255, 0]], dtype=torch.int64, device=self.device)

        output = torch.empty(batch_size, topk, dtype=torch.float32, device=self.device)

        BLOCK_SIZE = 1024
        TOPK_BLOCK_SIZE = triton.next_power_of_2(topk)

        _topk_log_softmax_kernel[(batch_size,)](
            output,
            logits,
            logits.stride(0),
            topk_ids,
            topk,
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
            TOPK_BLOCK_SIZE=TOPK_BLOCK_SIZE,
        )
        torch.npu.synchronize()

        expected = _topk_log_softmax_ref(logits.cpu(), topk_ids.cpu())

        torch.testing.assert_close(output.cpu(), expected, rtol=1e-5, atol=1e-5)
