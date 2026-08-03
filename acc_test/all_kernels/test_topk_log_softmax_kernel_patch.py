# vLLM-Ascend patched kernel: _topk_log_softmax_kernel from
# vllm-ascend/vllm_ascend/worker/v2/sample/logprob.py:29
# PATCH NOTE: This is an Ascend NPU adaptation of the original vLLM Triton kernel

"""
Precision test for patched _topk_log_softmax_kernel (Ascend NPU version).

Patch differences vs original vllm:
- Uses float("-inf") for padding instead of -float("inf")
- Uses BLOCK_SIZE=12944 (different from original for NPU optimization)
- Uses PADDED_TOPK=max(triton.next_power_of_2(num_logprobs), 2) for kernel specialization
- Uses propagate_nan=tl.PropagateNan.ALL in tl.maximum (explicit NaN propagation)
- Uses tl.where + mask pattern for loading topk_ids

Kernel signature:
    _topk_log_softmax_kernel(
        output_ptr,         # [batch_size, topk] output logprobs
        logits_ptr,         # [batch_size, vocab_size] input logits
        logits_stride,      # stride(0) of logits
        topk_ids_ptr,       # [batch_size, topk] token IDs to compute logprobs for
        topk,               # scalar: number of top-k IDs
        vocab_size,         # scalar: vocab size
        BLOCK_SIZE: tl.constexpr,   # block size (12944)
        PADDED_TOPK: tl.constexpr,  # padded topk (next power of 2)
    )

Computes log P(token_i | logits) for specified token IDs using the log-softmax
formula: logprob = logit - log(sum(exp(logits - max_val))) - max_val.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

import pytest


def _topk_log_softmax_ref(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """CPU reference for _topk_log_softmax_kernel using numerically stable log-softmax."""
    batch_size, vocab_size = logits.shape
    num_logprobs = token_ids.shape[1]
    output = torch.empty(batch_size, num_logprobs, dtype=torch.float32)

    for i in range(batch_size):
        row = logits[i].to(torch.float64)
        max_val = float(row.max())
        shifted = row - max_val
        logsumexp = max_val + float(torch.log(torch.sum(torch.exp(shifted))))
        for k in range(num_logprobs):
            tid = token_ids[i, k].item()
            output[i, k] = float(logits[i, tid].to(torch.float32)) - logsumexp

    return output


class TestTopkLogSoftmaxKernelPatch:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("vocab_size", [128, 1024, 8192, 32768])
    @pytest.mark.parametrize("topk", [1, 3, 5])
    def test_topk_log_softmax(self, batch_size, vocab_size, topk):
        """Compare NPU top-k log-softmax with CPU reference."""
        from vllm_ascend.worker.v2.sample.logprob import _topk_log_softmax_kernel

        logits = torch.randn(batch_size, vocab_size, dtype=torch.float32, device=self.device)
        token_ids = torch.randint(0, vocab_size, (batch_size, topk), dtype=torch.int64, device=self.device)

        output = torch.empty(batch_size, topk, dtype=torch.float32, device=self.device)
        padded_topk = max(triton.next_power_of_2(topk), 2)

        _topk_log_softmax_kernel[(batch_size,)](
            output,
            logits,
            logits.stride(0),
            token_ids,
            topk,
            vocab_size,
            BLOCK_SIZE=12944,
            PADDED_TOPK=padded_topk,
            multibuffer=False,
        )
        torch.npu.synchronize()

        expected = _topk_log_softmax_ref(logits.cpu(), token_ids.cpu())
        torch.testing.assert_close(output.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_with_extreme_values(self):
        """Test numerical stability with extreme logit values."""
        from vllm_ascend.worker.v2.sample.logprob import _topk_log_softmax_kernel

        batch_size, vocab_size, topk = 2, 1024, 3
        logits = torch.empty(batch_size, vocab_size, dtype=torch.float32, device=self.device)
        logits[0] = torch.randn(vocab_size, device=self.device) * 20  # Large range
        logits[1] = torch.randn(vocab_size, device=self.device) * 0.1  # Tiny range

        token_ids = torch.randint(0, vocab_size, (batch_size, topk), dtype=torch.int64, device=self.device)

        output = torch.empty(batch_size, topk, dtype=torch.float32, device=self.device)
        padded_topk = max(triton.next_power_of_2(topk), 2)

        _topk_log_softmax_kernel[(batch_size,)](
            output,
            logits,
            logits.stride(0),
            token_ids,
            topk,
            vocab_size,
            BLOCK_SIZE=12944,
            PADDED_TOPK=padded_topk,
            multibuffer=False,
        )
        torch.npu.synchronize()

        expected = _topk_log_softmax_ref(logits.cpu(), token_ids.cpu())
        torch.testing.assert_close(output.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_compute_token_logprobs_wrapper(self):
        """Test the compute_token_logprobs convenience wrapper."""
        from vllm_ascend.worker.v2.sample.logprob import compute_token_logprobs

        batch_size, vocab_size = 3, 4096
        logits = torch.randn(batch_size, vocab_size, dtype=torch.float32, device=self.device)
        token_ids = torch.randint(0, vocab_size, (batch_size, 4), dtype=torch.int64, device=self.device)

        result = compute_token_logprobs(logits, token_ids)

        expected = _topk_log_softmax_ref(logits.cpu(), token_ids.cpu())
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)
