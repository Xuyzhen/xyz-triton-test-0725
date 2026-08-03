# vLLM vanilla kernel: _ranks_kernel from
# vllm/vllm/v1/worker/gpu/sample/logprob.py

"""
Precision test for _ranks_kernel.

Kernel signature:
    _ranks_kernel(
        output_ptr,          # int64 ranks [batch_size]
        logits_ptr,          # fp32 logits [batch_size, vocab_size]
        logits_stride,       # stride(0) of logits
        token_ids_ptr,       # int64 token IDs [batch_size]
        vocab_size,
        BLOCK_SIZE: tl.constexpr,
    )

Computes the rank of a specified token ID per row.  Rank is the number of
tokens with logit >= target_token_logit.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.logprob import _ranks_kernel
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

import pytest


def _ranks_ref(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """CPU reference: rank of token_id in each row."""
    batch_size, vocab_size = logits.shape
    output = torch.empty(batch_size, dtype=torch.int64)
    for b in range(batch_size):
        tid = token_ids[b].item()
        x = logits[b, tid].item()
        count = 0
        for j in range(vocab_size):
            if logits[b, j].item() >= x:
                count += 1
        output[b] = count
    return output


class TestRanksKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("vocab_size", [128, 1024, 4096])
    def test_ranks(self, batch_size, vocab_size):
        """Compare GPU ranks with CPU reference."""
        logits = torch.randn(batch_size, vocab_size, dtype=torch.float32, device=self.device)
        token_ids = torch.randint(0, vocab_size, (batch_size,), dtype=torch.int64, device=self.device)

        output = torch.empty(batch_size, dtype=torch.int64, device=self.device)

        _ranks_kernel[(batch_size,)](
            output,
            logits,
            logits.stride(0),
            token_ids,
            vocab_size,
            BLOCK_SIZE=8192,
        )
        torch.npu.synchronize()

        expected = _ranks_ref(logits.cpu(), token_ids.cpu())

        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

    def test_rank_always_one_for_max(self):
        """The max-value token should have rank 1 (highest)."""
        batch_size, vocab_size = 4, 256
        logits = torch.randn(batch_size, vocab_size, dtype=torch.float32, device=self.device)
        max_indices = torch.argmax(logits, dim=-1).to(torch.int64, device=self.device)

        output = torch.empty(batch_size, dtype=torch.int64, device=self.device)

        _ranks_kernel[(batch_size,)](
            output,
            logits,
            logits.stride(0),
            max_indices,
            vocab_size,
            BLOCK_SIZE=8192,
        )
        torch.npu.synchronize()

        # The highest logit should have rank 1 (counting tokens with logit >= itself)
        torch.testing.assert_close(output.cpu(), torch.ones(batch_size, dtype=torch.int64), rtol=0, atol=0)

    def test_rank_increases_with_identical(self):
        """With identical logit values, all tokens have the same rank (= vocab_size)."""
        batch_size, vocab_size = 2, 100
        logits = torch.full((batch_size, vocab_size), 1.0, dtype=torch.float32, device=self.device)
        # All tokens have the same logit value, so each token's rank = vocab_size
        token_ids = torch.tensor([0, 50], dtype=torch.int64, device=self.device)

        output = torch.empty(batch_size, dtype=torch.int64, device=self.device)

        _ranks_kernel[(batch_size,)](
            output,
            logits,
            logits.stride(0),
            token_ids,
            vocab_size,
            BLOCK_SIZE=8192,
        )
        torch.npu.synchronize()

        expected = torch.full((batch_size,), vocab_size, dtype=torch.int64)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
