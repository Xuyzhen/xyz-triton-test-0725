# vLLM-Ascend patched kernel: _ranks_kernel from
# vllm-ascend/vllm_ascend/worker/v2/sample/logprob.py:88
# PATCH NOTE: This is an Ascend NPU adaptation of the original vLLM Triton kernel

"""
Precision test for patched _ranks_kernel (Ascend NPU version).

Patch differences vs original vllm:
- Uses do_not_specialize=["batch_size", "rows_per_core"] (parameterized for variable size)
- Uses load-balanced grid based on get_vectorcore_num() instead of fixed grid
- Uses core-based work distribution with start_row/end_row pattern
- Uses tl.zeros for n_vec initialization and tl.sum for reduction
- Uses tl.full for n_vec (different from original which may use different init)
- Uses BLOCK_SIZE=8192 for iteration

Kernel signature:
    _ranks_kernel(
        output_ptr,         # [batch_size] int64: rank of sampled token per request
        logits_ptr,         # [batch_size, vocab_size] input logits
        logits_stride,      # stride(0) of logits
        token_ids_ptr,      # [batch_size] int64: sampled token IDs
        vocab_size,         # scalar: vocab size
        batch_size,         # scalar: number of requests
        rows_per_core,      # scalar: rows per program
        BLOCK_SIZE: tl.constexpr,   # block size (8192)
    )

For each request, counts how many tokens in the vocabulary have strictly greater
logit values than the sampled token, i.e. rank = sum_{v}(logit[v] > logit[sampled]).
"""

import torch

from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

import pytest


def _ranks_ref(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """CPU reference: compute rank of each sampled token."""
    batch_size, vocab_size = logits.shape
    output = torch.empty(batch_size, dtype=torch.int64)

    for req_idx in range(batch_size):
        sampled_token = token_ids[req_idx].item()
        token_logit = logits[req_idx, sampled_token].item()
        count = int(torch.sum(logits[req_idx] > token_logit).item())
        output[req_idx] = count

    return output


class TestRanksKernelPatch:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")
        self.BLOCK_SIZE = 8192

    def _run_kernel(self, logits, token_ids):
        from vllm_ascend.worker.v2.sample.logprob import _ranks_kernel

        batch_size, vocab_size = logits.shape
        vec_core = get_vectorcore_num()
        num_cores = min(batch_size, vec_core)
        rows_per_core = triton.cdiv(batch_size, num_cores)

        output = torch.empty(batch_size, dtype=torch.int64, device=self.device)

        _ranks_kernel[(num_cores,)](
            output,
            logits,
            logits.stride(0),
            token_ids,
            vocab_size,
            batch_size,
            rows_per_core,
            BLOCK_SIZE=self.BLOCK_SIZE,
            multibuffer=False,
        )
        torch.npu.synchronize()

        return output

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    @pytest.mark.parametrize("vocab_size", [128, 1024, 8192, 16384])
    def test_ranks(self, batch_size, vocab_size):
        """Compare NPU ranks with CPU reference."""
        logits = torch.randn(batch_size, vocab_size, dtype=torch.float32, device=self.device)
        token_ids = torch.randint(0, vocab_size, (batch_size,), dtype=torch.int64, device=self.device)

        output_gpu = self._run_kernel(logits, token_ids)
        expected = _ranks_ref(logits.cpu(), token_ids.cpu())

        torch.testing.assert_close(output_gpu.cpu(), expected, rtol=0, atol=0)

    def test_rank_of_max_token(self):
        """The argmax token should have rank 0 (no tokens greater than it)."""
        batch_size, vocab_size = 4, 256
        logits = torch.randn(batch_size, vocab_size, dtype=torch.float32, device=self.device)
        argmax_ids = torch.argmax(logits, dim=-1)

        output_gpu = self._run_kernel(logits, argmax_ids)

        assert torch.all(output_gpu == 0).item(), \
            f"Argmax token should have rank 0, got {output_gpu}"

    def test_rank_of_min_token(self):
        """The argmin token should have rank vocab_size-1."""
        batch_size, vocab_size = 4, 256
        logits = torch.randn(batch_size, vocab_size, dtype=torch.float32, device=self.device)
        argmin_ids = torch.argmin(logits, dim=-1)

        output_gpu = self._run_kernel(logits, argmin_ids)

        # Min token: all other tokens have greater logits, except possibly equals
        expected = _ranks_ref(logits.cpu(), argmin_ids.cpu())
        torch.testing.assert_close(output_gpu.cpu(), expected, rtol=0, atol=0)

    def test_all_same_logits(self):
        """When all logits equal, rank should be 0 (no strictly greater)."""
        batch_size, vocab_size = 2, 128
        logits = torch.full((batch_size, vocab_size), 1.0, dtype=torch.float32, device=self.device)
        token_ids = torch.randint(0, vocab_size, (batch_size,), dtype=torch.int64, device=self.device)

        output_gpu = self._run_kernel(logits, token_ids)

        # No strictly greater values
        assert torch.all(output_gpu == 0).item(), \
            f"With all equal logits, rank should be 0, got {output_gpu}"
