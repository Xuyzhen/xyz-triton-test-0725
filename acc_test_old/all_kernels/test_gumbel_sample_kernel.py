# vLLM vanilla kernel: _gumbel_sample_kernel from
# vllm/vllm/v1/worker/gpu/sample/gumbel.py

"""
Precision test for _gumbel_sample_kernel.

Kernel signature:
    _gumbel_sample_kernel(
        local_argmax_ptr,           # [num_tokens, num_blocks] output token ids per block
        local_argmax_stride,        # stride(0) of local_argmax
        local_max_ptr,              # [num_tokens, num_blocks] output max values per block
        local_max_stride,           # stride(0) of local_max
        processed_logits_ptr,       # optional [max_num_reqs, col * vocab_size] or None
        processed_logits_stride,    # stride(0) of processed_logits
        processed_logits_col_ptr,   # optional col index
        logits_ptr,                 # fp32 logits [num_tokens, vocab_size]
        logits_stride,              # stride(0) of logits
        expanded_idx_mapping_ptr,   # [num_tokens] token_idx -> req_state_idx
        seeds_ptr,                  # [max_num_reqs]
        pos_ptr,                    # [num_tokens]
        temp_ptr,                   # [max_num_reqs]
        vocab_size,
        BLOCK_SIZE: tl.constexpr,
        APPLY_TEMPERATURE: tl.constexpr,
        USE_FP64: tl.constexpr,
        PER_TOKEN_COL: tl.constexpr,
    )

Full Gumbel sampling kernel.  Loads logits, optionally applies temperature,
adds Gumbel noise, finds block-level argmax, and stores block results for
reduction.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import _gumbel_sample_kernel
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

import pytest


def _gumbel_sample_ref(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    apply_temperature: bool,
    use_fp64: bool = False,
) -> torch.Tensor:
    """CPU reference for Gumbel sampling.

    NOTE: Does not reproduce the kernel's exact random numbers; this is a
    functional end-to-end check that the kernel returns valid token ids in
    range.
    """
    num_tokens, vocab_size = logits.shape
    result = torch.empty(num_tokens, dtype=torch.int64)
    for token_idx in range(num_tokens):
        req_state_idx = expanded_idx_mapping[token_idx].item()
        temp = temperature[req_state_idx].item()

        row = logits[token_idx].clone()
        if apply_temperature and temp != 0.0 and temp != 1.0:
            row = row / temp

        # Sample deterministic argmax (no noise in ref)
        argmax_idx = torch.argmax(row).item()
        result[token_idx] = argmax_idx
    return result


class TestGumbelSampleKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    def _run_sample(
        self,
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature=False,
        use_fp64=False,
        output_processed_logits=None,
        output_processed_logits_col=None,
    ):
        """Run gumbel_sample logic (kernel + reduction)."""
        num_tokens, vocab_size = logits.shape
        BLOCK_SIZE = 1024
        num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
        local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
        local_max_dtype = torch.float64 if use_fp64 else torch.float32
        local_max = logits.new_empty(num_tokens, num_blocks, dtype=local_max_dtype)
        per_token_col = (
            output_processed_logits_col is not None
            and output_processed_logits_col.dim() > 0
        )
        _gumbel_sample_kernel[(num_tokens, num_blocks)](
            local_argmax,
            local_argmax.stride(0),
            local_max,
            local_max.stride(0),
            output_processed_logits,
            output_processed_logits.stride(0) if output_processed_logits is not None else 0,
            output_processed_logits_col,
            logits,
            logits.stride(0),
            expanded_idx_mapping,
            seed,
            pos,
            temperature,
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
            APPLY_TEMPERATURE=apply_temperature,
            USE_FP64=use_fp64,
            PER_TOKEN_COL=per_token_col,
        )
        torch.npu.synchronize()

        # Reduction: argmax over blocks
        max_block_idx = local_max.argmax(dim=-1, keepdim=True)
        sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
        return sampled

    @pytest.mark.parametrize("num_tokens", [1, 4, 8])
    @pytest.mark.parametrize("vocab_size", [128, 1024, 4096])
    @pytest.mark.parametrize("apply_temp", [True, False])
    @pytest.mark.parametrize("use_fp64", [True, False])
    def test_valid_indices(self, num_tokens, vocab_size, apply_temp, use_fp64):
        """Sampled indices should be in [0, vocab_size)."""
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        num_reqs = 4
        expanded_idx_mapping = torch.randint(0, num_reqs, (num_tokens,), dtype=torch.int64, device=self.device)
        temperature = torch.full((num_reqs,), 1.0, dtype=torch.float32, device=self.device)
        seed = torch.full((num_reqs,), 42, dtype=torch.int64, device=self.device)
        pos = torch.randint(0, 100, (num_tokens,), dtype=torch.int64, device=self.device)

        sampled = self._run_sample(
            logits, expanded_idx_mapping, temperature, seed, pos,
            apply_temperature=apply_temp, use_fp64=use_fp64,
        )

        assert (sampled >= 0).all().item(), "Sampled indices should be >= 0"
        assert (sampled < vocab_size).all().item(), f"Sampled indices should be < {vocab_size}"

    def test_zero_temperature(self):
        """When temperature=0, Gumbel noise is not added -> deterministic argmax."""
        num_tokens = 4
        vocab_size = 256
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int64, device=self.device)
        temperature = torch.zeros(num_tokens, dtype=torch.float32, device=self.device)
        seed = torch.full((num_tokens,), 42, dtype=torch.int64, device=self.device)
        pos = torch.zeros(num_tokens, dtype=torch.int64, device=self.device)

        # Run the full pipeline
        BLOCK_SIZE = 1024
        num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
        local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
        local_max_dtype = torch.float32
        local_max = logits.new_empty(num_tokens, num_blocks, dtype=local_max_dtype)
        _gumbel_sample_kernel[(num_tokens, num_blocks)](
            local_argmax,
            local_argmax.stride(0),
            local_max,
            local_max.stride(0),
            None,
            0,
            None,
            logits,
            logits.stride(0),
            expanded_idx_mapping,
            seed,
            pos,
            temperature,
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
            APPLY_TEMPERATURE=True,
            USE_FP64=False,
            PER_TOKEN_COL=False,
        )
        torch.npu.synchronize()
        max_block_idx = local_max.argmax(dim=-1, keepdim=True)
        sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)

        # With temp=0, the kernel should produce deterministic argmax
        expected = torch.argmax(logits, dim=-1)
        torch.testing.assert_close(sampled.cpu(), expected.cpu(), rtol=0, atol=0)
