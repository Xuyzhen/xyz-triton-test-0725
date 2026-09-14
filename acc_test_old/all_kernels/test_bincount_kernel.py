# vLLM vanilla kernel: _bincount_kernel from vllm/vllm/v1/worker/gpu/sample/penalties.py

"""
Precision test for _bincount_kernel.

Kernel signature:
    _bincount_kernel(
        expanded_idx_mapping_ptr,     # int32 [num_tokens] token_idx -> req_state_idx
        all_token_ids_ptr,            # int32 [max_num_reqs, max_model_len]
        all_token_ids_stride,         # stride(0) of all_token_ids
        prompt_len_ptr,               # int32 [max_num_reqs]
        prefill_len_ptr,              # int32 [max_num_reqs]
        prompt_bin_mask_ptr,          # int32 [max_num_reqs, cdiv(vocab_size, 32)]
        prompt_bin_mask_stride,       # stride(0) of prompt_bin_mask
        output_bin_counts_ptr,        # int32 [max_num_reqs, vocab_size]
        output_bin_counts_stride,     # stride(0) of output_bin_counts
        BLOCK_SIZE: tl.constexpr,     # block size for iteration
    )

Counts token occurrences from all_token_ids into prompt_bin_mask (for prompt tokens)
and output_bin_counts (for output tokens). Uses atomic operations.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.penalties import _bincount_kernel
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

import pytest


class TestBincountKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_tokens", [1, 4])
    @pytest.mark.parametrize("prompt_len", [4, 16])
    @pytest.mark.parametrize("prefill_len", [8, 24])
    def test_bincount(self, num_tokens, prompt_len, prefill_len):
        """Compare bincount GPU output with CPU reference."""
        assert prefill_len >= prompt_len, "prefill_len must be >= prompt_len"
        max_model_len = 64
        vocab_size = 128
        max_num_reqs = 4
        max_prefill_len = prefill_len

        all_token_ids = torch.randint(0, vocab_size, (max_num_reqs, max_model_len), dtype=torch.int32, device=self.device)
        expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=self.device) % max_num_reqs
        prompt_len_t = torch.full((max_num_reqs,), prompt_len, dtype=torch.int32, device=self.device)
        prefill_len_t = torch.full((max_num_reqs,), prefill_len, dtype=torch.int32, device=self.device)

        num_bins = triton.cdiv(vocab_size, 32)
        prompt_bin_mask = torch.zeros(max_num_reqs, num_bins, dtype=torch.int32, device=self.device)
        output_bin_counts = torch.zeros(max_num_reqs, vocab_size, dtype=torch.int32, device=self.device)

        num_blocks = triton.cdiv(max_prefill_len, 1024)
        _bincount_kernel[(num_tokens, num_blocks)](
            expanded_idx_mapping,
            all_token_ids,
            all_token_ids.stride(0),
            prompt_len_t,
            prefill_len_t,
            prompt_bin_mask,
            prompt_bin_mask.stride(0),
            output_bin_counts,
            output_bin_counts.stride(0),
            BLOCK_SIZE=1024,
        )
        torch.npu.synchronize()

        # CPU reference
        prompt_bin_ref = torch.zeros(max_num_reqs, num_bins, dtype=torch.int32)
        output_ref = torch.zeros(max_num_reqs, vocab_size, dtype=torch.int32)
        for t in range(num_tokens):
            rs_idx = expanded_idx_mapping[t].item()
            plen = prompt_len
            prefill = prefill_len
            # Prompt part: set bits in bin mask
            for pos in range(plen):
                tok = all_token_ids[rs_idx, pos].item()
                bidx = tok // 32
                bit = tok % 32
                prompt_bin_ref[rs_idx, bidx] |= (1 << bit)
            # Output part: count tokens in [prompt_len, prefill_len)
            for pos in range(plen, prefill):
                tok = all_token_ids[rs_idx, pos].item()
                output_ref[rs_idx, tok] += 1

        torch.testing.assert_close(prompt_bin_mask.cpu(), prompt_bin_ref, rtol=0, atol=0)
        torch.testing.assert_close(output_bin_counts.cpu(), output_ref, rtol=0, atol=0)

    def test_bincount_respects_block_boundaries(self):
        """prefill_len < BLOCK_SIZE: kernel returns early for out-of-range blocks."""
        num_tokens = 1
        max_num_reqs = 1
        vocab_size = 64
        max_model_len = 32
        prompt_len = 2
        prefill_len = 5

        all_token_ids = torch.arange(max_model_len, dtype=torch.int32, device=self.device).unsqueeze(0)
        expanded_idx_mapping = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        prompt_len_t = torch.full((max_num_reqs,), prompt_len, dtype=torch.int32, device=self.device)
        prefill_len_t = torch.full((max_num_reqs,), prefill_len, dtype=torch.int32, device=self.device)

        num_bins = triton.cdiv(vocab_size, 32)
        prompt_bin_mask = torch.zeros(max_num_reqs, num_bins, dtype=torch.int32, device=self.device)
        output_bin_counts = torch.zeros(max_num_reqs, vocab_size, dtype=torch.int32, device=self.device)

        num_blocks = triton.cdiv(prefill_len, 1024)  # 1 block
        _bincount_kernel[(num_tokens, num_blocks)](
            expanded_idx_mapping,
            all_token_ids,
            all_token_ids.stride(0),
            prompt_len_t,
            prefill_len_t,
            prompt_bin_mask,
            prompt_bin_mask.stride(0),
            output_bin_counts,
            output_bin_counts.stride(0),
            BLOCK_SIZE=1024,
        )
        torch.npu.synchronize()

        # Reference: prompt tokens 0,1 set bits; output tokens 2,3,4 get counts
        prompt_bin_ref = torch.zeros(max_num_reqs, num_bins, dtype=torch.int32)
        prompt_bin_ref[0, 0] |= (1 << 0) | (1 << 1)
        output_ref = torch.zeros(max_num_reqs, vocab_size, dtype=torch.int32)
        output_ref[0, 2] = 1
        output_ref[0, 3] = 1
        output_ref[0, 4] = 1

        torch.testing.assert_close(prompt_bin_mask.cpu(), prompt_bin_ref, rtol=0, atol=0)
        torch.testing.assert_close(output_bin_counts.cpu(), output_ref, rtol=0, atol=0)

    def test_zero_prefill_early_return(self):
        """When block_idx * BLOCK_SIZE >= prefill_len, kernel returns early."""
        num_tokens = 1
        max_num_reqs = 1
        max_model_len = 10
        vocab_size = 16

        all_token_ids = torch.arange(max_model_len, dtype=torch.int32, device=self.device).unsqueeze(0)
        expanded_idx_mapping = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        prompt_len_t = torch.zeros(max_num_reqs, dtype=torch.int32, device=self.device)
        prefill_len_t = torch.zeros(max_num_reqs, dtype=torch.int32, device=self.device)

        num_bins = triton.cdiv(vocab_size, 32)
        prompt_bin_mask = torch.ones(max_num_reqs, num_bins, dtype=torch.int32, device=self.device)
        output_bin_counts = torch.ones(max_num_reqs, vocab_size, dtype=torch.int32, device=self.device)

        expected_prompt_bin = prompt_bin_mask.clone().cpu()
        expected_output = output_bin_counts.clone().cpu()

        num_blocks = 2
        _bincount_kernel[(num_tokens, num_blocks)](
            expanded_idx_mapping,
            all_token_ids,
            all_token_ids.stride(0),
            prompt_len_t,
            prefill_len_t,
            prompt_bin_mask,
            prompt_bin_mask.stride(0),
            output_bin_counts,
            output_bin_counts.stride(0),
            BLOCK_SIZE=1024,
        )
        torch.npu.synchronize()

        # Should be unchanged because block 0 * 1024 >= 0, so early return
        torch.testing.assert_close(prompt_bin_mask.cpu(), expected_prompt_bin, rtol=0, atol=0)
        torch.testing.assert_close(output_bin_counts.cpu(), expected_output, rtol=0, atol=0)
