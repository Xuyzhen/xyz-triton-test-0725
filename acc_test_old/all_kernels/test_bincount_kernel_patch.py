# vLLM-Ascend patched kernel: _bincount_kernel from
# vllm-ascend/vllm_ascend/worker/v2/sample/penalties.py:143
# PATCH NOTE: This is an Ascend NPU adaptation of the original vLLM Triton kernel

"""
Precision test for patched _bincount_kernel (Ascend NPU version).

Patch differences vs original vllm:
- Uses tl.atomic_or and tl.atomic_add for bin counting (NPU atomic support)
- Simplified conditional logic for separating prompt/output token counting
- Uses BLOCK_SIZE=1024 (consistent with original)
- Bit-level packing: 32 bits per int32 for prompt_bin_mask
- Computes idx = prompt_tokens // 32 and bit_idx = prompt_tokens % 32

Kernel signature:
    _bincount_kernel(
        expanded_idx_mapping_ptr,       # [num_tokens] token_idx -> req_state_idx
        all_token_ids_ptr,              # [max_num_reqs, max_prefill_len]
        all_token_ids_stride,           # stride(0) of all_token_ids
        prompt_len_ptr,                 # [max_num_reqs]
        prefill_len_ptr,                # [max_num_reqs]
        prompt_bin_mask_ptr,            # [max_num_reqs, padded_vocab//32]
        prompt_bin_mask_stride,         # stride(0) of prompt_bin_mask
        output_bin_counts_ptr,          # [max_num_reqs, vocab_size]
        output_bin_counts_stride,       # stride(0) of output_bin_counts
        BLOCK_SIZE: tl.constexpr,       # block size (1024)
    )

For each token, builds:
- prompt_bin_mask: bitmask of token IDs in the prompt region
- output_bin_counts: count of token IDs in the output region
"""

import torch

from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

import pytest


def _bincount_ref(
    expanded_idx_mapping: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    prefill_len: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
    max_prefill_len: int,
):
    """CPU reference for _bincount_kernel."""
    num_tokens = expanded_idx_mapping.shape[0]

    for token_idx in range(num_tokens):
        req_state_idx = expanded_idx_mapping[token_idx].item()
        plen = prompt_len[req_state_idx].item()
        prefill = prefill_len[req_state_idx].item()

        # Reset bins for this request
        prompt_bin_mask[req_state_idx] = 0
        output_bin_counts[req_state_idx] = 0

        for pos in range(plen):
            token = all_token_ids[req_state_idx, pos].item()
            word_idx = token // 32
            bit_idx = token % 32
            prompt_bin_mask[req_state_idx, word_idx] |= (1 << bit_idx)

        for pos in range(plen, prefill):
            token = all_token_ids[req_state_idx, pos].item()
            output_bin_counts[req_state_idx, token] += 1


class TestBincountKernelPatch:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    def _run_kernel(
        self,
        expanded_idx_mapping,
        all_token_ids,
        prompt_len,
        prefill_len,
        prompt_bin_mask,
        output_bin_counts,
        max_prefill_len,
    ):
        from vllm_ascend.worker.v2.sample.penalties import _bincount_kernel

        num_tokens = expanded_idx_mapping.shape[0]
        BLOCK_SIZE = 1024
        num_blocks = triton.cdiv(max_prefill_len, BLOCK_SIZE)

        _bincount_kernel[(num_tokens, num_blocks)](
            expanded_idx_mapping,
            all_token_ids,
            all_token_ids.stride(0),
            prompt_len,
            prefill_len,
            prompt_bin_mask,
            prompt_bin_mask.stride(0),
            output_bin_counts,
            output_bin_counts.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.npu.synchronize()

    @pytest.mark.parametrize("vocab_size", [128, 512])
    def test_bincount_basic(self, vocab_size):
        """Verify bincount produces correct mask and counts for a single request."""
        num_tokens = 1
        max_reqs = 1
        max_prefill_len = 16

        expanded_idx_mapping = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        all_token_ids = torch.zeros(max_reqs, max_prefill_len, dtype=torch.int32, device=self.device)
        prompt_len = torch.tensor([4], dtype=torch.int32, device=self.device)
        prefill_len = torch.tensor([8], dtype=torch.int32, device=self.device)

        # Prompt tokens (0..3)
        prompt_tokens = [10, 15, 20, 25]
        for i, t in enumerate(prompt_tokens):
            all_token_ids[0, i] = t
        # Output tokens (4..7)
        output_tokens = [30, 10, 30, 40]
        for i, t in enumerate(output_tokens):
            all_token_ids[0, 4 + i] = t

        padded_vocab = triton.cdiv(vocab_size, 32)
        prompt_bin_mask = torch.zeros(max_reqs, padded_vocab, dtype=torch.int32, device=self.device)
        output_bin_counts = torch.zeros(max_reqs, vocab_size, dtype=torch.int32, device=self.device)

        self._run_kernel(
            expanded_idx_mapping, all_token_ids, prompt_len, prefill_len,
            prompt_bin_mask, output_bin_counts, max_prefill_len,
        )

        # Verify prompt bitmask
        for t in prompt_tokens:
            word_idx = t // 32
            bit_idx = t % 32
            assert (prompt_bin_mask[0, word_idx].item() >> bit_idx) & 1, \
                f"Token {t} not set in prompt_bin_mask"

        # Verify output counts
        assert output_bin_counts[0, 10].item() == 1, "Token 10 should have count 1"
        assert output_bin_counts[0, 30].item() == 2, "Token 30 should have count 2"
        assert output_bin_counts[0, 40].item() == 1, "Token 40 should have count 1"

    def test_bincount_multiple_requests(self):
        """Verify bincount works correctly with multiple requests."""
        num_tokens = 2
        max_reqs = 2
        vocab_size = 256
        max_prefill_len = 8

        expanded_idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
        all_token_ids = torch.zeros(max_reqs, max_prefill_len, dtype=torch.int32, device=self.device)
        prompt_len = torch.tensor([3, 2], dtype=torch.int32, device=self.device)
        prefill_len = torch.tensor([5, 4], dtype=torch.int32, device=self.device)

        # Request 0: prompt=[1,2,3], output=[4,5]
        all_token_ids[0, 0:3] = torch.tensor([1, 2, 3])
        all_token_ids[0, 3:5] = torch.tensor([4, 5])
        # Request 1: prompt=[10,20], output=[30]
        all_token_ids[1, 0:2] = torch.tensor([10, 20])
        all_token_ids[1, 2:3] = torch.tensor([30])

        padded_vocab = triton.cdiv(vocab_size, 32)
        prompt_bin_mask = torch.zeros(max_reqs, padded_vocab, dtype=torch.int32, device=self.device)
        output_bin_counts = torch.zeros(max_reqs, vocab_size, dtype=torch.int32, device=self.device)

        self._run_kernel(
            expanded_idx_mapping, all_token_ids, prompt_len, prefill_len,
            prompt_bin_mask, output_bin_counts, max_prefill_len,
        )

        # Check prompt bits for request 0
        for t in [1, 2, 3]:
            w, b = t // 32, t % 32
            assert (prompt_bin_mask[0, w].item() >> b) & 1, f"Token {t} should be in prompt 0"

        # Check prompt bits for request 1
        for t in [10, 20]:
            w, b = t // 32, t % 32
            assert (prompt_bin_mask[1, w].item() >> b) & 1, f"Token {t} should be in prompt 1"

        # Check output counts
        assert output_bin_counts[0, 4].item() == 1 and output_bin_counts[0, 5].item() == 1
        assert output_bin_counts[1, 30].item() == 1
