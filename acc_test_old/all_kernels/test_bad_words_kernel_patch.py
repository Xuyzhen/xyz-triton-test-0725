# vLLM-Ascend patched kernel: _bad_words_kernel from
# vllm-ascend/vllm_ascend/worker/v2/sample/bad_words.py:31
# PATCH NOTE: This is an Ascend NPU adaptation of the original vLLM Triton kernel

"""
Precision test for patched _bad_words_kernel (Ascend NPU version).

Patch differences vs original vllm:
- Uses load-balanced grid based on get_vectorcore_num() instead of (num_tokens, max_num_bad_words)
- Processes tokens in chunks per core for better load balancing
- Uses tl.range and tl.minimum for loop bounds (Ascend NPU limits)
- Adds MAX_PREFIX_LEN constexpr (set to 32)
- Uses tl.minimum instead of min() for bounds clamping
- Removes early termination j variable (uses while loop with match flag)
- Optimized memory access patterns for NPU

Kernel signature:
    _bad_words_kernel(
        logits_ptr,                     # fp32 logits [num_tokens, vocab_size]
        logits_stride,                  # stride(0) of logits
        expanded_idx_mapping_ptr,       # [num_tokens] token_idx -> req_state_idx
        bad_word_token_ids_ptr,         # [max_num_reqs, MAX_BAD_WORDS_TOTAL_TOKENS]
        bad_word_token_ids_stride,      # stride(0) of bad_word_token_ids
        bad_word_offsets_ptr,           # [max_num_reqs, MAX_NUM_BAD_WORDS + 1]
        bad_word_offsets_stride,        # stride(0) of bad_word_offsets
        num_bad_words_ptr,              # [max_num_reqs]
        all_token_ids_ptr,              # [max_num_reqs, max_total_len]
        all_token_ids_stride,           # stride(0) of all_token_ids
        prompt_len_ptr,                 # [max_num_reqs]
        total_len_ptr,                  # [max_num_reqs]
        input_ids_ptr,                  # [num_tokens] input token IDs (for spec decode)
        expanded_local_pos_ptr,         # [num_tokens] local position within request
        num_tokens,                     # scalar: number of tokens
        max_num_bad_words,              # scalar: max bad words per request
        MAX_PREFIX_LEN: tl.constexpr,   # constexpr: max prefix length (32)
    )

Sets logits to -inf for tokens completing a bad word pattern in the request's output.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

import pytest

MAX_BAD_WORDS_TOTAL_TOKENS = 1024
MAX_NUM_BAD_WORDS = 128
MAX_PREFIX_LEN = 32


def _bad_words_ref(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    bad_word_token_ids: torch.Tensor,
    bad_word_offsets: torch.Tensor,
    num_bad_words: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    total_len: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
):
    """CPU reference for patched _bad_words_kernel."""
    num_tokens = logits.shape[0]
    for token_idx in range(num_tokens):
        req_state_idx = expanded_idx_mapping[token_idx].item()
        n_bw = num_bad_words[req_state_idx].item()
        pos = expanded_local_pos[token_idx].item()
        cur_req_first_pos = token_idx - pos
        prompt_len_ = prompt_len[req_state_idx].item()
        total_len_ = total_len[req_state_idx].item()
        output_len = total_len_ - prompt_len_
        effective_len = output_len + pos

        for bw_idx in range(n_bw):
            start = bad_word_offsets[req_state_idx, bw_idx].item()
            end = bad_word_offsets[req_state_idx, bw_idx + 1].item()
            bad_word_len = end - start
            prefix_len = bad_word_len - 1

            if prefix_len > effective_len:
                continue

            last_token = bad_word_token_ids[req_state_idx, end - 1].item()
            match = True
            for i in range(prefix_len):
                expected = bad_word_token_ids[req_state_idx, start + i].item()
                actual_pos = effective_len - prefix_len + i

                from_spec_input = actual_pos >= output_len
                if from_spec_input:
                    spec_offset = actual_pos - output_len
                    actual = input_ids[cur_req_first_pos + spec_offset].item()
                else:
                    actual = all_token_ids[req_state_idx, prompt_len_ + actual_pos].item()

                if expected != actual:
                    match = False
                    break

            if match:
                logits[token_idx, last_token] = float("-inf")


class TestBadWordsKernelPatch:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    def _run_kernel(
        self,
        logits,
        expanded_idx_mapping,
        bad_word_token_ids,
        bad_word_offsets,
        num_bad_words,
        all_token_ids,
        prompt_len,
        total_len,
        input_ids,
        expanded_local_pos,
        max_num_bad_words,
    ):
        num_tokens = logits.shape[0]
        core_num = get_vectorcore_num()
        # Import the patched kernel from vllm-ascend
        from vllm_ascend.worker.v2.sample.bad_words import _bad_words_kernel

        _bad_words_kernel[(core_num,)](
            logits,
            logits.stride(0),
            expanded_idx_mapping,
            bad_word_token_ids,
            bad_word_token_ids.stride(0),
            bad_word_offsets,
            bad_word_offsets.stride(0),
            num_bad_words,
            all_token_ids,
            all_token_ids.stride(0),
            prompt_len,
            total_len,
            input_ids,
            expanded_local_pos,
            num_tokens,
            max_num_bad_words,
            MAX_PREFIX_LEN,
        )
        torch.npu.synchronize()

    def test_no_bad_words(self):
        """When no bad words exist, logits must remain unchanged."""
        num_tokens, vocab_size = 4, 128
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        logits_copy = logits.clone()
        expanded_idx_mapping = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        bad_word_token_ids = torch.zeros(1, MAX_BAD_WORDS_TOTAL_TOKENS, dtype=torch.int32, device=self.device)
        bad_word_offsets = torch.zeros(1, MAX_NUM_BAD_WORDS + 1, dtype=torch.int32, device=self.device)
        num_bad_words = torch.zeros(1, dtype=torch.int32, device=self.device)
        all_token_ids = torch.zeros(1, 256, dtype=torch.int32, device=self.device)
        prompt_len = torch.zeros(1, dtype=torch.int32, device=self.device)
        total_len = torch.zeros(1, dtype=torch.int32, device=self.device)
        input_ids = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        expanded_local_pos = torch.arange(num_tokens, dtype=torch.int32, device=self.device)
        max_num_bad_words = 0

        self._run_kernel(
            logits, expanded_idx_mapping,
            bad_word_token_ids, bad_word_offsets, num_bad_words,
            all_token_ids, prompt_len, total_len, input_ids, expanded_local_pos,
            max_num_bad_words,
        )

        torch.testing.assert_close(logits.cpu(), logits_copy.cpu(), rtol=0, atol=0)

    @pytest.mark.parametrize("num_reqs", [1, 2, 4])
    @pytest.mark.parametrize("vocab_size", [128, 512])
    def test_bad_word_blocking(self, num_reqs, vocab_size):
        """Verify bad words are correctly blocked: logits match CPU reference."""
        num_tokens = num_reqs
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        expanded_local_pos = torch.ones(num_tokens, dtype=torch.int32, device=self.device)

        bad_word_token_ids = torch.zeros(num_reqs, MAX_BAD_WORDS_TOTAL_TOKENS, dtype=torch.int32, device=self.device)
        bad_word_offsets_base = torch.zeros(num_reqs, MAX_NUM_BAD_WORDS + 1, dtype=torch.int32, device=self.device)
        num_bad_words_t = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        prompt_len_v = torch.full((num_reqs,), 5, dtype=torch.int32, device=self.device)
        total_len_v = torch.full((num_reqs,), 10, dtype=torch.int32, device=self.device)

        for i in range(num_reqs):
            bad_word_tokens = [10, 20]
            bad_word_token_ids[i, 0] = bad_word_tokens[0]
            bad_word_token_ids[i, 1] = bad_word_tokens[1]
            bad_word_offsets_base[i, 0] = 0
            bad_word_offsets_base[i, 1] = 2
            num_bad_words_t[i] = 1
            all_token_ids = torch.zeros(num_reqs, 256, dtype=torch.int32, device=self.device)
            all_token_ids[i, 5] = bad_word_tokens[0]

        max_num_bad_words = num_bad_words_t.max().item()
        input_ids = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        logits_copy = logits.clone().cpu()

        self._run_kernel(
            logits, expanded_idx_mapping,
            bad_word_token_ids, bad_word_offsets_base, num_bad_words_t,
            all_token_ids, prompt_len_v, total_len_v, input_ids, expanded_local_pos,
            max_num_bad_words,
        )

        expected = logits_copy.clone()
        _bad_words_ref(
            expected, expanded_idx_mapping.cpu(),
            bad_word_token_ids.cpu(), bad_word_offsets_base.cpu(), num_bad_words_t.cpu(),
            all_token_ids.cpu(), prompt_len_v.cpu(), total_len_v.cpu(),
            input_ids.cpu(), expanded_local_pos.cpu(),
        )

        torch.testing.assert_close(logits.cpu(), expected, rtol=0, atol=0)

    def test_multi_token_bad_word(self):
        """Test with a longer bad word (3 tokens)."""
        num_reqs = 1
        num_tokens = 1
        vocab_size = 64
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        expanded_local_pos = torch.ones(num_tokens, dtype=torch.int32, device=self.device)

        bad_word_token_ids = torch.zeros(num_reqs, MAX_BAD_WORDS_TOTAL_TOKENS, dtype=torch.int32, device=self.device)
        bad_word_offsets = torch.zeros(num_reqs, MAX_NUM_BAD_WORDS + 1, dtype=torch.int32, device=self.device)
        num_bad_words_t = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        prompt_len_v = torch.full((num_reqs,), 3, dtype=torch.int32, device=self.device)
        total_len_v = torch.full((num_reqs,), 8, dtype=torch.int32, device=self.device)

        # Bad word: [5, 15, 25] (3 tokens, so prefix_len=2, last_token=25)
        bad_word_tokens = [5, 15, 25]
        for i, t in enumerate(bad_word_tokens):
            bad_word_token_ids[0, i] = t
        bad_word_offsets[0, 0] = 0
        bad_word_offsets[0, 1] = 3
        num_bad_words_t[0] = 1

        # Place prefix tokens in output (offset from prompt_len)
        all_token_ids = torch.zeros(num_reqs, 256, dtype=torch.int32, device=self.device)
        all_token_ids[0, 3] = bad_word_tokens[0]  # prompt_len=3, so idx 3
        all_token_ids[0, 4] = bad_word_tokens[1]  # prompt_len+1=4

        max_num_bad_words = 1
        input_ids = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        logits_copy = logits.clone().cpu()

        self._run_kernel(
            logits, expanded_idx_mapping,
            bad_word_token_ids, bad_word_offsets, num_bad_words_t,
            all_token_ids, prompt_len_v, total_len_v, input_ids, expanded_local_pos,
            max_num_bad_words,
        )

        expected = logits_copy.clone()
        _bad_words_ref(
            expected, expanded_idx_mapping.cpu(),
            bad_word_token_ids.cpu(), bad_word_offsets.cpu(), num_bad_words_t.cpu(),
            all_token_ids.cpu(), prompt_len_v.cpu(), total_len_v.cpu(),
            input_ids.cpu(), expanded_local_pos.cpu(),
        )

        torch.testing.assert_close(logits.cpu(), expected, rtol=0, atol=0)

    def test_bad_word_not_in_output(self):
        """When the prefix doesn't appear in context, logits must remain unchanged."""
        num_tokens = 2
        vocab_size = 128
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        logits_copy = logits.clone()
        expanded_idx_mapping = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        expanded_local_pos = torch.tensor([1, 2], dtype=torch.int32, device=self.device)

        bad_word_token_ids = torch.zeros(1, MAX_BAD_WORDS_TOTAL_TOKENS, dtype=torch.int32, device=self.device)
        bad_word_offsets = torch.zeros(1, MAX_NUM_BAD_WORDS + 1, dtype=torch.int32, device=self.device)
        num_bad_words_t = torch.zeros(1, dtype=torch.int32, device=self.device)
        prompt_len_v = torch.full((1,), 5, dtype=torch.int32, device=self.device)
        total_len_v = torch.full((1,), 10, dtype=torch.int32, device=self.device)

        # Bad word: [99, 100]
        bad_word_token_ids[0, 0] = 99
        bad_word_token_ids[0, 1] = 100
        bad_word_offsets[0, 0] = 0
        bad_word_offsets[0, 1] = 2
        num_bad_words_t[0] = 1

        all_token_ids = torch.zeros(1, 256, dtype=torch.int32, device=self.device)
        all_token_ids[0, 5] = 50  # Different token, not 99

        max_num_bad_words = 1
        input_ids = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)

        self._run_kernel(
            logits, expanded_idx_mapping,
            bad_word_token_ids, bad_word_offsets, num_bad_words_t,
            all_token_ids, prompt_len_v, total_len_v, input_ids, expanded_local_pos,
            max_num_bad_words,
        )

        torch.testing.assert_close(logits.cpu(), logits_copy.cpu(), rtol=0, atol=0)
