# vLLM vanilla kernel: _bad_words_kernel from
# vllm/vllm/v1/worker/gpu/sample/bad_words.py

"""
Precision test for _bad_words_kernel.

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
        word tokens        all_token_ids_stride,           # stride(0) of all_token_ids
        prompt_len_ptr,                 # [max_num_reqs]
        total_len_ptr,                  # [max_num_reqs]
        input_ids_ptr,                  # [num_tokens] input token IDs (for spec decode)
        expanded_local_pos_ptr,         # [num_tokens] local position within request
    )

For each request, sets logits to -inf for tokens that complete a bad word
pattern in the request's output.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.bad_words import _bad_words_kernel
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

import pytest


MAX_BAD_WORDS_TOTAL_TOKENS = 1024
MAX_NUM_BAD_WORDS = 128


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
    """CPU reference for _bad_words_kernel."""
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


class TestBadWordsKernel:

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
    ):
        num_tokens, vocab_size = logits.shape
        max_num_bw = num_bad_words.max().item()
        if max_num_bw == 0:
            return  # kernel not launched
        _bad_words_kernel[(num_tokens, max_num_bw)](
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
        )
        torch.npu.synchronize()

    def test_no_bad_words(self):
        """When no request uses bad words, logits are unchanged."""
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

        self._run_kernel(
            logits, expanded_idx_mapping,
            bad_word_token_ids, bad_word_offsets, num_bad_words,
            all_token_ids, prompt_len, total_len, input_ids, expanded_local_pos,
        )

        torch.testing.assert_close(logits.cpu(), logits_copy.cpu(), rtol=0, atol=0)

    @pytest.mark.parametrize("num_reqs", [1, 2, 4])
    @pytest.mark.parametrize("vocab_size", [128, 512])
    def test_bad_word_blocking(self, num_reqs, vocab_size):
        """Verify bad words are correctly blocked: logits match CPU reference."""
        num_tokens = num_reqs
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        expanded_local_pos = torch.ones(num_tokens, dtype=torch.int32, device=self.device)  # decode step

        bad_word_token_ids = torch.zeros(num_reqs, MAX_BAD_WORDS_TOTAL_TOKENS, dtype=torch.int32, device=self.device)
        bad_word_offsets_base = torch.zeros(num_reqs, MAX_NUM_BAD_WORDS + 1, dtype=torch.int32, device=self.device)
        num_bad_words_t = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        prompt_len_v = torch.full((num_reqs,), 5, dtype=torch.int32, device=self.device)
        total_len_v = torch.full((num_reqs,), 10, dtype=torch.int32, device=self.device)

        for i in range(num_reqs):
            bad_word_tokens = [10, 20]  # a 2-token bad word
            bad_word_token_ids[i, 0] = bad_word_tokens[0]
            bad_word_token_ids[i, 1] = bad_word_tokens[1]
            bad_word_offsets_base[i, 0] = 0
            bad_word_offsets_base[i, 1] = 2
            num_bad_words_t[i] = 1
            # Place the prefix token in the output
            all_token_ids[i, 5] = bad_word_tokens[0]

        # The prefix already exists in output; the last token is what we sample
        # expanded_local_pos=1 means effective_len = output_len + 1 = 6
        # prefix_len = 1, effective_len = 6, so it matches
        # last_token = 20 -> logits[token_idx, 20] should be -inf

        input_ids = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)

        logits_copy = logits.clone().cpu()

        self._run_kernel(
            logits, expanded_idx_mapping,
            bad_word_token_ids, bad_word_offsets_base, num_bad_words_t,
            all_token_ids, prompt_len_v, total_len_v, input_ids, expanded_local_pos,
        )

        # CPU reference
        expected = logits_copy.clone()
        _bad_words_ref(
            expected, expanded_idx_mapping.cpu(),
            bad_word_token_ids.cpu(), bad_word_offsets_base.cpu(), num_bad_words_t.cpu(),
            all_token_ids.cpu(), prompt_len_v.cpu(), total_len_v.cpu(),
            input_ids.cpu(), expanded_local_pos.cpu(),
        )

        torch.testing.assert_close(logits.cpu(), expected, rtol=0, atol=0)
