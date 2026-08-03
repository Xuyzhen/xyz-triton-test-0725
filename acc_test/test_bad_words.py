# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.gpu.sample.bad_words import _bad_words_kernel

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _apply_bad_words_cpu(
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
) -> torch.Tensor:
    """Independent PyTorch CPU reference for bad words filtering.

    Sets logits to -inf for tokens that would complete a bad word.
    """
    num_tokens = logits.shape[0]
    vocab_size = logits.shape[1]
    output = logits.clone()

    for token_idx in range(num_tokens):
        req_state_idx = int(expanded_idx_mapping[token_idx])
        n_bad = int(num_bad_words[req_state_idx])

        if n_bad == 0:
            continue

        pos = int(expanded_local_pos[token_idx])
        prompt_len_val = int(prompt_len[req_state_idx])
        total_len_val = int(total_len[req_state_idx])
        output_len = total_len_val - prompt_len_val
        effective_len = output_len + pos

        for bw_idx in range(n_bad):
            start = int(bad_word_offsets[req_state_idx, bw_idx])
            end = int(bad_word_offsets[req_state_idx, bw_idx + 1])
            bad_word_len = end - start
            prefix_len = bad_word_len - 1

            if prefix_len > effective_len:
                continue

            last_token = int(bad_word_token_ids[req_state_idx, end - 1])

            # Check prefix match
            match = True
            for i in range(prefix_len):
                expected = int(bad_word_token_ids[req_state_idx, start + i])
                actual_pos = effective_len - prefix_len + i

                from_spec_input = actual_pos >= output_len
                if from_spec_input:
                    spec_offset = actual_pos - output_len
                    actual = int(input_ids[token_idx - pos + spec_offset])
                else:
                    actual = int(all_token_ids[req_state_idx, prompt_len_val + actual_pos])

                if expected != actual:
                    match = False
                    break

            if match:
                output[token_idx, last_token] = float("-inf")

    return output


def _setup_bad_words_test(
    vocab_size: int = 64,
    max_num_reqs: int = 4,
) -> dict:
    """Helper to create test data for bad words tests.

    Creates a scenario with:
    - req 0: 2 bad words (short tokens)
    - req 1: 1 bad word
    - req 2: no bad words
    - req 3: 1 bad word (does not match)
    """
    prompt_len = torch.tensor([10, 10, 10, 10], dtype=torch.int32)
    total_len = torch.tensor([15, 15, 15, 15], dtype=torch.int32)
    output_len = (total_len - prompt_len).tolist()  # [5, 5, 5, 5]

    # all_token_ids: [max_num_reqs, max_total_len]
    max_total_len = 20
    all_token_ids = torch.randint(
        0, vocab_size, (max_num_reqs, max_total_len), dtype=torch.int32
    )
    # Set specific tokens in the output region for matching bad words
    # req 0: output tokens (indices 10-14) should contain [42, 7] at positions 10-11
    all_token_ids[0, prompt_len[0]:prompt_len[0] + 2] = torch.tensor([42, 7])
    # req 1: output contains [8, 3, 15] at positions 10-12
    all_token_ids[1, prompt_len[1]:prompt_len[1] + 3] = torch.tensor([8, 3, 15])

    # input_ids: [num_tokens] for speculative decoding inputs
    # For simplicity, use an empty/dummy tensor since we test without spec tokens
    input_ids = torch.empty(0, dtype=torch.int32)

    # bad word definitions: [max_num_reqs, max_bad_words_total_tokens]
    max_bad_words_total_tokens = 32
    bad_word_token_ids = torch.zeros(
        (max_num_reqs, max_bad_words_total_tokens), dtype=torch.int32
    )
    # bad word offsets: [max_num_reqs, max_num_bad_words + 1]
    max_num_bad_words = 4
    bad_word_offsets = torch.zeros(
        (max_num_reqs, max_num_bad_words + 1), dtype=torch.int32
    )
    # num_bad_words per request
    num_bad_words = torch.zeros(max_num_reqs, dtype=torch.int32)

    # req 0: bad word = [42, 7] (two tokens), and [99] (single token)
    # The bad word [42, 7] matches the output at prompt_len+0, prompt_len+1
    bad_word_token_ids[0, 0:2] = torch.tensor([42, 7])
    bad_word_token_ids[0, 2] = torch.tensor([99])  # single token bad word
    bad_word_offsets[0, 0] = 0
    bad_word_offsets[0, 1] = 2
    bad_word_offsets[0, 2] = 3
    num_bad_words[0] = 2

    # req 1: bad word = [8, 3, 15] (three tokens) - matches output
    bad_word_token_ids[1, 0:3] = torch.tensor([8, 3, 15])
    bad_word_offsets[1, 0] = 0
    bad_word_offsets[1, 1] = 3
    num_bad_words[1] = 1

    # req 2: no bad words

    # req 3: bad word = [5, 6, 7] - does NOT match any output
    bad_word_token_ids[3, 0:3] = torch.tensor([5, 6, 7])
    bad_word_offsets[3, 0] = 0
    bad_word_offsets[3, 1] = 3
    num_bad_words[3] = 1

    # expanded_idx_mapping: [num_tokens] token_idx -> req_state_idx
    # expanded_local_pos: [num_tokens] position within request
    # For simplicity, 1 token per request at output position `output_len - 1`
    # This is the position where the bad word would be completed
    num_reqs = 4

    # Generate tokens at the position just after the matching prefix
    # For req 0: bad word [42, 7] -> check when effective_len >= 1 (prefix_len=1)
    #   We want to verify the token at position pos where last_token=7 is set to -inf
    #   So we create a token at effective_len = 1 (the position after token 42 in output)
    # For req 1: bad word [8, 3, 15] -> check when effective_len >= 2
    #   Create a token at effective_len = 2 (position after [8,3])
    # For req 3: bad word [5, 6, 7] -> check when effective_len >= 2
    #   Token at effective_len = 2 but expected tokens don't match

    # We'll create tokens for all requests at positions that should trigger checks
    # Token positions:
    # req 0: output pos 5 + 1 = 6 -> effective_len = 0 + 6 = 6,
    #   but we want effective_len where prefix matches
    # Actually the way the kernel works:
    #   pos = expanded_local_pos[token_idx]
    #   effective_len = output_len + pos
    #   prefix_len = bad_word_len - 1
    #   It checks if all prefix_len tokens match, then last_token is penalized
    #
    # The 'pos' is expanded_local_pos which is the position within the logits,
    # not the actual sequence position. The effective_len maps to
    # actual sequence position = prompt_len + effective_len - prefix_len ... etc.
    #
    # For req 0, bad word [42, 7]: prefix_len=1, last_token=7
    #   At the first output position (pos=0), effective_len = output_len + 0 = 5
    #   We need prefix_len (1) tokens before to match: all_token_ids[pos=prompt_len+5-1=14]
    #   Actually for all_token_ids, it accesses:
    #     actual_pos = effective_len - prefix_len + i
    #     actual = all_token_ids[req_state_idx, prompt_len + actual_pos]
    #   So for i=0: actual_pos = 5 - 1 + 0 = 4
    #   actual = all_token_ids[0, 10 + 4] = all_token_ids[0, 14]
    #
    # Let's simplify: compute what effective_len to use to match

    # For req 0: bad word [42, 7], prefix_len=1
    #   We need actual_pos = 0 (first output token) to be 42
    #   actual_pos = effective_len - 1
    #   So effective_len = 1
    #   pos = effective_len - output_len = 1 - 5 = -4 -> Not possible
    #
    # Hmm, let me re-read the kernel logic more carefully...
    #   pos = tl.load(expanded_local_pos_ptr + token_idx)
    #   effective_len = output_len + pos
    #   prefix_len = bad_word_len - 1
    #   for i in range(prefix_len):
    #     expected = bad_word_tokens[start + i]
    #     actual_pos = effective_len - prefix_len + i
    #     actual = all_token_ids[prompt_len + actual_pos]
    #
    # So actual_pos ranges from effective_len-prefix_len to effective_len-1
    # The LAST token of the bad word is NOT in all_token_ids - it's the one
    # we're about to sample, which gets penalized.
    #
    # For req 0 bad word [42, 7], to match we need:
    #   actual_pos = 0 -> effective_len = 1, so actual = all_token_ids[0, 10+0] = 42
    #   pos = 1 - 5 = -4... not possible since pos >= 0
    #
    # Wait, pos = expanded_local_pos, which starts at 0.
    # So effective_len = output_len + pos >= output_len = 5
    # actual_pos = effective_len - prefix_len
    # For prefix_len=1: actual_pos = effective_len - 1 >= 4
    # We need actual_pos=0 and actual=42... but 4 != 0
    #
    # Hmm, I think the kernel handles the case differently. Let me re-read...
    #
    # The expanded_local_pos at a given token_idx represents the local position
    # within THIS request's logits. For output tokens, this corresponds to
    # positions within the output (generated) region.
    #
    # For the first output logit: pos=0, effective_len = output_len + 0 = 5
    # A bad word [42, 7] with prefix_len=1:
    #   actual_pos = 5 - 1 = 4
    #   actual = all_token_ids[10 + 4] = all_token_ids[14]
    #
    # So for a prefix match at the first token, we need:
    #   all_token_ids[10 + effective_len - 1] = 42
    #   With effective_len = 5: all_token_ids[14] = 42
    #
    # OK let me just set up data that works with the math.

    # Let's set up the scenario more carefully:
    # For req 0: all_token_ids[0, prompt_len + (effective_len - 1)] should be 42
    # at the position where we check. Let's just set key positions.
    # We'll put 42 at all_token_ids[0, 10 + 4] = all_token_ids[0, 14]
    # Then for effective_len = 5 (pos=0): actual_pos = 4, matches 42
    all_token_ids[0, 14] = 42  # So that at effective_len=5, actual_pos=4 gives 42

    # We also need the second token check... but prefix_len=1 means only 1 check
    # which is 42. Then last_token=7 would be penalized at token_idx=0.

    # For req 1: bad word [8, 3, 15], prefix_len=2
    #   i=0: actual_pos = effective_len - 2, need all_token_ids[10+actual_pos] = 8
    #   i=1: actual_pos = effective_len - 1, need all_token_ids[10+actual_pos] = 3
    #   With pos=0, effective_len=5:
    #   i=0: actual_pos=3, need all_token_ids[0, 13] = 8
    #   i=1: actual_pos=4, need all_token_ids[0, 14] = 3
    # But we already set req 0 all_token_ids at column 14 to 42.
    # Let's adjust positions...
    # We'll put tokens at the right places
    all_token_ids[1, 13] = 8
    all_token_ids[1, 14] = 3
    # Then at pos=0, effective_len=5: i=0 checks col 13 (=8, match), i=1 checks col 14 (=3, match)
    # So bad word matches, last_token=15 gets penalized at this token_idx

    # For req 3: bad word [5, 6, 7], prefix_len=2
    #   We don't want a match. Ensure all_token_ids at relevant positions are not 5,6
    all_token_ids[3, 13] = 1
    all_token_ids[3, 14] = 2
    # So at pos=0, effective_len=5: i=0 checks col 13 (=1 vs 5, no match)

    # Create expanded arrays: 1 token per request
    num_tokens = num_reqs
    expanded_idx_mapping = torch.arange(num_reqs, dtype=torch.int32)
    expanded_local_pos = torch.zeros(num_reqs, dtype=torch.int32)

    # Logits: [num_tokens, vocab_size]
    logits = torch.full(
        (num_tokens, vocab_size), 0.0, dtype=torch.float32
    )
    # Set some non-zero logits for the bad word last tokens to verify suppression
    logits[0, 7] = 5.0  # req 0, last_token of bad word [42, 7]
    logits[1, 15] = 3.0  # req 1, last_token of bad word [8, 3, 15]
    logits[3, 7] = 10.0  # req 3, last_token 7 of bad word [5, 6, 7] - should NOT be suppressed

    max_num_bad_words = int(num_bad_words.max().item())

    return {
        "logits": logits,
        "expanded_idx_mapping": expanded_idx_mapping,
        "bad_word_token_ids": bad_word_token_ids,
        "bad_word_offsets": bad_word_offsets,
        "num_bad_words": num_bad_words,
        "all_token_ids": all_token_ids,
        "prompt_len": prompt_len,
        "total_len": total_len,
        "input_ids": input_ids,
        "expanded_local_pos": expanded_local_pos,
        "max_num_bad_words": max_num_bad_words,
        "max_num_reqs": max_num_reqs,
        "num_tokens": num_tokens,
    }


@pytest.mark.parametrize("vocab_size", [64, 128])
def test_bad_words_basic(vocab_size: int) -> None:
    """Test that bad words kernel correctly suppresses matching bad word tokens."""
    init_device_properties_triton()

    data = _setup_bad_words_test(vocab_size=vocab_size)

    logits = data["logits"]
    expanded_idx_mapping = data["expanded_idx_mapping"]
    bad_word_token_ids = data["bad_word_token_ids"]
    bad_word_offsets = data["bad_word_offsets"]
    num_bad_words = data["num_bad_words"]
    all_token_ids = data["all_token_ids"]
    prompt_len = data["prompt_len"]
    total_len = data["total_len"]
    input_ids = data["input_ids"]
    expanded_local_pos = data["expanded_local_pos"]

    # Compute CPU reference
    expected = _apply_bad_words_cpu(
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
    )

    device = torch.device("npu")
    logits_npu = logits.to(device)
    expanded_idx_mapping_npu = expanded_idx_mapping.to(device)
    bad_word_token_ids_npu = bad_word_token_ids.to(device)
    bad_word_offsets_npu = bad_word_offsets.to(device)
    num_bad_words_npu = num_bad_words.to(device)
    all_token_ids_npu = all_token_ids.to(device)
    prompt_len_npu = prompt_len.to(device)
    total_len_npu = total_len.to(device)
    input_ids_npu = input_ids.to(device)
    expanded_local_pos_npu = expanded_local_pos.to(device)

    output = torch.full_like(logits_npu, float("nan"))
    num_tokens = data["num_tokens"]
    max_num_bad_words = data["max_num_bad_words"]

    _bad_words_kernel[(num_tokens, max_num_bad_words)](
        logits_npu,
        logits_npu.stride(0),
        expanded_idx_mapping_npu,
        bad_word_token_ids_npu,
        bad_word_token_ids_npu.stride(0),
        bad_word_offsets_npu,
        bad_word_offsets_npu.stride(0),
        num_bad_words_npu,
        all_token_ids_npu,
        all_token_ids_npu.stride(0),
        prompt_len_npu,
        total_len_npu,
        input_ids_npu,
        expanded_local_pos_npu,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        logits_npu.cpu(), expected, rtol=0, atol=0
    )


def test_bad_words_no_bad_words() -> None:
    """Test with no bad words (should be a no-op)."""
    init_device_properties_triton()

    vocab_size = 64
    num_reqs = 3
    max_num_reqs = 3

    logits = torch.randn(num_reqs, vocab_size, dtype=torch.float32)
    expected = logits.clone()

    prompt_len = torch.tensor([10, 10, 10], dtype=torch.int32)
    total_len = torch.tensor([15, 15, 15], dtype=torch.int32)

    all_token_ids = torch.randint(
        0, vocab_size, (max_num_reqs, 20), dtype=torch.int32
    )
    input_ids = torch.empty(0, dtype=torch.int32)

    max_bad_words_total_tokens = 32
    bad_word_token_ids = torch.zeros(
        (max_num_reqs, max_bad_words_total_tokens), dtype=torch.int32
    )
    max_num_bad_words = 4
    bad_word_offsets = torch.zeros(
        (max_num_reqs, max_num_bad_words + 1), dtype=torch.int32
    )
    num_bad_words = torch.zeros(max_num_reqs, dtype=torch.int32)

    expanded_idx_mapping = torch.arange(num_reqs, dtype=torch.int32)
    expanded_local_pos = torch.zeros(num_reqs, dtype=torch.int32)

    device = torch.device("npu")
    logits_npu = logits.to(device)
    output = torch.full_like(logits_npu, float("nan"))

    _bad_words_kernel[(num_reqs, 1)](
        logits_npu,
        logits_npu.stride(0),
        expanded_idx_mapping.to(device),
        bad_word_token_ids.to(device),
        bad_word_token_ids.stride(0),
        bad_word_offsets.to(device),
        bad_word_offsets.stride(0),
        num_bad_words.to(device),
        all_token_ids.to(device),
        all_token_ids.stride(0),
        prompt_len.to(device),
        total_len.to(device),
        input_ids.to(device),
        expanded_local_pos.to(device),
    )
    torch.npu.synchronize()

    # The kernel modifies logits in-place; with no bad words it should be unchanged
    torch.testing.assert_close(
        logits_npu.cpu(), expected, rtol=0, atol=0
    )
