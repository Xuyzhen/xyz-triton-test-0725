"""bad_words: logits numeric class (apply_bad_words, in-place).

GPU side: vanilla vllm (vllm.v1.worker.gpu.sample.bad_words.apply_bad_words).
NPU side: vllm-ascend (vllm_ascend.worker.v2.sample.bad_words.apply_bad_words).
Wrapper signatures are positionally identical (logits + 9 tensors + python
int num_bad_words_per_req).

Data construction mirrors the strict UT (create_test_data): each request gets
num_bad_words_per_req bad words of bad_word_length tokens; input_ids are
planted so the current token ends a bad-word match (expanded_local_pos =
bad_word_length - 1). Matched vocab positions are set to -inf.

Outputs: logits (fp32, 1e-5; positions set to -inf compare as equal inf).
"""

from __future__ import annotations

import torch

import capture_runtime as cr
from capture_runtime import CaseSpec

MAX_BAD_WORDS_TOTAL_TOKENS = 1024
MAX_NUM_BAD_WORDS = 128
MAX_SEQ_LEN = 1024


def build_inputs(params: dict, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_tok, vocab = params["num_tokens"], params["vocab_size"]
    n_req, n_bw, bw_len = params["num_requests"], params["num_bad_words_per_req"], params["bad_word_length"]

    logits = torch.randn(n_tok, vocab, generator=g, dtype=torch.float32)
    mapping = torch.randint(0, n_req, (n_tok,), generator=g, dtype=torch.int32)

    bad_word_token_ids = torch.zeros(n_req, MAX_BAD_WORDS_TOTAL_TOKENS, dtype=torch.int32)
    bad_word_offsets = torch.zeros(n_req, MAX_NUM_BAD_WORDS + 1, dtype=torch.int32)
    num_bad_words = torch.zeros(n_req, dtype=torch.int32)
    for r in range(n_req):
        offset, actual = 0, 0
        for bw in range(n_bw):
            if offset + bw_len > MAX_BAD_WORDS_TOTAL_TOKENS:
                break
            word = torch.full((bw_len,), 100 + r * 10 + bw, dtype=torch.int32)
            bad_word_token_ids[r, offset:offset + bw_len] = word
            bad_word_offsets[r, bw] = offset
            offset += bw_len
            actual += 1
        bad_word_offsets[r, actual] = offset
        num_bad_words[r] = actual

    all_token_ids = torch.randint(0, vocab, (n_req, MAX_SEQ_LEN), generator=g, dtype=torch.int32)
    prompt_len = torch.full((n_req,), 50, dtype=torch.int32)
    total_len = torch.full((n_req,), MAX_SEQ_LEN, dtype=torch.int32)

    input_ids = torch.randint(0, vocab, (n_tok,), generator=g, dtype=torch.int32)
    if n_bw > 0:
        for tok in range(n_tok):
            r = mapping[tok].item()
            if num_bad_words[r] > 0:
                word = bad_word_token_ids[r, :bw_len]
                for i in range(bw_len):
                    if tok - i >= 0:
                        input_ids[tok - i] = word[bw_len - 1 - i]

    expanded_local_pos = torch.full((n_tok,), bw_len - 1, dtype=torch.int32)

    return {
        "logits": logits, "expanded_idx_mapping": mapping,
        "bad_word_token_ids": bad_word_token_ids, "bad_word_offsets": bad_word_offsets,
        "num_bad_words": num_bad_words, "all_token_ids": all_token_ids,
        "prompt_len": prompt_len, "total_len": total_len,
        "input_ids": input_ids, "expanded_local_pos": expanded_local_pos,
    }


def run(side: str, t: dict[str, torch.Tensor], params: dict) -> dict[str, torch.Tensor]:
    if side == "gpu":
        from vllm.v1.worker.gpu.sample.bad_words import apply_bad_words
    else:
        from vllm_ascend.worker.v2.sample.bad_words import apply_bad_words

    logits = t["logits"].clone()
    apply_bad_words(
        logits,
        t["expanded_idx_mapping"],
        t["bad_word_token_ids"],
        t["bad_word_offsets"],
        t["num_bad_words"],
        t["all_token_ids"],
        t["prompt_len"],
        t["total_len"],
        t["input_ids"],
        t["expanded_local_pos"],
        params["num_bad_words_per_req"],
    )
    if side == "gpu":
        torch.cuda.synchronize()
    else:
        torch.npu.synchronize()
    return {"logits": logits}


def ref(t: dict[str, torch.Tensor], params: dict) -> dict[str, torch.Tensor]:
    """fp64 golden mirroring _apply_bad_words_kernel match rule:
      prefix_len = word_len - 1; skip if prefix_len > effective_len
      effective_len = output_len + pos, output_len = total_len - prompt_len
      actual_pos = effective_len - prefix_len + i
      from_spec_input = actual_pos >= output_len  -> input_ids[base + spec_off + 1]
      else                                        -> all_token_ids[r, prompt_len + actual_pos]
      all prefix tokens match -> logits[tok, last_token] = -inf
    """
    logits = t["logits"].to(torch.float64).clone()
    n_tok = logits.shape[0]
    mapping = t["expanded_idx_mapping"].long()
    pos_all = t["expanded_local_pos"].long()
    bw_ids = t["bad_word_token_ids"].long()
    bw_off = t["bad_word_offsets"].long()
    num_bw = t["num_bad_words"].long()
    all_ids = t["all_token_ids"].long()
    prompt_len = t["prompt_len"].long()
    total_len = t["total_len"].long()
    input_ids = t["input_ids"].long()

    for tok in range(n_tok):
        r = int(mapping[tok])
        if int(num_bw[r]) == 0:
            continue
        pos = int(pos_all[tok])
        base = tok - pos
        out_len = int(total_len[r] - prompt_len[r])
        eff = out_len + pos
        for bw in range(int(num_bw[r])):
            start, end = int(bw_off[r, bw]), int(bw_off[r, bw + 1])
            bw_len = end - start
            prefix_len = bw_len - 1
            if prefix_len > eff:
                continue
            last = int(bw_ids[r, end - 1])
            matched = True
            for i in range(prefix_len):
                actual_pos = eff - prefix_len + i
                if actual_pos >= out_len:
                    actual = int(input_ids[base + (actual_pos - out_len) + 1])
                else:
                    actual = int(all_ids[r, int(prompt_len[r]) + actual_pos])
                if actual != int(bw_ids[r, start + i]):
                    matched = False
                    break
            if matched:
                logits[tok, last] = float("-inf")
    return {"logits": logits}


def _mk(name: str, n_tok: int, vocab: int, n_req: int, n_bw: int, bw_len: int) -> CaseSpec:
    return CaseSpec(
        kernel="bad_words", name=name,
        params={"num_tokens": n_tok, "vocab_size": vocab, "num_requests": n_req,
                "num_bad_words_per_req": n_bw, "bad_word_length": bw_len},
        seed=42,
        output_modes={"logits": cr.MODE_F32},
    )


CASES = [
    _mk("small_512t_50257v_16r_3bw", 512, 50257, 16, 3, 2),
    _mk("deepseek_8t_129280v_4r_3bw", 8, 129280, 4, 3, 2),
    _mk("kimi_8t_163840v_4r_3bw", 8, 163840, 4, 3, 2),
    _mk("no_bad_words_256t_1000v_4r_0bw", 256, 1000, 4, 0, 3),
]
