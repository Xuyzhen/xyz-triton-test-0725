"""bias: logits numeric class (_bias_kernel, direct launch).

Both sides import the SAME upstream kernel
(vllm.v1.worker.gpu.sample.logit_bias._bias_kernel), so the launch recipe is
identical. Applies, per request state:
  1. allowed token ids: everything else -> -inf
  2. logit bias: += bias at listed token ids
  3. min tokens: stop-token logits -> -inf while pos+1 < min_len

Padded per-request tables use the UT's MAX_* capacities
(allowed 1024, bias 1024, stop 128). Outputs: logits (fp32, 1e-5).
"""

from __future__ import annotations

import torch

import capture_runtime as cr
from capture_runtime import CaseSpec

MAX_NUM_ALLOWED_TOKEN_IDS = 1024
MAX_NUM_LOGIT_BIAS_TOKENS = 1024
MAX_NUM_STOP_TOKEN_IDS = 128
LOGITS_BLOCK_SIZE = 8192


def build_inputs(params: dict, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_tok, vocab, n_req = params["num_tokens"], params["vocab_size"], params["num_requests"]
    features = params["features"]

    logits = torch.randn(n_tok, vocab, generator=g, dtype=torch.float32)
    mapping = torch.randint(0, n_req, (n_tok,), generator=g, dtype=torch.int32)
    pos = torch.randint(0, 16, (n_tok,), generator=g, dtype=torch.int32)

    num_allowed = torch.zeros(n_req, dtype=torch.int32)
    allowed = torch.zeros(n_req, MAX_NUM_ALLOWED_TOKEN_IDS, dtype=torch.int32)
    num_bias = torch.zeros(n_req, dtype=torch.int32)
    bias_ids = torch.zeros(n_req, MAX_NUM_LOGIT_BIAS_TOKENS, dtype=torch.int32)
    bias_vals = torch.zeros(n_req, MAX_NUM_LOGIT_BIAS_TOKENS, dtype=torch.float32)
    min_lens = torch.zeros(n_req, dtype=torch.int32)
    num_stop = torch.zeros(n_req, dtype=torch.int32)
    stop_ids = torch.zeros(n_req, MAX_NUM_STOP_TOKEN_IDS, dtype=torch.int32)

    if "allowed" in features:
        for r in range(n_req):
            toks = torch.randperm(vocab, generator=g)[:4]
            num_allowed[r] = 4
            allowed[r, :4] = toks.to(torch.int32)
    if "bias" in features:
        for r in range(n_req):
            toks = torch.randint(0, vocab, (4,), generator=g)
            num_bias[r] = 4
            bias_ids[r, :4] = toks.to(torch.int32)
            bias_vals[r, :4] = torch.randn(4, generator=g, dtype=torch.float32)
    if "min_tokens" in features:
        for r in range(n_req):
            min_lens[r] = 32  # > pos+1 for all tokens -> suppression active
            toks = torch.randint(0, vocab, (4,), generator=g)
            num_stop[r] = 4
            stop_ids[r, :4] = toks.to(torch.int32)

    return {
        "logits": logits, "expanded_idx_mapping": mapping, "pos": pos,
        "num_allowed_token_ids": num_allowed, "allowed_token_ids": allowed,
        "num_logit_bias": num_bias, "bias_token_ids": bias_ids, "bias": bias_vals,
        "min_lens": min_lens, "num_stop_token_ids": num_stop, "stop_token_ids": stop_ids,
    }


def run(side: str, t: dict[str, torch.Tensor], params: dict) -> dict[str, torch.Tensor]:
    from vllm.triton_utils import triton
    from vllm.v1.worker.gpu.sample.logit_bias import _bias_kernel

    logits = t["logits"].clone()
    n_tok, vocab = logits.shape
    BLOCK_SIZE = triton.next_power_of_2(
        max(
            t["allowed_token_ids"].shape[-1],
            t["bias_token_ids"].shape[-1],
            t["stop_token_ids"].shape[-1],
        )
    )
    _bias_kernel[(n_tok,)](
        logits, logits.stride(0), vocab,
        t["expanded_idx_mapping"],
        t["num_allowed_token_ids"], t["allowed_token_ids"], t["allowed_token_ids"].stride(0),
        t["num_logit_bias"], t["bias_token_ids"], t["bias_token_ids"].stride(0),
        t["bias"], t["bias"].stride(0),
        t["pos"], t["min_lens"],
        t["num_stop_token_ids"], t["stop_token_ids"], t["stop_token_ids"].stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        LOGITS_BLOCK_SIZE=LOGITS_BLOCK_SIZE,
    )
    if side == "gpu":
        torch.cuda.synchronize()
    else:
        torch.npu.synchronize()
    return {"logits": logits}


def ref(t: dict[str, torch.Tensor], params: dict) -> dict[str, torch.Tensor]:
    """fp64 golden mirroring _apply_token_bias_kernel (order: allowed ->
    logit_bias -> min_tokens_stop):
      1. allowed_token_ids[r] non-empty: save those logits, set row -inf, restore
      2. add bias[r] at bias_token_ids[r]
      3. pos+1 < min_lens[r]: set stop_token_ids[r] to -inf
    """
    logits = t["logits"].to(torch.float64).clone()
    n_tok = logits.shape[0]
    mapping = t["expanded_idx_mapping"].long()
    for tok in range(n_tok):
        r = int(mapping[tok])
        num_allowed = int(t["num_allowed_token_ids"][r])
        if num_allowed > 0:
            allowed = t["allowed_token_ids"][r, :num_allowed].long()
            saved = logits[tok, allowed].clone()
            logits[tok, :] = float("-inf")
            logits[tok, allowed] = saved
        num_bias = int(t["num_logit_bias"][r])
        if num_bias > 0:
            # sequential adds so duplicate token ids accumulate exactly like
            # the kernel's load-add-store per (tok, bias) pair
            for j in range(num_bias):
                tid = int(t["bias_token_ids"][r, j])
                logits[tok, tid] = logits[tok, tid] + float(t["bias"][r, j])
        num_stops = int(t["num_stop_token_ids"][r])
        if num_stops > 0 and int(t["pos"][tok]) + 1 < int(t["min_lens"][r]):
            stops = t["stop_token_ids"][r, :num_stops].long()
            logits[tok, stops] = float("-inf")
    return {"logits": logits}


def _mk(name: str, n_tok: int, vocab: int, n_req: int, features: list[str]) -> CaseSpec:
    return CaseSpec(
        kernel="bias", name=name,
        params={"num_tokens": n_tok, "vocab_size": vocab, "num_requests": n_req,
                "features": features},
        seed=42,
        output_modes={"logits": cr.MODE_F32},
    )


CASES = [
    _mk("all_features_4t_1024v_4r", 4, 1024, 4, ["allowed", "bias", "min_tokens"]),
    _mk("bias_min_tokens_4t_129280v_4r", 4, 129280, 4, ["bias", "min_tokens"]),
    _mk("no_features_2t_64v_2r", 2, 64, 2, []),
]
