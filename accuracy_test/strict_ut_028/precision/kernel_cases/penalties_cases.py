"""penalties: logits numeric class (apply_penalties).

GPU side: vanilla vllm Triton kernel (vllm.v1.worker.gpu.sample.penalties).
NPU side: vllm-ascend kernel (vllm_ascend.worker.v2.sample.penalties).
Wrapper signatures are positionally identical.

Inputs are rebuilt on CPU only (the source UT used device RNG, which is not
reproducible across cuda/npu). The draft-lookback invariant token_idx-pos>=0
is respected (see gpu/test_penalties.py history for the OOB lesson).
"""

from __future__ import annotations

import torch

import capture_runtime as cr
from capture_runtime import CaseSpec

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
_MODES = {"bfloat16": cr.MODE_BF16, "float16": cr.MODE_F16, "float32": cr.MODE_F32}


def build_inputs(params: dict, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_tok, vocab = params["num_tokens"], params["vocab_size"]
    n_status, n_spec = params["num_status"], params["num_speculative_tokens"]
    dtype = _DTYPES[params["dtype"]]

    logits = torch.randn(n_tok, vocab, generator=g, dtype=torch.float32).to(dtype)

    rep = torch.ones(n_status, dtype=torch.float32)
    freq = torch.zeros(n_status, dtype=torch.float32)
    pres = torch.zeros(n_status, dtype=torch.float32)
    for i in range(n_status):
        if torch.rand(1, generator=g) > 0.3:
            rep[i] = torch.rand(1, generator=g) * 0.8 + 0.6
        if torch.rand(1, generator=g) > 0.5:
            freq[i] = torch.rand(1, generator=g) * 0.2
        if torch.rand(1, generator=g) > 0.5:
            pres[i] = torch.rand(1, generator=g) * 0.2

    idx_mapping = torch.randint(0, n_status, (n_tok,), generator=g, dtype=torch.int32)
    token_ids = torch.randint(0, vocab, (n_tok,), generator=g, dtype=torch.int32)
    pos = torch.tensor([int(torch.randint(0, min(i, n_spec) + 1, (1,), generator=g)) for i in range(n_tok)], dtype=torch.int32)

    num_packed = (vocab + 31) // 32
    prompt_bin_mask = torch.zeros(n_status, num_packed, dtype=torch.int32)
    n_prompt = max(1, vocab // 20)
    for s in range(n_status):
        for token_id in torch.randperm(vocab, generator=g)[:n_prompt].tolist():
            prompt_bin_mask[s, token_id // 32] |= 1 << (token_id % 32)

    output_bin_counts = torch.zeros(n_status, vocab, dtype=torch.int32)
    n_out = max(1, vocab // 20)
    for s in range(n_status):
        # Unique indices (randperm): a scatter with duplicated indices is
        # officially non-deterministic in PyTorch, and torch 2.10 vs 2.13
        # resolve the 660 duplicate writes differently (probe 2026-08-28:
        # identical nonzero bins, different surviving counts). Unique
        # indices make the result independent of write order on any build.
        toks = torch.randperm(vocab, generator=g)[:n_out]
        cnts = torch.randint(1, 10, (n_out,), generator=g)
        output_bin_counts[s, toks] = cnts.to(torch.int32)

    return {
        "logits": logits,
        "idx_mapping": idx_mapping,
        "token_ids": token_ids,
        "expanded_local_pos": pos,
        "repetition_penalty": rep,
        "frequency_penalty": freq,
        "presence_penalty": pres,
        "prompt_bin_mask": prompt_bin_mask,
        "output_bin_counts": output_bin_counts,
    }


def run(side: str, t: dict[str, torch.Tensor], params: dict) -> dict[str, torch.Tensor]:
    if side == "gpu":
        from vllm.v1.worker.gpu.sample.penalties import apply_penalties
    else:
        from vllm_ascend.worker.v2.sample.penalties import apply_penalties

    logits = t["logits"].clone()
    # output_bin_counts is an in/out tensor: the kernel updates it in place.
    # Pass a clone so the recorded IN digest reflects the true pre-run input
    # (the harness computes digests AFTER run(); without the clone the IN
    # digest silently captured the post-kernel counts and lit up as a fake
    # IN MISMATCH, plus the in-memory input was polluted).
    bin_counts = t["output_bin_counts"].clone()
    apply_penalties(
        logits,
        t["idx_mapping"],
        t["token_ids"],
        t["expanded_local_pos"],
        t["repetition_penalty"],
        t["frequency_penalty"],
        t["presence_penalty"],
        t["prompt_bin_mask"],
        bin_counts,
    )
    if side == "gpu":
        torch.cuda.synchronize()
    else:
        torch.npu.synchronize()
    return {"logits": logits}


def ref(t: dict[str, torch.Tensor], params: dict) -> dict[str, torch.Tensor]:
    """fp64 golden mirroring _apply_penalties_kernel:
      early-out unless rep != 1 or freq != 0 or pres != 0
      oc = output_bin_counts[r] + count(this step's committed tokens at
          base + spec_offset + 1 = already sampled draft tokens)
      rep: logits *= (logits > 0) ? 1/rep : rep   on oc>0 | prompt_mask
      logits -= freq * oc;  logits -= pres * (oc > 0)
    prompt_bin_mask is unpacked little-endian per 32-token block.
    """
    logits = t["logits"].to(torch.float64).clone()
    n_tok, vocab = logits.shape
    mapping = t["idx_mapping"].long()
    token_ids = t["token_ids"].long()
    pos_t = t["expanded_local_pos"].long()
    rep = t["repetition_penalty"].to(torch.float64)
    freq = t["frequency_penalty"].to(torch.float64)
    pres = t["presence_penalty"].to(torch.float64)
    packed = t["prompt_bin_mask"].long()                       # [s, num_packed]
    counts = t["output_bin_counts"].long()                     # [s, vocab]

    bits = (packed.unsqueeze(-1) >> torch.arange(32)) & 1      # [s, p, 32]
    prompt_mask = bits.reshape(packed.shape[0], -1)[:, :vocab].bool()

    for tok in range(n_tok):
        r = int(mapping[tok])
        use_rep = rep[r] != 1.0
        use_freq = freq[r] != 0.0
        use_pres = pres[r] != 0.0
        if not (use_rep or use_freq or use_pres):
            continue
        pos = int(pos_t[tok])
        base = tok - pos
        oc = counts[r].clone()
        for pp in range(pos):
            oc[int(token_ids[base + pp + 1])] += 1
        omask = oc > 0
        row = logits[tok]
        if use_rep:
            scale = torch.where(prompt_mask[r] | omask, rep[r], torch.ones_like(rep[r]))
            row = row * torch.where(row > 0, 1.0 / scale, scale)
        row = row - freq[r] * oc.to(torch.float64)
        row = row - pres[r] * omask.to(torch.float64)
        logits[tok] = row
    return {"logits": logits}


def _mk(name: str, n_tok: int, vocab: int, n_status: int, n_spec: int, dtype: str) -> CaseSpec:
    params = {"num_tokens": n_tok, "vocab_size": vocab, "num_status": n_status,
              "num_speculative_tokens": n_spec, "dtype": dtype}
    return CaseSpec(kernel="penalties", name=name, params=params, seed=42,
                    output_modes={"logits": _MODES[dtype]})


CASES = [
    _mk("basic_4t_1000v_4s_3spec_bf16", 4, 1000, 4, 3, "bfloat16"),
    _mk("basic_4t_1000v_4s_3spec_fp16", 4, 1000, 4, 3, "float16"),
    _mk("single_1t_1000v_1s_0spec_bf16", 1, 1000, 1, 0, "bfloat16"),
    _mk("large_4t_129280v_4s_3spec_bf16", 4, 129280, 4, 3, "bfloat16"),
]
