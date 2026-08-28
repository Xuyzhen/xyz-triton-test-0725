"""gumbel_sample: stochastic class - compare deterministic intermediates only.

API divergence (verified on both checkouts):
  GPU vanilla : gumbel_sample(..., apply_temperature, logits_cache=[R, C, V],
                logits_cache_col=scalar_or_vec)  -> cache stores PRE-temperature
                logits (vllm #50910/#53017).
  NPU ascend  : gumbel_sample(..., apply_temperature, output_processed_logits=
                [R, V], output_processed_logits_col=...) -> cache stores
                POST-temperature logits (divided inside the kernel).

Compare basis: PRE-temperature logits. The normalize hook multiplies the NPU
cache rows back by their per-request temperature (temp==0 rows were never
divided, factor stays 1). The `sampled` token ids depend on per-device RNG
streams and are declared MODE_SKIP.

Determinism invariant: every case uses a 1:1 token->request mapping
(permutation). With a random many-to-one mapping, several tokens write the
same cache row (all at column 0), the surviving row depends on write order,
and the logits_cache digest drifts run-to-run on BOTH platforms (verified
2026-08-27/28 on GPU 010925/030652 and NPU 171050/173219/175106).
"""

from __future__ import annotations

import torch

import capture_runtime as cr
from capture_runtime import CaseSpec


def build_inputs(params: dict, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_tok, n_req, vocab = params["num_tokens"], params["num_reqs"], params["vocab_size"]
    logits = torch.randn(n_tok, vocab, generator=g, dtype=torch.float32)
    # 1:1 permutation mapping: each cache row is written by exactly one token
    # -> deterministic logits_cache regardless of kernel write order.
    assert n_tok == n_req, "gumbel cases require 1:1 token<->request mapping"
    mapping = torch.randperm(n_req, generator=g).to(torch.int32)
    temperature = torch.rand(n_req, generator=g, dtype=torch.float32) * 1.5 + 0.5
    temperature[0] = 1.0  # exercise the no-op divide path on both sides
    seed_t = torch.randint(0, 2**31, (n_req,), generator=g, dtype=torch.int64)
    pos = torch.arange(n_tok, dtype=torch.int32)
    return {
        "logits": logits,
        "expanded_idx_mapping": mapping,
        "temperature": temperature,
        "seed": seed_t,
        "pos": pos,
    }


def run(side: str, t: dict[str, torch.Tensor], params: dict) -> dict[str, torch.Tensor]:
    n_req, vocab = params["num_reqs"], params["vocab_size"]
    col0 = torch.zeros((), dtype=torch.int32, device=t["logits"].device)
    if side == "gpu":
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
        cache = torch.zeros(n_req, 1, vocab, dtype=torch.float32, device=t["logits"].device)
        sampled = gumbel_sample(
            t["logits"], t["expanded_idx_mapping"], t["temperature"], t["seed"], t["pos"],
            apply_temperature=True, logits_cache=cache, logits_cache_col=col0,
        )
        torch.cuda.synchronize()
        logits_cache = cache[:, 0, :].contiguous()
    else:
        from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample
        cache = torch.zeros(n_req, vocab, dtype=torch.float32, device=t["logits"].device)
        sampled = gumbel_sample(
            t["logits"], t["expanded_idx_mapping"], t["temperature"], t["seed"], t["pos"],
            True, output_processed_logits=cache, output_processed_logits_col=col0,
        )
        torch.npu.synchronize()
        logits_cache = cache
    return {"sampled": sampled, "logits_cache": logits_cache}


def normalize(output_name: str, side: str, tensor: torch.Tensor,
              inputs: dict[str, torch.Tensor], params: dict) -> torch.Tensor:
    """Bring both sides to the PRE-temperature compare basis."""
    if output_name != "logits_cache" or side != "npu":
        return tensor
    temp = inputs["temperature"].to(torch.float32)
    factor = torch.where(temp == 0.0, torch.ones_like(temp), temp)
    return tensor.to(torch.float32) * factor.view(-1, 1)


def ref(t: dict[str, torch.Tensor], params: dict) -> dict[str, torch.Tensor]:
    """Golden: logits_cache rows written by each token = that token's input
    logits row (PRE-temperature compare basis). 'sampled' is stochastic
    (MODE_SKIP) and is intentionally omitted."""
    mapping = t["expanded_idx_mapping"].long()
    # 1:1 token->request mapping: inverse permutation recovers the writer row
    inv = torch.argsort(mapping)
    cache = t["logits"][inv].to(torch.float64)
    return {"logits_cache": cache}


def _mk(name: str, n_tok: int, n_req: int, vocab: int) -> CaseSpec:
    return CaseSpec(
        kernel="gumbel_sample", name=name, stochastic=True,
        params={"num_tokens": n_tok, "num_reqs": n_req, "vocab_size": vocab},
        seed=42,
        output_modes={"sampled": cr.MODE_SKIP, "logits_cache": cr.MODE_F32},
        normalize=normalize,
    )


CASES = [
    _mk("basic_4t_4r_32000v", 4, 4, 32000),
    _mk("batch_8t_8r_32000v", 8, 8, 32000),
    _mk("deepseek_2t_2r_129280v", 2, 2, 129280),
]
