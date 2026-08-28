"""topk_log_softmax: logits numeric class (direct kernel launch).

GPU side: vllm.v1.worker.gpu.sample.logprob._topk_log_softmax_kernel.
NPU side: vllm_ascend.worker.v2.sample.logprob._topk_log_softmax_kernel.

API divergence (known, see ut_failure_analysis.md): positional args are
identical, but the constexpr names differ --
  GPU : TOPK_BLOCK_SIZE = min(next_power_of_2(k), 1024)
  NPU : PADDED_TOPK     = max(next_power_of_2(k), 2)
Both sides launch with the values their own host wrapper expects.

Output: gathered log-probs for the requested token ids (float32).
Tolerance: UT uses rtol=atol=1e-3 -> MODE_F16 tolerances.
"""

from __future__ import annotations

import torch

import capture_runtime as cr
from capture_runtime import CaseSpec


def build_inputs(params: dict, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    batch, vocab, k = params["batch_size"], params["vocab_size"], params["num_logprobs"]
    logits = torch.randn(batch, vocab, generator=g, dtype=torch.float32)
    token_ids = torch.randint(0, vocab, (batch, k), generator=g, dtype=torch.int64)
    return {"logits": logits, "token_ids": token_ids}


def run(side: str, t: dict[str, torch.Tensor], params: dict) -> dict[str, torch.Tensor]:
    batch, vocab, k = params["batch_size"], params["vocab_size"], params["num_logprobs"]
    if side == "gpu":
        from vllm.triton_utils import triton
        from vllm.v1.worker.gpu.sample.logprob import _topk_log_softmax_kernel
        constexpr = {"BLOCK_SIZE": 1024,
                     "TOPK_BLOCK_SIZE": min(triton.next_power_of_2(k), 1024)}
    else:
        from vllm.triton_utils import triton
        from vllm_ascend.worker.v2.sample.logprob import _topk_log_softmax_kernel
        constexpr = {"BLOCK_SIZE": 1024,
                     "PADDED_TOPK": max(triton.next_power_of_2(k), 2)}

    logits, token_ids = t["logits"], t["token_ids"]
    out = torch.empty(batch, k, dtype=torch.float32, device=logits.device)
    _topk_log_softmax_kernel[(batch,)](
        out, logits, logits.stride(0), token_ids, k, vocab, **constexpr
    )
    if side == "gpu":
        torch.cuda.synchronize()
    else:
        torch.npu.synchronize()
    return {"logprobs": out}


def ref(t: dict[str, torch.Tensor], params: dict) -> dict[str, torch.Tensor]:
    """fp64 golden: logprob = logit[token_id] - row_max - log(sum(exp(row - row_max)))
    (mirrors _topk_log_softmax_kernel)."""
    logits = t["logits"].to(torch.float64)
    max_val = logits.max(dim=-1, keepdim=True).values
    lse = torch.log(torch.exp(logits - max_val).sum(dim=-1, keepdim=True))
    gathered = torch.gather(logits, -1, t["token_ids"].long())
    return {"logprobs": gathered - max_val - lse}


def _mk(name: str, batch: int, vocab: int, k: int) -> CaseSpec:
    return CaseSpec(
        kernel="topk_log_softmax", name=name,
        params={"batch_size": batch, "vocab_size": vocab, "num_logprobs": k},
        seed=42,
        output_modes={"logprobs": cr.MODE_F16},
    )


CASES = [
    _mk("basic_8r_32000v_k1", 8, 32000, 1),
    _mk("glm_24r_151936v_k8", 24, 151936, 8),
    _mk("deepseek_8r_129280v_k6", 8, 129280, 6),
    _mk("kimi_8r_163840v_k8", 8, 163840, 8),
]
