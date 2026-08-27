"""min_p: logits numeric class (apply_min_p, in-place).

GPU side: vanilla vllm (vllm.v1.worker.gpu.sample.min_p.apply_min_p).
NPU side: vllm-ascend (vllm_ascend.worker.v2.sample.min_p.apply_min_p).
Wrapper signatures are positionally identical. Tokens below
max + log(min_p) are set to -inf; min_p == 0.0 rows stay unchanged.
Compare basis: the processed logits themselves (float32).
"""

from __future__ import annotations

import torch

import capture_runtime as cr
from capture_runtime import CaseSpec


def build_inputs(params: dict, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_req, vocab = params["num_reqs"], params["vocab_size"]

    logits = torch.randn(n_req, vocab, generator=g, dtype=torch.float32)
    # reversed arange mapping (mirrors the UT): row i uses min_p of req n-1-i
    mapping = torch.arange(n_req - 1, -1, -1, dtype=torch.int32)
    min_p = torch.rand(n_req, generator=g, dtype=torch.float32) * 0.49 + 0.01
    if n_req >= 2:
        min_p[0] = 0.0  # disabled path: logits unchanged
    return {"logits": logits, "expanded_idx_mapping": mapping, "min_p": min_p}


def run(side: str, t: dict[str, torch.Tensor], params: dict) -> dict[str, torch.Tensor]:
    if side == "gpu":
        from vllm.v1.worker.gpu.sample.min_p import apply_min_p
    else:
        from vllm_ascend.worker.v2.sample.min_p import apply_min_p

    logits = t["logits"].clone()
    apply_min_p(logits, t["expanded_idx_mapping"], t["min_p"])
    if side == "gpu":
        torch.cuda.synchronize()
    else:
        torch.npu.synchronize()
    return {"logits": logits}


def _mk(name: str, n_req: int, vocab: int) -> CaseSpec:
    return CaseSpec(
        kernel="min_p", name=name,
        params={"num_reqs": n_req, "vocab_size": vocab},
        seed=42,
        output_modes={"logits": cr.MODE_F32},
    )


CASES = [
    _mk("basic_1r_32000v", 1, 32000),
    _mk("llama_4r_32000v", 4, 32000),
    _mk("deepseek_4r_129280v", 4, 129280),
    _mk("qwen_4r_248320v", 4, 248320),
]
