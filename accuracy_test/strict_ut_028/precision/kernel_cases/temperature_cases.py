"""temperature: logits numeric class (apply_temperature, in-place).

GPU side: vanilla vllm (vllm.v1.worker.gpu.sample.gumbel.apply_temperature).
NPU side: vllm-ascend (vllm_ascend.worker.v2.sample.gumbel.apply_temperature).
Wrapper signatures are positionally identical; logits divided in-place by
per-request temperature. temp == 0.0 or 1.0 rows must stay unchanged
(edge semantics shared by both sides).
"""

from __future__ import annotations

import torch

import capture_runtime as cr
from capture_runtime import CaseSpec

_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
_MODES = {"bfloat16": cr.MODE_BF16, "float32": cr.MODE_F32}


def build_inputs(params: dict, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_tok, n_req, vocab = params["num_tokens"], params["num_reqs"], params["vocab_size"]
    dtype = _DTYPES[params["dtype"]]

    logits = torch.randn(n_tok, vocab, generator=g, dtype=torch.float32).to(dtype)
    mapping = torch.arange(n_tok, dtype=torch.int32) % n_req
    temperature = torch.rand(n_req, generator=g, dtype=torch.float32) * 1.8 + 0.2
    if n_req >= 3:
        temperature[0] = 0.0  # greedy: unchanged
        temperature[1] = 1.0  # identity: unchanged
    return {"logits": logits, "expanded_idx_mapping": mapping, "temperature": temperature}


def run(side: str, t: dict[str, torch.Tensor], params: dict) -> dict[str, torch.Tensor]:
    if side == "gpu":
        from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
    else:
        from vllm_ascend.worker.v2.sample.gumbel import apply_temperature

    logits = t["logits"].clone()
    apply_temperature(logits, t["expanded_idx_mapping"], t["temperature"])
    if side == "gpu":
        torch.cuda.synchronize()
    else:
        torch.npu.synchronize()
    return {"logits": logits}


def _mk(name: str, n_tok: int, n_req: int, vocab: int, dtype: str) -> CaseSpec:
    return CaseSpec(
        kernel="temperature", name=name,
        params={"num_tokens": n_tok, "num_reqs": n_req, "vocab_size": vocab, "dtype": dtype},
        seed=42,
        output_modes={"logits": _MODES[dtype]},
    )


CASES = [
    _mk("basic_4t_4r_1000v_f32", 4, 4, 1000, "float32"),
    _mk("llama_8t_4r_32000v_f32", 8, 4, 32000, "float32"),
    _mk("deepseek_2t_2r_129280v_f32", 2, 2, 129280, "float32"),
    _mk("mixed_4t_4r_32000v_bf16", 4, 4, 32000, "bfloat16"),
]
