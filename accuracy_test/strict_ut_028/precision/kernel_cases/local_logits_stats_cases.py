"""compute_local_logits_stats: spec-decode logits numeric class (direct launch).

Both sides import the SAME upstream kernel
(vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils), so positional args
and constexprs are identical (BLOCK_SIZE=VOCAB_BLOCK_SIZE, HAS_DRAFT_LOGITS).

Kernel semantics (post PR #47041-era): draft logits are divided by the
per-request temperature inside the kernel before max/sumexp -- both sides run
the same code, so the compare basis is the raw outputs.

Outputs (block-level stats per (logit, vocab-block)):
  target_local_argmax : int64 -> int_exact
  target_local_max    : fp32  -> 1e-5 (pure max reduction, stable)
  target_local_sumexp : fp32  -> 1e-3 (exp-sum, cross-device reduction order)
  draft_local_max     : fp32  -> 1e-5
  draft_local_sumexp  : fp32  -> 1e-3
"""

from __future__ import annotations

import torch

import capture_runtime as cr
from capture_runtime import CaseSpec

VOCAB_BLOCK_SIZE = 8192


def build_inputs(params: dict, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_logits, vocab, n_spec = params["num_logits"], params["vocab_size"], params["num_speculative_steps"]
    max_num_reqs = 4

    target_logits = torch.randn(n_logits, vocab, generator=g, dtype=torch.float32)
    draft_logits = torch.randn(max_num_reqs, n_spec, vocab, generator=g, dtype=torch.float32)
    mapping = torch.arange(n_logits, dtype=torch.int64) % max_num_reqs
    local_pos = torch.zeros(n_logits, dtype=torch.int64)
    # temperatures: 1.0 (identity) and 0.8 (real scaling); avoid 0.0 (greedy
    # argmax path) so draft max/sumexp stay on the well-defined divide path.
    temperature = torch.tensor([1.0, 0.8, 1.0, 0.8], dtype=torch.float32)
    return {
        "target_logits": target_logits,
        "draft_logits": draft_logits,
        "expanded_idx_mapping": mapping,
        "expanded_local_pos": local_pos,
        "temperature": temperature,
    }


def run(side: str, t: dict[str, torch.Tensor], params: dict) -> dict[str, torch.Tensor]:
    from vllm.triton_utils import triton
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        _compute_local_logits_stats_kernel,
    )

    n_logits, vocab, n_spec = params["num_logits"], params["vocab_size"], params["num_speculative_steps"]
    vocab_num_blocks = triton.cdiv(vocab, VOCAB_BLOCK_SIZE)
    dev = t["target_logits"].device

    target_local_argmax = torch.zeros(n_logits, vocab_num_blocks, dtype=torch.int64, device=dev)
    target_local_max = torch.zeros(n_logits, vocab_num_blocks, dtype=torch.float32, device=dev)
    target_local_sumexp = torch.zeros(n_logits, vocab_num_blocks, dtype=torch.float32, device=dev)
    draft_local_max = torch.zeros(n_logits, vocab_num_blocks, dtype=torch.float32, device=dev)
    draft_local_sumexp = torch.zeros(n_logits, vocab_num_blocks, dtype=torch.float32, device=dev)

    _compute_local_logits_stats_kernel[(n_logits, vocab_num_blocks)](
        target_local_argmax, target_local_argmax.stride(0),
        target_local_max, target_local_max.stride(0),
        target_local_sumexp, target_local_sumexp.stride(0),
        draft_local_max, draft_local_max.stride(0),
        draft_local_sumexp, draft_local_sumexp.stride(0),
        t["target_logits"], t["target_logits"].stride(0),
        t["draft_logits"], t["draft_logits"].stride(0), t["draft_logits"].stride(1),
        t["expanded_idx_mapping"], t["expanded_local_pos"], t["temperature"],
        vocab, n_spec,
        BLOCK_SIZE=VOCAB_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=True,
    )
    if side == "gpu":
        torch.cuda.synchronize()
    else:
        torch.npu.synchronize()
    return {
        "target_local_argmax": target_local_argmax,
        "target_local_max": target_local_max,
        "target_local_sumexp": target_local_sumexp,
        "draft_local_max": draft_local_max,
        "draft_local_sumexp": draft_local_sumexp,
    }


def _mk(name: str, n_logits: int, vocab: int, n_spec: int) -> CaseSpec:
    return CaseSpec(
        kernel="compute_local_logits_stats", name=name,
        params={"num_logits": n_logits, "vocab_size": vocab, "num_speculative_steps": n_spec},
        seed=42,
        output_modes={
            "target_local_argmax": cr.MODE_INT_EXACT,
            "target_local_max": cr.MODE_F32,
            "target_local_sumexp": cr.MODE_F16,
            "draft_local_max": cr.MODE_F32,
            "draft_local_sumexp": cr.MODE_F16,
        },
    )


CASES = [
    _mk("small_2l_1024v_1spec", 2, 1024, 1),
    _mk("multi_4l_16384v_2spec", 4, 16384, 2),
    _mk("deepseek_2l_129280v_3spec", 2, 129280, 3),
]
