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


def ref(t: dict[str, torch.Tensor], params: dict) -> dict[str, torch.Tensor]:
    """fp64 golden mirroring _compute_local_logits_stats_kernel.

    For each verified row (local_pos < n_spec) and each vocab block b:
      greedy (temp==0): target_am = block-global argmax, target_max = max,
                        sums stay 0, draft stats stay 0
      else:             target stats over target_logits[row, block]
                        draft stats over draft_logits[req, local_pos, block]
                        / temp (per-request temperature)
    NOTE: the kernel only stores target_local_argmax in the greedy branch
    (temp==0); the non-greedy branch writes max/sumexp only, leaving argmax
    at its zero-initialized value. ref() must mirror that or every non-greedy
    case fails with a full argmax mismatch (verified 2026-08-28: gpu/npu
    argmax were all 0 while the old ref wrote real indices).
    Rows with local_pos == n_spec (bonus) write nothing: outputs stay at their
    zero-initialized values.
    """
    BLOCK = VOCAB_BLOCK_SIZE
    n_logits = params["num_logits"]
    vocab = params["vocab_size"]
    n_spec = params["num_speculative_steps"]
    n_blocks = (vocab + BLOCK - 1) // BLOCK

    target = t["target_logits"].to(torch.float64)
    draft = t["draft_logits"].to(torch.float64)
    mapping = t["expanded_idx_mapping"].long()
    pos = t["expanded_local_pos"].long()
    temp = t["temperature"].to(torch.float64)

    t_am = torch.zeros(n_logits, n_blocks, dtype=torch.int64)
    t_mx = torch.zeros(n_logits, n_blocks, dtype=torch.float64)
    t_se = torch.zeros(n_logits, n_blocks, dtype=torch.float64)
    d_mx = torch.zeros(n_logits, n_blocks, dtype=torch.float64)
    d_se = torch.zeros(n_logits, n_blocks, dtype=torch.float64)

    for li in range(n_logits):
        p = int(pos[li])
        if p >= n_spec:
            continue
        r = int(mapping[li])
        tp = float(temp[r])
        for b in range(n_blocks):
            seg = target[li, b * BLOCK:(b + 1) * BLOCK]
            if tp == 0.0:
                t_am[li, b] = b * BLOCK + int(seg.argmax())
                t_mx[li, b] = seg.max()
            else:
                m = seg.max()
                t_mx[li, b] = m
                t_se[li, b] = torch.exp(seg - m).sum()
                dseg = draft[r, p, b * BLOCK:(b + 1) * BLOCK] / tp
                dm = dseg.max()
                d_mx[li, b] = dm
                d_se[li, b] = torch.exp(dseg - dm).sum()
    return {"target_local_argmax": t_am, "target_local_max": t_mx,
            "target_local_sumexp": t_se, "draft_local_max": d_mx,
            "draft_local_sumexp": d_se}


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
