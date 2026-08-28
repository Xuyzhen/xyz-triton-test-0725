"""rejection (_rejection_kernel / _probabilistic_rejection_kernel):
spec-decode logits numeric class - decision-level compare.

API divergence (verified on both checkouts):
  GPU vanilla : vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils.
                _rejection_kernel -- extra tail args (synthetic_conditional_rates,
                cumulative_log_p, local_residual_mass + stride) and constexprs
                SYNTHETIC_MODE / USE_BLOCK_VERIFICATION.
  NPU ascend  : vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils.
                _probabilistic_rejection_kernel -- tail ends at
                vocab_num_blocks / PADDED_VOCAB_NUM_BLOCKS / HAS_DRAFT_LOGITS.

Input dtype divergence: draft_sampled int64[num_logits+1] & seed int32 &
cu_num_logits int64 on GPU; draft_sampled int32[num_logits] & seeds int64 &
cu_num_logits int32 on NPU. run() converts per side.

Compare basis (per 精度标准 2.1 - 浮点计算类):
  The acceptance decision consumes a philox draw u, so `sampled` is RNG
  dependent in non-greedy mode. Every case is therefore constructed with a
  SIGN-DEFINITE acceptance margin (mirrors the strict UTs):
    greedy           : temp=0, decision = argmax comparison (no RNG).
    always_accept    : target near-one-hot at the draft token ->
                       target_log_prob == 0 >= log(u) + draft_log_prob.
    always_reject    : draft token impossible under target ->
                       margin far below log(2^-31) (philox u floor).
  With the decision point deterministic, the fp32 outputs
  target/draft_rejected_logsumexp are deterministic on both sides and are
  compared numerically (MODE_F16); `sampled` is INT_EXACT for greedy cases
  only (argmax path, no RNG) and MODE_SKIP for non-greedy cases (the
  rejection-point token is resampled with per-device RNG).
"""

from __future__ import annotations

import torch

import capture_runtime as cr
from capture_runtime import CaseSpec

VOCAB_BLOCK_SIZE = 8192


def build_inputs(params: dict, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_req, n_spec, vocab = params["num_reqs"], params["num_speculative_steps"], params["vocab_size"]
    mode = params["mode"]  # "greedy_accept" | "greedy_reject" | "accept" | "reject"
    num_logits = n_req * (n_spec + 1)

    target_logits = torch.randn(num_logits, vocab, generator=g, dtype=torch.float32)
    draft_logits = torch.randn(n_req, n_spec, vocab, generator=g, dtype=torch.float32)

    # draft token per draft position; index li+1 mirrors both strict UTs.
    draft_sampled = torch.randint(0, vocab, (num_logits + 1,), generator=g, dtype=torch.int64)
    if mode == "greedy_accept":
        for ri in range(n_req):
            for di in range(n_spec):
                li = ri * (n_spec + 1) + di
                draft_sampled[li + 1] = int(target_logits[li].argmax().item())
    elif mode == "greedy_reject":
        for ri in range(n_req):
            for di in range(n_spec):
                li = ri * (n_spec + 1) + di
                draft_sampled[li + 1] = (int(target_logits[li].argmax().item()) + 1) % vocab
    elif mode == "accept":
        for ri in range(n_req):
            for di in range(n_spec):
                li = ri * (n_spec + 1) + di
                target_logits[li, draft_sampled[li + 1]] = 100.0
    elif mode == "reject":
        for ri in range(n_req):
            for di in range(n_spec):
                li = ri * (n_spec + 1) + di
                target_logits[li, draft_sampled[li + 1]] = -100.0
    else:
        raise ValueError(f"unknown mode {mode}")

    temperature = torch.zeros(n_req, dtype=torch.float32) if mode.startswith("greedy") \
        else torch.ones(n_req, dtype=torch.float32)

    expanded_idx_mapping = torch.zeros(num_logits, dtype=torch.int64)
    expanded_local_pos = torch.zeros(num_logits, dtype=torch.int64)
    for ri in range(n_req):
        for di in range(n_spec + 1):
            li = ri * (n_spec + 1) + di
            expanded_idx_mapping[li] = ri
            expanded_local_pos[li] = di

    seeds = torch.randint(0, 2**31, (n_req,), generator=g, dtype=torch.int64)
    pos = torch.arange(num_logits, dtype=torch.int64)
    cu_num_logits = torch.arange(n_req + 1, dtype=torch.int64) * (n_spec + 1)
    idx_mapping = torch.arange(n_req, dtype=torch.int32)

    return {
        "target_logits": target_logits,
        "draft_logits": draft_logits,
        "draft_sampled": draft_sampled,
        "temperature": temperature,
        "seeds": seeds,
        "pos": pos,
        "cu_num_logits": cu_num_logits,
        "idx_mapping": idx_mapping,
        "expanded_idx_mapping": expanded_idx_mapping,
        "expanded_local_pos": expanded_local_pos,
    }


def _compute_stats(side: str, t: dict, params: dict, dev) -> dict[str, torch.Tensor]:
    """Block stats via the shared vllm kernel (alias-shimmed like the NPU UT)."""
    from vllm.triton_utils import triton
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as vllm_utils

    block_stats = getattr(vllm_utils, "_compute_local_logits_stats_kernel", None)
    if block_stats is None:
        block_stats = getattr(vllm_utils, "_compute_block_stats_kernel", None)
    if block_stats is None:
        raise ImportError("vllm rejection_sampler_utils lacks a block-stats kernel")

    n_req, n_spec, vocab = params["num_reqs"], params["num_speculative_steps"], params["vocab_size"]
    num_logits = n_req * (n_spec + 1)
    vocab_num_blocks = triton.cdiv(vocab, VOCAB_BLOCK_SIZE)

    out = {
        "target_local_argmax": torch.zeros(num_logits, vocab_num_blocks, dtype=torch.int64, device=dev),
        "target_local_max": torch.zeros(num_logits, vocab_num_blocks, dtype=torch.float32, device=dev),
        "target_local_sumexp": torch.zeros(num_logits, vocab_num_blocks, dtype=torch.float32, device=dev),
        "draft_local_max": torch.zeros(num_logits, vocab_num_blocks, dtype=torch.float32, device=dev),
        "draft_local_sumexp": torch.zeros(num_logits, vocab_num_blocks, dtype=torch.float32, device=dev),
    }
    block_stats[(num_logits, vocab_num_blocks)](
        out["target_local_argmax"], out["target_local_argmax"].stride(0),
        out["target_local_max"], out["target_local_max"].stride(0),
        out["target_local_sumexp"], out["target_local_sumexp"].stride(0),
        out["draft_local_max"], out["draft_local_max"].stride(0),
        out["draft_local_sumexp"], out["draft_local_sumexp"].stride(0),
        t["target_logits"], t["target_logits"].stride(0),
        t["draft_logits"], t["draft_logits"].stride(0), t["draft_logits"].stride(1),
        t["expanded_idx_mapping"], t["expanded_local_pos"], t["temperature"],
        vocab, n_spec,
        BLOCK_SIZE=VOCAB_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=True,
    )
    return out


def run(side: str, t: dict[str, torch.Tensor], params: dict) -> dict[str, torch.Tensor]:
    from vllm.triton_utils import triton

    n_req, n_spec, vocab = params["num_reqs"], params["num_speculative_steps"], params["vocab_size"]
    num_logits = n_req * (n_spec + 1)
    dev = t["target_logits"].device
    vocab_num_blocks = triton.cdiv(vocab, VOCAB_BLOCK_SIZE)
    padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)

    stats = _compute_stats(side, t, params, dev)

    sampled = torch.zeros(n_req, n_spec + 1, dtype=torch.int64, device=dev)
    rejected_steps = torch.zeros(n_req, dtype=torch.int32, device=dev)
    target_rejected_lse = torch.zeros(n_req, dtype=torch.float32, device=dev)
    draft_rejected_lse = torch.zeros(n_req, dtype=torch.float32, device=dev)

    if side == "gpu":
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import _rejection_kernel
        _rejection_kernel[(n_req,)](
            sampled, sampled.stride(0),
            rejected_steps, target_rejected_lse, draft_rejected_lse,
            t["target_logits"], t["target_logits"].stride(0),
            stats["target_local_argmax"], stats["target_local_argmax"].stride(0),
            stats["target_local_max"], stats["target_local_max"].stride(0),
            stats["target_local_sumexp"], stats["target_local_sumexp"].stride(0),
            t["draft_sampled"].to(torch.int64),
            t["draft_logits"], t["draft_logits"].stride(0), t["draft_logits"].stride(1),
            stats["draft_local_max"], stats["draft_local_max"].stride(0),
            stats["draft_local_sumexp"], stats["draft_local_sumexp"].stride(0),
            t["cu_num_logits"].to(torch.int64), t["idx_mapping"], t["temperature"],
            t["seeds"].to(torch.int32), t["pos"],
            None,  # synthetic_conditional_rates
            None,  # cumulative_log_p
            None,  # local_residual_mass
            0,     # local_residual_mass_stride
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
            HAS_DRAFT_LOGITS=True,
            SYNTHETIC_MODE=False,
            USE_BLOCK_VERIFICATION=False,
            num_warps=1,
        )
        torch.cuda.synchronize()
    else:
        from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import (
            _probabilistic_rejection_kernel,
        )
        _probabilistic_rejection_kernel[(n_req,)](
            sampled, sampled.stride(0),
            rejected_steps, target_rejected_lse, draft_rejected_lse,
            t["target_logits"], t["target_logits"].stride(0),
            stats["target_local_argmax"], stats["target_local_argmax"].stride(0),
            stats["target_local_max"], stats["target_local_max"].stride(0),
            stats["target_local_sumexp"], stats["target_local_sumexp"].stride(0),
            t["draft_sampled"].to(torch.int32)[:num_logits],
            t["draft_logits"], t["draft_logits"].stride(0), t["draft_logits"].stride(1),
            stats["draft_local_max"], stats["draft_local_max"].stride(0),
            stats["draft_local_sumexp"], stats["draft_local_sumexp"].stride(0),
            t["cu_num_logits"].to(torch.int32), t["idx_mapping"], t["temperature"],
            t["seeds"].to(torch.int64), t["pos"],
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
            HAS_DRAFT_LOGITS=True,
            num_warps=1,
        )
        torch.npu.synchronize()

    return {
        "sampled": sampled,
        "rejected_steps": rejected_steps,
        "target_rejected_logsumexp": target_rejected_lse,
        "draft_rejected_logsumexp": draft_rejected_lse,
    }


def ref(t: dict[str, torch.Tensor], params: dict) -> dict[str, torch.Tensor]:
    """fp64 golden mirroring _rejection_kernel (SYNTHETIC_MODE=False,
    USE_BLOCK_VERIFICATION=False, HAS_DRAFT_LOGITS=True).

    Per request, walk the n_spec draft positions:
      greedy (temp==0): accepted = (target row argmax == draft token);
                        sampled[i] = draft if accepted else target argmax;
                        lse outputs stay 0 (greedy branch never computes them)
      non-greedy:       ratio test p(x) > u*q(x); case construction plants
                        p/q far from 1 (logit +-100), so the u-dependence
                        vanishes: accept iff log_p >= log_q.
                        lse outputs = LSE values of the LAST verified step
                        (target row / draft row / temp).
    rejected_steps = accepted_length. sampled[bonus] stays 0 (never written).
    """
    n_req, n_spec = params["num_reqs"], params["num_speculative_steps"]
    f64 = torch.float64
    target = t["target_logits"].to(f64)
    draft = t["draft_logits"].to(f64)
    draft_sampled = t["draft_sampled"].long()
    temp = t["temperature"].to(f64)

    sampled = torch.zeros(n_req, n_spec + 1, dtype=torch.int64)
    rejected_steps = torch.zeros(n_req, dtype=torch.int32)
    t_lse = torch.zeros(n_req, dtype=f64)
    d_lse = torch.zeros(n_req, dtype=f64)

    for ri in range(n_req):
        start = ri * (n_spec + 1)
        tp = float(temp[ri])
        greedy = tp == 0.0
        accepted_length = 0
        tl_val, dl_val = 0.0, 0.0
        verifying = True
        for i in range(n_spec):
            if not verifying:
                break
            li = start + i
            dtok = int(draft_sampled[li + 1])
            if greedy:
                t_am = int(target[li].argmax())
                accepted = t_am == dtok
                sampled[ri, i] = dtok if accepted else t_am
            else:
                trow = target[li]
                tl_val = float(torch.logsumexp(trow, dim=-1))
                drow = draft[ri, i] / tp
                dl_val = float(torch.logsumexp(drow, dim=-1))
                accepted = bool(trow[dtok] - tl_val >= drow[dtok] - dl_val)
                sampled[ri, i] = dtok
            verifying = accepted
            accepted_length += int(accepted)
        rejected_steps[ri] = accepted_length
        if not greedy:
            t_lse[ri] = tl_val
            d_lse[ri] = dl_val
    return {"sampled": sampled, "rejected_steps": rejected_steps,
            "target_rejected_logsumexp": t_lse,
            "draft_rejected_logsumexp": d_lse}


def _mk(name: str, n_req: int, n_spec: int, vocab: int, mode: str) -> CaseSpec:
    greedy = mode.startswith("greedy")
    return CaseSpec(
        kernel="rejection", name=name,
        params={"num_reqs": n_req, "num_speculative_steps": n_spec,
                "vocab_size": vocab, "mode": mode},
        seed=42,
        output_modes={
            # greedy decisions are argmax-based (no RNG) -> exact;
            # non-greedy rejection-point token is RNG-resampled -> skip.
            "sampled": cr.MODE_INT_EXACT if greedy else cr.MODE_SKIP,
            "rejected_steps": cr.MODE_INT_EXACT,
            "target_rejected_logsumexp": cr.MODE_F16,
            "draft_rejected_logsumexp": cr.MODE_F16,
        },
    )


CASES = [
    _mk("greedy_accept_1r_2spec_128v", 1, 2, 128, "greedy_accept"),
    _mk("greedy_reject_1r_2spec_128v", 1, 2, 128, "greedy_reject"),
    _mk("greedy_multi_2r_3spec_1024v", 2, 3, 1024, "greedy_accept"),
    _mk("nongreedy_accept_1r_2spec_128v", 1, 2, 128, "accept"),
    _mk("nongreedy_reject_1r_2spec_128v", 1, 2, 128, "reject"),
    _mk("nongreedy_accept_1r_2spec_129280v", 1, 2, 129280, "accept"),
]
