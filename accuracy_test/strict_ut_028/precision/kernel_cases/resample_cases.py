"""resample (_resample_kernel): spec-decode logits numeric class - deterministic
bonus-token path only.

API divergence (verified on both checkouts):
  GPU vanilla : vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils.
                _resample_kernel -- optional cumulative_log_p positional +
                USE_BLOCK_VERIFICATION / USE_FP64 constexprs (detected via
                arg_names, legacy installs lack them).
  NPU ascend  : vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils.
                _resample_kernel -- tail ends at vocab_size.

Input dtype divergence: rejected_step int64 / draft_sampled int64 / seed
int32 / pos int32 on GPU; rejected_step int32 / draft_sampled int32 / seed
int64 / pos int64 on NPU. run() converts per side.

Compare basis (per 精度标准 2.1 - 浮点计算类):
  At temp != 0 the kernel adds Gumbel noise from the per-device philox
  stream, so only temp == 0 cases are cross-platform comparable:
    bonus     : rejected_step points at the bonus token -> residual logits
                are the raw target row; outputs are per-block argmax (int,
                exact) and per-block max (fp32 reduction).
    greedy_noop: non-bonus + temp=0 -> kernel early-returns; outputs keep
                the -1 sentinel (proves the no-op guard on both sides).
  The draft-ratio residual path (log1p(1 - p_draft/p_target)) is only
  reachable at temp != 0 (RNG) or for non-bonus tokens (greedy no-op), so
  it is covered by the platform-local strict UTs, not by this capture.
"""

from __future__ import annotations

import torch

import capture_runtime as cr
from capture_runtime import CaseSpec

RESAMPLE_BLOCK_SIZE = 1024


def build_inputs(params: dict, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_req, n_spec, vocab = params["num_reqs"], params["num_speculative_steps"], params["vocab_size"]
    scenario = params["scenario"]  # "bonus" | "greedy_noop"
    has_draft = params["has_draft_logits"]
    num_logits = n_req * (n_spec + 1)

    target_logits = torch.randn(num_logits, vocab, generator=g, dtype=torch.float32)
    draft_logits = torch.randn(n_req, n_spec, vocab, generator=g, dtype=torch.float32) if has_draft \
        else torch.zeros(1, 1, 1, dtype=torch.float32)
    draft_sampled = torch.randint(0, vocab, (num_logits + 1,), generator=g, dtype=torch.int64)

    if scenario == "bonus":
        # Rejection at the last (bonus) position of every request.
        rejected_step = torch.full((n_req,), n_spec, dtype=torch.int64)
    else:
        # Mid-sequence rejection: non-bonus + temp=0 -> kernel no-op.
        rejected_step = torch.ones(n_req, dtype=torch.int64)

    temperature = torch.zeros(n_req, dtype=torch.float32)
    seeds = torch.randint(0, 2**31, (n_req,), generator=g, dtype=torch.int64)
    pos = torch.arange(num_logits, dtype=torch.int64)
    cu_num_logits = torch.arange(n_req + 1, dtype=torch.int32) * (n_spec + 1)
    expanded_idx_mapping = torch.arange(n_req, dtype=torch.int32).repeat_interleave(n_spec + 1)

    return {
        "target_logits": target_logits,
        "draft_logits": draft_logits,
        "draft_sampled": draft_sampled,
        "rejected_step": rejected_step,
        "temperature": temperature,
        "seeds": seeds,
        "pos": pos,
        "cu_num_logits": cu_num_logits,
        "expanded_idx_mapping": expanded_idx_mapping,
    }


def run(side: str, t: dict[str, torch.Tensor], params: dict) -> dict[str, torch.Tensor]:
    from vllm.triton_utils import triton

    n_req, n_spec, vocab = params["num_reqs"], params["num_speculative_steps"], params["vocab_size"]
    has_draft = params["has_draft_logits"]
    dev = t["target_logits"].device
    num_blocks = triton.cdiv(vocab, RESAMPLE_BLOCK_SIZE)
    padded_num_blocks = triton.next_power_of_2(num_blocks)

    # -1 sentinel: the greedy_noop scenario must leave outputs untouched.
    resampled_local_argmax = -torch.ones(n_req, padded_num_blocks, dtype=torch.int64, device=dev)
    resampled_local_max = -torch.ones(n_req, padded_num_blocks, dtype=torch.float32, device=dev)

    target_rejected_lse = torch.zeros(n_req, dtype=torch.float32, device=dev)
    draft_rejected_lse = torch.zeros(n_req, dtype=torch.float32, device=dev)

    if side == "gpu":
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import _resample_kernel
        arg_names = set(_resample_kernel.arg_names)
        has_bv = "cumulative_log_p_ptr" in arg_names and "USE_BLOCK_VERIFICATION" in arg_names
        num_logits = n_req * (n_spec + 1)
        cumulative_log_p = torch.zeros(num_logits, dtype=torch.float32, device=dev)
        args = [
            resampled_local_argmax, resampled_local_argmax.stride(0),
            resampled_local_max, resampled_local_max.stride(0),
            t["target_logits"], t["target_logits"].stride(0),
            target_rejected_lse,
            t["draft_logits"] if has_draft else torch.empty(1, 1, 1, dtype=torch.float32, device=dev),
            t["draft_logits"].stride(0) if has_draft else 0,
            t["draft_logits"].stride(1) if has_draft else 0,
            draft_rejected_lse,
            t["rejected_step"].to(torch.int64),
            t["cu_num_logits"],
            t["expanded_idx_mapping"],
            t["draft_sampled"].to(torch.int64),
            t["temperature"],
            t["seeds"].to(torch.int32),
            t["pos"].to(torch.int32),
        ]
        if has_bv:
            args.append(cumulative_log_p)
        args.append(vocab)
        kwargs = {"BLOCK_SIZE": RESAMPLE_BLOCK_SIZE, "HAS_DRAFT_LOGITS": has_draft}
        if "USE_FP64" in arg_names:
            kwargs["USE_FP64"] = False
        if has_bv:
            kwargs["USE_BLOCK_VERIFICATION"] = False
        _resample_kernel[(n_req, num_blocks)](*args, **kwargs)
        torch.cuda.synchronize()
    else:
        from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import _resample_kernel
        arg_names = set(_resample_kernel.arg_names)
        has_bv = "cumulative_log_p_ptr" in arg_names and "USE_BLOCK_VERIFICATION" in arg_names
        num_logits = n_req * (n_spec + 1)
        cumulative_log_p = torch.zeros(num_logits, dtype=torch.float32, device=dev)
        args = [
            resampled_local_argmax, resampled_local_argmax.stride(0),
            resampled_local_max, resampled_local_max.stride(0),
            t["target_logits"], t["target_logits"].stride(0),
            target_rejected_lse,
            t["draft_logits"] if has_draft else torch.empty(1, 1, 1, dtype=torch.float32, device=dev),
            t["draft_logits"].stride(0) if has_draft else 0,
            t["draft_logits"].stride(1) if has_draft else 0,
            draft_rejected_lse,
            t["rejected_step"].to(torch.int32),
            t["cu_num_logits"],
            t["expanded_idx_mapping"],
            t["draft_sampled"].to(torch.int32)[:num_logits],
            t["temperature"],
            t["seeds"].to(torch.int64),
            t["pos"].to(torch.int64),
        ]
        if has_bv:
            args.append(cumulative_log_p)
        args.append(vocab)
        kwargs = {"BLOCK_SIZE": RESAMPLE_BLOCK_SIZE, "HAS_DRAFT_LOGITS": has_draft}
        if "USE_FP64" in arg_names:
            kwargs["USE_FP64"] = False
        if has_bv:
            kwargs["USE_BLOCK_VERIFICATION"] = False
        _resample_kernel[(n_req, num_blocks)](*args, **kwargs)
        torch.npu.synchronize()

    return {
        "resampled_local_argmax": resampled_local_argmax,
        "resampled_local_max": resampled_local_max,
    }


def ref(t: dict[str, torch.Tensor], params: dict) -> dict[str, torch.Tensor]:
    """fp64 golden mirroring _resample_kernel for the temp==0 paths this
    capture covers (per the module docstring, temp!=0 adds per-device philox
    Gumbel noise and is not cross-platform comparable):

      bonus      : resample_token_idx == end-1 -> residual logits are the raw
                   target row; per-block plain argmax (no noise at temp 0).
                   token_id = block*BLOCK_SIZE + local argmax; value = max.
      greedy_noop: temp==0 and non-bonus -> kernel early-returns, outputs
                   keep the -1 sentinel.

    Blocks >= num_blocks (padding to next_power_of_2) also keep -1: the grid
    only launches num_blocks blocks.
    """
    n_req, n_spec, vocab = params["num_reqs"], params["num_speculative_steps"], params["vocab_size"]
    block_size = RESAMPLE_BLOCK_SIZE
    num_blocks = (vocab + block_size - 1) // block_size
    padded = 1
    while padded < num_blocks:
        padded *= 2

    argmax_out = -torch.ones(n_req, padded, dtype=torch.int64)
    max_out = -torch.ones(n_req, padded, dtype=torch.float64)

    if params["scenario"] == "bonus":
        target = t["target_logits"].to(torch.float64)
        temp = t["temperature"].to(torch.float64)
        rejected_step = t["rejected_step"].long()
        mapping = t["expanded_idx_mapping"].long()
        for ri in range(n_req):
            start = ri * (n_spec + 1)
            end = start + n_spec + 1
            ridx = int(rejected_step[ri])
            tok_idx = start + ridx
            is_bonus = tok_idx == end - 1
            tp = float(temp[int(mapping[tok_idx])])
            if tp == 0.0 and not is_bonus:
                continue  # greedy no-op guard: keep -1 sentinel
            row = target[tok_idx]
            for b in range(num_blocks):
                seg = row[b * block_size:(b + 1) * block_size]
                argmax_out[ri, b] = b * block_size + int(seg.argmax())
                max_out[ri, b] = seg.max()
    return {"resampled_local_argmax": argmax_out, "resampled_local_max": max_out}


def _mk(name: str, n_req: int, n_spec: int, vocab: int, scenario: str,
        has_draft: bool) -> CaseSpec:
    return CaseSpec(
        kernel="resample", name=name,
        params={"num_reqs": n_req, "num_speculative_steps": n_spec,
                "vocab_size": vocab, "scenario": scenario,
                "has_draft_logits": has_draft},
        seed=42,
        output_modes={
            "resampled_local_argmax": cr.MODE_INT_EXACT,
            "resampled_local_max": cr.MODE_F16,
        },
    )


CASES = [
    _mk("bonus_nodraft_1r_1spec_1024v", 1, 1, 1024, "bonus", False),
    _mk("bonus_draft_2r_2spec_1024v", 2, 2, 1024, "bonus", True),
    _mk("bonus_deepseek_1r_1spec_129280v", 1, 1, 129280, "bonus", False),
    _mk("greedy_noop_2r_2spec_512v", 2, 2, 512, "greedy_noop", True),
]
