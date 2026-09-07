# SPDX-License-Identifier: Apache-2.0
# strict_ut_027 shared dual-side UT for
# _prepare_input_hidden_states_and_embeddings_kernel.
# Source: vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py
# (upstream, no vllm-ascend patch). Category: non-compute + float copy/gather
# fusion (target hidden states shifted right by re-prefill gap; cached hidden
# states / embeddings fill the gap). No arithmetic on float data -> bitwise
# comparison is appropriate (fp32 values are copied verbatim).
#
# Dual-side design: re-exported by gpu/ and npu/ test entry modules; the
# side runtime (CUDA or Ascend NPU) is injected via the ``rt`` pytest fixture.
#
# Kernel summary (grid = (num_reqs, cdiv(max_query_len, BQ), cdiv(H, BH))):
#   Per (req_idx, query_block_idx, dim_block_idx):
#     1. req_state_idx = idx_mapping[req_idx]; query_start/end from
#        query_start_loc; num_rejected from num_rejected_ptr.
#     2. num_reprefill_hs = max(0, num_rejected - 1);
#        num_input_hs = query_len - num_rejected.
#     3. Copy target_hidden_states[query_start + query_block] to
#        draft_input_hidden_states[query_start + num_reprefill_hs + query_block]
#        for query_block < num_input_hs (masked).
#     4. If query_block_idx == 0: for i in 0..num_reprefill_hs-1, copy
#        cached_target_hidden_states[req_state_idx,
#        num_spec_steps-1-num_reprefill_hs+i] to draft_input_hidden_states[
#        query_start + i]. If USE_INPUT_EMBEDS: copy
#        cached_draft_input_embeds[...] to input_embeds[query_start + i].

from __future__ import annotations

import traceback
from typing import Any

import pytest
import torch

kernel = None
_import_error: Exception | None = None
_import_traceback: str | None = None
try:
    from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (
        _prepare_input_hidden_states_and_embeddings_kernel as kernel,
    )
except Exception as exc:  # pragma: no cover
    _import_error = exc
    _import_traceback = traceback.format_exc()


_SENTINEL = float("nan")


def _ref(
    num_reqs: int,
    num_speculative_steps: int,
    hidden_size: int,
    idx_mapping: torch.Tensor,
    target_hidden_states: torch.Tensor,  # [num_tokens, H]
    cached_target_hidden_states: torch.Tensor,  # [max_num_reqs, spec-1, H]
    input_embeds: torch.Tensor | None,  # [num_tokens, H] or None
    cached_draft_input_embeds: torch.Tensor | None,  # [max_num_reqs, spec-1, H]
    num_rejected: torch.Tensor,
    query_start_loc: torch.Tensor,
    draft_input_hidden_states: torch.Tensor,  # [num_tokens, H] output
) -> dict[str, torch.Tensor]:
    """Independent CPU reference. Returns expected output tensors."""
    dihs = draft_input_hidden_states.cpu().clone()
    iem = input_embeds.cpu().clone() if input_embeds is not None else None

    ths = target_hidden_states.cpu()
    cths = cached_target_hidden_states.cpu()
    cdie = cached_draft_input_embeds.cpu() if cached_draft_input_embeds is not None else None
    im = idx_mapping.cpu().to(torch.int64).tolist()
    nr = num_rejected.cpu().to(torch.int64).tolist()
    qsl = query_start_loc.cpu().to(torch.int64).tolist()

    for req_idx in range(num_reqs):
        req_state_idx = im[req_idx]
        query_start = qsl[req_idx]
        query_end = qsl[req_idx + 1]
        query_len = query_end - query_start
        num_rej = nr[req_idx]
        num_reprefill = max(0, num_rej - 1)
        num_input_hs = query_len - num_rej

        # Copy target -> draft (shifted right by num_reprefill)
        for b in range(num_input_hs):
            src = query_start + b
            dst = query_start + num_reprefill + b
            dihs[dst] = ths[src]

        # Fill re-prefill gap from cached hidden states
        for i in range(num_reprefill):
            cache_read_slot = num_speculative_steps - 1 - num_reprefill + i
            cached_hs = cths[req_state_idx, cache_read_slot]
            dihs[query_start + i] = cached_hs
            # Fill embeds if MM model
            if iem is not None and cdie is not None:
                cached_emb = cdie[req_state_idx, cache_read_slot]
                iem[query_start + i] = cached_emb

    result = {"draft_input_hidden_states": dihs}
    if iem is not None:
        result["input_embeds"] = iem
    return result


def _gen_inputs(
    num_reqs: int,
    max_num_reqs: int,
    num_tokens: int,
    hidden_size: int,
    num_speculative_steps: int,
    *,
    scenario: str,
    use_embeds: bool,
    device,
    block_size_q: int = 4,
    block_size_h: int = 8,
) -> dict[str, Any]:
    torch.manual_seed(42 + num_reqs * 7 + hidden_size + num_speculative_steps)

    idx_mapping = torch.arange(num_reqs, dtype=torch.int64, device=device)

    # Build query_lens and query_start_loc
    if scenario == "no_reject":
        query_lens = torch.randint(3, 8, (num_reqs,), dtype=torch.int32,
                                   device=device)
        num_rejected_vals = [0] * num_reqs
    elif scenario == "with_reject":
        query_lens = torch.randint(6, 12, (num_reqs,), dtype=torch.int32,
                                   device=device)
        # num_rejected must be <= num_speculative_steps: num_reprefill =
        # num_rejected - 1 fills at most num_speculative_steps - 1 cache slots.
        num_rejected_vals = [min(int(torch.randint(1, 4, ()).item()),
                                  int(query_lens[r].item()) - 2,
                                  num_speculative_steps)
                             for r in range(num_reqs)]
    elif scenario == "tile_boundary":
        # num_input_hs = block_size_q exactly
        query_lens = torch.full((num_reqs,), block_size_q + 2, dtype=torch.int32,
                                device=device)
        num_rejected_vals = [2] * num_reqs
    elif scenario == "max_reprefill":
        # num_reprefill = num_speculative_steps - 1 (full gap)
        query_lens = torch.full((num_reqs,), num_speculative_steps + 3,
                                dtype=torch.int32, device=device)
        num_rejected_vals = [num_speculative_steps] * num_reqs
    else:  # mixed
        query_lens = torch.randint(4, 12, (num_reqs,), dtype=torch.int32,
                                   device=device)
        num_rejected_vals = []
        for r in range(num_reqs):
            ql = int(query_lens[r].item())
            nr = min(int(torch.randint(0, max(1, ql // 2), ()).item()),
                     ql - 1, num_speculative_steps)
            num_rejected_vals.append(nr)

    query_start_loc = torch.zeros(max_num_reqs + 1, dtype=torch.int32,
                                  device=device)
    qsl_cpu = query_start_loc.cpu()
    qsl_cpu[0] = 0
    for r in range(num_reqs):
        qsl_cpu[r + 1] = qsl_cpu[r] + int(query_lens[r].item())
    qsl_cpu[num_reqs + 1:] = -1
    query_start_loc.copy_(qsl_cpu)

    num_rejected = torch.tensor(num_rejected_vals, dtype=torch.int32,
                                device=device)

    # Hidden states: random fp32
    target_hidden_states = torch.randn(num_tokens, hidden_size,
                                       dtype=torch.float32, device=device)
    cached_target_hidden_states = torch.randn(
        max_num_reqs, max(1, num_speculative_steps - 1), hidden_size,
        dtype=torch.float32, device=device)

    if use_embeds:
        input_embeds = torch.full((num_tokens, hidden_size), _SENTINEL,
                                  dtype=torch.float32, device=device)
        cached_draft_input_embeds = torch.randn(
            max_num_reqs, max(1, num_speculative_steps - 1), hidden_size,
            dtype=torch.float32, device=device)
    else:
        input_embeds = None
        cached_draft_input_embeds = None

    # Output buffer: pre-fill with NaN sentinel
    draft_input_hidden_states = torch.full((num_tokens, hidden_size), _SENTINEL,
                                           dtype=torch.float32, device=device)

    return {
        "num_reqs": num_reqs,
        "max_num_reqs": max_num_reqs,
        "num_tokens": num_tokens,
        "hidden_size": hidden_size,
        "num_speculative_steps": num_speculative_steps,
        "idx_mapping": idx_mapping,
        "target_hidden_states": target_hidden_states,
        "cached_target_hidden_states": cached_target_hidden_states,
        "input_embeds": input_embeds,
        "cached_draft_input_embeds": cached_draft_input_embeds,
        "num_rejected": num_rejected,
        "query_start_loc": query_start_loc,
        "draft_input_hidden_states": draft_input_hidden_states,
        "use_embeds": use_embeds,
        "block_size_q": block_size_q,
        "block_size_h": block_size_h,
    }


def _run_kernel(inputs: dict[str, Any], rt) -> dict[str, torch.Tensor]:
    hs = inputs["hidden_size"]
    nss = inputs["num_speculative_steps"]
    use_embeds = inputs["use_embeds"]

    # Compute max_query_len for grid
    qsl = inputs["query_start_loc"].cpu()
    max_query_len = 0
    for r in range(inputs["num_reqs"]):
        ql = int(qsl[r + 1].item()) - int(qsl[r].item())
        max_query_len = max(max_query_len, ql)

    import triton
    grid = (
        inputs["num_reqs"],
        triton.cdiv(max_query_len, inputs["block_size_q"]),
        triton.cdiv(hs, inputs["block_size_h"]),
    )

    dihs = inputs["draft_input_hidden_states"].clone()
    iem = inputs["input_embeds"].clone() if inputs["input_embeds"] is not None else None
    cths = inputs["cached_target_hidden_states"]
    cdie = inputs["cached_draft_input_embeds"]

    kernel[grid](
        dihs,
        dihs.stride(0),
        inputs["target_hidden_states"],
        inputs["target_hidden_states"].stride(0),
        cths,
        cths.stride(0) if cths is not None else 0,
        cths.stride(1) if cths is not None else 0,
        iem,
        iem.stride(0) if iem is not None else 0,
        cdie,
        cdie.stride(0) if cdie is not None else 0,
        cdie.stride(1) if cdie is not None else 0,
        inputs["idx_mapping"],
        inputs["num_rejected"],
        inputs["query_start_loc"],
        nss,
        hs,
        BLOCK_SIZE_Q=inputs["block_size_q"],
        BLOCK_SIZE_H=inputs["block_size_h"],
        USE_INPUT_EMBEDS=use_embeds,
        num_warps=1,
    )
    rt.synchronize()
    result = {"draft_input_hidden_states": dihs}
    if iem is not None:
        result["input_embeds"] = iem
    return result


def _assert_bitwise(name: str, expected: torch.Tensor, actual: torch.Tensor) -> None:
    exp_cpu = expected.cpu().contiguous()
    act_cpu = actual.cpu().contiguous()
    if exp_cpu.dtype != act_cpu.dtype or exp_cpu.shape != act_cpu.shape:
        raise AssertionError(
            f"[{name}] shape/dtype mismatch: expected {tuple(exp_cpu.shape)} "
            f"{exp_cpu.dtype} vs actual {tuple(act_cpu.shape)} {act_cpu.dtype}"
        )
    if exp_cpu.is_floating_point():
        # Compare raw bits so NaN sentinels compare equal. torch.equal treats
        # NaN != NaN, which falsely reports bit-identical NaN padding slots.
        view_dtype = {
            torch.float16: torch.int16,
            torch.bfloat16: torch.int16,
            torch.float32: torch.int32,
            torch.float64: torch.int64,
        }[exp_cpu.dtype]
        eq = exp_cpu.view(view_dtype) == act_cpu.view(view_dtype)
    else:
        eq = exp_cpu == act_cpu
    if not bool(torch.all(eq)):
        diffs = (~eq).nonzero()
        max_show = min(10, diffs.shape[0])
        detail = []
        for i in range(max_show):
            idx = tuple(diffs[i].tolist())
            detail.append(
                f"  idx={idx} expected={exp_cpu[idx].item()} "
                f"actual={act_cpu[idx].item()}"
            )
        raise AssertionError(
            f"[{name}] bitwise mismatch: {diffs.shape[0]} elements differ. "
            f"First {max_show}:\n" + "\n".join(detail)
        )


@pytest.mark.skipif(kernel is None, reason="kernel import failed")
@pytest.mark.parametrize(
    "num_reqs,hidden_size,num_speculative_steps,scenario,use_embeds,bq,bh",
    [
        # Basic, no rejections, no embeds
        (1, 16, 2, "no_reject", False, 4, 8),
        (2, 32, 3, "no_reject", False, 4, 8),
        # With rejections (re-prefill gap from cache)
        (1, 16, 2, "with_reject", False, 4, 8),
        (2, 32, 3, "with_reject", False, 4, 8),
        # With embeds (MM model)
        (1, 16, 2, "no_reject", True, 4, 8),
        (2, 32, 3, "with_reject", True, 4, 8),
        # Tile boundary: num_input_hs = block_size_q
        (1, 16, 2, "tile_boundary", False, 4, 8),
        (2, 32, 3, "tile_boundary", True, 4, 8),
        # Max re-prefill (full gap = num_spec_steps - 1)
        (1, 16, 3, "max_reprefill", False, 4, 8),
        (2, 32, 4, "max_reprefill", True, 4, 8),
        # Hidden_size tile boundary
        (1, 8, 2, "with_reject", False, 4, 8),  # H = BH exactly
        (1, 9, 2, "with_reject", False, 4, 8),  # H = BH + 1
        (1, 7, 2, "with_reject", False, 4, 8),  # H = BH - 1
        # Mixed scenarios
        (4, 64, 3, "mixed", False, 4, 16),
        (4, 64, 3, "mixed", True, 4, 16),
        # Larger num_speculative_steps
        (2, 32, 5, "with_reject", True, 4, 8),
        # Realistic block sizes
        (2, 256, 3, "with_reject", True, 16, 256),
        # --- strict_ut_027 high-spec additions ---
        (32, 2048, 3, "with_reject", True, 16, 256),   # large batch
        (64, 512, 3, "no_reject", False, 16, 64),      # max batch
        (16, 5120, 3, "with_reject", True, 16, 128),   # production hidden (Llama-class)
        (8, 7168, 4, "with_reject", True, 32, 256),    # large hidden (Kimi/Qwen-class)
        (32, 1024, 5, "mixed", True, 16, 128),         # large batch + max steps
        (16, 2048, 4, "max_reprefill", True, 16, 256), # full gap at scale
        (16, 2048, 3, "tile_boundary", True, 32, 256), # boundary at larger bq
        (8, 5120, 5, "mixed", True, 32, 512),          # production hidden + max steps
    ],
    ids=lambda v: str(v),
)
def test_prepare_input_hidden_states_and_embeddings(
    num_reqs: int,
    hidden_size: int,
    num_speculative_steps: int,
    scenario: str,
    use_embeds: bool,
    bq: int,
    bh: int,
    rt,
) -> None:
    """Bitwise comparison: pure copy/gather, no float arithmetic."""
    if _import_error is not None:
        pytest.skip(f"kernel import failed: {_import_error}")
    rt.init_device_properties_triton()

    max_num_reqs = max(8, num_reqs * 2)
    # query lens go up to ~12 (mixed/with_reject) or nss+3 (max_reprefill);
    # size the token buffer so every request fits.
    num_tokens = max(256, num_reqs * (num_speculative_steps + 16))

    inputs = _gen_inputs(
        num_reqs=num_reqs,
        max_num_reqs=max_num_reqs,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        num_speculative_steps=num_speculative_steps,
        scenario=scenario,
        use_embeds=use_embeds,
        device=rt.STRICT_DEVICE,
        block_size_q=bq,
        block_size_h=bh,
    )

    expected = _ref(
        num_reqs=inputs["num_reqs"],
        num_speculative_steps=inputs["num_speculative_steps"],
        hidden_size=inputs["hidden_size"],
        idx_mapping=inputs["idx_mapping"],
        target_hidden_states=inputs["target_hidden_states"],
        cached_target_hidden_states=inputs["cached_target_hidden_states"],
        input_embeds=inputs["input_embeds"],
        cached_draft_input_embeds=inputs["cached_draft_input_embeds"],
        num_rejected=inputs["num_rejected"],
        query_start_loc=inputs["query_start_loc"],
        draft_input_hidden_states=inputs["draft_input_hidden_states"],
    )

    actual = _run_kernel(inputs, rt)

    _assert_bitwise("draft_input_hidden_states",
                    expected["draft_input_hidden_states"],
                    actual["draft_input_hidden_states"])
    if "input_embeds" in expected:
        _assert_bitwise("input_embeds", expected["input_embeds"],
                        actual["input_embeds"])


def test_import_error() -> None:
    if _import_error is not None:
        pytest.fail(
            f"Failed to import _prepare_input_hidden_states_and_embeddings_kernel:\n"
            f"{_import_traceback}"
        )
