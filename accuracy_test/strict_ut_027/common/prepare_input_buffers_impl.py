# SPDX-License-Identifier: Apache-2.0
# strict_ut_027 shared dual-side UT for _prepare_input_buffers_kernel.
# Source: vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py
# (upstream, no vllm-ascend patch). Category: non-compute + integer index
# manipulation (input-id / position shifts, cache re-prefill, padding).
# All outputs are integer -> bitwise comparison.
#
# Dual-side design: this module holds the full test implementation and is
# re-exported by gpu/test_prepare_input_buffers_kernel.py and
# npu/test_prepare_input_buffers_kernel.py. The side runtime (CUDA or
# Ascend NPU) is injected through the ``rt`` pytest fixture.
#
# Kernel summary (per req_idx in [0, num_reqs)):
#   1. Read query_start/query_end, compute query_len, num_rejected,
#      num_reprefill_tokens = max(0, num_rejected - 1),
#      num_input_tokens = query_len - num_rejected.
#   2. Decrement seq_len by num_reprefill_tokens; store draft_seq_lens[req_idx].
#   3. next_token = last_sampled[req_state_idx].to(int32) if num_sampled>0
#      else next_prefill_tokens[0, req_state_idx].
#   4. For i in 1..num_speculative_steps-1: write override
#      (=-1 if num_sampled>0 else future_prefill_token) to
#      draft_input_id_overrides[req_idx, i-1].
#   5. Copy target_input_ids[query_start+1 .. query_start+num_input_tokens-1]
#      to draft_input_ids[query_start+num_reprefill_tokens-1+1 ..
#      query_start+num_reprefill_tokens-1+num_input_tokens-1].
#   6. last_token_index = query_start + num_reprefill_tokens +
#      num_input_tokens - 1; store to last_token_indices[req_idx];
#      draft_input_ids[last_token_index] = next_token.
#   7. Copy target_positions[query_start .. query_start+num_input_tokens-1]
#      to draft_positions[query_start+num_reprefill_tokens ..
#      query_start+num_reprefill_tokens+num_input_tokens-1].
#   8. Fill re-prefill gap: for i in 0..num_reprefill_tokens-1,
#      draft_input_ids[query_start+i] = cached_draft_input_ids[
#      req_state_idx, num_speculative_steps-1-num_reprefill_tokens+i];
#      draft_positions[query_start+i] = first_position -
#      num_reprefill_tokens + i.
#   9. If req_idx == num_reqs-1: pad query_start_loc[num_reqs..max_num_reqs+1)
#      with query_end; pad draft_seq_lens[num_reqs..max_num_reqs) with 0;
#      pad last_token_indices[num_reqs..max_num_reqs) with 0.

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
        _prepare_input_buffers_kernel as kernel,
    )
except Exception as exc:  # pragma: no cover
    _import_error = exc
    _import_traceback = traceback.format_exc()


_SENTINEL_ID = -12345  # sentinel for int32/int64 slots (never a legal token id)
_SENTINEL_POS = -99999  # sentinel for positions
_SENTINEL_LEN = -1  # sentinel for seq_lens / last_token_indices padding


def _ref(
    num_reqs: int,
    max_num_reqs: int,
    num_speculative_steps: int,
    # Read-only inputs
    idx_mapping: torch.Tensor,        # [num_reqs] int64
    target_input_ids: torch.Tensor,   # [max_num_tokens] int32
    target_positions: torch.Tensor,   # [max_num_tokens] int64
    target_seq_lens: torch.Tensor,    # [num_reqs] int32
    cached_draft_input_ids: torch.Tensor,  # [max_num_reqs, num_spec_steps-1] int64
    last_sampled: torch.Tensor,       # [max_num_reqs] int64
    next_prefill_tokens: torch.Tensor,  # [num_prefill_lookahead, max_num_reqs] int32
    num_sampled: torch.Tensor,        # [num_reqs] int32
    num_rejected: torch.Tensor,       # [num_reqs] int32
    query_start_loc_in: torch.Tensor,  # [num_reqs+1] int32 (read-only portion)
    # Output buffers (pre-filled with sentinels)
    draft_input_ids: torch.Tensor,    # [max_num_tokens] int32
    draft_positions: torch.Tensor,    # [max_num_tokens] int64
    draft_seq_lens: torch.Tensor,     # [max_num_reqs] int32
    last_token_indices: torch.Tensor,  # [max_num_reqs] int64
    draft_input_id_overrides: torch.Tensor,  # [max_num_reqs, num_spec_steps-1] int64
    query_start_loc_out: torch.Tensor,  # [max_num_reqs+1] int32
    max_num_tokens: int,
) -> dict[str, torch.Tensor]:
    """Independent CPU reference. Mirrors the kernel's per-request logic and
    the trailing padding done by the last request. Returns a dict of expected
    output tensors (cloned from the input buffers)."""
    # Work on CPU clones so the caller's tensors are untouched.
    did = draft_input_ids.cpu().clone()
    dpos = draft_positions.cpu().clone()
    dsl = draft_seq_lens.cpu().clone()
    lti = last_token_indices.cpu().clone()
    dio = draft_input_id_overrides.cpu().clone()
    qsl = query_start_loc_out.cpu().clone()

    im = idx_mapping.cpu().to(torch.int64).tolist()
    tid = target_input_ids.cpu().to(torch.int64).tolist()
    tpos = target_positions.cpu().to(torch.int64).tolist()
    tsl = target_seq_lens.cpu().to(torch.int64).tolist()
    cdi = cached_draft_input_ids.cpu().to(torch.int64).tolist()
    ls = last_sampled.cpu().to(torch.int64).tolist()
    npt = next_prefill_tokens.cpu().to(torch.int64).tolist()
    ns = num_sampled.cpu().to(torch.int64).tolist()
    nr = num_rejected.cpu().to(torch.int64).tolist()
    qsl_in = query_start_loc_in.cpu().to(torch.int64).tolist()

    npt_stride0 = next_prefill_tokens.shape[1] if next_prefill_tokens.dim() > 1 else max_num_reqs
    # Actually next_prefill_tokens is [num_prefill_lookahead, max_num_reqs]
    # so stride(0) = max_num_reqs. We index as npt[i][req_state_idx].
    npt_rows = next_prefill_tokens.shape[0]

    for req_idx in range(num_reqs):
        req_state_idx = im[req_idx]
        query_start = qsl_in[req_idx]
        query_end = qsl_in[req_idx + 1]
        query_len = query_end - query_start
        seq_len = tsl[req_idx]
        num_rej = nr[req_idx]
        num_reprefill = max(0, num_rej - 1)
        num_input_tokens = query_len - num_rej
        seq_len -= num_reprefill
        dsl[req_idx] = seq_len

        num_samp = ns[req_idx]
        if num_samp > 0:
            next_token = int(ls[req_state_idx]) & 0xFFFFFFFF
            # Truncate to int32 (matching kernel's .to(tl.int32))
            if next_token >= 0x80000000:
                next_token -= 0x100000000
        else:
            next_token = int(npt[0][req_state_idx])

        # Write overrides for draft steps 1..num_speculative_steps-1
        for i in range(1, num_speculative_steps):
            if i < npt_rows:
                future_token = int(npt[i][req_state_idx])
            else:
                future_token = 0
            override = -1 if num_samp > 0 else future_token
            dio[req_idx][i - 1] = override

        # Copy target_input_ids[query_start+1 .. query_start+num_input_tokens-1]
        # to draft_input_ids[query_start+num_reprefill-1+1 ..
        #                   query_start+num_reprefill-1+num_input_tokens-1]
        for b in range(1, num_input_tokens):
            src = query_start + b
            dst = query_start + num_reprefill - 1 + b
            did[dst] = tid[src]

        last_token_index = query_start + num_reprefill + num_input_tokens - 1
        lti[req_idx] = last_token_index
        did[last_token_index] = next_token

        # Copy target_positions[query_start .. query_start+num_input_tokens-1]
        # to draft_positions[query_start+num_reprefill ..
        #                    query_start+num_reprefill+num_input_tokens-1]
        for b in range(0, num_input_tokens):
            src = query_start + b
            dst = query_start + num_reprefill + b
            dpos[dst] = tpos[src]

        # Fill re-prefill gap with cached tokens
        first_position = tpos[query_start]
        for i in range(num_reprefill):
            cache_read_slot = num_speculative_steps - 1 - num_reprefill + i
            cached_token_id = int(cdi[req_state_idx][cache_read_slot])
            # Truncate to int32 (kernel stores int64 -> int32 buffer)
            cached_token_id_i32 = cached_token_id & 0xFFFFFFFF
            if cached_token_id_i32 >= 0x80000000:
                cached_token_id_i32 -= 0x100000000
            did[query_start + i] = cached_token_id_i32
            dpos[query_start + i] = first_position - num_reprefill + i

    # Padding by last request
    if num_reqs > 0:
        last_query_end = qsl_in[num_reqs]  # query_end of last req
        for i in range(num_reqs, max_num_reqs + 1):
            qsl[i] = last_query_end
        for i in range(num_reqs, max_num_reqs):
            dsl[i] = 0
            lti[i] = 0

    return {
        "draft_input_ids": did.to(torch.int32),
        "draft_positions": dpos.to(torch.int64),
        "draft_seq_lens": dsl.to(torch.int32),
        "last_token_indices": lti.to(torch.int64),
        "draft_input_id_overrides": dio.to(torch.int64),
        "query_start_loc": qsl.to(torch.int32),
    }


def _gen_inputs(
    num_reqs: int,
    max_num_reqs: int,
    max_num_tokens: int,
    num_speculative_steps: int,
    *,
    scenario: str,
    device,
    block_size: int = 8,
) -> dict[str, Any]:
    """Build inputs for a given scenario.

    Scenarios:
      - "decode_no_reject": num_sampled>0, num_rejected=0 (no re-prefill)
      - "decode_with_reject": num_sampled>0, num_rejected>0 (re-prefill from cache)
      - "chunked_prefill": num_sampled=0 (seed with next_prefill_tokens)
      - "mixed": some reqs decode, some chunked-prefill
      - "tile_boundary": num_input_tokens at block_size boundary
      - "minimal_input": num_input_tokens=1 (only next_token)
      - "reject_one": num_rejected=1 (num_reprefill_tokens=0)
    """
    torch.manual_seed(42 + num_reqs * 7 + num_speculative_steps)

    # idx_mapping: identity for simplicity
    idx_mapping = torch.arange(num_reqs, dtype=torch.int64, device=device)

    # query_start_loc: contiguous allocation, random query_lens
    if scenario == "tile_boundary":
        # Each query_len = block_size + num_rejected so num_input_tokens = block_size
        query_lens = torch.full((num_reqs,), block_size + 2, dtype=torch.int32,
                                device=device)
        num_rejected_vals = [2] * num_reqs
    elif scenario == "minimal_input":
        # query_len = 1, num_rejected = 0 -> num_input_tokens = 1
        query_lens = torch.ones(num_reqs, dtype=torch.int32, device=device)
        num_rejected_vals = [0] * num_reqs
    elif scenario == "reject_one":
        # num_rejected = 1 -> num_reprefill_tokens = 0
        query_lens = torch.full((num_reqs,), 4, dtype=torch.int32, device=device)
        num_rejected_vals = [1] * num_reqs
    elif scenario == "decode_no_reject":
        query_lens = torch.randint(3, 10, (num_reqs,), dtype=torch.int32,
                                   device=device)
        num_rejected_vals = [0] * num_reqs
    elif scenario == "decode_with_reject":
        query_lens = torch.randint(5, 12, (num_reqs,), dtype=torch.int32,
                                   device=device)
        num_rejected_vals = [min(int(torch.randint(1, 4, ()).item()),
                                  int(query_lens[r].item()) - 1,
                                  num_speculative_steps)
                             for r in range(num_reqs)]
    elif scenario == "chunked_prefill":
        query_lens = torch.randint(3, 10, (num_reqs,), dtype=torch.int32,
                                   device=device)
        num_rejected_vals = [0] * num_reqs
    else:  # mixed
        query_lens = torch.randint(4, 12, (num_reqs,), dtype=torch.int32,
                                   device=device)
        num_rejected_vals = []
        for r in range(num_reqs):
            ql = int(query_lens[r].item())
            if r % 2 == 0:
                nr = min(int(torch.randint(0, max(1, ql // 2), ()).item()),
                         ql - 1, num_speculative_steps)
            else:
                nr = 0
            num_rejected_vals.append(nr)

    query_start_loc = torch.zeros(max_num_reqs + 1, dtype=torch.int32,
                                  device=device)
    qsl_cpu = query_start_loc.cpu()
    qsl_cpu[0] = 0
    for r in range(num_reqs):
        qsl_cpu[r + 1] = qsl_cpu[r] + int(query_lens[r].item())
    # Fill remaining with sentinel (will be overwritten by padding)
    qsl_cpu[num_reqs + 1:] = _SENTINEL_LEN
    query_start_loc.copy_(qsl_cpu)

    # num_sampled: > 0 for decode, == 0 for chunked prefill
    num_sampled = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    for r in range(num_reqs):
        if scenario == "chunked_prefill":
            num_sampled[r] = 0
        elif scenario == "mixed" and r % 3 == 0:
            num_sampled[r] = 0  # some chunked-prefill
        else:
            num_sampled[r] = int(torch.randint(1, num_speculative_steps + 1,
                                               ()).item())

    num_rejected = torch.tensor(num_rejected_vals, dtype=torch.int32,
                                device=device)

    # target_input_ids: random token ids in [0, 1000)
    target_input_ids = torch.randint(0, 1000, (max_num_tokens,),
                                     dtype=torch.int32, device=device)
    # target_positions: contiguous positions per request
    target_positions = torch.zeros(max_num_tokens, dtype=torch.int64,
                                    device=device)
    tp_cpu = target_positions.cpu()
    for r in range(num_reqs):
        qs = int(qsl_cpu[r].item())
        qe = int(qsl_cpu[r + 1].item())
        base = r * 100  # arbitrary base position per request
        for j in range(qe - qs):
            tp_cpu[qs + j] = base + j
    target_positions.copy_(tp_cpu)

    # target_seq_lens: arbitrary, > query_len
    target_seq_lens = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    tsl_cpu = target_seq_lens.cpu()
    for r in range(num_reqs):
        ql = int(query_lens[r].item())
        tsl_cpu[r] = ql + int(torch.randint(10, 50, ()).item())
    target_seq_lens.copy_(tsl_cpu)

    # cached_draft_input_ids: random, shape [max_num_reqs, num_spec_steps-1]
    cached_draft_input_ids = torch.randint(0, 1000,
                                           (max_num_reqs,
                                            max(1, num_speculative_steps - 1)),
                                           dtype=torch.int64, device=device)

    # last_sampled: random, shape [max_num_reqs]
    last_sampled = torch.randint(0, 1000, (max_num_reqs,), dtype=torch.int64,
                                 device=device)

    # next_prefill_tokens: [num_prefill_lookahead, max_num_reqs]
    # num_prefill_lookahead = num_speculative_steps
    next_prefill_tokens = torch.randint(0, 1000,
                                        (num_speculative_steps, max_num_reqs),
                                        dtype=torch.int32, device=device)

    # Output buffers: pre-fill with sentinels
    draft_input_ids = torch.full((max_num_tokens,), _SENTINEL_ID,
                                 dtype=torch.int32, device=device)
    draft_positions = torch.full((max_num_tokens,), _SENTINEL_POS,
                                 dtype=torch.int64, device=device)
    draft_seq_lens = torch.full((max_num_reqs,), _SENTINEL_LEN,
                                dtype=torch.int32, device=device)
    last_token_indices = torch.full((max_num_reqs,), _SENTINEL_LEN,
                                    dtype=torch.int64, device=device)
    draft_input_id_overrides = torch.full((max_num_reqs,
                                           max(1, num_speculative_steps - 1)),
                                          _SENTINEL_ID, dtype=torch.int64,
                                          device=device)

    return {
        "num_reqs": num_reqs,
        "max_num_reqs": max_num_reqs,
        "max_num_tokens": max_num_tokens,
        "num_speculative_steps": num_speculative_steps,
        "idx_mapping": idx_mapping,
        "target_input_ids": target_input_ids,
        "target_positions": target_positions,
        "target_seq_lens": target_seq_lens,
        "cached_draft_input_ids": cached_draft_input_ids,
        "last_sampled": last_sampled,
        "next_prefill_tokens": next_prefill_tokens,
        "num_sampled": num_sampled,
        "num_rejected": num_rejected,
        "query_start_loc_in": query_start_loc,
        "draft_input_ids": draft_input_ids,
        "draft_positions": draft_positions,
        "draft_seq_lens": draft_seq_lens,
        "last_token_indices": last_token_indices,
        "draft_input_id_overrides": draft_input_id_overrides,
        "query_start_loc_out": query_start_loc.clone(),
        "block_size": block_size,
    }


def _run_kernel(inputs: dict[str, Any], rt) -> dict[str, torch.Tensor]:
    """Run the kernel on the test device and return the output tensors."""
    nss = inputs["num_speculative_steps"]
    # Clone output buffers so we can run the kernel and keep the originals
    did = inputs["draft_input_ids"].clone()
    dpos = inputs["draft_positions"].clone()
    dsl = inputs["draft_seq_lens"].clone()
    lti = inputs["last_token_indices"].clone()
    dio = inputs["draft_input_id_overrides"].clone()
    qsl = inputs["query_start_loc_out"].clone()

    kernel[(inputs["num_reqs"],)](
        lti,
        did,
        dpos,
        dsl,
        inputs["target_input_ids"],
        inputs["target_positions"],
        inputs["cached_draft_input_ids"],
        inputs["cached_draft_input_ids"].stride(0),
        dio,
        dio.stride(0),
        inputs["idx_mapping"],
        inputs["last_sampled"],
        inputs["next_prefill_tokens"],
        inputs["next_prefill_tokens"].stride(0),
        inputs["num_sampled"],
        inputs["num_rejected"],
        inputs["target_seq_lens"],
        qsl,
        inputs["max_num_reqs"],
        nss,
        BLOCK_SIZE=inputs["block_size"],
        num_warps=1,
    )
    rt.synchronize()
    return {
        "draft_input_ids": did,
        "draft_positions": dpos,
        "draft_seq_lens": dsl,
        "last_token_indices": lti,
        "draft_input_id_overrides": dio,
        "query_start_loc": qsl,
    }


def _assert_bitwise(name: str, expected: torch.Tensor,
                    actual: torch.Tensor) -> None:
    """Assert bitwise equality (raw bits for floats so NaN sentinels match)."""
    exp_cpu = expected.cpu().contiguous()
    act_cpu = actual.cpu().contiguous()
    if exp_cpu.dtype != act_cpu.dtype or exp_cpu.shape != act_cpu.shape:
        raise AssertionError(
            f"[{name}] shape/dtype mismatch: expected {tuple(exp_cpu.shape)} "
            f"{exp_cpu.dtype} vs actual {tuple(act_cpu.shape)} {act_cpu.dtype}"
        )
    if exp_cpu.is_floating_point():
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


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@pytest.mark.skipif(kernel is None, reason="kernel import failed")
@pytest.mark.parametrize(
    "num_reqs,max_num_reqs,max_num_tokens,num_speculative_steps,scenario,block_size",
    [
        # Basic decode, no rejections
        (1, 4, 64, 2, "decode_no_reject", 8),
        (1, 4, 64, 3, "decode_no_reject", 8),
        # Decode with rejections (re-prefill from cache)
        (1, 4, 64, 2, "decode_with_reject", 8),
        (2, 8, 128, 3, "decode_with_reject", 8),
        (4, 8, 256, 4, "decode_with_reject", 8),
        # Chunked prefill (num_sampled=0, overrides = future tokens)
        (1, 4, 64, 2, "chunked_prefill", 8),
        (3, 8, 128, 4, "chunked_prefill", 8),
        # Mixed decode + chunked prefill
        (4, 8, 256, 3, "mixed", 8),
        (2, 8, 128, 2, "mixed", 4),
        # Tile boundary: num_input_tokens = block_size
        (1, 4, 64, 2, "tile_boundary", 8),
        (2, 8, 128, 3, "tile_boundary", 8),
        # Minimal input: num_input_tokens = 1 (only next_token, no copy)
        (1, 4, 16, 2, "minimal_input", 8),
        (3, 8, 64, 3, "minimal_input", 8),
        # num_rejected = 1 -> num_reprefill_tokens = 0 (no cache fill)
        (1, 4, 32, 2, "reject_one", 8),
        (2, 8, 64, 3, "reject_one", 4),
        # Larger batch with padding verification
        (1, 16, 64, 2, "decode_no_reject", 8),
        (2, 16, 128, 3, "decode_with_reject", 8),
        # Larger num_speculative_steps
        (2, 8, 128, 5, "decode_with_reject", 8),
        # Realistic BLOCK_SIZE=1024
        (2, 8, 4096, 3, "decode_no_reject", 1024),
        # --- strict_ut_027 high-spec additions ---
        (8, 32, 2048, 3, "mixed", 64),
        (16, 64, 4096, 4, "decode_with_reject", 256),
        (8, 32, 2048, 5, "mixed", 1024),
        (32, 64, 8192, 3, "decode_no_reject", 8),   # large batch
        (16, 64, 4096, 2, "chunked_prefill", 1024),
        (64, 128, 8192, 3, "mixed", 8),             # max batch
    ],
    ids=lambda v: str(v),
)
def test_prepare_input_buffers(
    num_reqs: int,
    max_num_reqs: int,
    max_num_tokens: int,
    num_speculative_steps: int,
    scenario: str,
    block_size: int,
    rt,
) -> None:
    """Bitwise comparison of device kernel output vs independent CPU reference."""
    if _import_error is not None:
        pytest.skip(f"kernel import failed: {_import_error}")
    rt.init_device_properties_triton()

    inputs = _gen_inputs(
        num_reqs=num_reqs,
        max_num_reqs=max_num_reqs,
        max_num_tokens=max_num_tokens,
        num_speculative_steps=num_speculative_steps,
        scenario=scenario,
        device=rt.STRICT_DEVICE,
        block_size=block_size,
    )

    # Compute CPU reference
    expected = _ref(
        num_reqs=inputs["num_reqs"],
        max_num_reqs=inputs["max_num_reqs"],
        num_speculative_steps=inputs["num_speculative_steps"],
        idx_mapping=inputs["idx_mapping"],
        target_input_ids=inputs["target_input_ids"],
        target_positions=inputs["target_positions"],
        target_seq_lens=inputs["target_seq_lens"],
        cached_draft_input_ids=inputs["cached_draft_input_ids"],
        last_sampled=inputs["last_sampled"],
        next_prefill_tokens=inputs["next_prefill_tokens"],
        num_sampled=inputs["num_sampled"],
        num_rejected=inputs["num_rejected"],
        query_start_loc_in=inputs["query_start_loc_in"],
        draft_input_ids=inputs["draft_input_ids"],
        draft_positions=inputs["draft_positions"],
        draft_seq_lens=inputs["draft_seq_lens"],
        last_token_indices=inputs["last_token_indices"],
        draft_input_id_overrides=inputs["draft_input_id_overrides"],
        query_start_loc_out=inputs["query_start_loc_out"],
        max_num_tokens=inputs["max_num_tokens"],
    )

    # Run kernel on the device
    actual = _run_kernel(inputs, rt)

    # Bitwise comparison for all outputs
    _assert_bitwise("draft_input_ids", expected["draft_input_ids"],
                    actual["draft_input_ids"])
    _assert_bitwise("draft_positions", expected["draft_positions"],
                    actual["draft_positions"])
    _assert_bitwise("draft_seq_lens", expected["draft_seq_lens"],
                    actual["draft_seq_lens"])
    _assert_bitwise("last_token_indices", expected["last_token_indices"],
                    actual["last_token_indices"])
    _assert_bitwise("draft_input_id_overrides",
                    expected["draft_input_id_overrides"],
                    actual["draft_input_id_overrides"])
    _assert_bitwise("query_start_loc", expected["query_start_loc"],
                    actual["query_start_loc"])


def test_import_error() -> None:
    """If the kernel import failed, surface the traceback for debugging."""
    if _import_error is not None:
        pytest.fail(
            f"Failed to import _prepare_input_buffers_kernel:\n"
            f"{_import_traceback}"
        )
