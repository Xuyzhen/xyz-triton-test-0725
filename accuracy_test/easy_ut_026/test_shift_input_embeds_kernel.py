# SPDX-License-Identifier: Apache-2.0
# easy_ut_026 strict UT for _shift_input_embeds_kernel.
# Source: vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py
# (upstream, no vllm-ascend patch). Category: non-compute + float copy/shift
# fusion (left-shift input_embeds by 1 within each request's query window,
# then insert the draft embed at the last position). No arithmetic on float
# data -> bitwise comparison is appropriate (fp32 values are copied verbatim).
#
# Kernel summary (grid = (num_reqs, cdiv(H, BH))):
#   Per (req_idx, block_idx):
#     req_state_idx = idx_mapping[req_idx]; skip if < 0.
#     query_start = query_start_loc[req_idx]
#     last_token_index = last_token_indices[req_idx]
#     query_len = last_token_index - query_start + 1
#     For i in range(1, query_len, BLOCK_SIZE_Q):
#       query_block = i + arange(0, BLOCK_SIZE_Q)
#       query_mask = query_block < query_len
#       mask = query_mask[:, None] & dim_mask[None, :]
#       input_embed = input_embeds[query_start + query_block, dim_block]
#         (masked load)
#       input_embeds[query_start + query_block - 1, dim_block] = input_embed
#         (masked store)
#     draft_embed = draft_embeds[req_idx, dim_block]  (masked load)
#     input_embeds[last_token_index, dim_block] = draft_embed  (masked store)

from accuracy_test.easy_ut_026.runtime_npu import (
    STRICT_DEVICE as _STRICT_DEVICE,
)
from accuracy_test.easy_ut_026.runtime_npu import (
    init_device_properties_triton,
    synchronize,
)

import traceback
from typing import Any

import pytest
import torch

kernel = None
_import_error: Exception | None = None
_import_traceback: str | None = None
try:
    from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (
        _shift_input_embeds_kernel as kernel,
    )
except Exception as exc:  # pragma: no cover
    _import_error = exc
    _import_traceback = traceback.format_exc()


_SENTINEL = float("nan")


def _ref(
    num_reqs: int,
    hidden_size: int,
    input_embeds: torch.Tensor,       # [num_tokens, H] float32
    draft_embeds: torch.Tensor,       # [num_reqs, H] float32
    idx_mapping: torch.Tensor,        # [num_reqs] int64
    query_start_loc: torch.Tensor,    # [num_reqs+1] int32
    last_token_indices: torch.Tensor,  # [num_reqs] int64
) -> torch.Tensor:
    """Independent CPU reference. In-place left-shift by 1 within each
    request's query window, then insert draft_embeds[req_idx] at the last
    token index. Reads from the original input_embeds buffer (not the
    in-place modified one) to avoid data hazards."""
    out = input_embeds.cpu().clone()
    ie = input_embeds.cpu()
    de = draft_embeds.cpu()
    im = idx_mapping.cpu().to(torch.int64).tolist()
    qsl = query_start_loc.cpu().to(torch.int64).tolist()
    lti = last_token_indices.cpu().to(torch.int64).tolist()

    for req_idx in range(num_reqs):
        rsi = im[req_idx]
        if rsi < 0:
            continue
        qs = qsl[req_idx]
        last_idx = lti[req_idx]
        query_len = last_idx - qs + 1
        # Left-shift by 1: new[j] = old[j+1] for j in [0, query_len-2]
        for j in range(0, query_len - 1):
            out[qs + j] = ie[qs + j + 1]
        # Insert draft embed at the last position
        out[last_idx] = de[req_idx]

    return out


def _gen_inputs(
    num_reqs: int,
    max_num_reqs: int,
    num_tokens: int,
    hidden_size: int,
    *,
    scenario: str,
    device: str,
    block_size_q: int = 4,
    block_size_h: int = 8,
) -> dict[str, Any]:
    torch.manual_seed(42 + num_reqs * 7 + hidden_size)

    if scenario == "skip_padded":
        idx_mapping = torch.arange(num_reqs, dtype=torch.int64,
                                   device=device)
        idx_mapping[-1] = -1
    else:
        idx_mapping = torch.arange(num_reqs, dtype=torch.int64,
                                   device=device)

    # Build query_start_loc and last_token_indices
    if scenario == "single_token":
        query_lens = torch.ones(num_reqs, dtype=torch.int32,
                                device=device)
    elif scenario == "two_tokens":
        query_lens = torch.full((num_reqs,), 2, dtype=torch.int32,
                               device=device)
    elif scenario == "tile_boundary_q":
        query_lens = torch.full((num_reqs,), block_size_q, dtype=torch.int32,
                               device=device)
    elif scenario == "tile_boundary_q_plus_one":
        query_lens = torch.full((num_reqs,), block_size_q + 1, dtype=torch.int32,
                               device=device)
    else:  # random
        query_lens = torch.randint(1, 16, (num_reqs,), dtype=torch.int32,
                                   device=device)

    query_start_loc = torch.zeros(max_num_reqs + 1, dtype=torch.int32,
                                  device=device)
    qsl_cpu = query_start_loc.cpu()
    qsl_cpu[0] = 0
    for r in range(num_reqs):
        qsl_cpu[r + 1] = qsl_cpu[r] + int(query_lens[r].item())
    qsl_cpu[num_reqs + 1:] = -1
    query_start_loc.copy_(qsl_cpu)

    last_token_indices = (query_start_loc[1:num_reqs + 1] - 1).to(torch.int64)

    # input_embeds: pre-fill with NaN sentinel, fill in-request windows
    # with random fp32
    input_embeds = torch.full((num_tokens, hidden_size), _SENTINEL,
                              dtype=torch.float32, device=device)
    ie_cpu = input_embeds.cpu()
    for r in range(num_reqs):
        qs = int(qsl_cpu[r].item())
        qe = int(qsl_cpu[r + 1].item())
        ie_cpu[qs:qe] = torch.randn(qe - qs, hidden_size, dtype=torch.float32)
    input_embeds.copy_(ie_cpu)

    # draft_embeds: random, shape [num_reqs, H]
    draft_embeds = torch.randn(num_reqs, hidden_size, dtype=torch.float32,
                                device=device)

    return {
        "num_reqs": num_reqs,
        "max_num_reqs": max_num_reqs,
        "num_tokens": num_tokens,
        "hidden_size": hidden_size,
        "input_embeds": input_embeds,
        "draft_embeds": draft_embeds,
        "idx_mapping": idx_mapping,
        "query_start_loc": query_start_loc,
        "last_token_indices": last_token_indices,
        "block_size_q": block_size_q,
        "block_size_h": block_size_h,
    }


def _run_kernel(inputs: dict[str, Any], device: str) -> torch.Tensor:
    import triton
    hs = inputs["hidden_size"]
    ie = inputs["input_embeds"].clone()
    de = inputs["draft_embeds"]

    grid = (inputs["num_reqs"], triton.cdiv(hs, inputs["block_size_h"]))
    kernel[grid](
        ie,
        ie.stride(0),
        de,
        de.stride(0),
        inputs["idx_mapping"],
        inputs["query_start_loc"],
        inputs["last_token_indices"],
        hs,
        BLOCK_SIZE_Q=inputs["block_size_q"],
        BLOCK_SIZE_H=inputs["block_size_h"],
        num_warps=1,
    )
    synchronize()
    return ie


def _assert_bitwise(name: str, expected: torch.Tensor,
                    actual: torch.Tensor) -> None:
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


@pytest.mark.skipif(kernel is None, reason="kernel import failed")
@pytest.mark.parametrize(
    "num_reqs,hidden_size,scenario,bq,bh",
    [
        # Single token query (no shift, only insert)
        (1, 16, "single_token", 4, 8),
        (2, 32, "single_token", 4, 8),
        # Two tokens (shift 1, insert draft)
        (1, 16, "two_tokens", 4, 8),
        (2, 32, "two_tokens", 4, 8),
        # Tile boundary (query_len == BQ)
        (1, 16, "tile_boundary_q", 4, 8),
        (2, 32, "tile_boundary_q", 4, 8),
        # Tile boundary + 1 (partial tail block)
        (1, 16, "tile_boundary_q_plus_one", 4, 8),
        (2, 32, "tile_boundary_q_plus_one", 4, 8),
        # Hidden_size tile boundary
        (1, 8, "random", 4, 8),   # H = BH
        (1, 9, "random", 4, 8),  # H = BH + 1
        (1, 7, "random", 4, 8),  # H = BH - 1
        # Random query lengths
        (4, 64, "random", 4, 8),
        (8, 64, "random", 4, 8),
        # Skip cudagraph padded requests (idx_mapping < 0)
        (4, 32, "skip_padded", 4, 8),
        # Larger batch
        (16, 64, "random", 4, 8),
        # Different block sizes
        (4, 64, "random", 8, 16),
        (4, 64, "random", 2, 4),
        # Realistic block sizes
        (4, 256, "random", 16, 256),
    ],
    ids=lambda v: str(v),
)
def test_shift_input_embeds(
    num_reqs: int,
    hidden_size: int,
    scenario: str,
    bq: int,
    bh: int,
) -> None:
    """Bitwise comparison: pure left-shift + insert of fp32 copies."""
    if _import_error is not None:
        pytest.skip(f"kernel import failed: {_import_error}")
    init_device_properties_triton()

    max_num_reqs = max(8, num_reqs * 2)
    num_tokens = 256

    inputs = _gen_inputs(
        num_reqs=num_reqs,
        max_num_reqs=max_num_reqs,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        scenario=scenario,
        device=_STRICT_DEVICE,
        block_size_q=bq,
        block_size_h=bh,
    )

    expected = _ref(
        num_reqs=inputs["num_reqs"],
        hidden_size=inputs["hidden_size"],
        input_embeds=inputs["input_embeds"],
        draft_embeds=inputs["draft_embeds"],
        idx_mapping=inputs["idx_mapping"],
        query_start_loc=inputs["query_start_loc"],
        last_token_indices=inputs["last_token_indices"],
    )

    actual = _run_kernel(inputs, _STRICT_DEVICE)
    _assert_bitwise("input_embeds", expected, actual)


def test_import_error() -> None:
    if _import_error is not None:
        pytest.fail(
            f"Failed to import _shift_input_embeds_kernel:\n"
            f"{_import_traceback}"
        )
