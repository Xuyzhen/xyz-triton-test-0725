# SPDX-License-Identifier: Apache-2.0
# strict_ut_027 shared dual-side UT for _shift_input_ids_kernel.
# Source: vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py
# (upstream, no vllm-ascend patch). Category: non-compute shift (left-shift
# input_ids by 1 within each request's query window, then insert the draft
# token at the last position). All values int32 -> bitwise comparison.
#
# Dual-side design: this module holds the full test implementation and is
# re-exported by gpu/test_shift_input_ids_kernel.py and
# npu/test_shift_input_ids_kernel.py. The side runtime (CUDA or Ascend NPU)
# is injected through the ``rt`` pytest fixture, which provides
# STRICT_DEVICE / init_device_properties_triton() / synchronize().
#
# Kernel summary (grid = (num_reqs,)):
#   Per req_idx:
#     req_state_idx = idx_mapping[req_idx]; skip if < 0.
#     query_start = query_start_loc[req_idx]
#     last_token_index = last_token_indices[req_idx]
#     query_len = last_token_index - query_start + 1
#     For i in range(1, query_len, BLOCK_SIZE):
#       offs = i + arange(0, BLOCK_SIZE)
#       mask = offs < query_len
#       shifted = load(input_ids[query_start + offs], mask=mask, other=0)
#       store(input_ids[query_start + offs - 1], shifted, mask=mask)
#     draft = draft_token_ids[req_idx]
#     store(input_ids[last_token_index], draft)

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
        _shift_input_ids_kernel as kernel,
    )
except Exception as exc:  # pragma: no cover
    _import_error = exc
    _import_traceback = traceback.format_exc()


_SENTINEL = -12345


def _ref(
    num_reqs: int,
    input_ids: torch.Tensor,           # [num_tokens] int32
    draft_token_ids: torch.Tensor,     # [num_reqs] int32
    idx_mapping: torch.Tensor,         # [num_reqs] int64
    query_start_loc: torch.Tensor,     # [num_reqs+1] int32
    last_token_indices: torch.Tensor,  # [num_reqs] int64
) -> torch.Tensor:
    """Independent CPU reference. In-place left-shift by 1 within each
    request's query window, then insert draft_token_ids[req_idx] at the
    last token index. Reads from the original input_ids buffer (not the
    in-place modified one) to avoid data hazards."""
    out = input_ids.cpu().clone()
    ii = input_ids.cpu()
    dt = draft_token_ids.cpu()
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
            out[qs + j] = ii[qs + j + 1]
        # Insert draft token at the last position
        out[last_idx] = dt[req_idx]

    return out


def _gen_inputs(
    num_reqs: int,
    max_num_reqs: int,
    num_tokens: int,
    *,
    scenario: str,
    device,
    block_size: int = 8,
) -> dict[str, Any]:
    torch.manual_seed(42 + num_reqs * 7 + block_size)

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
    elif scenario == "tile_boundary":
        query_lens = torch.full((num_reqs,), block_size, dtype=torch.int32,
                                device=device)
    elif scenario == "tile_boundary_plus_one":
        query_lens = torch.full((num_reqs,), block_size + 1,
                                dtype=torch.int32, device=device)
    elif scenario == "long_query":
        # Long per-request queries (64..256 tokens): exercises multi-block
        # shift loops at production-like sequence lengths.
        query_lens = torch.randint(64, 257, (num_reqs,),
                                   dtype=torch.int32, device=device)
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

    # input_ids: pre-fill with sentinel, fill in-request windows with ids
    input_ids = torch.full((num_tokens,), _SENTINEL, dtype=torch.int32,
                           device=device)
    ii_cpu = input_ids.cpu()
    for r in range(num_reqs):
        qs = int(qsl_cpu[r].item())
        qe = int(qsl_cpu[r + 1].item())
        ii_cpu[qs:qe] = torch.randint(0, 1000, (qe - qs,),
                                      dtype=torch.int32)
    input_ids.copy_(ii_cpu)

    # draft_token_ids: random, shape [num_reqs]
    draft_token_ids = torch.randint(0, 1000, (num_reqs,),
                                    dtype=torch.int32, device=device)

    return {
        "num_reqs": num_reqs,
        "max_num_reqs": max_num_reqs,
        "num_tokens": num_tokens,
        "input_ids": input_ids,
        "draft_token_ids": draft_token_ids,
        "idx_mapping": idx_mapping,
        "query_start_loc": query_start_loc,
        "last_token_indices": last_token_indices,
        "block_size": block_size,
    }


def _run_kernel(inputs: dict[str, Any], rt) -> torch.Tensor:
    ii = inputs["input_ids"].clone()

    grid = (inputs["num_reqs"],)
    kernel[grid](
        ii,
        inputs["idx_mapping"],
        inputs["query_start_loc"],
        inputs["last_token_indices"],
        inputs["draft_token_ids"],
        BLOCK_SIZE=inputs["block_size"],
        num_warps=1,
    )
    rt.synchronize()
    return ii


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
    "num_reqs,scenario,block_size",
    [
        # Single token query (no shift, only insert)
        (1, "single_token", 8),
        (2, "single_token", 8),
        # Two tokens (shift 1, insert draft)
        (1, "two_tokens", 8),
        (2, "two_tokens", 8),
        # Tile boundary (query_len == BLOCK_SIZE)
        (1, "tile_boundary", 8),
        (2, "tile_boundary", 8),
        (1, "tile_boundary", 4),
        # Tile boundary + 1 (partial tail block)
        (1, "tile_boundary_plus_one", 8),
        (2, "tile_boundary_plus_one", 8),
        # Random query lengths
        (4, "random", 8),
        (8, "random", 8),
        # Skip cudagraph padded requests (idx_mapping < 0)
        (4, "skip_padded", 8),
        # Larger batch
        (16, "random", 8),
        # Different block sizes
        (4, "random", 4),
        (4, "random", 16),
        (4, "random", 1024),
        # --- strict_ut_027 high-spec additions ---
        (32, "random", 8),               # large batch
        (64, "random", 16),              # max-batch random queries
        (8, "tile_boundary", 32),        # boundary at larger block
        (8, "tile_boundary_plus_one", 64),
        (16, "skip_padded", 16),         # padded slots at larger batch
        (32, "random", 1024),            # large batch, realistic block
        (16, "long_query", 8),           # long queries, small block
        (16, "long_query", 1024),        # long queries, realistic block
        (64, "long_query", 256),         # long queries, large batch
    ],
    ids=lambda v: str(v),
)
def test_shift_input_ids(
    num_reqs: int,
    scenario: str,
    block_size: int,
    rt,
) -> None:
    """Bitwise comparison: pure left-shift + insert of int32 values."""
    if _import_error is not None:
        pytest.skip(f"kernel import failed: {_import_error}")
    rt.init_device_properties_triton()

    max_num_reqs = max(8, num_reqs * 2)
    if scenario == "long_query":
        # per-req lens up to 256 -> total up to num_reqs * 256
        num_tokens = max(4096, num_reqs * 320)
    else:
        # per-req lens up to block_size + 1 (tile_boundary_plus_one)
        num_tokens = max(256, num_reqs * (block_size + 2))

    inputs = _gen_inputs(
        num_reqs=num_reqs,
        max_num_reqs=max_num_reqs,
        num_tokens=num_tokens,
        scenario=scenario,
        device=rt.STRICT_DEVICE,
        block_size=block_size,
    )

    expected = _ref(
        num_reqs=inputs["num_reqs"],
        input_ids=inputs["input_ids"],
        draft_token_ids=inputs["draft_token_ids"],
        idx_mapping=inputs["idx_mapping"],
        query_start_loc=inputs["query_start_loc"],
        last_token_indices=inputs["last_token_indices"],
    )

    actual = _run_kernel(inputs, rt)
    _assert_bitwise("input_ids", expected, actual)


def test_import_error() -> None:
    if _import_error is not None:
        pytest.fail(
            f"Failed to import _shift_input_ids_kernel:\n"
            f"{_import_traceback}"
        )
