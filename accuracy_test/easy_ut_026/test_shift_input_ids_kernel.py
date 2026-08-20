# SPDX-License-Identifier: Apache-2.0
# easy_ut_026 strict UT for _shift_input_ids_kernel.
# Source: vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py
# (upstream, no vllm-ascend patch). Category: non-compute shift (left-shift
# input_ids by 1 within each request's query window, then insert the draft
# token at the last position). All outputs int32 -> bitwise comparison.
#
# Kernel summary (grid = (num_reqs,)):
#   req_state_idx = idx_mapping[req_idx]; skip if < 0.
#   query_start = query_start_loc[req_idx]
#   last_token_index = last_token_indices[req_idx]
#   query_len = last_token_index - query_start + 1
#   For i in range(1, query_len, BLOCK_SIZE):
#     block = i + arange(0, BLOCK_SIZE)
#     mask = block < query_len
#     input_ids = input_ids_ptr[query_start + block]  (masked load)
#     input_ids_ptr[query_start + block - 1] = input_ids  (masked store)
#   draft_token = draft_tokens_ptr[req_idx]  (int64)
#   input_ids_ptr[last_token_index] = draft_token  (store, downcast to int32)
#
# Note: the kernel reads and writes the SAME buffer. The CPU reference must
# respect the order of the kernel's in-place shift (left-to-right, in blocks of
# BLOCK_SIZE) to avoid clobbering source data before it's read. Since the shift
# is by -1 (left), and the kernel iterates i = 1, BLOCK_SIZE+1, 2*BLOCK_SIZE+1,
# ... a simple Python list copy is safe because we always read slot j+1 and
# write slot j, and the write to slot j never clobbers a source slot of a
# later block (later blocks start at higher i).

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
        _shift_input_ids_kernel as kernel,
    )
except Exception as exc:  # pragma: no cover
    _import_error = exc
    _import_traceback = traceback.format_exc()


_SENTINEL_ID = -12345  # catches stray writes outside [query_start, last_token_index]


def _ref(
    num_reqs: int,
    input_ids: torch.Tensor,        # [num_tokens] int32
    idx_mapping: torch.Tensor,     # [num_reqs] int64
    query_start_loc: torch.Tensor,  # [num_reqs+1] int32
    last_token_indices: torch.Tensor,  # [num_reqs] int64
    draft_tokens: torch.Tensor,     # [num_reqs] int64
) -> torch.Tensor:
    """Independent CPU reference. In-place left-shift by 1 within each
    request's query window, then insert draft_tokens[req_idx] at the last
    token index. Reads from the original input_ids buffer (not the in-place
    modified one) to avoid data hazards."""
    out = input_ids.cpu().clone()
    im = idx_mapping.cpu().to(torch.int64).tolist()
    qsl = query_start_loc.cpu().to(torch.int64).tolist()
    lti = last_token_indices.cpu().to(torch.int64).tolist()
    dt = draft_tokens.cpu().to(torch.int64).tolist()
    iid = input_ids.cpu().to(torch.int64).tolist()

    for req_idx in range(num_reqs):
        rsi = im[req_idx]
        if rsi < 0:
            continue
        qs = qsl[req_idx]
        last_idx = lti[req_idx]
        query_len = last_idx - qs + 1
        # Left-shift by 1: new[j] = old[j+1] for j in [0, query_len-2]
        for j in range(0, query_len - 1):
            out[qs + j] = iid[qs + j + 1]
        # Insert draft token at the last position
        # The kernel loads draft_token as int64 and stores to int32 buffer.
        # Truncate to int32 to match.
        draft_val = dt[req_idx] & 0xFFFFFFFF
        if draft_val >= 0x80000000:
            draft_val -= 0x100000000
        out[last_idx] = draft_val

    return out.to(torch.int32)


def _gen_inputs(
    num_reqs: int,
    max_num_reqs: int,
    num_tokens: int,
    *,
    scenario: str,
    device: str,
    block_size: int = 8,
) -> dict[str, Any]:
    torch.manual_seed(42 + num_reqs * 7)

    if scenario == "skip_padded":
        idx_mapping = torch.arange(num_reqs, dtype=torch.int64,
                                   device=device)
        idx_mapping[-1] = -1
    else:
        idx_mapping = torch.arange(num_reqs, dtype=torch.int64,
                                   device=device)

    # Build query_start_loc and last_token_indices
    if scenario == "single_token":
        # query_len = 1 (only insert draft_token, no shift)
        query_lens = torch.ones(num_reqs, dtype=torch.int32,
                                device=device)
    elif scenario == "two_tokens":
        # query_len = 2 (shift 1 token, insert draft)
        query_lens = torch.full((num_reqs,), 2, dtype=torch.int32,
                               device=device)
    elif scenario == "tile_boundary":
        # query_len = block_size exactly
        query_lens = torch.full((num_reqs,), block_size, dtype=torch.int32,
                               device=device)
    elif scenario == "tile_boundary_plus_one":
        query_lens = torch.full((num_reqs,), block_size + 1, dtype=torch.int32,
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

    # last_token_indices = query_end - 1 for all
    last_token_indices = (query_start_loc[1:num_reqs + 1] - 1).to(torch.int64)

    # input_ids: random token ids in [0, 1000), pre-filled with sentinel
    # outside any request's window to catch stray writes
    input_ids = torch.full((num_tokens,), _SENTINEL_ID, dtype=torch.int32,
                           device=device)
    # Fill in-request windows with random tokens
    iid_cpu = input_ids.cpu()
    for r in range(num_reqs):
        qs = int(qsl_cpu[r].item())
        qe = int(qsl_cpu[r + 1].item())
        iid_cpu[qs:qe] = torch.randint(0, 1000, (qe - qs,), dtype=torch.int32)
    input_ids.copy_(iid_cpu)

    # draft_tokens: random, shape [num_reqs], int64 (matching upstream)
    draft_tokens = torch.randint(0, 1000, (num_reqs,), dtype=torch.int64,
                                 device=device)

    return {
        "num_reqs": num_reqs,
        "max_num_reqs": max_num_reqs,
        "num_tokens": num_tokens,
        "input_ids": input_ids,
        "idx_mapping": idx_mapping,
        "query_start_loc": query_start_loc,
        "last_token_indices": last_token_indices,
        "draft_tokens": draft_tokens,
        "block_size": block_size,
    }


def _run_kernel(inputs: dict[str, Any], device: str) -> torch.Tensor:
    input_ids = inputs["input_ids"].clone()
    kernel[(inputs["num_reqs"],)](
        input_ids,
        inputs["idx_mapping"],
        inputs["query_start_loc"],
        inputs["last_token_indices"],
        inputs["draft_tokens"],
        BLOCK_SIZE=inputs["block_size"],
        num_warps=1,
    )
    synchronize()
    return input_ids


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
        # Tile boundary: query_len == block_size
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
        # Realistic block size
        (4, "random", 1024),
    ],
    ids=lambda v: str(v),
)
def test_shift_input_ids(
    num_reqs: int,
    scenario: str,
    block_size: int,
) -> None:
    """Bitwise comparison: pure left-shift + insert, no arithmetic."""
    if _import_error is not None:
        pytest.skip(f"kernel import failed: {_import_error}")
    init_device_properties_triton()

    max_num_reqs = max(8, num_reqs * 2)
    num_tokens = 256

    inputs = _gen_inputs(
        num_reqs=num_reqs,
        max_num_reqs=max_num_reqs,
        num_tokens=num_tokens,
        scenario=scenario,
        device=_STRICT_DEVICE,
        block_size=block_size,
    )

    expected = _ref(
        num_reqs=inputs["num_reqs"],
        input_ids=inputs["input_ids"],
        idx_mapping=inputs["idx_mapping"],
        query_start_loc=inputs["query_start_loc"],
        last_token_indices=inputs["last_token_indices"],
        draft_tokens=inputs["draft_tokens"],
    )

    actual = _run_kernel(inputs, _STRICT_DEVICE)
    _assert_bitwise("input_ids", expected, actual)


def test_import_error() -> None:
    if _import_error is not None:
        pytest.fail(
            f"Failed to import _shift_input_ids_kernel:\n{_import_traceback}"
        )
