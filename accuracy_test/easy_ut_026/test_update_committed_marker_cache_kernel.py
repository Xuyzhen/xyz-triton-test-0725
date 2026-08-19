# SPDX-License-Identifier: Apache-2.0
# easy_ut_026 strict UT for _update_committed_marker_cache_kernel.
# Source: vllm/v1/worker/gpu/sample/thinking_budget.py (upstream, no vllm-ascend
# patch). Category: integer compute (marker scanning with cold/incremental
# paths). All outputs int32 -> bitwise. budget<0 path emits no writes; sentinel
# pre-fill catches accidental over-writes.
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
    from vllm.v1.worker.gpu.sample.thinking_budget import (
        _update_committed_marker_cache_kernel as kernel,
    )
except Exception as exc:  # pragma: no cover
    _import_error = exc
    _import_traceback = traceback.format_exc()


_SENTINEL = -2  # never a legal scan_pos / last_start / last_end value


def _scan_match(tokens_row: list[int], pos: int, marker: list[int],
                total_len: int) -> bool:
    """Return True if tokens_row[pos:pos+len(marker)] == marker (within total_len)."""
    if pos + len(marker) > total_len:
        return False
    for j, expected in enumerate(marker):
        if tokens_row[pos + j] != expected:
            return False
    return True


def _ref(
    req_ids: torch.Tensor,
    thinking_token_budget: torch.Tensor,
    all_token_ids: torch.Tensor,
    all_token_ids_stride: int,
    total_len: torch.Tensor,
    cached_last_start: torch.Tensor,
    cached_last_end: torch.Tensor,
    cached_scan_pos: torch.Tensor,
    reasoning_start_token_ids: torch.Tensor,
    natural_reasoning_end_token_ids: torch.Tensor,
    start_len: int,
    natural_end_len: int,
    max_len: int,
    block: int,
) -> dict[str, torch.Tensor]:
    """Independent CPU reference mirroring all kernel branches.

    Branches:
      1. budget<0 -> early return (no writes).
      2. scan_pos>total_len -> reset (scan_pos=0, last_start=-1, last_end=-1),
         then fall through to scan selection below.
      3. scan_pos==0 AND last_start<0 AND last_end<0 -> cold scan (backward,
         BLOCK-sized chunks, stop at first chunk with any marker).
      4. else -> incremental scan (forward from scan_pos to total_len, latest
         match wins).
    """
    req_ids_np = req_ids.cpu().to(torch.int64).numpy()
    budget_np = thinking_token_budget.cpu().to(torch.int64).numpy()
    tokens_np = all_token_ids.cpu().to(torch.int64).numpy()
    total_len_np = total_len.cpu().to(torch.int64).numpy()
    cls_np = cached_last_start.cpu().to(torch.int64).numpy().copy()
    cle_np = cached_last_end.cpu().to(torch.int64).numpy().copy()
    csp_np = cached_scan_pos.cpu().to(torch.int64).numpy().copy()
    start_marker = list(reasoning_start_token_ids.cpu().to(torch.int64).tolist())
    end_marker = list(natural_reasoning_end_token_ids.cpu().to(torch.int64).tolist())

    for prog_id in range(req_ids_np.shape[0]):
        rsi = int(req_ids_np[prog_id])
        budget = int(budget_np[rsi])
        if budget < 0:
            continue

        tl_ = int(total_len_np[rsi])
        scan_pos = int(csp_np[rsi])
        last_start = int(cls_np[rsi])
        last_end = int(cle_np[rsi])

        if scan_pos > tl_:
            scan_pos = 0
            last_start = -1
            last_end = -1

        tokens_row = list(tokens_np[rsi, :tl_])

        if scan_pos == 0 and last_start < 0 and last_end < 0:
            # Cold scan: walk backward in BLOCK-sized chunks.
            block_hi = tl_
            while block_hi > 0 and last_start < 0 and last_end < 0:
                block_lo = block_hi - block
                if block_lo < 0:
                    block_lo = 0
                chunk_start_matches: list[int] = []
                chunk_end_matches: list[int] = []
                for lane in range(block):
                    offs = block_lo + lane
                    if offs >= block_hi:
                        continue  # mask=False lane
                    if _scan_match(tokens_row, offs, start_marker, tl_):
                        chunk_start_matches.append(offs)
                    if _scan_match(tokens_row, offs, end_marker, tl_):
                        chunk_end_matches.append(offs)
                last_start = max(chunk_start_matches) if chunk_start_matches else -1
                last_end = max(chunk_end_matches) if chunk_end_matches else -1
                block_hi = block_lo
        else:
            # Incremental scan: forward from scan_pos, latest match wins.
            for i in range(scan_pos, tl_):
                if _scan_match(tokens_row, i, start_marker, tl_):
                    last_start = i
                if _scan_match(tokens_row, i, end_marker, tl_):
                    last_end = i

        cls_np[rsi] = last_start
        cle_np[rsi] = last_end
        new_scan_pos = tl_ - (max_len - 1)
        if new_scan_pos < 0:
            new_scan_pos = 0
        csp_np[rsi] = new_scan_pos

    return {
        "cached_last_start": torch.from_numpy(cls_np).to(torch.int32),
        "cached_last_end": torch.from_numpy(cle_np).to(torch.int32),
        "cached_scan_pos": torch.from_numpy(csp_np).to(torch.int32),
    }


def _build_tokens(
    total_len: int,
    start_marker: list[int],
    end_marker: list[int],
    scenario: str,
    block: int,
    vocab_size: int,
    device: str,
) -> torch.Tensor:
    """Build a 1D token row of length total_len that exercises the scenario.

    Scenarios (cold-scan / incremental):
      - "cold_none":              no markers anywhere.
      - "cold_both_last_chunk":   both markers in the last BLOCK chunk.
      - "cold_both_diff_chunks":  start in earlier chunk, end in last chunk.
      - "cold_start_only":        only start marker in last chunk.
      - "cold_end_only":          only end marker in last chunk.
      - "cold_boundary":          start marker at offs where offs+START_LEN==total_len.
      - "incr_none":              no markers after scan_pos.
      - "incr_new_start":         start marker after scan_pos.
      - "incr_new_end":           end marker after scan_pos.
      - "incr_both":              both markers after scan_pos.
    """
    torch.manual_seed(123 + total_len + block + len(start_marker) * 7)
    # Random non-marker tokens (avoid the marker sequences by chance).
    tokens = torch.randint(0, vocab_size, (total_len,), dtype=torch.int32, device=device)

    def place_marker(pos: int, marker: list[int]) -> None:
        for j, tok in enumerate(marker):
            tokens[pos + j] = tok

    # Choose safe random tokens that avoid the first element of either marker
    # so we don't accidentally create matches elsewhere.
    avoid = set([start_marker[0] if start_marker else -1,
                end_marker[0] if end_marker else -1])
    if len(avoid) >= vocab_size:
        # Pathological case: vocab too small. Just use the original random data.
        pass
    else:
        # Re-roll any token that hits a marker head until it doesn't.
        for i in range(total_len):
            attempts = 0
            while int(tokens[i].item()) in avoid and attempts < 16:
                tokens[i] = torch.randint(0, vocab_size, (1,), dtype=torch.int32, device=device)[0]
                attempts += 1

    if scenario == "cold_none":
        pass
    elif scenario == "cold_both_last_chunk":
        # Last chunk is [block_lo, block_hi) where block_hi=total_len.
        # block_lo = max(total_len - block, 0). Place markers within this chunk.
        lo = max(total_len - block, 0)
        # Ensure both fit: start at lo, end at lo + max(1, len(start_marker))
        start_pos = lo
        end_pos = lo + max(1, len(start_marker))
        # Make sure end marker fits within total_len
        if end_pos + len(end_marker) > total_len:
            end_pos = max(lo, total_len - len(end_marker))
        if start_pos + len(start_marker) > total_len:
            start_pos = total_len - len(start_marker)
        place_marker(start_pos, start_marker)
        place_marker(end_pos, end_marker)
    elif scenario == "cold_both_diff_chunks":
        # Two chunks: [0, block) and [block, 2*block) (assuming total_len >= 2*block).
        # Place start in earlier chunk, end in last chunk.
        if total_len < 2 * block:
            # Cannot construct: fall back to both in last chunk.
            lo = max(total_len - block, 0)
            place_marker(lo, start_marker)
            place_marker(min(lo + 1, total_len - len(end_marker)), end_marker)
        else:
            place_marker(0, start_marker)  # earliest chunk
            place_marker(total_len - len(end_marker), end_marker)  # last chunk
    elif scenario == "cold_start_only":
        lo = max(total_len - block, 0)
        place_marker(min(lo, total_len - len(start_marker)), start_marker)
    elif scenario == "cold_end_only":
        lo = max(total_len - block, 0)
        place_marker(min(lo, total_len - len(end_marker)), end_marker)
    elif scenario == "cold_boundary":
        # Start marker ends exactly at total_len: offs + len(start_marker) == total_len.
        pos = total_len - len(start_marker)
        place_marker(pos, start_marker)
    elif scenario == "incr_none":
        pass
    elif scenario == "incr_new_start":
        # Place start marker somewhere after scan_pos. Caller will set scan_pos.
        # For now, place at midpoint; caller adjusts scan_pos below midpoint.
        pos = total_len // 2
        place_marker(pos, start_marker)
    elif scenario == "incr_new_end":
        pos = total_len // 2
        place_marker(pos, end_marker)
    elif scenario == "incr_both":
        pos = total_len // 2
        place_marker(pos, start_marker)
        if pos + len(end_marker) <= total_len and pos + len(start_marker) + len(end_marker) <= total_len:
            place_marker(pos + len(start_marker), end_marker)
        else:
            place_marker(max(0, pos - len(end_marker)), end_marker)
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    return tokens


def _gen_inputs(
    scenario: str,
    start_len: int,
    natural_end_len: int,
    max_len: int,
    block: int,
    vocab_size: int,
    max_state_slots: int,
    device: str,
) -> dict[str, Any]:
    """Build inputs for a single request, exercising the chosen scenario.

    The kernel is per-request (grid = (num_reqs,)), so we run one request per
    test case. max_state_slots=1 keeps the test focused on branch behavior.
    """
    # Pick a total_len that has room for markers and respects the chunk layout.
    # For cold scan scenarios, total_len must be >= 2*block to exercise the
    # multi-chunk path; for boundary, total_len >= len(start_marker).
    if scenario == "cold_both_diff_chunks":
        total_len = max(2 * block + 1, start_len + natural_end_len + 1)
    elif scenario in ("cold_both_last_chunk", "cold_start_only", "cold_end_only"):
        total_len = max(block + 1, start_len + 1, natural_end_len + 1)
    elif scenario == "cold_boundary":
        total_len = max(start_len, natural_end_len) + max(block, 1)
    elif scenario == "cold_none":
        total_len = max(2 * block + 1, 8)
    elif scenario.startswith("incr_"):
        total_len = max(2 * block + 1, 16)
    else:
        total_len = max(2 * block + 1, 16)

    # Choose markers that don't collide with each other.
    start_marker = list(range(1, 1 + start_len))
    end_marker = list(range(100, 100 + natural_end_len))
    # Ensure they don't share a head token (avoids accidental cross-matches).
    if start_len and natural_end_len and start_marker[0] == end_marker[0]:
        end_marker[0] = end_marker[0] + 1
    # Ensure within vocab.
    assert all(0 <= t < vocab_size for t in start_marker + end_marker), \
        f"markers {start_marker} / {end_marker} exceed vocab_size={vocab_size}"

    tokens = _build_tokens(total_len, start_marker, end_marker, scenario, block, vocab_size, device)

    # All-zero budget means kernel processes. Use -1 for budget_skip via separate path.
    budget = torch.tensor([0], dtype=torch.int32, device=device)

    total_len_t = torch.tensor([total_len], dtype=torch.int32, device=device)

    # Default: cold scan path. Set scan_pos=0, last_start=-1, last_end=-1.
    # For incremental scenarios, set scan_pos to a value that:
    #   (a) is > 0 (so we take the else branch), AND
    #   (b) is <= total_len (so we don't hit the reset path), AND
    #   (c) is before the marker we want to find in the incremental scan.
    scan_pos_init = 0
    last_start_init = -1
    last_end_init = -1
    if scenario.startswith("incr_"):
        # Need scan_pos in [1, total_len]. Reset path triggers when scan_pos > total_len.
        # Incremental path triggers when NOT (scan_pos==0 AND last_start<0 AND last_end<0).
        # So set scan_pos=1 (and last_start=last_end=-1) -> goes to incremental.
        scan_pos_init = 1

    cached_last_start = torch.tensor([last_start_init], dtype=torch.int32, device=device)
    cached_last_end = torch.tensor([last_end_init], dtype=torch.int32, device=device)
    cached_scan_pos = torch.tensor([scan_pos_init], dtype=torch.int32, device=device)

    start_marker_t = torch.tensor(start_marker, dtype=torch.int32, device=device)
    end_marker_t = torch.tensor(end_marker, dtype=torch.int32, device=device)

    # req_ids: one program.
    req_ids = torch.tensor([0], dtype=torch.int32, device=device)

    # Embed tokens into the [max_state_slots, max_model_len] layout. The kernel
    # reads tokens[req_state_idx, :total_len] via all_token_ids_stride.
    max_model_len = max(total_len, 32)
    all_token_ids = torch.zeros((max_state_slots, max_model_len), dtype=torch.int32, device=device)
    all_token_ids[0, :total_len] = tokens

    return {
        "req_ids": req_ids,
        "thinking_token_budget": budget,
        "all_token_ids": all_token_ids,
        "all_token_ids_stride": max_model_len,
        "total_len": total_len_t,
        "cached_last_start": cached_last_start,
        "cached_last_end": cached_last_end,
        "cached_scan_pos": cached_scan_pos,
        "reasoning_start_token_ids": start_marker_t,
        "natural_reasoning_end_token_ids": end_marker_t,
        "start_len": start_len,
        "natural_end_len": natural_end_len,
        "max_len": max_len,
        "block": block,
        "total_len_value": total_len,
    }


def _launch(k, inputs: dict[str, Any]) -> None:
    k[(inputs["req_ids"].shape[0],)](
        inputs["req_ids"],
        inputs["thinking_token_budget"],
        inputs["all_token_ids"],
        inputs["all_token_ids_stride"],
        inputs["total_len"],
        inputs["cached_last_start"],
        inputs["cached_last_end"],
        inputs["cached_scan_pos"],
        inputs["reasoning_start_token_ids"],
        inputs["natural_reasoning_end_token_ids"],
        START_LEN=inputs["start_len"],
        NATURAL_END_LEN=inputs["natural_end_len"],
        MAX_LEN=inputs["max_len"],
        BLOCK=inputs["block"],
    )


# Shape parameters: (start_len, natural_end_len, max_len, block).
# block is kept small (4, 8) to exercise multi-chunk cold scan, plus 1024 to
# exercise the single-chunk path used in production.
SHAPE_PARAMS = [
    (1, 1, 1, 4),
    (2, 2, 2, 8),
    (1, 2, 2, 4),
    (3, 2, 3, 1024),
    (2, 3, 3, 8),
]

SCENARIOS = [
    "cold_none",
    "cold_both_last_chunk",
    "cold_both_diff_chunks",
    "cold_start_only",
    "cold_end_only",
    "cold_boundary",
    "incr_none",
    "incr_new_start",
    "incr_new_end",
    "incr_both",
]


@pytest.mark.parametrize("start_len,natural_end_len,max_len,block", SHAPE_PARAMS)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_update_committed_marker_cache(
    start_len: int,
    natural_end_len: int,
    max_len: int,
    block: int,
    scenario: str,
):
    """Compare upstream kernel (NPU) against an independent CPU reference.

    All outputs are int32. budget<0 path is exercised by an extra check below.
    Sentinel pre-fill catches accidental over-writes of slots that should not
    be touched (e.g., slots for budget<0 requests).
    """
    if kernel is None:
        pytest.fail(
            "_update_committed_marker_cache_kernel import failed; this is not "
            "a precision failure and no kernel was tested.\n"
            f"error={_import_error}\ntraceback:\n{_import_traceback}",
            pytrace=False,
        )

    init_device_properties_triton()
    device = str(_STRICT_DEVICE)

    vocab_size = 256
    max_state_slots = 1
    inputs = _gen_inputs(
        scenario=scenario,
        start_len=start_len,
        natural_end_len=natural_end_len,
        max_len=max_len,
        block=block,
        vocab_size=vocab_size,
        max_state_slots=max_state_slots,
        device=device,
    )

    # CPU reference.
    ref_inputs = {
        "req_ids": inputs["req_ids"].cpu().clone(),
        "thinking_token_budget": inputs["thinking_token_budget"].cpu().clone(),
        "all_token_ids": inputs["all_token_ids"].cpu().clone(),
        "all_token_ids_stride": inputs["all_token_ids_stride"],
        "total_len": inputs["total_len"].cpu().clone(),
        "cached_last_start": inputs["cached_last_start"].cpu().clone(),
        "cached_last_end": inputs["cached_last_end"].cpu().clone(),
        "cached_scan_pos": inputs["cached_scan_pos"].cpu().clone(),
        "reasoning_start_token_ids": inputs["reasoning_start_token_ids"].cpu().clone(),
        "natural_reasoning_end_token_ids": inputs["natural_reasoning_end_token_ids"].cpu().clone(),
        "start_len": start_len,
        "natural_end_len": natural_end_len,
        "max_len": max_len,
        "block": block,
    }
    expected = _ref(**ref_inputs)

    # NPU run: clone mutated tensors.
    npu_inputs = {
        "req_ids": inputs["req_ids"].clone(),
        "thinking_token_budget": inputs["thinking_token_budget"].clone(),
        "all_token_ids": inputs["all_token_ids"].clone(),
        "all_token_ids_stride": inputs["all_token_ids_stride"],
        "total_len": inputs["total_len"].clone(),
        "cached_last_start": inputs["cached_last_start"].clone(),
        "cached_last_end": inputs["cached_last_end"].clone(),
        "cached_scan_pos": inputs["cached_scan_pos"].clone(),
        "reasoning_start_token_ids": inputs["reasoning_start_token_ids"].clone(),
        "natural_reasoning_end_token_ids": inputs["natural_reasoning_end_token_ids"].clone(),
        "start_len": start_len,
        "natural_end_len": natural_end_len,
        "max_len": max_len,
        "block": block,
    }
    _launch(kernel, npu_inputs)
    synchronize()

    mutated = ("cached_last_start", "cached_last_end", "cached_scan_pos")
    for name in mutated:
        npu_out = npu_inputs[name].cpu()
        ref_out = expected[name]
        assert npu_out.dtype == ref_out.dtype, f"{name}: dtype mismatch"
        assert npu_out.shape == ref_out.shape, f"{name}: shape mismatch"
        if not torch.equal(npu_out, ref_out):
            mismatched = torch.ne(npu_out, ref_out)
            count = int(mismatched.sum().item())
            total = int(mismatched.numel())
            first_idx = torch.nonzero(mismatched, as_tuple=False)
            first_info = ""
            if first_idx.numel() > 0:
                loc = tuple(first_idx[0].tolist())
                first_info = (
                    f"; first mismatch at {loc}: "
                    f"npu={npu_out[loc].item()} ref={ref_out[loc].item()}"
                )
            pytest.fail(
                f"_update_committed_marker_cache_kernel mismatch for {name} "
                f"(scenario={scenario}): {count}/{total} elements differ{first_info}"
            )


@pytest.mark.parametrize("start_len,natural_end_len,max_len,block", SHAPE_PARAMS)
def test_update_committed_marker_cache_budget_skip(
    start_len: int,
    natural_end_len: int,
    max_len: int,
    block: int,
):
    """budget<0 path: kernel must not write anything.

    Pre-fill cached_last_start/end/scan_pos with a sentinel; if the kernel
    respects the early return, all three outputs remain at the sentinel.
    """
    if kernel is None:
        pytest.fail(
            "_update_committed_marker_cache_kernel import failed",
            pytrace=False,
        )

    init_device_properties_triton()
    device = str(_STRICT_DEVICE)

    vocab_size = 256
    total_len = max(block + 1, 8)
    max_model_len = max(total_len, 32)

    req_ids = torch.tensor([0], dtype=torch.int32, device=device)
    budget = torch.tensor([-1], dtype=torch.int32, device=device)  # SKIP
    all_token_ids = torch.zeros((1, max_model_len), dtype=torch.int32, device=device)
    total_len_t = torch.tensor([total_len], dtype=torch.int32, device=device)
    cached_last_start = torch.full((1,), _SENTINEL, dtype=torch.int32, device=device)
    cached_last_end = torch.full((1,), _SENTINEL, dtype=torch.int32, device=device)
    cached_scan_pos = torch.full((1,), _SENTINEL, dtype=torch.int32, device=device)
    start_marker = torch.tensor(list(range(1, 1 + start_len)), dtype=torch.int32, device=device)
    end_marker = torch.tensor(list(range(100, 100 + natural_end_len)), dtype=torch.int32, device=device)

    npu_inputs = {
        "req_ids": req_ids,
        "thinking_token_budget": budget,
        "all_token_ids": all_token_ids,
        "all_token_ids_stride": max_model_len,
        "total_len": total_len_t,
        "cached_last_start": cached_last_start,
        "cached_last_end": cached_last_end,
        "cached_scan_pos": cached_scan_pos,
        "reasoning_start_token_ids": start_marker,
        "natural_reasoning_end_token_ids": end_marker,
        "start_len": start_len,
        "natural_end_len": natural_end_len,
        "max_len": max_len,
        "block": block,
    }
    _launch(kernel, npu_inputs)
    synchronize()

    for name in ("cached_last_start", "cached_last_end", "cached_scan_pos"):
        out = npu_inputs[name].cpu()
        assert int(out[0].item()) == _SENTINEL, (
            f"budget<0 early-return violated: {name}={out[0].item()} "
            f"(expected sentinel {_SENTINEL})"
        )
