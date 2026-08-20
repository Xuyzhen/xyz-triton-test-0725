# SPDX-License-Identifier: Apache-2.0
# easy_ut_026 strict UT for _thinking_budget_kernel.
# Source: vllm/v1/worker/gpu/sample/thinking_budget.py (upstream, no vllm-ascend
# patch). Category: floating-point + discrete fused kernel.
#   - Discrete output: force_token_id (which logits column is overwritten).
#     Must be exact (integer semantics).
#   - Float output: the 1e9 constant written at logits[token_idx, force_token_id].
#     Bitwise-identical since 1e9 is exactly representable in float32 and the
#     write is a literal store (no arithmetic). Other logits entries must
#     remain untouched (sentinel = NaN).
# Per the ASCEND_OPERATOR_ACCURACY_2_1_TEST_PLAN.md, fusion kernels must use
# the strictest applicable criterion per output class. Here: force_token_id is
# bitwise, logits overwrite value is bitwise (literal 1e9), untouched entries
# stay at NaN sentinel.
from accuracy_test.easy_ut_026.runtime_npu import (
    STRICT_DEVICE as _STRICT_DEVICE,
)
from accuracy_test.easy_ut_026.runtime_npu import (
    init_device_properties_triton,
    synchronize,
)

import math
import traceback
from typing import Any

import pytest
import torch

kernel = None
_import_error: Exception | None = None
_import_traceback: str | None = None
try:
    from vllm.v1.worker.gpu.sample.thinking_budget import (
        _thinking_budget_kernel as kernel,
    )
except Exception as exc:  # pragma: no cover
    _import_error = exc
    _import_traceback = traceback.format_exc()


# NaN sentinel pre-fill for the logits row. The kernel only writes to ONE
# column per row when budget is exceeded, so all other entries must remain NaN.
def _nan_row(num_tokens: int, vocab_size: int, device: str) -> torch.Tensor:
    return torch.full((num_tokens, vocab_size), float("nan"), dtype=torch.float32, device=device)


def _load_effective_token_cpu(
    all_token_ids_np,
    all_token_ids_stride: int,
    input_ids_np,
    cur_req_first_pos: int,
    req_state_idx: int,
    total_len: int,
    pos: int,
) -> int:
    """CPU mirror of _load_effective_token in thinking_budget.py.

    For pos < total_len: read all_token_ids[req_state_idx, pos].
    Otherwise: read input_ids[cur_req_first_pos + pos - total_len + 1]
    (the draft-prefix tokens).
    """
    if pos < total_len:
        return int(all_token_ids_np[req_state_idx * all_token_ids_stride + pos])
    input_pos = cur_req_first_pos + pos - total_len + 1
    return int(input_ids_np[input_pos])


def _ref(
    logits: torch.Tensor,
    logits_stride: int,
    expanded_idx_mapping: torch.Tensor,
    thinking_token_budget: torch.Tensor,
    all_token_ids: torch.Tensor,
    all_token_ids_stride: int,
    total_len: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    cached_last_start: torch.Tensor,
    cached_last_end: torch.Tensor,
    reasoning_start_token_ids: torch.Tensor,
    natural_reasoning_end_token_ids: torch.Tensor,
    reasoning_end_token_ids: torch.Tensor,
    start_len: int,
    natural_end_len: int,
    end_len: int,
) -> dict[str, torch.Tensor]:
    """Independent CPU reference.

    Per token_idx (program_id(0)):
      1. Look up req_state_idx via expanded_idx_mapping. If budget<0, skip.
      2. effective_len = total_len + local_pos. cur_req_first_pos = token_idx - local_pos.
      3. Scan [start_lo, effective_len - START_LEN + 1) for start marker;
         update last_start if found.
      4. Scan [end_lo, effective_len - NATURAL_END_LEN + 1) for end marker;
         update last_end if found.
      5. If last_start<0 or last_start<=last_end: skip (no write).
      6. reasoning_start = last_start + START_LEN. If effective_len -
         reasoning_start < budget: skip (under budget).
      7. Compute end_prefix_len: longest suffix of effective tokens that
         matches a prefix of the forced end marker (1..END_LEN-1).
      8. Write logits[token_idx, end_marker[end_prefix_len]] = 1e9.
    """
    logits_out = logits.cpu().clone()
    eim_np = expanded_idx_mapping.cpu().to(torch.int64).numpy()
    budget_np = thinking_token_budget.cpu().to(torch.int64).numpy()
    tokens_np = all_token_ids.cpu().to(torch.int64).numpy()
    total_len_np = total_len.cpu().to(torch.int64).numpy()
    input_ids_np = input_ids.cpu().to(torch.int64).numpy()
    local_pos_np = expanded_local_pos.cpu().to(torch.int64).numpy()
    cls_np = cached_last_start.cpu().to(torch.int64).numpy().copy()
    cle_np = cached_last_end.cpu().to(torch.int64).numpy().copy()
    start_marker = list(reasoning_start_token_ids.cpu().to(torch.int64).tolist())
    end_marker = list(natural_reasoning_end_token_ids.cpu().to(torch.int64).tolist())
    forced_end = list(reasoning_end_token_ids.cpu().to(torch.int64).tolist())

    num_tokens = logits.shape[0]
    for token_idx in range(num_tokens):
        rsi = int(eim_np[token_idx])
        budget = int(budget_np[rsi])
        if budget < 0:
            continue

        local_pos = int(local_pos_np[token_idx])
        cur_req_first_pos = token_idx - local_pos
        tl_ = int(total_len_np[rsi])
        effective_len = tl_ + local_pos

        last_start = int(cls_np[rsi])
        last_end = int(cle_np[rsi])

        # Scan for start marker.
        start_lo = tl_ - start_len + 1
        if start_lo < 0:
            start_lo = 0
        for i in range(start_lo, effective_len - start_len + 1):
            match = True
            for j in range(start_len):
                expected = start_marker[j]
                actual = _load_effective_token_cpu(
                    tokens_np, all_token_ids_stride, input_ids_np,
                    cur_req_first_pos, rsi, tl_, i + j,
                )
                if actual != expected:
                    match = False
                    break
            if match:
                last_start = i

        # Scan for natural end marker.
        end_lo = tl_ - natural_end_len + 1
        if end_lo < 0:
            end_lo = 0
        for i in range(end_lo, effective_len - natural_end_len + 1):
            match = True
            for j in range(natural_end_len):
                expected = end_marker[j]
                actual = _load_effective_token_cpu(
                    tokens_np, all_token_ids_stride, input_ids_np,
                    cur_req_first_pos, rsi, tl_, i + j,
                )
                if actual != expected:
                    match = False
                    break
            if match:
                last_end = i

        if last_start < 0 or last_start <= last_end:
            continue

        reasoning_start = last_start + start_len
        num_reasoning_tokens = effective_len - reasoning_start
        if num_reasoning_tokens < budget:
            continue

        # end_prefix_len: longest suffix of effective tokens that matches a
        # prefix of the forced end marker (1..END_LEN-1).
        end_prefix_len = 0
        max_prefix_len = end_len - 1
        if effective_len < max_prefix_len:
            max_prefix_len = effective_len

        for prefix_len in range(1, end_len):
            if prefix_len > max_prefix_len:
                continue
            suffix_start = effective_len - prefix_len
            match = True
            for j in range(end_len):
                if j < prefix_len:
                    expected = forced_end[j]
                    actual = _load_effective_token_cpu(
                        tokens_np, all_token_ids_stride, input_ids_np,
                        cur_req_first_pos, rsi, tl_, suffix_start + j,
                    )
                    if actual != expected:
                        match = False
                        break
            if match:
                end_prefix_len = prefix_len

        force_token_id = forced_end[end_prefix_len]
        logits_out[token_idx, force_token_id] = 1.0e9

    return {"logits": logits_out}


def _gen_inputs(
    scenario: str,
    start_len: int,
    natural_end_len: int,
    end_len: int,
    vocab_size: int,
    device: str,
) -> dict[str, Any]:
    """Build a single-token, single-request scenario.

    Scenarios:
      - "skip_budget":         budget=-1, no write.
      - "skip_no_start":       no start marker in scan range, last_start=-1.
      - "skip_last_start_le":  last_start=1, last_end=2 (last_start<=last_end).
      - "skip_under_budget":   budget exceeded marker count, but
                                num_reasoning_tokens<budget.
      - "write_no_prefix":    budget exceeded, no end-prefix -> write at end_marker[0].
      - "write_prefix_len_1": tail matches end_marker[0], write at end_marker[1].
      - "write_prefix_len_2": tail matches end_marker[0:2], write at end_marker[2]
                                (requires end_len >= 3).
    """
    torch.manual_seed(42 + start_len * 7 + natural_end_len * 3 + end_len)

    # Markers: pick distinct token ids within vocab.
    start_marker = list(range(10, 10 + start_len))
    natural_end_marker = list(range(20, 20 + natural_end_len))
    forced_end = list(range(30, 30 + end_len))
    # Avoid accidental collisions between marker heads.
    assert all(0 <= t < vocab_size for t in start_marker + natural_end_marker + forced_end)

    # Single token, single req.
    num_tokens = 1
    max_state_slots = 1
    max_model_len = 64  # plenty of room

    # Pick total_len and local_pos depending on scenario.
    # For "write_*" scenarios, we need:
    #   - A start marker in [start_lo, effective_len - start_len + 1)
    #   - No natural end marker (or last_start > last_end after scan)
    #   - num_reasoning_tokens >= budget
    #   - Optional end-prefix at the tail.
    if scenario == "skip_budget":
        budget_val = -1
        total_len = 8
        local_pos = 0
    elif scenario == "skip_no_start":
        budget_val = 4
        total_len = 8
        local_pos = 0
    elif scenario == "skip_last_start_le":
        budget_val = 4
        total_len = 8
        local_pos = 0
    elif scenario == "skip_under_budget":
        budget_val = 100
        total_len = 8
        local_pos = 0
    elif scenario == "write_no_prefix":
        budget_val = 1
        total_len = 8
        local_pos = 0
    elif scenario == "write_prefix_len_1":
        budget_val = 1
        total_len = 8
        local_pos = 0
    elif scenario == "write_prefix_len_2":
        assert end_len >= 3, "write_prefix_len_2 requires end_len>=3"
        budget_val = 1
        total_len = 8
        local_pos = 0
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    effective_len = total_len + local_pos

    # Build all_token_ids (committed history) and input_ids (draft tokens).
    # Random non-marker tokens, then place markers per scenario.
    all_token_ids = torch.zeros((max_state_slots, max_model_len), dtype=torch.int32, device=device)
    # Use a non-marker token id as filler.
    filler = (max(vocab_size - 1, 0)) if vocab_size > 0 else 0
    # Avoid filler being any marker head; pick a value not in any marker.
    avoid = set(start_marker + natural_end_marker + forced_end)
    while filler in avoid:
        filler = (filler - 1) % vocab_size
    all_token_ids[0, :total_len] = filler
    input_ids = torch.full((num_tokens,), filler, dtype=torch.int32, device=device)

    # Cached last_start/end: pre-set per scenario.
    cached_last_start_init = -1
    cached_last_end_init = -1

    if scenario == "skip_no_start":
        # No start marker anywhere -> last_start stays -1 -> skip.
        pass
    elif scenario == "skip_last_start_le":
        # Pre-set cached values such that last_start <= last_end, and no
        # scan updates them.
        cached_last_start_init = 1
        cached_last_end_init = 2
    elif scenario == "skip_under_budget":
        # Place a start marker so last_start becomes >=0, no end marker.
        # reasoning_start = last_start + start_len.
        # num_reasoning_tokens = effective_len - reasoning_start.
        # Set last_start = total_len - start_len (marker at end of committed),
        # so reasoning_start = total_len, num_reasoning_tokens = local_pos = 0.
        # budget=100 -> 0 < 100, skip.
        marker_pos = total_len - start_len
        for j in range(start_len):
            all_token_ids[0, marker_pos + j] = start_marker[j]
        cached_last_start_init = marker_pos  # in case scan doesn't update (it should)
        cached_last_end_init = -1
    elif scenario == "write_no_prefix":
        # Place start marker at end of committed, budget=1, num_reasoning_tokens=0+?.
        # We need num_reasoning_tokens >= budget=1, so we need at least one
        # reasoning token. Place start marker such that reasoning_start =
        # last_start + start_len = effective_len - 1 (so num_reasoning_tokens=1).
        # That means last_start = effective_len - 1 - start_len.
        # If start_len=1: last_start = effective_len - 2.
        # If start_len=2: last_start = effective_len - 3.
        marker_pos = effective_len - 1 - start_len
        if marker_pos >= 0:
            # Place marker in committed history (marker_pos < total_len).
            if marker_pos < total_len:
                for j in range(start_len):
                    all_token_ids[0, marker_pos + j] = start_marker[j]
            else:
                # Place in draft tokens.
                # marker_pos - total_len + 1 -> input_ids index.
                # local_pos=0 so input_ids has only 1 token (token_idx=0).
                # This case requires local_pos > 0; for local_pos=0 we can't.
                # Fall back to placing marker in history at total_len - start_len - 1.
                marker_pos = max(0, total_len - start_len - 1)
                for j in range(start_len):
                    if marker_pos + j < total_len:
                        all_token_ids[0, marker_pos + j] = start_marker[j]
        # No end-prefix: tail of effective tokens != forced_end[0].
        # filler != forced_end[0] by construction.
    elif scenario == "write_prefix_len_1":
        # Tail matches forced_end[0] only.
        marker_pos = max(0, total_len - start_len - 1)
        for j in range(start_len):
            if marker_pos + j < total_len:
                all_token_ids[0, marker_pos + j] = start_marker[j]
        # Place forced_end[0] at the last committed token (effective position total_len-1).
        all_token_ids[0, total_len - 1] = forced_end[0]
    elif scenario == "write_prefix_len_2":
        # Tail matches forced_end[0:2].
        marker_pos = max(0, total_len - start_len - 1)
        for j in range(start_len):
            if marker_pos + j < total_len:
                all_token_ids[0, marker_pos + j] = start_marker[j]
        # Place forced_end[0:2] at the last two committed tokens.
        if total_len >= 2:
            all_token_ids[0, total_len - 2] = forced_end[0]
            all_token_ids[0, total_len - 1] = forced_end[1]
        else:
            # Not enough room: fall back to no-prefix behavior.
            all_token_ids[0, total_len - 1] = forced_end[0]

    # Tensors for kernel call.
    expanded_idx_mapping = torch.tensor([0], dtype=torch.int32, device=device)
    thinking_token_budget = torch.tensor([budget_val], dtype=torch.int32, device=device)
    total_len_t = torch.tensor([total_len], dtype=torch.int32, device=device)
    expanded_local_pos = torch.tensor([local_pos], dtype=torch.int32, device=device)
    cached_last_start = torch.tensor([cached_last_start_init], dtype=torch.int32, device=device)
    cached_last_end = torch.tensor([cached_last_end_init], dtype=torch.int32, device=device)
    reasoning_start_t = torch.tensor(start_marker, dtype=torch.int32, device=device)
    natural_reasoning_end_t = torch.tensor(natural_end_marker, dtype=torch.int32, device=device)
    reasoning_end_t = torch.tensor(forced_end, dtype=torch.int32, device=device)

    # Logits: pre-fill with NaN.
    logits = _nan_row(num_tokens, vocab_size, device)
    logits_stride = vocab_size

    return {
        "logits": logits,
        "logits_stride": logits_stride,
        "expanded_idx_mapping": expanded_idx_mapping,
        "thinking_token_budget": thinking_token_budget,
        "all_token_ids": all_token_ids,
        "all_token_ids_stride": max_model_len,
        "total_len": total_len_t,
        "input_ids": input_ids,
        "expanded_local_pos": expanded_local_pos,
        "cached_last_start": cached_last_start,
        "cached_last_end": cached_last_end,
        "reasoning_start_token_ids": reasoning_start_t,
        "natural_reasoning_end_token_ids": natural_reasoning_end_t,
        "reasoning_end_token_ids": reasoning_end_t,
        "start_len": start_len,
        "natural_end_len": natural_end_len,
        "end_len": end_len,
        "scenario": scenario,
        "expected_force_token_id": _expected_force_token_id(
            scenario, start_marker, natural_end_marker, forced_end,
        ),
        "expect_write": scenario.startswith("write_"),
    }


def _expected_force_token_id(
    scenario: str,
    start_marker: list[int],
    natural_end_marker: list[int],
    forced_end: list[int],
) -> int | None:
    """Return the expected force_token_id, or None if no write should occur."""
    if scenario == "write_no_prefix":
        return forced_end[0]
    elif scenario == "write_prefix_len_1":
        return forced_end[1] if len(forced_end) > 1 else forced_end[0]
    elif scenario == "write_prefix_len_2":
        return forced_end[2] if len(forced_end) > 2 else (
            forced_end[1] if len(forced_end) > 1 else forced_end[0]
        )
    return None


def _launch(k, inputs: dict[str, Any]) -> None:
    k[(inputs["logits"].shape[0],)](
        inputs["logits"],
        inputs["logits_stride"],
        inputs["expanded_idx_mapping"],
        inputs["thinking_token_budget"],
        inputs["all_token_ids"],
        inputs["all_token_ids_stride"],
        inputs["total_len"],
        inputs["input_ids"],
        inputs["expanded_local_pos"],
        inputs["cached_last_start"],
        inputs["cached_last_end"],
        inputs["reasoning_start_token_ids"],
        inputs["natural_reasoning_end_token_ids"],
        inputs["reasoning_end_token_ids"],
        START_LEN=inputs["start_len"],
        NATURAL_END_LEN=inputs["natural_end_len"],
        END_LEN=inputs["end_len"],
    )


SHAPE_PARAMS = [
    # (start_len, natural_end_len, end_len)
    (1, 1, 1),
    (2, 2, 2),
    (1, 2, 3),  # enables write_prefix_len_2
    (3, 2, 4),
]

SCENARIOS = [
    "skip_budget",
    "skip_no_start",
    "skip_last_start_le",
    "skip_under_budget",
    "write_no_prefix",
    "write_prefix_len_1",
    "write_prefix_len_2",
]


@pytest.mark.parametrize("start_len,natural_end_len,end_len", SHAPE_PARAMS)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_thinking_budget(
    start_len: int,
    natural_end_len: int,
    end_len: int,
    scenario: str,
):
    """Compare upstream kernel (NPU) against an independent CPU reference.

    Output verification:
      1. If scenario expects a write: logits[0, force_token_id] must equal
         1.0e9 bitwise. All other entries must remain NaN (untouched).
      2. If scenario expects no write: all logits entries must remain NaN.
    """
    if scenario == "write_prefix_len_2" and end_len < 3:
        pytest.skip("write_prefix_len_2 requires end_len >= 3")
    if scenario == "write_prefix_len_1" and end_len < 2:
        pytest.skip("write_prefix_len_1 requires end_len >= 2")

    if kernel is None:
        pytest.fail(
            "_thinking_budget_kernel import failed; this is not a precision "
            "failure and no kernel was tested.\n"
            f"error={_import_error}\ntraceback:\n{_import_traceback}",
            pytrace=False,
        )

    init_device_properties_triton()
    device = str(_STRICT_DEVICE)

    # Pick a vocab_size comfortably larger than any marker token id used.
    # The markers use ids < 100 in _gen_inputs.
    vocab_size = 256

    inputs = _gen_inputs(
        scenario=scenario,
        start_len=start_len,
        natural_end_len=natural_end_len,
        end_len=end_len,
        vocab_size=vocab_size,
        device=device,
    )

    # CPU reference.
    ref_inputs = {
        "logits": inputs["logits"].cpu().clone(),
        "logits_stride": inputs["logits_stride"],
        "expanded_idx_mapping": inputs["expanded_idx_mapping"].cpu().clone(),
        "thinking_token_budget": inputs["thinking_token_budget"].cpu().clone(),
        "all_token_ids": inputs["all_token_ids"].cpu().clone(),
        "all_token_ids_stride": inputs["all_token_ids_stride"],
        "total_len": inputs["total_len"].cpu().clone(),
        "input_ids": inputs["input_ids"].cpu().clone(),
        "expanded_local_pos": inputs["expanded_local_pos"].cpu().clone(),
        "cached_last_start": inputs["cached_last_start"].cpu().clone(),
        "cached_last_end": inputs["cached_last_end"].cpu().clone(),
        "reasoning_start_token_ids": inputs["reasoning_start_token_ids"].cpu().clone(),
        "natural_reasoning_end_token_ids": inputs["natural_reasoning_end_token_ids"].cpu().clone(),
        "reasoning_end_token_ids": inputs["reasoning_end_token_ids"].cpu().clone(),
        "start_len": start_len,
        "natural_end_len": natural_end_len,
        "end_len": end_len,
    }
    expected = _ref(**ref_inputs)

    # NPU run.
    npu_inputs = {
        "logits": inputs["logits"].clone(),
        "logits_stride": inputs["logits_stride"],
        "expanded_idx_mapping": inputs["expanded_idx_mapping"].clone(),
        "thinking_token_budget": inputs["thinking_token_budget"].clone(),
        "all_token_ids": inputs["all_token_ids"].clone(),
        "all_token_ids_stride": inputs["all_token_ids_stride"],
        "total_len": inputs["total_len"].clone(),
        "input_ids": inputs["input_ids"].clone(),
        "expanded_local_pos": inputs["expanded_local_pos"].clone(),
        "cached_last_start": inputs["cached_last_start"].clone(),
        "cached_last_end": inputs["cached_last_end"].clone(),
        "reasoning_start_token_ids": inputs["reasoning_start_token_ids"].clone(),
        "natural_reasoning_end_token_ids": inputs["natural_reasoning_end_token_ids"].clone(),
        "reasoning_end_token_ids": inputs["reasoning_end_token_ids"].clone(),
        "start_len": start_len,
        "natural_end_len": natural_end_len,
        "end_len": end_len,
    }
    _launch(kernel, npu_inputs)
    synchronize()

    npu_logits = npu_inputs["logits"].cpu()
    ref_logits = expected["logits"]

    assert npu_logits.dtype == ref_logits.dtype
    assert npu_logits.shape == ref_logits.shape

    if not torch.equal(npu_logits, ref_logits):
        # Identify the failure mode for diagnostics.
        nps_np = npu_logits.numpy()
        ref_np = ref_logits.numpy()
        # Find differing entries.
        import numpy as _np
        # Treat NaN == NaN as equal (NaN is sentinel for "untouched").
        nan_mask_npu = _np.isnan(nps_np)
        nan_mask_ref = _np.isnan(ref_np)
        both_nan = nan_mask_npu & nan_mask_ref
        differ = ~(both_nan | (nps_np == ref_np))
        # Where one is NaN and the other isn't, that's also a difference.
        differ = differ | (nan_mask_npu != nan_mask_ref)
        count = int(differ.sum())
        total = int(differ.size)
        first_idx = _np.argwhere(differ)
        first_info = ""
        if len(first_idx) > 0:
            loc = tuple(int(x) for x in first_idx[0])
            npu_val = float(nps_np[loc]) if not _np.isnan(nps_np[loc]) else float("nan")
            ref_val = float(ref_np[loc]) if not _np.isnan(ref_np[loc]) else float("nan")
            first_info = (
                f"; first mismatch at {loc}: npu={npu_val} ref={ref_val}"
            )
        pytest.fail(
            f"_thinking_budget_kernel logits mismatch (scenario={scenario}): "
            f"{count}/{total} entries differ{first_info}"
        )
