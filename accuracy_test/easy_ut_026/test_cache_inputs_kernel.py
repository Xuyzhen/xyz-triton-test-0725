# SPDX-License-Identifier: Apache-2.0
# easy_ut_026 strict UT for _cache_inputs_kernel.
# Source: vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py
# (upstream, no vllm-ascend patch). Category: non-compute + float copy fusion
# (snapshot last num_spec_steps-1 draft input ids/embeds/hidden states into
# cached buffers). No arithmetic on data -> bitwise comparison.
#
# Kernel summary (grid = (num_reqs, cdiv(H, BLOCK_SIZE))):
#   Per (req_idx, block_idx):
#     req_state_idx = idx_mapping[req_idx]; skip if < 0.
#     query_start = query_start_loc[req_idx]; last_token_index =
#       last_token_indices[req_idx].
#     cache_window_size = num_speculative_steps - 1.
#     window_start = last_token_index - cache_window_size + 1.
#     For i in range(max(window_start, query_start), last_token_index + 1):
#       cache_write_slot = i - window_start.
#       If block_idx == 0: cached_draft_input_ids[req_state_idx,
#         cache_write_slot] = draft_input_ids[i].
#       If USE_INPUT_EMBEDS: cached_draft_input_embeds[req_state_idx,
#         cache_write_slot, :] = draft_input_embeds[i, :].
#       cached_target_hidden_states[req_state_idx, cache_write_slot, :] =
#         draft_input_hidden_states[i, :].

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
        _cache_inputs_kernel as kernel,
    )
except Exception as exc:  # pragma: no cover
    _import_error = exc
    _import_traceback = traceback.format_exc()


_SENTINEL_ID = -12345
_SENTINEL_FP = float("nan")


def _ref(
    num_reqs: int,
    num_speculative_steps: int,
    hidden_size: int,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    last_token_indices: torch.Tensor,
    draft_input_ids: torch.Tensor,
    draft_input_embeds: torch.Tensor | None,
    draft_input_hidden_states: torch.Tensor,
    cached_draft_input_ids: torch.Tensor,
    cached_draft_input_embeds: torch.Tensor | None,
    cached_target_hidden_states: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Independent CPU reference."""
    cdi = cached_draft_input_ids.cpu().clone()
    cde = (cached_draft_input_embeds.cpu().clone()
           if cached_draft_input_embeds is not None else None)
    cth = cached_target_hidden_states.cpu().clone()

    im = idx_mapping.cpu().to(torch.int64).tolist()
    qsl = query_start_loc.cpu().to(torch.int64).tolist()
    lti = last_token_indices.cpu().to(torch.int64).tolist()
    did = draft_input_ids.cpu().to(torch.int64).tolist()
    die = draft_input_embeds.cpu() if draft_input_embeds is not None else None
    dihs = draft_input_hidden_states.cpu()

    cache_window_size = num_speculative_steps - 1

    for req_idx in range(num_reqs):
        rsi = im[req_idx]
        if rsi < 0:
            continue
        qs = qsl[req_idx]
        lti_val = lti[req_idx]
        window_start = lti_val - cache_window_size + 1
        for i in range(max(window_start, qs), lti_val + 1):
            cache_write_slot = i - window_start
            cdi[rsi][cache_write_slot] = did[i]
            if cde is not None:
                cde[rsi, cache_write_slot] = die[i]
            cth[rsi, cache_write_slot] = dihs[i]

    result = {
        "cached_draft_input_ids": cdi,
        "cached_target_hidden_states": cth,
    }
    if cde is not None:
        result["cached_draft_input_embeds"] = cde
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
    device: str,
    block_size: int = 8,
) -> dict[str, Any]:
    torch.manual_seed(42 + num_reqs * 7 + hidden_size + num_speculative_steps)

    cache_window_size = num_speculative_steps - 1

    # idx_mapping: identity, or with negatives for skip tests
    if scenario == "skip_padded":
        idx_mapping = torch.arange(num_reqs, dtype=torch.int64,
                                   device=device)
        # Make last request a cudagraph padded slot
        idx_mapping[-1] = -1
    else:
        idx_mapping = torch.arange(num_reqs, dtype=torch.int64,
                                   device=device)

    # Build query_start_loc and last_token_indices
    if scenario == "full_window":
        # query_len >= cache_window_size, so full window is cached
        min_ql = cache_window_size + 1
        query_lens = torch.randint(min_ql, min_ql + 5, (num_reqs,),
                                   dtype=torch.int32, device=device)
    elif scenario == "short_query":
        # query_len < cache_window_size (partial window)
        max_ql = max(1, cache_window_size - 1)
        query_lens = torch.randint(1, max_ql + 1, (num_reqs,),
                                   dtype=torch.int32, device=device)
    elif scenario == "exact_window":
        # query_len == cache_window_size exactly
        query_lens = torch.full((num_reqs,), cache_window_size,
                                dtype=torch.int32, device=device)
    elif scenario == "skip_padded":
        query_lens = torch.randint(cache_window_size + 1,
                                   cache_window_size + 5, (num_reqs,),
                                   dtype=torch.int32, device=device)
    else:  # mixed
        query_lens = torch.randint(1, cache_window_size + 5, (num_reqs,),
                                   dtype=torch.int32, device=device)

    query_start_loc = torch.zeros(max_num_reqs + 1, dtype=torch.int32,
                                  device=device)
    qsl_cpu = query_start_loc.cpu()
    qsl_cpu[0] = 0
    for r in range(num_reqs):
        qsl_cpu[r + 1] = qsl_cpu[r] + int(query_lens[r].item())
    qsl_cpu[num_reqs + 1:] = -1
    query_start_loc.copy_(qsl_cpu)

    # last_token_indices: query_end - 1 for all
    last_token_indices = (query_start_loc[1:num_reqs + 1] - 1).to(torch.int32)

    # Draft buffers: random data
    draft_input_ids = torch.randint(0, 1000, (num_tokens,),
                                    dtype=torch.int32, device=device)
    draft_input_hidden_states = torch.randn(num_tokens, hidden_size,
                                            dtype=torch.float32,
                                            device=device)
    if use_embeds:
        draft_input_embeds = torch.randn(num_tokens, hidden_size,
                                         dtype=torch.float32, device=device)
    else:
        draft_input_embeds = None

    # Cached buffers: pre-fill with sentinels
    cached_draft_input_ids = torch.full(
        (max_num_reqs, max(1, cache_window_size)), _SENTINEL_ID,
        dtype=torch.int64, device=device)
    cached_target_hidden_states = torch.full(
        (max_num_reqs, max(1, cache_window_size), hidden_size), _SENTINEL_FP,
        dtype=torch.float32, device=device)
    if use_embeds:
        cached_draft_input_embeds = torch.full(
            (max_num_reqs, max(1, cache_window_size), hidden_size),
            _SENTINEL_FP, dtype=torch.float32, device=device)
    else:
        cached_draft_input_embeds = None

    return {
        "num_reqs": num_reqs,
        "max_num_reqs": max_num_reqs,
        "num_tokens": num_tokens,
        "hidden_size": hidden_size,
        "num_speculative_steps": num_speculative_steps,
        "idx_mapping": idx_mapping,
        "query_start_loc": query_start_loc,
        "last_token_indices": last_token_indices,
        "draft_input_ids": draft_input_ids,
        "draft_input_embeds": draft_input_embeds,
        "draft_input_hidden_states": draft_input_hidden_states,
        "cached_draft_input_ids": cached_draft_input_ids,
        "cached_draft_input_embeds": cached_draft_input_embeds,
        "cached_target_hidden_states": cached_target_hidden_states,
        "use_embeds": use_embeds,
        "block_size": block_size,
    }


def _run_kernel(inputs: dict[str, Any], device: str) -> dict[str, torch.Tensor]:
    import triton
    hs = inputs["hidden_size"]
    nss = inputs["num_speculative_steps"]
    use_embeds = inputs["use_embeds"]
    bs = inputs["block_size"]

    cdi = inputs["cached_draft_input_ids"].clone()
    cde = (inputs["cached_draft_input_embeds"].clone()
           if inputs["cached_draft_input_embeds"] is not None else None)
    cth = inputs["cached_target_hidden_states"].clone()
    die = inputs["draft_input_embeds"]

    grid = (inputs["num_reqs"], triton.cdiv(hs, bs))
    kernel[grid](
        inputs["draft_input_ids"],
        die,
        die.stride(0) if die is not None else 0,
        inputs["draft_input_hidden_states"],
        inputs["draft_input_hidden_states"].stride(0),
        cdi,
        cdi.stride(0),
        cde,
        cde.stride(0) if cde is not None else 0,
        cde.stride(1) if cde is not None else 0,
        cth,
        cth.stride(0),
        cth.stride(1),
        inputs["idx_mapping"],
        inputs["last_token_indices"],
        inputs["query_start_loc"],
        nss,
        hs,
        BLOCK_SIZE=bs,
        USE_INPUT_EMBEDS=use_embeds,
        num_warps=1,
    )
    synchronize()
    result = {
        "cached_draft_input_ids": cdi,
        "cached_target_hidden_states": cth,
    }
    if cde is not None:
        result["cached_draft_input_embeds"] = cde
    return result


def _assert_bitwise(name: str, expected: torch.Tensor,
                    actual: torch.Tensor) -> None:
    exp_cpu = expected.cpu()
    act_cpu = actual.cpu()
    if not torch.equal(exp_cpu, act_cpu):
        diffs = (exp_cpu != act_cpu).nonzero()
        max_show = min(10, diffs.shape[0])
        detail = []
        for i in range(max_show):
            idx = diffs[i].tolist()
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
    "num_reqs,hidden_size,num_speculative_steps,scenario,use_embeds,block_size",
    [
        # Full window (query_len >= cache_window_size)
        (1, 16, 2, "full_window", False, 8),
        (2, 32, 3, "full_window", False, 8),
        (2, 32, 3, "full_window", True, 8),
        # Short query (partial window, some cache slots stay sentinel)
        (1, 16, 3, "short_query", False, 8),
        (2, 32, 4, "short_query", True, 8),
        # Exact window (query_len == cache_window_size)
        (1, 16, 3, "exact_window", False, 8),
        (2, 32, 4, "exact_window", True, 8),
        # Skip cudagraph padded requests (idx_mapping < 0)
        (4, 32, 3, "skip_padded", False, 8),
        (4, 32, 3, "skip_padded", True, 8),
        # Mixed
        (4, 64, 3, "mixed", False, 8),
        (4, 64, 3, "mixed", True, 8),
        # Hidden_size tile boundary
        (1, 8, 2, "full_window", False, 8),   # H = BLOCK_SIZE
        (1, 9, 2, "full_window", False, 8),   # H = BLOCK_SIZE + 1
        (1, 7, 2, "full_window", False, 8),   # H = BLOCK_SIZE - 1
        # Larger num_speculative_steps
        (2, 32, 5, "full_window", True, 8),
        # Realistic block size
        (2, 1024, 3, "full_window", True, 1024),
    ],
    ids=lambda v: str(v),
)
def test_cache_inputs(
    num_reqs: int,
    hidden_size: int,
    num_speculative_steps: int,
    scenario: str,
    use_embeds: bool,
    block_size: int,
) -> None:
    """Bitwise comparison: pure copy/snapshot, no float arithmetic."""
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
        num_speculative_steps=num_speculative_steps,
        scenario=scenario,
        use_embeds=use_embeds,
        device=_STRICT_DEVICE,
        block_size=block_size,
    )

    expected = _ref(
        num_reqs=inputs["num_reqs"],
        num_speculative_steps=inputs["num_speculative_steps"],
        hidden_size=inputs["hidden_size"],
        idx_mapping=inputs["idx_mapping"],
        query_start_loc=inputs["query_start_loc"],
        last_token_indices=inputs["last_token_indices"],
        draft_input_ids=inputs["draft_input_ids"],
        draft_input_embeds=inputs["draft_input_embeds"],
        draft_input_hidden_states=inputs["draft_input_hidden_states"],
        cached_draft_input_ids=inputs["cached_draft_input_ids"],
        cached_draft_input_embeds=inputs["cached_draft_input_embeds"],
        cached_target_hidden_states=inputs["cached_target_hidden_states"],
    )

    actual = _run_kernel(inputs, _STRICT_DEVICE)

    _assert_bitwise("cached_draft_input_ids",
                    expected["cached_draft_input_ids"],
                    actual["cached_draft_input_ids"])
    _assert_bitwise("cached_target_hidden_states",
                    expected["cached_target_hidden_states"],
                    actual["cached_target_hidden_states"])
    if "cached_draft_input_embeds" in expected:
        _assert_bitwise("cached_draft_input_embeds",
                        expected["cached_draft_input_embeds"],
                        actual["cached_draft_input_embeds"])


def test_import_error() -> None:
    if _import_error is not None:
        pytest.fail(
            f"Failed to import _cache_inputs_kernel:\n{_import_traceback}"
        )
