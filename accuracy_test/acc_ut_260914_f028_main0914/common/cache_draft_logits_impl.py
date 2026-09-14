# SPDX-License-Identifier: Apache-2.0
# acc_ut_260914_f028_main0914 shared UT for _cache_draft_logits_kernel.
# Source: vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py
# Category: sparse->dense scatter + incremental cleanup. The kernel
# scatters top-k candidate scores into a dense draft_logits matrix and
# clears the previous round's cached candidate positions.
#
# Kernel signature:
#   @triton.jit
#   def _cache_draft_logits_kernel(
#       draft_logits_ptr,         # [max_num_reqs, num_steps, vocab_size] fp32
#       cached_candidate_ptr,     # [max_num_reqs, num_steps, top_k] int64
#       candidate_ptr,            # [num_reqs * num_steps, top_k] int64
#       scores_ptr,               # [num_reqs * num_steps, top_k] fp32
#       req_state_ptr,            # [num_reqs * num_steps] -> req_state_idx (-1 skip)
#       draft_logits_stride_0,
#       draft_logits_stride_1,
#       num_steps: tl.constexpr,
#       top_k: tl.constexpr,
#       BLOCK_K: tl.constexpr,
#   )
#
# Grid: (num_sample,) where num_sample = num_reqs * num_steps.
# Per program (flat):
#   1. Load old cached token ids, set their logits positions to -inf
#   2. Load new token ids + scores, store scores at their positions
#   3. Update cached_candidate_ptr with new token ids
#
# All outputs are fp32 / int64 -> bitwise exact (no compute, pure scatter).

from __future__ import annotations

import traceback
from typing import Any

import pytest
import torch

from vllm.triton_utils import tl, triton

kernel = None
_import_error: Exception | None = None
_import_traceback: str | None = None
try:
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
        _cache_draft_logits_kernel as kernel,
    )
except Exception as exc:  # pragma: no cover
    _import_error = exc
    _import_traceback = traceback.format_exc()


def _ref(
    draft_logits: torch.Tensor,        # [max_num_reqs, num_steps, vocab_size]
    cached_candidates: torch.Tensor,    # [max_num_reqs, num_steps, top_k]
    candidates: torch.Tensor,           # [num_sample, top_k]
    scores: torch.Tensor,               # [num_sample, top_k]
    req_state: torch.Tensor,            # [num_sample]
    num_steps: int,
    top_k: int,
    max_num_reqs: int,
    vocab_size: int,
    num_sample: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference for the cache_draft_logits kernel.

    Returns (draft_logits, cached_candidates) — both modified in-place.
    """
    dl = draft_logits.clone()
    cc = cached_candidates.clone()

    for flat in range(num_sample):
        rs = int(req_state[flat].item())
        step = flat % num_steps
        if rs < 0:
            continue

        # 1. Clear old cached positions
        old_ids = cc[rs, step, :].tolist()
        for tid in old_ids:
            dl[rs, step, tid] = float("-inf")

        # 2. Store new scores at new positions
        new_ids = candidates[flat, :].tolist()
        new_scores = scores[flat, :].tolist()
        for i, tid in enumerate(new_ids):
            dl[rs, step, tid] = new_scores[i]

        # 3. Update cached candidates
        cc[rs, step, :] = candidates[flat, :]

    return dl, cc


def _gen_inputs(
    num_reqs: int,
    num_steps: int,
    top_k: int,
    max_num_reqs: int,
    vocab_size: int,
    *,
    with_invalid: bool,
    has_cached: bool,
    device,
) -> dict[str, Any]:
    """Build inputs for cache_draft_logits test.

    Branches:
      - with_invalid: some req_state entries are -1 (skip)
      - has_cached: cached_candidates has non-zero entries from a previous round
    """
    torch.manual_seed(42 + num_reqs * 7 + num_steps * 3 + top_k * 11)

    num_sample = num_reqs * num_steps

    # Draft logits: initialized to -inf (per spec: "fp32 so the walk and the
    # rejection that checks it read the same distribution; -inf because the
    # cache kernel writes only the K candidates.")
    draft_logits = torch.full(
        (max_num_reqs, num_steps, vocab_size),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )

    # Cached candidates: zeros by default (first round)
    cached_candidates = torch.zeros(
        (max_num_reqs, num_steps, top_k),
        dtype=torch.int64,
        device=device,
    )

    if has_cached:
        # Simulate a previous round: some positions already have scores
        # and cached_candidates has old token ids
        for flat in range(num_sample):
            # Assign a req_state for setup
            if with_invalid and flat % 4 == 3:
                continue
            rs = flat // num_steps
            step = flat % num_steps
            old_ids = torch.randint(0, vocab_size, (top_k,), device=device)
            cached_candidates[rs, step, :] = old_ids
            # Set old scores at old positions (so kernel needs to clear them)
            old_scores = torch.randn(top_k, dtype=torch.float32, device=device)
            for i in range(top_k):
                draft_logits[rs, step, int(old_ids[i].item())] = old_scores[i].item()

    # New candidates: random token ids in [0, vocab_size)
    candidates = torch.randint(
        0, vocab_size, (num_sample, top_k), dtype=torch.int64, device=device
    )

    # New scores: random floats
    scores = torch.randn(num_sample, top_k, dtype=torch.float32, device=device)

    # req_state: maps flat index -> req_state_idx
    req_state = torch.zeros(num_sample, dtype=torch.int64, device=device)
    for flat in range(num_sample):
        if with_invalid and flat % 4 == 3:
            req_state[flat] = -1
        else:
            req_state[flat] = flat // num_steps

    return {
        "draft_logits": draft_logits,
        "cached_candidates": cached_candidates,
        "candidates": candidates,
        "scores": scores,
        "req_state": req_state,
        "num_steps": num_steps,
        "top_k": top_k,
        "max_num_reqs": max_num_reqs,
        "vocab_size": vocab_size,
        "num_sample": num_sample,
    }


def _launch(k, inputs: dict[str, Any]) -> dict[str, Any]:
    num_sample = inputs["num_sample"]
    top_k = inputs["top_k"]
    block_k = triton.next_power_of_2(top_k)
    draft_logits = inputs["draft_logits"].clone()
    cached = inputs["cached_candidates"].clone()

    k[(num_sample,)](
        draft_logits,
        cached,
        inputs["candidates"].contiguous(),
        inputs["scores"].contiguous(),
        inputs["req_state"].contiguous(),
        draft_logits.stride(0),
        draft_logits.stride(1),
        num_steps=inputs["num_steps"],
        top_k=top_k,
        BLOCK_K=block_k,
        num_warps=1,
    )

    return {"draft_logits": draft_logits, "cached_candidates": cached}


# Realistic shapes for DFlash2:
#   - max_num_reqs: production concurrency, 256/512
#   - num_steps: num_speculative_steps, 1/2/4/8
#   - top_k: selector_top_k, 4/8/16/32
#   - vocab_size: model vocab, 128000 (Llama) / 152064 (Qwen3)
# For test performance, use smaller vocab but cover the range
SHAPE_PARAMS = [
    # (num_reqs, num_steps, top_k, max_num_reqs, vocab_size)
    (1, 1, 4, 4, 128),
    (1, 2, 4, 4, 128),
    (4, 1, 4, 8, 256),
    (4, 2, 8, 8, 256),
    (8, 1, 8, 16, 512),
    (8, 2, 8, 16, 512),
    (8, 4, 16, 16, 512),
    (16, 2, 8, 32, 1024),
    (16, 4, 4, 32, 1024),
    (32, 2, 8, 64, 1024),
    (32, 4, 16, 64, 2048),
    (64, 1, 4, 128, 256),
    (64, 2, 8, 128, 256),
    (128, 1, 4, 256, 512),
    # Edge: top_k=1 (degenerate)
    (4, 2, 1, 8, 128),
    # Edge: single step, large top_k
    (4, 1, 32, 8, 256),
    # Edge: num_steps > max_num_reqs (multiple steps per req)
    (2, 8, 4, 4, 128),
    # Edge: vocab_size < top_k (every candidate appears)
    (4, 1, 8, 8, 8),
]

BRANCH_PARAMS = [
    # (with_invalid, has_cached)
    (False, False),  # first round, all valid
    (False, True),   # second round, all valid -> test incremental cleanup
    (True, False),    # first round, some invalid
    (True, True),     # second round, some invalid -> cleanup skips invalid
]


@pytest.mark.parametrize("num_reqs,num_steps,top_k,max_num_reqs,vocab_size", SHAPE_PARAMS)
@pytest.mark.parametrize("with_invalid,has_cached", BRANCH_PARAMS)
def test_cache_draft_logits(
    num_reqs: int,
    num_steps: int,
    top_k: int,
    max_num_reqs: int,
    vocab_size: int,
    with_invalid: bool,
    has_cached: bool,
    rt,
):
    """Compare kernel against independent CPU reference.

    The kernel is a pure scatter (no compute beyond -inf assignment and
    score copy). Pass requires:
      - draft_logits: bitwise equality (fp32, including -inf positions)
      - cached_candidates: bitwise equality (int64)
    """
    if kernel is None:
        pytest.fail(
            "_cache_draft_logits_kernel import failed.\n"
            f"error={_import_error}\ntraceback:\n{_import_traceback}",
            pytrace=False,
        )

    rt.init_device_properties_triton()
    device = rt.STRICT_DEVICE

    inputs = _gen_inputs(
        num_reqs=num_reqs,
        num_steps=num_steps,
        top_k=top_k,
        max_num_reqs=max_num_reqs,
        vocab_size=vocab_size,
        with_invalid=with_invalid,
        has_cached=has_cached,
        device=device,
    )

    # CPU reference
    expected_dl, expected_cc = _ref(
        draft_logits=inputs["draft_logits"].cpu(),
        cached_candidates=inputs["cached_candidates"].cpu(),
        candidates=inputs["candidates"].cpu(),
        scores=inputs["scores"].cpu(),
        req_state=inputs["req_state"].cpu(),
        num_steps=num_steps,
        top_k=top_k,
        max_num_reqs=max_num_reqs,
        vocab_size=vocab_size,
        num_sample=inputs["num_sample"],
    )

    # Device run
    result = _launch(kernel, inputs)
    rt.synchronize()

    # Compare draft_logits: must match exactly, including -inf positions
    dev_dl = result["draft_logits"].cpu()
    assert dev_dl.dtype == expected_dl.dtype, "draft_logits dtype mismatch"
    assert dev_dl.shape == expected_dl.shape, "draft_logits shape mismatch"

    # Check special values (NaN/inf pattern) match
    from accuracy_test.acc_ut_260914_f028_main0914.metrics import assert_special_values
    assert_special_values(dev_dl, expected_dl)

    # Check finite values match
    finite_mask = torch.isfinite(expected_dl)
    if not torch.equal(dev_dl[finite_mask], expected_dl[finite_mask]):
        mismatched = torch.ne(dev_dl[finite_mask], expected_dl[finite_mask])
        count = int(mismatched.sum().item())
        total = int(mismatched.numel())
        pytest.fail(
            f"_cache_draft_logits_kernel draft_logits mismatch: "
            f"{count}/{total} finite elements differ"
        )

    # Check -inf pattern matches (old positions cleared, new positions set)
    neginf_dev = torch.isneginf(dev_dl)
    neginf_ref = torch.isneginf(expected_dl)
    if not torch.equal(neginf_dev, neginf_ref):
        mismatched = torch.ne(neginf_dev, neginf_ref)
        count = int(mismatched.sum().item())
        pytest.fail(
            f"_cache_draft_logits_kernel -inf pattern mismatch: "
            f"{count} positions differ"
        )

    # Compare cached_candidates
    dev_cc = result["cached_candidates"].cpu()
    assert dev_cc.dtype == expected_cc.dtype, "cached_candidates dtype mismatch"
    assert dev_cc.shape == expected_cc.shape, "cached_candidates shape mismatch"
    if not torch.equal(dev_cc, expected_cc):
        mismatched = torch.ne(dev_cc, expected_cc)
        count = int(mismatched.sum().item())
        total = int(mismatched.numel())
        first_idx = torch.nonzero(mismatched, as_tuple=False)
        first_info = ""
        if first_idx.numel() > 0:
            loc = tuple(first_idx[0].tolist())
            first_info = (
                f"; first mismatch at {loc}: "
                f"device={dev_cc[loc].item()} ref={expected_cc[loc].item()}"
            )
        pytest.fail(
            f"_cache_draft_logits_kernel cached_candidates mismatch: "
            f"{count}/{total} elements differ{first_info}"
        )


def test_cache_draft_logits_first_round(rt):
    """First round (cached_candidates all zero): clears token 0, sets new scores."""
    if kernel is None:
        pytest.fail("kernel import failed", pytrace=False)

    rt.init_device_properties_triton()
    device = rt.STRICT_DEVICE

    num_reqs = 1
    num_steps = 1
    top_k = 2
    max_num_reqs = 1
    vocab_size = 8
    num_sample = 1

    draft_logits = torch.full(
        (max_num_reqs, num_steps, vocab_size),
        float("-inf"),
        dtype=torch.float32, device=device,
    )
    cached = torch.zeros(
        (max_num_reqs, num_steps, top_k),
        dtype=torch.int64, device=device,
    )
    candidates = torch.tensor([[3, 5]], dtype=torch.int64, device=device)
    scores = torch.tensor([[1.0, 2.0]], dtype=torch.float32, device=device)
    req_state = torch.tensor([0], dtype=torch.int64, device=device)

    inputs = {
        "draft_logits": draft_logits,
        "cached_candidates": cached,
        "candidates": candidates,
        "scores": scores,
        "req_state": req_state,
        "num_steps": num_steps, "top_k": top_k,
        "num_sample": num_sample,
    }

    result = _launch(kernel, inputs)
    rt.synchronize()

    dl = result["draft_logits"][0, 0, :].cpu()
    # Token 0 should be cleared (was already -inf, so no change)
    assert dl[0].item() == float("-inf"), "token 0 should be -inf"
    # Token 3 should have score 1.0
    assert dl[3].item() == 1.0, f"token 3 should be 1.0, got {dl[3].item()}"
    # Token 5 should have score 2.0
    assert dl[5].item() == 2.0, f"token 5 should be 2.0, got {dl[5].item()}"
    # Others should be -inf
    for i in [1, 2, 4, 6, 7]:
        assert dl[i].item() == float("-inf"), f"token {i} should be -inf"

    # Cached should be updated
    cc = result["cached_candidates"][0, 0, :].cpu()
    assert cc[0].item() == 3, f"cached[0] should be 3, got {cc[0].item()}"
    assert cc[1].item() == 5, f"cached[1] should be 5, got {cc[1].item()}"


def test_cache_draft_logits_incremental_cleanup(rt):
    """Second round: old positions cleared, new positions set, no residue."""
    if kernel is None:
        pytest.fail("kernel import failed", pytrace=False)

    rt.init_device_properties_triton()
    device = rt.STRICT_DEVICE

    num_reqs = 1
    num_steps = 1
    top_k = 2
    max_num_reqs = 1
    vocab_size = 8
    num_sample = 1

    # Simulate first round already happened
    draft_logits = torch.full(
        (max_num_reqs, num_steps, vocab_size),
        float("-inf"),
        dtype=torch.float32, device=device,
    )
    # Old round cached tokens: 1 and 3
    draft_logits[0, 0, 1] = 0.5
    draft_logits[0, 0, 3] = 0.8
    cached = torch.tensor([[[1, 3]]], dtype=torch.int64, device=device)

    # New round: tokens 3 and 5
    candidates = torch.tensor([[3, 5]], dtype=torch.int64, device=device)
    scores = torch.tensor([[1.0, 2.0]], dtype=torch.float32, device=device)
    req_state = torch.tensor([0], dtype=torch.int64, device=device)

    inputs = {
        "draft_logits": draft_logits,
        "cached_candidates": cached,
        "candidates": candidates,
        "scores": scores,
        "req_state": req_state,
        "num_steps": num_steps, "top_k": top_k,
        "num_sample": num_sample,
    }

    result = _launch(kernel, inputs)
    rt.synchronize()

    dl = result["draft_logits"][0, 0, :].cpu()

    # Old token 1: should be cleared to -inf
    assert dl[1].item() == float("-inf"), (
        f"old token 1 should be cleared to -inf, got {dl[1].item()}"
    )
    # Old token 3: should be cleared then overwritten with new score 1.0
    # (clear happens first, then new store)
    assert dl[3].item() == 1.0, (
        f"token 3 (reused) should have new score 1.0, got {dl[3].item()}"
    )
    # New token 5: should have score 2.0
    assert dl[5].item() == 2.0, (
        f"new token 5 should be 2.0, got {dl[5].item()}"
    )
    # Old token 3's old score (0.8) should NOT survive
    # (it's overwritten by 1.0 in the new round)

    # Cached should be updated to new tokens
    cc = result["cached_candidates"][0, 0, :].cpu()
    assert cc[0].item() == 3 and cc[1].item() == 5, (
        f"cached should be [3, 5], got {cc.tolist()}"
    )


def test_cache_draft_logits_skip_invalid(rt):
    """Invalid requests (req_state=-1): no writes, no cleanup."""
    if kernel is None:
        pytest.fail("kernel import failed", pytrace=False)

    rt.init_device_properties_triton()
    device = rt.STRICT_DEVICE

    num_reqs = 2
    num_steps = 1
    top_k = 2
    max_num_reqs = 4
    vocab_size = 8
    num_sample = 2

    draft_logits = torch.full(
        (max_num_reqs, num_steps, vocab_size),
        float("-inf"),
        dtype=torch.float32, device=device,
    )
    # Set some pre-existing values for req 0
    draft_logits[0, 0, 1] = 0.5
    cached = torch.zeros(
        (max_num_reqs, num_steps, top_k),
        dtype=torch.int64, device=device,
    )
    cached[0, 0, :] = torch.tensor([1, 2], device=device)

    candidates = torch.tensor([[3, 5], [7, 6]], dtype=torch.int64, device=device)
    scores = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=device)
    # req 0 valid, req 1 invalid
    req_state = torch.tensor([0, -1], dtype=torch.int64, device=device)

    inputs = {
        "draft_logits": draft_logits,
        "cached_candidates": cached,
        "candidates": candidates,
        "scores": scores,
        "req_state": req_state,
        "num_steps": num_steps, "top_k": top_k,
        "num_sample": num_sample,
    }

    # Record state before kernel for invalid req
    # req_state[1] = -1 -> the kernel reads rs=-1 and skips
    # But req 1 maps to no req_state_idx, so nothing is written or cleared

    result = _launch(kernel, inputs)
    rt.synchronize()

    dl = result["draft_logits"].cpu()

    # req 0: old token 1 cleared to -inf, new tokens 3/5 set
    assert dl[0, 0, 1].item() == float("-inf"), "old token 1 should be cleared"
    assert dl[0, 0, 3].item() == 1.0, "new token 3 should be 1.0"
    assert dl[0, 0, 5].item() == 2.0, "new token 5 should be 2.0"

    # req 1 (invalid): no changes, all -inf (no pre-existing values either)
    for v in range(vocab_size):
        assert dl[1, 0, v].item() == float("-inf"), (
            f"invalid req 1, token {v} should be -inf, got {dl[1,0,v].item()}"
        )

    # req 1 cached should also be unchanged (0)
    cc = result["cached_candidates"].cpu()
    assert torch.all(cc[1, 0, :] == 0), "invalid req cached should stay 0"


def test_import_error() -> None:
    if _import_error is not None:
        pytest.fail(
            f"Failed to import _cache_draft_logits_kernel:\n"
            f"{_import_traceback}"
        )
