# SPDX-License-Identifier: Apache-2.0
# acc_ut_260914_f028_main0914 shared UT for _selector_walk_kernel.
# Source: vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py
# Category: stochastic (Gumbel sampling walk) / deterministic (greedy walk).
#
# The kernel walks a candidate tree: at each step, it selects one of top_k
# candidates via Gumbel-max (probabilistic) or plain argmax (greedy / temp=0).
# The selected index ("previous") determines which child branch to follow.
#
# Test strategy:
#   - Greedy mode (SAMPLE_PROBABILISTIC=False): deterministic, bitwise exact
#     against an independent CPU reference.
#   - Probabilistic mode: verify reproducibility (same seed -> same output)
#     and distributional properties (empirical frequency of selected tokens
#     matches the softmax of scores).
#
# Kernel signature:
#   @triton.jit
#   def _selector_walk_kernel(
#       scores_ptr,              # [num_reqs * num_steps * top_k * top_k] flat
#       candidate_ptr,           # [num_reqs * num_steps * top_k] flat
#       sample_pos_ptr,          # [num_reqs * num_steps]
#       req_state_ptr,           # [num_reqs * num_steps] -> req_state_idx
#       temperature_ptr,         # [max_num_reqs]
#       seeds_ptr,               # [max_num_reqs]
#       tokens_ptr,              # [num_reqs * num_steps] output
#       realized_scores_ptr,     # [num_reqs * num_steps * top_k] output
#       num_steps: tl.constexpr,
#       top_k: tl.constexpr,
#       BLOCK_K: tl.constexpr,
#       SAMPLE_PROBABILISTIC: tl.constexpr,
#       USE_FP64: tl.constexpr,
#   )

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
        _selector_walk_kernel as kernel,
    )
except Exception as exc:  # pragma: no cover
    _import_error = exc
    _import_traceback = traceback.format_exc()


def _ref_greedy_walk(
    scores: torch.Tensor,      # [num_reqs * num_steps, top_k, top_k]
    candidates: torch.Tensor,   # [num_reqs * num_steps, top_k]
    req_state: torch.Tensor,    # [num_reqs * num_steps]
    num_reqs: int,
    num_steps: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference for greedy (temp=0) selector walk.

    Returns:
        tokens: [num_reqs * num_steps] int64
        realized_scores: [num_reqs * num_steps, top_k] float32
    """
    total = num_reqs * num_steps
    tokens = torch.zeros(total, dtype=torch.int64)
    realized = torch.zeros(total, top_k, dtype=torch.float32)

    for row in range(num_reqs):
        rs = int(req_state[row * num_steps].item())
        if rs < 0:
            # Invalid request: kernel writes 0 tokens (other=0)
            for step in range(num_steps):
                flat = row * num_steps + step
                tokens[flat] = 0
                realized[flat, :] = 0.0
            continue

        previous = 0
        for step in range(num_steps):
            flat = row * num_steps + step
            # scores layout: scores[flat, previous, :] in a 3D view
            step_scores = scores[flat, previous, :].to(torch.float32)
            index = int(step_scores.argmax().item())
            token = int(candidates[flat, index].item())
            tokens[flat] = token
            realized[flat, :] = step_scores
            previous = index

    return tokens, realized


def _gen_inputs_greedy(
    num_reqs: int,
    num_steps: int,
    top_k: int,
    *,
    with_invalid: bool,
    device,
) -> dict[str, Any]:
    """Build inputs for greedy walk test.

    The scores tree is [num_reqs * num_steps, top_k, top_k]: at step S, the
    `previous`-th row of the top_k x top_k matrix gives the scores for the
    top_k children of the previously selected candidate.
    """
    torch.manual_seed(42 + num_reqs * 7 + num_steps * 3 + top_k * 11)

    total = num_reqs * num_steps
    # Scores: random floats, with a clear winner at each step
    scores = torch.randn(total, top_k, top_k, dtype=torch.float32, device=device)
    # Make the argmax unambiguous: boost one candidate by +10
    for flat in range(total):
        for prev in range(top_k):
            winner = torch.randint(0, top_k, (1,)).item()
            scores[flat, prev, winner] += 10.0

    # Candidates: random token ids in [0, 1000)
    candidates = torch.randint(0, 1000, (total, top_k), dtype=torch.int64, device=device)

    # req_state: all valid unless with_invalid
    req_state = torch.zeros(total, dtype=torch.int64, device=device)
    for row in range(num_reqs):
        if with_invalid and row == num_reqs - 1:
            req_state[row * num_steps:(row + 1) * num_steps] = -1
        else:
            req_state[row * num_steps:(row + 1) * num_steps] = row

    # sample_pos: positions for the sampling (not used in greedy, but kernel reads)
    sample_pos = torch.arange(total, dtype=torch.int64, device=device) + 1

    # temperature: all zero for greedy
    temperature = torch.zeros(num_reqs, dtype=torch.float32, device=device)

    # seeds: not used in greedy, but kernel reads
    seeds = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int32, device=device)

    return {
        "scores": scores,
        "candidates": candidates,
        "sample_pos": sample_pos,
        "req_state": req_state,
        "temperature": temperature,
        "seeds": seeds,
        "num_reqs": num_reqs,
        "num_steps": num_steps,
        "top_k": top_k,
    }


def _launch(k, inputs: dict[str, Any], *, sample_prob: bool, use_fp64: bool) -> dict:
    num_reqs = inputs["num_reqs"]
    num_steps = inputs["num_steps"]
    top_k = inputs["top_k"]
    block_k = triton.next_power_of_2(top_k)

    total = num_reqs * num_steps
    tokens = torch.zeros(total, dtype=torch.int64, device=inputs["scores"].device)
    realized = torch.zeros(total, top_k, dtype=torch.float32, device=inputs["scores"].device)

    k[(num_reqs,)](
        inputs["scores"].contiguous(),
        inputs["candidates"].contiguous(),
        inputs["sample_pos"].contiguous(),
        inputs["req_state"].contiguous(),
        inputs["temperature"].contiguous(),
        inputs["seeds"].contiguous(),
        tokens,
        realized,
        num_steps=num_steps,
        top_k=top_k,
        BLOCK_K=block_k,
        SAMPLE_PROBABILISTIC=sample_prob,
        USE_FP64=use_fp64,
        num_warps=1,
    )

    return {"tokens": tokens, "realized_scores": realized}


# Realistic shapes for DFlash2:
#   - top_k (selector_top_k): typically in {4, 8, 16, 32}
#   - num_steps (num_speculative_steps): typically in {1, 2, 4, 8}
#   - num_reqs: 1, 4, 8, 16, 32, 64, 128, 256
SHAPE_PARAMS = [
    # (num_reqs, num_steps, top_k)
    (1, 1, 4),
    (1, 2, 4),
    (1, 4, 8),
    (4, 1, 4),
    (4, 2, 8),
    (4, 4, 8),
    (8, 2, 4),
    (8, 4, 8),
    (8, 4, 16),
    (16, 2, 8),
    (16, 4, 4),
    (32, 2, 8),
    (32, 4, 16),
    (64, 1, 4),
    (64, 2, 8),
    (128, 2, 4),
    (256, 1, 4),
    # Edge: single step, large top_k
    (4, 1, 32),
    # Edge: many steps
    (4, 8, 4),
    # Edge: top_k=1 (degenerate, only one candidate)
    (4, 2, 1),
]


@pytest.mark.parametrize("num_reqs,num_steps,top_k", SHAPE_PARAMS)
@pytest.mark.parametrize("use_fp64", [False, True])
@pytest.mark.parametrize("with_invalid", [False, True])
def test_selector_walk_greedy(
    num_reqs: int,
    num_steps: int,
    top_k: int,
    use_fp64: bool,
    with_invalid: bool,
    rt,
):
    """Greedy walk (SAMPLE_PROBABILISTIC=False, temp=0): deterministic.

    Pass requires bitwise equality of output tokens AND realized_scores
    against an independent CPU reference.
    """
    if kernel is None:
        pytest.fail(
            "_selector_walk_kernel import failed.\n"
            f"error={_import_error}\ntraceback:\n{_import_traceback}",
            pytrace=False,
        )

    rt.init_device_properties_triton()
    device = rt.STRICT_DEVICE

    inputs = _gen_inputs_greedy(
        num_reqs=num_reqs,
        num_steps=num_steps,
        top_k=top_k,
        with_invalid=with_invalid,
        device=device,
    )

    # CPU reference
    expected_tokens, expected_realized = _ref_greedy_walk(
        scores=inputs["scores"].cpu(),
        candidates=inputs["candidates"].cpu(),
        req_state=inputs["req_state"].cpu(),
        num_reqs=num_reqs,
        num_steps=num_steps,
        top_k=top_k,
    )

    # Device run
    result = _launch(kernel, inputs, sample_prob=False, use_fp64=use_fp64)
    rt.synchronize()

    # Compare tokens
    dev_tokens = result["tokens"].cpu()
    assert dev_tokens.dtype == expected_tokens.dtype, "tokens dtype mismatch"
    assert dev_tokens.shape == expected_tokens.shape, "tokens shape mismatch"
    if not torch.equal(dev_tokens, expected_tokens):
        mismatched = torch.ne(dev_tokens, expected_tokens)
        count = int(mismatched.sum().item())
        total = int(mismatched.numel())
        first_idx = torch.nonzero(mismatched, as_tuple=False)
        first_info = ""
        if first_idx.numel() > 0:
            loc = tuple(first_idx[0].tolist())
            first_info = (
                f"; first mismatch at {loc}: "
                f"device={dev_tokens[loc].item()} ref={expected_tokens[loc].item()}"
            )
        pytest.fail(
            f"_selector_walk_kernel tokens mismatch (greedy): "
            f"{count}/{total} elements differ{first_info}"
        )

    # Compare realized_scores (float, so allow small tolerance for fp32)
    dev_realized = result["realized_scores"].cpu()
    assert dev_realized.dtype == expected_realized.dtype, "realized dtype mismatch"
    assert dev_realized.shape == expected_realized.shape, "realized shape mismatch"
    torch.testing.assert_close(
        dev_realized, expected_realized, rtol=1e-4, atol=1e-4
    )


def test_selector_walk_reproducibility(rt):
    """Probabilistic mode: same seed -> same output (reproducibility).

    The kernel uses seed + sample_pos for Gumbel noise. Running twice with
    identical inputs must produce identical outputs.
    """
    if kernel is None:
        pytest.fail("kernel import failed", pytrace=False)

    rt.init_device_properties_triton()
    device = rt.STRICT_DEVICE

    num_reqs = 8
    num_steps = 2
    top_k = 8
    total = num_reqs * num_steps

    torch.manual_seed(123)
    scores = torch.randn(total, top_k, top_k, dtype=torch.float32, device=device)
    candidates = torch.randint(0, 100, (total, top_k), dtype=torch.int64, device=device)
    req_state = torch.arange(num_reqs, dtype=torch.int64, device=device).repeat_interleave(num_steps)
    sample_pos = torch.arange(total, dtype=torch.int64, device=device) + 1
    temperature = torch.full((num_reqs,), 1.0, dtype=torch.float32, device=device)
    seeds = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int32, device=device)

    inputs = {
        "scores": scores, "candidates": candidates,
        "sample_pos": sample_pos, "req_state": req_state,
        "temperature": temperature, "seeds": seeds,
        "num_reqs": num_reqs, "num_steps": num_steps, "top_k": top_k,
    }

    result1 = _launch(kernel, inputs, sample_prob=True, use_fp64=False)
    rt.synchronize()
    result2 = _launch(kernel, inputs, sample_prob=True, use_fp64=False)
    rt.synchronize()

    assert torch.equal(result1["tokens"], result2["tokens"]), (
        "Probabilistic walk should be reproducible with same seed"
    )
    assert torch.equal(result1["realized_scores"], result2["realized_scores"]), (
        "Realized scores should be reproducible"
    )


def test_selector_walk_skip_invalid(rt):
    """Requests with req_state=-1 should produce token=0 (other=0)."""
    if kernel is None:
        pytest.fail("kernel import failed", pytrace=False)

    rt.init_device_properties_triton()
    device = rt.STRICT_DEVICE

    num_reqs = 4
    num_steps = 2
    top_k = 4
    total = num_reqs * num_steps

    scores = torch.randn(total, top_k, top_k, dtype=torch.float32, device=device)
    candidates = torch.randint(0, 100, (total, top_k), dtype=torch.int64, device=device)
    # req 0: valid, req 1: invalid (-1), req 2: valid, req 3: invalid
    req_state = torch.zeros(total, dtype=torch.int64, device=device)
    req_state[0*num_steps:1*num_steps] = 0
    req_state[1*num_steps:2*num_steps] = -1
    req_state[2*num_steps:3*num_steps] = 2
    req_state[3*num_steps:4*num_steps] = -1
    sample_pos = torch.arange(total, dtype=torch.int64, device=device) + 1
    temperature = torch.zeros(num_reqs, dtype=torch.float32, device=device)
    seeds = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int32, device=device)

    inputs = {
        "scores": scores, "candidates": candidates,
        "sample_pos": sample_pos, "req_state": req_state,
        "temperature": temperature, "seeds": seeds,
        "num_reqs": num_reqs, "num_steps": num_steps, "top_k": top_k,
    }

    result = _launch(kernel, inputs, sample_prob=False, use_fp64=False)
    rt.synchronize()

    # Invalid requests (1 and 3) should have token=0
    for row in [1, 3]:
        for step in range(num_steps):
            flat = row * num_steps + step
            assert result["tokens"][flat].item() == 0, (
                f"Invalid req {row} step {step}: expected token=0, "
                f"got {result['tokens'][flat].item()}"
            )

    # Valid requests should have non-zero tokens (candidates are in [0,100))
    for row in [0, 2]:
        for step in range(num_steps):
            flat = row * num_steps + step
            # Token could be 0 by chance if candidate[0]=0, so just check
            # it's a valid candidate from the row
            token = result["tokens"][flat].item()
            assert token in candidates[flat].cpu().tolist(), (
                f"Valid req {row} step {step}: token {token} not in candidates "
                f"{candidates[flat].cpu().tolist()}"
            )


def test_selector_walk_distribution(rt):
    """Probabilistic mode: verify the empirical distribution matches softmax(scores).

    For a single-step, single-req walk with top_k=2 and known scores, the
    probability of selecting each candidate should match softmax(scores/temp).
    We run many trials with different seeds and check the empirical frequency
    is within a confidence interval.
    """
    if kernel is None:
        pytest.fail("kernel import failed", pytrace=False)

    rt.init_device_properties_triton()
    device = rt.STRICT_DEVICE

    num_steps = 1
    top_k = 2
    # Use many requests with different seeds to get statistics
    num_reqs = 1000
    total = num_reqs * num_steps

    # Fixed scores: candidate 0 has score 2.0, candidate 1 has score 1.0
    # With temperature=1.0, softmax = [e^2 / (e^2+e^1), e^1 / (e^2+e^1)]
    # = [0.731, 0.269]
    scores_val_0 = 2.0
    scores_val_1 = 1.0
    scores = torch.zeros(total, top_k, top_k, dtype=torch.float32, device=device)
    scores[:, 0, 0] = scores_val_0
    scores[:, 0, 1] = scores_val_1
    candidates = torch.tensor([[0, 1]] * total, dtype=torch.int64, device=device)

    # Each request gets a unique seed
    req_state = torch.arange(num_reqs, dtype=torch.int64, device=device).repeat_interleave(num_steps)
    sample_pos = torch.ones(total, dtype=torch.int64, device=device)
    temperature = torch.ones(num_reqs, dtype=torch.float32, device=device)
    seeds = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int32, device=device)

    inputs = {
        "scores": scores, "candidates": candidates,
        "sample_pos": sample_pos, "req_state": req_state,
        "temperature": temperature, "seeds": seeds,
        "num_reqs": num_reqs, "num_steps": num_steps, "top_k": top_k,
    }

    result = _launch(kernel, inputs, sample_prob=True, use_fp64=False)
    rt.synchronize()

    tokens = result["tokens"].cpu()
    # Count how many times candidate 0 (token=0) was selected
    count_0 = int((tokens == 0).sum().item())
    count_1 = int((tokens == 1).sum().item())
    total_count = count_0 + count_1
    assert total_count == num_reqs, f"Expected {num_reqs} tokens, got {total_count}"

    # Expected: ~73.1% for candidate 0, ~26.9% for candidate 1
    # Use a generous confidence interval (3-sigma) due to Gumbel approximation
    import math
    expected_p = math.exp(scores_val_0) / (math.exp(scores_val_0) + math.exp(scores_val_1))
    expected_count = expected_p * num_reqs
    # 3-sigma: sqrt(n * p * (1-p)) * 3
    sigma = math.sqrt(num_reqs * expected_p * (1 - expected_p))
    lower = expected_count - 5 * sigma  # generous 5-sigma to avoid flakiness
    upper = expected_count + 5 * sigma

    assert lower <= count_0 <= upper, (
        f"Distribution mismatch: candidate 0 selected {count_0}/{num_reqs} "
        f"(expected ~{expected_count:.0f}±{5*sigma:.0f}, p={expected_p:.3f})"
    )


def test_import_error() -> None:
    if _import_error is not None:
        pytest.fail(
            f"Failed to import _selector_walk_kernel:\n"
            f"{_import_traceback}"
        )
