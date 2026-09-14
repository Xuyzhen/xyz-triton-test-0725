# SPDX-License-Identifier: Apache-2.0
# acc_ut_260914_f028_main0914 shared UT for _fill_num_accepted_kernel.
# Source: vllm/v1/worker/gpu/model_states/mamba_hybrid.py
# Category: integer/index compute (broadcast scalar -> per-req GPU array).
# All outputs int32 -> bitwise exact.
#
# Kernel signature (from source):
#   @triton.jit
#   def _fill_num_accepted_kernel(
#       idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx (-1 to skip)
#       num_accepted_ptr,  # [max_num_reqs] output
#       num_sampled,       # scalar int: the accepted count to broadcast
#   )
#
# The kernel broadcasts a single int value (num_sampled) to num_accepted[req_state_idx]
# for every row where idx_mapping[row] >= 0. Rows with -1 are skipped.
# The caller clamps to max(num_sampled, 1) before passing.

from __future__ import annotations

import traceback

import pytest
import torch

kernel = None
_import_error: Exception | None = None
_import_traceback: str | None = None
try:
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
        _fill_num_accepted_kernel as kernel,
    )
except Exception as exc:  # pragma: no cover
    _import_error = exc
    _import_traceback = traceback.format_exc()


_SENTINEL = -2


def _ref(
    idx_mapping: torch.Tensor,
    num_sampled: int,
    max_num_reqs: int,
) -> torch.Tensor:
    """CPU reference: broadcast num_sampled to num_accepted[req_state_idx].

    idx_mapping[row] = -1 -> skip. Otherwise num_accepted[req_state_idx] = num_sampled.
    """
    out = torch.full((max_num_reqs,), _SENTINEL, dtype=torch.int32)
    for row in range(idx_mapping.shape[0]):
        rsi = int(idx_mapping[row].item())
        if rsi < 0:
            continue
        out[rsi] = num_sampled
    return out


# Realistic shapes:
#   - num_reqs in {1, 3, 8, 16, 64, 128, 256} (production concurrency range)
#   - max_num_reqs >= num_reqs (padding for CUDAGraph)
#   - num_sampled in {0, 1, 2, 3, 4, 5, 8, 16} (covers neutral=1 and spec-decode
#     acceptance values; 0 is chunked prefill case where caller does max(0,1)=1)
SHAPE_PARAMS = [
    # (num_reqs, max_num_reqs, num_sampled)
    (1, 4, 1),
    (1, 4, 3),
    (3, 8, 1),
    (3, 8, 0),     # chunked prefill: caller does max(0,1)=1
    (8, 16, 2),
    (8, 16, 5),
    (16, 32, 1),
    (16, 32, 4),
    (64, 128, 3),
    (64, 128, 8),
    (128, 256, 1),
    (128, 256, 16),
    (256, 512, 2),
    (256, 512, 4),
    # Edge: num_reqs == max_num_reqs (no padding)
    (4, 4, 1),
    (8, 8, 3),
    # Edge: single request, large max_num_reqs
    (1, 256, 1),
    # Edge: large num_sampled (many accepted spec tokens)
    (4, 8, 32),
]


def _gen_inputs(
    num_reqs: int,
    max_num_reqs: int,
    num_sampled: int,
    *,
    with_negatives: bool,
    device,
) -> dict:
    torch.manual_seed(42 + num_reqs * 7 + num_sampled * 3)

    if with_negatives:
        # Scatter some -1 sentinels (PP filtered rows)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
        if num_reqs > 2:
            # Set every 3rd entry to -1
            idx_mapping[2::3] = -1
        # Shuffle the valid indices to random positions
        valid_mask = idx_mapping >= 0
        valid_indices = idx_mapping[valid_mask].clone()
        perm = torch.randperm(valid_indices.shape[0], device=device)
        idx_mapping[valid_mask] = valid_indices[perm]
    else:
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)

    # Pre-fill with sentinel to detect unwritten slots
    num_accepted = torch.full(
        (max_num_reqs,), _SENTINEL, dtype=torch.int32, device=device
    )

    return {
        "idx_mapping": idx_mapping,
        "num_accepted": num_accepted,
        "num_sampled": num_sampled,
        "num_reqs": num_reqs,
        "max_num_reqs": max_num_reqs,
    }


@pytest.mark.parametrize("num_reqs,max_num_reqs,num_sampled", SHAPE_PARAMS)
@pytest.mark.parametrize("with_negatives", [False, True])
def test_fill_num_accepted(
    num_reqs: int,
    max_num_reqs: int,
    num_sampled: int,
    with_negatives: bool,
    rt,
):
    """Compare kernel against independent CPU reference.

    The kernel is pure int32 scatter. Pass requires bitwise equality on every
    written slot AND sentinel preservation on unwritten slots (catches
    wrong-index / over-write bugs).

    Note: the caller in production does max(num_sampled, 1). Here we pass
    num_sampled directly to test the kernel's raw behavior, including 0.
    """
    if kernel is None:
        pytest.fail(
            "_fill_num_accepted_kernel import failed.\n"
            f"error={_import_error}\ntraceback:\n{_import_traceback}",
            pytrace=False,
        )

    rt.init_device_properties_triton()
    device = rt.STRICT_DEVICE

    inputs = _gen_inputs(
        num_reqs=num_reqs,
        max_num_reqs=max_num_reqs,
        num_sampled=num_sampled,
        with_negatives=with_negatives,
        device=device,
    )

    # CPU reference
    expected = _ref(
        idx_mapping=inputs["idx_mapping"].cpu(),
        num_sampled=inputs["num_sampled"],
        max_num_reqs=inputs["max_num_reqs"],
    )

    # Device run
    dev_num_accepted = inputs["num_accepted"].clone()
    kernel[(inputs["num_reqs"],)](
        inputs["idx_mapping"],
        dev_num_accepted,
        inputs["num_sampled"],
    )
    rt.synchronize()

    dev_out = dev_num_accepted.cpu()
    assert dev_out.dtype == expected.dtype, "dtype mismatch"
    assert dev_out.shape == expected.shape, "shape mismatch"
    if not torch.equal(dev_out, expected):
        mismatched = torch.ne(dev_out, expected)
        count = int(mismatched.sum().item())
        total = int(mismatched.numel())
        first_idx = torch.nonzero(mismatched, as_tuple=False)
        first_info = ""
        if first_idx.numel() > 0:
            loc = tuple(first_idx[0].tolist())
            first_info = (
                f"; first mismatch at {loc}: "
                f"device={dev_out[loc].item()} ref={expected[loc].item()}"
            )
        pytest.fail(
            f"_fill_num_accepted_kernel mismatch: "
            f"{count}/{total} elements differ{first_info}"
        )


def test_fill_num_accepted_clamp_one(rt):
    """Production caller does max(num_sampled, 1). Test that path explicitly."""
    if kernel is None:
        pytest.fail("kernel import failed", pytrace=False)

    rt.init_device_properties_triton()
    device = rt.STRICT_DEVICE

    num_reqs = 4
    max_num_reqs = 8
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    num_accepted = torch.full((max_num_reqs,), _SENTINEL, dtype=torch.int32, device=device)

    # Simulate: num_sampled=0 (chunked prefill), caller clamps to 1
    num_sampled = 0
    clamped = max(num_sampled, 1)

    kernel[(num_reqs,)](idx_mapping, num_accepted, clamped)
    rt.synchronize()

    expected = torch.full((max_num_reqs,), _SENTINEL, dtype=torch.int32)
    expected[:num_reqs] = 1

    torch.testing.assert_close(
        num_accepted.cpu(), expected, rtol=0, atol=0
    )


def test_fill_num_accepted_all_skip(rt):
    """All rows have idx_mapping=-1: no writes, output stays at sentinel."""
    if kernel is None:
        pytest.fail("kernel import failed", pytrace=False)

    rt.init_device_properties_triton()
    device = rt.STRICT_DEVICE

    num_reqs = 4
    max_num_reqs = 8
    idx_mapping = torch.full((num_reqs,), -1, dtype=torch.int32, device=device)
    num_accepted = torch.full((max_num_reqs,), _SENTINEL, dtype=torch.int32, device=device)

    kernel[(num_reqs,)](idx_mapping, num_accepted, 3)
    rt.synchronize()

    expected = torch.full((max_num_reqs,), _SENTINEL, dtype=torch.int32)
    torch.testing.assert_close(
        num_accepted.cpu(), expected, rtol=0, atol=0
    )


def test_import_error() -> None:
    if _import_error is not None:
        pytest.fail(
            f"Failed to import _fill_num_accepted_kernel:\n"
            f"{_import_traceback}"
        )
