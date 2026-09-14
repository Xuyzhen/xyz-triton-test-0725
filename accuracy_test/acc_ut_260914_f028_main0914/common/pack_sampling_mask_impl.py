# SPDX-License-Identifier: Apache-2.0
# acc_ut_260914_f028_main0914 shared UT for _pack_sampling_mask_kernel.
# Source: vllm/v1/worker/gpu/sample/output.py (base commit 50ba4bc6b2).
# This kernel was REMOVED in HEAD (560ef78bfe) and replaced by
# _compact_sampling_mask_kernel. We re-declare the kernel locally from its
# original source so the test is self-contained.
#
# Category: integer/index compute (bit-packing of logits masks). All outputs
# are uint8/int32 -> bitwise exact.
#
# Kernel signature (from base commit):
#   @triton.jit
#   def _pack_sampling_mask_kernel(
#       logits_ptr,
#       logits_row_stride,
#       logits_col_stride,
#       num_sampled_tokens_ptr,
#       packed_mask_ptr,
#       packed_mask_row_stride,
#       counts_ptr,
#       vocab_size,
#       BLOCK_SIZE: tl.constexpr,
#   )
#
# The kernel scans the vocab dimension of each request's logits and packs
# "finite-value" positions into a bit-packed uint8 mask (little-endian bit
# order). It also counts the number of finite positions per request.

from __future__ import annotations

import traceback
from typing import Any

import pytest
import torch

from vllm.triton_utils import tl, triton

kernel = None
_import_error: Exception | None = None
_import_traceback: str | None = None

# Re-declare the kernel from the base commit source, since it was removed.
try:
    @triton.jit
    def _pack_sampling_mask_kernel(
        logits_ptr,
        logits_row_stride,
        logits_col_stride,
        num_sampled_tokens_ptr,
        packed_mask_ptr,
        packed_mask_row_stride,
        counts_ptr,
        vocab_size,
        BLOCK_SIZE: tl.constexpr,
    ):
        req_idx = tl.program_id(0)
        is_active = tl.load(num_sampled_tokens_ptr + req_idx) > 0
        count = tl.zeros((), dtype=tl.int32)

        for start_idx in range(0, vocab_size, BLOCK_SIZE):
            offsets = start_idx + tl.arange(0, BLOCK_SIZE)
            valid = offsets < vocab_size
            logits = tl.load(
                logits_ptr + req_idx * logits_row_stride + offsets * logits_col_stride,
                mask=valid,
                other=-float("inf"),
            )
            keep = (logits > -float("inf")) & (logits < float("inf")) & is_active
            count += tl.sum(keep).to(tl.int32)

            keep = tl.reshape(keep.to(tl.int32), (BLOCK_SIZE // 8, 8))
            bit_shifts = tl.arange(0, 8)[None, :]
            packed = tl.sum(keep << bit_shifts, axis=1).to(tl.uint8)
            byte_offsets = start_idx // 8 + tl.arange(0, BLOCK_SIZE // 8)
            tl.store(
                packed_mask_ptr + req_idx * packed_mask_row_stride + byte_offsets,
                packed,
                mask=byte_offsets < tl.cdiv(vocab_size, 8),
            )

        tl.store(counts_ptr + req_idx, count)

    kernel = _pack_sampling_mask_kernel
except Exception as exc:  # pragma: no cover
    _import_error = exc
    _import_traceback = traceback.format_exc()


def _ref(
    logits: torch.Tensor,
    num_sampled_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference: pack finite-logit positions into a bit mask.

    Returns (packed_mask [num_reqs, ceil(vocab/8)] uint8, counts [num_reqs] int32)
    """
    num_reqs, vocab_size = logits.shape
    packed_width = (vocab_size + 7) // 8

    packed_mask = torch.zeros((num_reqs, packed_width), dtype=torch.uint8)
    counts = torch.zeros(num_reqs, dtype=torch.int32)

    for r in range(num_reqs):
        active = int(num_sampled_tokens[r].item()) > 0
        for v in range(vocab_size):
            val = float(logits[r, v].item())
            is_finite = (val > float("-inf")) and (val < float("inf"))
            if active and is_finite:
                counts[r] += 1
                byte_idx = v // 8
                bit_idx = v % 8
                packed_mask[r, byte_idx] |= (1 << bit_idx)

    return packed_mask, counts


def _gen_inputs(
    num_reqs: int,
    vocab_size: int,
    *,
    all_active: bool,
    all_inactive: bool,
    mixed_finite: bool,
    device,
) -> dict[str, Any]:
    """Build logits with a mix of finite and infinite values.

    Branches:
      - all_active: every request has num_sampled > 0
      - all_inactive: every request has num_sampled = 0 (full mask is zero)
      - mixed_finite: logits have a mix of -inf, +inf, and finite values
    """
    torch.manual_seed(42 + num_reqs * 7 + vocab_size * 3)

    # Create logits: default all finite, then sprinkle infinities
    logits = torch.randn(num_reqs, vocab_size, dtype=torch.float32, device=device)

    if mixed_finite:
        # Set ~30% to -inf (masked out tokens)
        mask = torch.rand(num_reqs, vocab_size, device=device) < 0.3
        logits[mask] = float("-inf")
        # Set ~5% to +inf (corrupted/anomalous, should also be excluded)
        mask2 = torch.rand(num_reqs, vocab_size, device=device) < 0.05
        logits[mask2] = float("+inf")

    if all_inactive:
        num_sampled = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    elif all_active:
        num_sampled = torch.randint(1, 5, (num_reqs,), dtype=torch.int32, device=device)
    else:
        # Mix of active and inactive
        num_sampled = torch.randint(0, 3, (num_reqs,), dtype=torch.int32, device=device)
        # Ensure at least one active and one inactive
        if num_reqs > 1:
            num_sampled[0] = 1
            num_sampled[1] = 0

    return {
        "logits": logits,
        "num_sampled_tokens": num_sampled,
    }


# Realistic shapes:
#   - vocab_size: 128 (tiny), 1024, 32000, 152064 (Qwen3), 256 (edge)
#   - BLOCK_SIZE must be power-of-2 and >= 8 (for the reshape)
#   - num_reqs: 1, 4, 8, 32, 128
SHAPE_PARAMS = [
    # (num_reqs, vocab_size, BLOCK_SIZE)
    (1, 128, 128),
    (1, 256, 256),
    (4, 128, 128),
    (4, 1024, 1024),
    (8, 256, 256),
    (8, 1024, 1024),
    (8, 32000, 8192),
    (16, 128, 128),
    (16, 1024, 1024),
    (32, 256, 256),
    (32, 32000, 8192),
    (128, 128, 128),
    (128, 256, 256),
    # Edge: vocab_size not a multiple of 8 (trailing partial byte)
    (4, 129, 128),
    (4, 131, 128),
    (8, 255, 256),
    # Edge: vocab_size < BLOCK_SIZE (single block)
    (4, 64, 128),
    (4, 7, 128),    # extremely small vocab
    # Edge: BLOCK_SIZE > 8192
    (4, 32000, 16384),
    # Production-scale: Qwen3 vocab
    (8, 152064, 8192),
]

BRANCH_PARAMS = [
    # (all_active, all_inactive, mixed_finite)
    (True, False, False),    # all active, all finite -> full mask
    (True, False, True),     # all active, mixed finite -> partial mask
    (False, True, False),    # all inactive -> all-zero mask
    (False, False, True),    # mixed active, mixed finite
]


@pytest.mark.parametrize("num_reqs,vocab_size,BLOCK_SIZE", SHAPE_PARAMS)
@pytest.mark.parametrize("all_active,all_inactive,mixed_finite", BRANCH_PARAMS)
def test_pack_sampling_mask(
    num_reqs: int,
    vocab_size: int,
    BLOCK_SIZE: int,
    all_active: bool,
    all_inactive: bool,
    mixed_finite: bool,
    rt,
):
    """Compare kernel against independent CPU reference.

    Outputs: packed_mask (uint8) and counts (int32). Both are bitwise exact.
    The CPU reference unpacks logits element-by-element and packs bits in
    little-endian byte order.
    """
    if kernel is None:
        pytest.fail(
            "_pack_sampling_mask_kernel declaration failed.\n"
            f"error={_import_error}\ntraceback:\n{_import_traceback}",
            pytrace=False,
        )

    rt.init_device_properties_triton()
    device = rt.STRICT_DEVICE

    inputs = _gen_inputs(
        num_reqs=num_reqs,
        vocab_size=vocab_size,
        all_active=all_active,
        all_inactive=all_inactive,
        mixed_finite=mixed_finite,
        device=device,
    )

    logits = inputs["logits"].contiguous()
    num_sampled = inputs["num_sampled_tokens"].contiguous()

    packed_width = (vocab_size + 7) // 8
    packed_mask = torch.zeros(
        (num_reqs, packed_width), dtype=torch.uint8, device=device
    )
    counts = torch.zeros(num_reqs, dtype=torch.int32, device=device)

    kernel[(num_reqs,)](
        logits,
        logits.stride(0),
        logits.stride(1),
        num_sampled,
        packed_mask,
        packed_mask.stride(0),
        counts,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    rt.synchronize()

    # CPU reference
    ref_mask, ref_counts = _ref(logits.cpu(), num_sampled.cpu())

    # Compare packed_mask
    dev_mask = packed_mask.cpu()
    assert dev_mask.dtype == ref_mask.dtype, "packed_mask dtype mismatch"
    assert dev_mask.shape == ref_mask.shape, "packed_mask shape mismatch"
    if not torch.equal(dev_mask, ref_mask):
        mismatched = torch.ne(dev_mask, ref_mask)
        count = int(mismatched.sum().item())
        total = int(mismatched.numel())
        first_idx = torch.nonzero(mismatched, as_tuple=False)
        first_info = ""
        if first_idx.numel() > 0:
            loc = tuple(first_idx[0].tolist())
            first_info = (
                f"; first mismatch at {loc}: "
                f"device={dev_mask[loc].item()} ref={ref_mask[loc].item()}"
            )
        pytest.fail(
            f"_pack_sampling_mask_kernel packed_mask mismatch: "
            f"{count}/{total} bytes differ{first_info}"
        )

    # Compare counts
    dev_counts = counts.cpu()
    assert dev_counts.dtype == ref_counts.dtype, "counts dtype mismatch"
    assert dev_counts.shape == ref_counts.shape, "counts shape mismatch"
    if not torch.equal(dev_counts, ref_counts):
        for r in range(num_reqs):
            if dev_counts[r].item() != ref_counts[r].item():
                pytest.fail(
                    f"_pack_sampling_mask_kernel counts mismatch at req {r}: "
                    f"device={dev_counts[r].item()} ref={ref_counts[r].item()}"
                )


def test_pack_sampling_mask_all_inf(rt):
    """All logits are -inf: packed mask is all-zero, counts=0."""
    if kernel is None:
        pytest.fail("kernel not available", pytrace=False)

    rt.init_device_properties_triton()
    device = rt.STRICT_DEVICE

    num_reqs = 4
    vocab_size = 128
    logits = torch.full((num_reqs, vocab_size), float("-inf"), dtype=torch.float32, device=device)
    num_sampled = torch.ones(num_reqs, dtype=torch.int32, device=device)

    packed_width = (vocab_size + 7) // 8
    packed_mask = torch.zeros((num_reqs, packed_width), dtype=torch.uint8, device=device)
    counts = torch.zeros(num_reqs, dtype=torch.int32, device=device)

    kernel[(num_reqs,)](
        logits, logits.stride(0), logits.stride(1),
        num_sampled, packed_mask, packed_mask.stride(0), counts,
        vocab_size, BLOCK_SIZE=128,
    )
    rt.synchronize()

    assert torch.all(packed_mask == 0), "packed_mask should be all-zero"
    assert torch.all(counts == 0), "counts should be all-zero"


def test_pack_sampling_mask_inactive_req(rt):
    """Inactive request (num_sampled=0): packed mask is all-zero even if logits are finite."""
    if kernel is None:
        pytest.fail("kernel not available", pytrace=False)

    rt.init_device_properties_triton()
    device = rt.STRICT_DEVICE

    num_reqs = 2
    vocab_size = 128
    logits = torch.randn(num_reqs, vocab_size, dtype=torch.float32, device=device)
    num_sampled = torch.tensor([1, 0], dtype=torch.int32, device=device)

    packed_width = (vocab_size + 7) // 8
    packed_mask = torch.zeros((num_reqs, packed_width), dtype=torch.uint8, device=device)
    counts = torch.zeros(num_reqs, dtype=torch.int32, device=device)

    kernel[(num_reqs,)](
        logits, logits.stride(0), logits.stride(1),
        num_sampled, packed_mask, packed_mask.stride(0), counts,
        vocab_size, BLOCK_SIZE=128,
    )
    rt.synchronize()

    # req 0: active, all finite -> all bits set
    assert counts[0].item() == vocab_size, f"req 0 count should be {vocab_size}, got {counts[0].item()}"
    expected_byte = 0xFF
    assert torch.all(packed_mask[0] == expected_byte), "req 0 packed_mask should be all 0xFF"

    # req 1: inactive -> all zero
    assert counts[1].item() == 0, f"req 1 count should be 0, got {counts[1].item()}"
    assert torch.all(packed_mask[1] == 0), "req 1 packed_mask should be all zero"


def test_import_error() -> None:
    if _import_error is not None:
        pytest.fail(
            f"Failed to declare _pack_sampling_mask_kernel:\n"
            f"{_import_traceback}"
        )
