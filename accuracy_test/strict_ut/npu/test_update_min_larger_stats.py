# GENERATED STRICT UT. Source: accuracy_test/codex/existing_accuracy_tests/from_vllm/test_update_min_larger_stats.py
# Do not edit mechanically; update the reviewed Codex source or strict generator.
from accuracy_test.strict_ut.runtime_npu import STRICT_DEVICE as _STRICT_DEVICE
# Standalone Ascend A3 adaptation of an upstream vLLM accuracy path.
# Accuracy UT source: vllm/tests/v1/sample/test_topk_topp_sampler.py
# Kernel source: vllm/vllm/v1/sample/ops/topk_topp_triton.py
# Coverage: _update_min_larger_stats

# vLLM vanilla kernel: _update_min_larger_stats from
# vllm/vllm/v1/sample/ops/topk_topp_triton.py

"""
Precision test for _update_min_larger_stats.

Helper for top-k/top-p that tracks the minimum value strictly above a pivot,
and the count of that minimum value, across tiles. Merge rule:
  - tile min < running min -> replace both
  - tile min == running min -> accumulate count
  - tile min > running min -> keep running values

This is a pure Triton JIT function (not a kernel) that gets inlined into
_topk_topp_kernel. We test it by launching a small wrapper kernel.

Kernel/wrapper signature:
    Called via _update_min_larger_stats(data, above_mask, min_larger,
                                        num_min_larger, sentinel)
    Returns (updated_min_larger, updated_num_min_larger)

Realistic shapes:
  - BLOCK_SIZE: tile size used inside _topk_topp_kernel (32, 64, 128)
  - data: one tile of logits (BLOCK_SIZE float32 values)
  - above_mask: which entries are strictly above the current pivot
  - min_larger / num_min_larger: running (min, count) state carried across tiles
  - sentinel: +inf (marks masked-out entries in the tile)
"""

from __future__ import annotations

import pytest
import torch

from accuracy_test.strict_ut.runtime_npu import DEVICE, init_device_properties_triton, synchronize
from vllm.triton_utils import tl, triton
from vllm.v1.sample.ops.topk_topp_triton import _update_min_larger_stats

pytestmark = [pytest.mark.npu]


@triton.jit
def _update_min_larger_wrapper(
    data_ptr,
    above_mask_ptr,
    min_larger_ptr,
    num_min_larger_ptr,
    sentinel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Wrapper that calls _update_min_larger_stats on one tile.

    Loads data and above_mask, initializes the running state from
    min_larger_ptr / num_min_larger_ptr, calls the helper, and stores
    the updated state back.
    """
    offs = tl.arange(0, BLOCK_SIZE)
    data = tl.load(data_ptr + offs, mask=offs < BLOCK_SIZE, other=sentinel)
    above_mask = tl.load(above_mask_ptr + offs, mask=offs < BLOCK_SIZE, other=0)
    min_larger = tl.load(min_larger_ptr)
    num_min_larger = tl.load(num_min_larger_ptr)
    new_min_larger, new_num_min_larger = _update_min_larger_stats(
        data, above_mask, min_larger, num_min_larger, sentinel
    )
    tl.store(min_larger_ptr, new_min_larger)
    tl.store(num_min_larger_ptr, new_num_min_larger)


def _update_min_larger_ref(
    data: torch.Tensor,
    above_mask: torch.Tensor,
    min_larger: float,
    num_min_larger: int,
    sentinel: float = float("inf"),
) -> tuple[float, int]:
    """CPU reference for _update_min_larger_stats."""
    above_data = data[above_mask]
    if above_data.numel() == 0:
        return min_larger, num_min_larger
    tile_min = above_data.min().item()
    tile_eq = (above_data - tile_min).abs() < 1e-9
    tile_cnt = tile_eq.sum().item()
    is_new = tile_min < min_larger - 1e-12
    is_same = abs(tile_min - min_larger) < 1e-9
    if is_new:
        num_min_larger = tile_cnt
    elif is_same:
        num_min_larger += tile_cnt
    if tile_min < min_larger:
        min_larger = tile_min
    return min_larger, num_min_larger


def _launch(data, above_mask, min_larger, num_min_larger, block_size):
    sentinel = float("inf")
    _update_min_larger_wrapper[(1,)](
        data,
        above_mask,
        min_larger,
        num_min_larger,
        sentinel=sentinel,
        BLOCK_SIZE=block_size,
    )
    synchronize()


@pytest.mark.parametrize("BLOCK_SIZE", [32, 64, 128])
def test_basic_merge_new_min(BLOCK_SIZE):
    """Tile min < running min: should replace both values."""
    init_device_properties_triton()
    torch.manual_seed(42)

    data = torch.randn(BLOCK_SIZE, dtype=torch.float32, device=DEVICE)
    above_mask = torch.ones(BLOCK_SIZE, dtype=torch.int32, device=DEVICE)
    min_larger = torch.tensor([100.0], dtype=torch.float32, device=DEVICE)
    num_min_larger = torch.tensor([0], dtype=torch.int32, device=DEVICE)

    _launch(data, above_mask, min_larger, num_min_larger, BLOCK_SIZE)

    expected_min, expected_cnt = _update_min_larger_ref(
        data.cpu(), above_mask.cpu().bool(), 100.0, 0
    )
    torch.testing.assert_close(
        min_larger.cpu(), torch.tensor([expected_min]), rtol=1e-5, atol=1e-5
    )
    assert num_min_larger.item() == expected_cnt, (
        f"Expected cnt={expected_cnt}, got {num_min_larger.item()}"
    )


@pytest.mark.parametrize("BLOCK_SIZE", [32, 64, 128])
def test_merge_same_min(BLOCK_SIZE):
    """Tile min == running min: should accumulate count."""
    init_device_properties_triton()
    torch.manual_seed(42)

    data = torch.full((BLOCK_SIZE,), 5.0, dtype=torch.float32, device=DEVICE)
    above_mask = torch.ones(BLOCK_SIZE, dtype=torch.int32, device=DEVICE)
    min_larger = torch.tensor([5.0], dtype=torch.float32, device=DEVICE)
    num_min_larger = torch.tensor([3], dtype=torch.int32, device=DEVICE)

    _launch(data, above_mask, min_larger, num_min_larger, BLOCK_SIZE)

    assert min_larger.item() == 5.0, "min_larger should remain 5.0"
    assert num_min_larger.item() == 3 + BLOCK_SIZE, (
        f"Expected {3 + BLOCK_SIZE}, got {num_min_larger.item()}"
    )


@pytest.mark.parametrize("BLOCK_SIZE", [32, 64, 128])
def test_merge_larger_min(BLOCK_SIZE):
    """Tile min > running min: should keep running values unchanged."""
    init_device_properties_triton()
    torch.manual_seed(42)

    data = torch.full((BLOCK_SIZE,), 10.0, dtype=torch.float32, device=DEVICE)
    above_mask = torch.ones(BLOCK_SIZE, dtype=torch.int32, device=DEVICE)
    min_larger = torch.tensor([5.0], dtype=torch.float32, device=DEVICE)
    num_min_larger = torch.tensor([7], dtype=torch.int32, device=DEVICE)

    _launch(data, above_mask, min_larger, num_min_larger, BLOCK_SIZE)

    assert min_larger.item() == 5.0, "min_larger should remain 5.0"
    assert num_min_larger.item() == 7, (
        f"Expected 7, got {num_min_larger.item()}"
    )


@pytest.mark.parametrize("BLOCK_SIZE", [32, 64])
def test_partial_above_mask(BLOCK_SIZE):
    """Only some entries above pivot: verify correct min/count on survivors."""
    init_device_properties_triton()
    torch.manual_seed(42)

    data = torch.randn(BLOCK_SIZE, dtype=torch.float32, device=DEVICE)
    # Mask out the lower half (simulating entries below the pivot)
    above_mask = torch.zeros(BLOCK_SIZE, dtype=torch.int32, device=DEVICE)
    above_mask[BLOCK_SIZE // 2 :] = 1
    min_larger = torch.tensor([100.0], dtype=torch.float32, device=DEVICE)
    num_min_larger = torch.tensor([0], dtype=torch.int32, device=DEVICE)

    _launch(data, above_mask, min_larger, num_min_larger, BLOCK_SIZE)

    expected_min, expected_cnt = _update_min_larger_ref(
        data.cpu(), above_mask.cpu().bool(), 100.0, 0
    )
    torch.testing.assert_close(
        min_larger.cpu(), torch.tensor([expected_min]), rtol=1e-5, atol=1e-5
    )
    assert num_min_larger.item() == expected_cnt, (
        f"Expected cnt={expected_cnt}, got={num_min_larger.item()}"
    )


def test_no_above_entries():
    """When above_mask is all-zero, running state must be unchanged."""
    init_device_properties_triton()
    torch.manual_seed(42)

    BLOCK_SIZE = 64
    data = torch.randn(BLOCK_SIZE, dtype=torch.float32, device=DEVICE)
    above_mask = torch.zeros(BLOCK_SIZE, dtype=torch.int32, device=DEVICE)
    min_larger = torch.tensor([3.14], dtype=torch.float32, device=DEVICE)
    num_min_larger = torch.tensor([5], dtype=torch.int32, device=DEVICE)

    _launch(data, above_mask, min_larger, num_min_larger, BLOCK_SIZE)

    assert min_larger.item() == 3.14, "min_larger should remain unchanged"
    assert num_min_larger.item() == 5, "num_min_larger should remain unchanged"
