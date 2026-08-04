# Standalone Ascend A3 adaptation of an upstream vLLM accuracy path.
# Accuracy UT source: vllm/tests/v1/sample/test_topk_topp_sampler.py
# Kernel source: vllm/vllm/v1/sample/ops/topk_topp_triton.py
# Coverage: _update_min_larger_stats

# vLLM vanilla kernel: _update_min_larger_stats from
# vllm/vllm/v1/sample/ops/topk_topp_triton.py

"""
Precision test for _update_min_larger_stats (vanilla vLLM version).

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
"""

import torch
import pytest

from vllm.triton_utils import tl, triton
from vllm.v1.sample.ops.topk_topp_triton import _update_min_larger_stats
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


# We test _update_min_larger_stats by writing a small wrapper kernel that
# applies it to a single tile of data.


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


class TestUpdateMinLargerStats:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("BLOCK_SIZE", [16, 32, 64])
    def test_basic_merge_new_min(self, BLOCK_SIZE):
        """Tile min < running min: should replace both values."""
        sentinel = float("inf")
        data = torch.randn(BLOCK_SIZE, dtype=torch.float32, device=self.device)
        above_mask = torch.ones(BLOCK_SIZE, dtype=torch.int32, device=self.device)
        min_larger = torch.tensor([100.0], dtype=torch.float32, device=self.device)
        num_min_larger = torch.tensor([0], dtype=torch.int32, device=self.device)

        _update_min_larger_wrapper[(1,)](
            data, above_mask, min_larger, num_min_larger,
            sentinel=sentinel, BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.npu.synchronize()

        expected_min, expected_cnt = _update_min_larger_ref(
            data.cpu(), above_mask.cpu().bool(), 100.0, 0, sentinel
        )
        torch.testing.assert_close(min_larger.cpu(), torch.tensor([expected_min]), rtol=1e-5, atol=1e-5)
        assert num_min_larger.item() == expected_cnt, (
            f"Expected cnt={expected_cnt}, got {num_min_larger.item()}"
        )

    @pytest.mark.parametrize("BLOCK_SIZE", [16, 32])
    def test_merge_same_min(self, BLOCK_SIZE):
        """Tile min == running min: should accumulate count."""
        sentinel = float("inf")
        # Set all data values to the same number
        data = torch.full((BLOCK_SIZE,), 5.0, dtype=torch.float32, device=self.device)
        above_mask = torch.ones(BLOCK_SIZE, dtype=torch.int32, device=self.device)
        min_larger = torch.tensor([5.0], dtype=torch.float32, device=self.device)
        num_min_larger = torch.tensor([3], dtype=torch.int32, device=self.device)

        _update_min_larger_wrapper[(1,)](
            data, above_mask, min_larger, num_min_larger,
            sentinel=sentinel, BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.npu.synchronize()

        assert min_larger.item() == 5.0, "min_larger should remain 5.0"
        assert num_min_larger.item() == 3 + BLOCK_SIZE, (
            f"Expected {3 + BLOCK_SIZE}, got {num_min_larger.item()}"
        )

    @pytest.mark.parametrize("BLOCK_SIZE", [16, 32])
    def test_merge_larger_min(self, BLOCK_SIZE):
        """Tile min > running min: should keep running values unchanged."""
        sentinel = float("inf")
        data = torch.full((BLOCK_SIZE,), 10.0, dtype=torch.float32, device=self.device)
        above_mask = torch.ones(BLOCK_SIZE, dtype=torch.int32, device=self.device)
        min_larger = torch.tensor([5.0], dtype=torch.float32, device=self.device)
        num_min_larger = torch.tensor([7], dtype=torch.int32, device=self.device)

        _update_min_larger_wrapper[(1,)](
            data, above_mask, min_larger, num_min_larger,
            sentinel=sentinel, BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.npu.synchronize()

        assert min_larger.item() == 5.0, "min_larger should remain 5.0"
        assert num_min_larger.item() == 7, (
            f"Expected 7, got {num_min_larger.item()}"
        )

    @pytest.mark.parametrize("BLOCK_SIZE", [16, 32])
    def test_no_above_data(self, BLOCK_SIZE):
        """When no data is above the mask, running state should be unchanged."""
        sentinel = float("inf")
        data = torch.randn(BLOCK_SIZE, dtype=torch.float32, device=self.device)
        above_mask = torch.zeros(BLOCK_SIZE, dtype=torch.int32, device=self.device)
        min_larger = torch.tensor([5.0], dtype=torch.float32, device=self.device)
        num_min_larger = torch.tensor([3], dtype=torch.int32, device=self.device)

        _update_min_larger_wrapper[(1,)](
            data, above_mask, min_larger, num_min_larger,
            sentinel=sentinel, BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.npu.synchronize()

        assert min_larger.item() == 5.0, "min_larger should remain 5.0"
        assert num_min_larger.item() == 3, (
            f"Expected 3, got {num_min_larger.item()}"
        )

    @pytest.mark.parametrize("BLOCK_SIZE", [16, 32])
    def test_sentinel_filtering(self, BLOCK_SIZE):
        """Data equal to sentinel should not affect results."""
        sentinel = float("inf")
        # Fill with sentinel-like large values and a few real values
        data = torch.full((BLOCK_SIZE,), sentinel, dtype=torch.float32, device=self.device)
        data[:5] = 3.0  # A few values above pivot
        data[5:8] = 7.0  # Higher values (should be min_larger=3.0)
        above_mask = torch.ones(BLOCK_SIZE, dtype=torch.int32, device=self.device)
        min_larger = torch.tensor([10.0], dtype=torch.float32, device=self.device)
        num_min_larger = torch.tensor([0], dtype=torch.int32, device=self.device)

        _update_min_larger_wrapper[(1,)](
            data, above_mask, min_larger, num_min_larger,
            sentinel=sentinel, BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.npu.synchronize()

        expected_min, expected_cnt = _update_min_larger_ref(
            data.cpu(), above_mask.cpu().bool(), 10.0, 0, sentinel
        )
        torch.testing.assert_close(min_larger.cpu(), torch.tensor([expected_min]), rtol=1e-5, atol=1e-5)
        assert num_min_larger.item() == expected_cnt, (
            f"Expected cnt={expected_cnt}, got {num_min_larger.item()}"
        )
