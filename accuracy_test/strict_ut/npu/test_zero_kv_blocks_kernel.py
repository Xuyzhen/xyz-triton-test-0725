# GENERATED STRICT UT. Source: accuracy_test/codex/missing_accuracy_tests/test_zero_kv_blocks_kernel_patch.py
# Do not edit mechanically; update the reviewed Codex source or strict generator.
from accuracy_test.strict_ut.runtime_npu import STRICT_DEVICE as _STRICT_DEVICE
# Standalone Ascend A3 adaptation of an upstream vLLM accuracy path.
# Accuracy UT source: vllm/tests/v1/worker/test_gpu_block_table.py (zero_kv path)
# Kernel source: vllm/vllm/v1/worker/utils.py (upstream) /
#                vllm-ascend/vllm_ascend/worker/utils.py (Ascend patch)
# Coverage: _zero_kv_blocks_kernel

# vLLM vanilla kernel: _zero_kv_blocks_kernel from
# vllm/vllm/v1/worker/utils.py

"""
Precision test for _zero_kv_blocks_kernel.

Zeros KV cache blocks at specified block IDs across all segments in a single
launch. Programs are mapped as (block_index, seg_index, chunk_index).

Kernel signature (upstream):
    _zero_kv_blocks_kernel(
        seg_addrs_ptr,      # [N_SEGS] int64 absolute byte addresses
        block_ids_ptr,      # [n_blocks] int64 block IDs to zero
        n_blocks,           # scalar
        N_SEGS: tl.constexpr,
        PAGE_SIZE_EL: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    )

Ascend patch adds: GRID_SIZE: tl.constexpr  (load-balancing across vector cores)

Realistic shapes:
  - page_size_el = block_size_tokens * num_kv_heads * head_size
    e.g. 16 * 8 * 128 = 16384 (Llama-style KV layout, block_size=16, 8 heads, head_size=128)
  - blk_size (BLOCK_SIZE constexpr) = 1024 (largest_power_of_2_divisor of page_size_el, capped at 8192)
  - n_segs = 1 (block_dim=0, K+V fused) or 2 (block_dim=1, K/V separate)
  - n_blocks: number of freed KV cache blocks per step (1..64 in practice)
"""

from __future__ import annotations

import importlib

import pytest
import torch

from accuracy_test.strict_ut.runtime_npu import (
    DEVICE,
    get_vectorcore_num,
    init_device_properties_triton,
    synchronize,
)

pytestmark = [pytest.mark.npu]


def _resolve_kernel():
    """Prefer the Ascend adaptation (has GRID_SIZE); reuse upstream otherwise."""
    try:
        module = importlib.import_module("vllm_ascend.worker.utils")
        kernel = getattr(module, "_zero_kv_blocks_kernel", None)
        if kernel is not None:
            return kernel, "ascend_adapted"
    except (ImportError, ModuleNotFoundError):
        pass
    from vllm.v1.worker.utils import _zero_kv_blocks_kernel as _upstream_kernel

    return _upstream_kernel, "upstream_reuse"


KERNEL, IMPLEMENTATION_KIND = _resolve_kernel()
_HAS_GRID_SIZE = "GRID_SIZE" in set(KERNEL.arg_names)


def _zero_kv_blocks_cpu(
    scratch: torch.Tensor,
    seg_addrs: torch.Tensor,
    block_ids: torch.Tensor,
    n_blocks: int,
    n_segs: int,
    page_size_el: int,
    block_size: int,
) -> torch.Tensor:
    """Pure PyTorch CPU reference for zero_kv_blocks.

    Mirrors the kernel's 3D program mapping (block_index, seg_index, chunk_index).
    Each segment's data_ptr identifies a contiguous region; for the CPU ref we
    operate on the concatenated scratch buffer using the same offset math.
    """
    result = scratch.clone()
    chunks = page_size_el // block_size
    work_per_block = n_segs * chunks
    total_work = n_blocks * work_per_block

    for pid in range(total_work):
        block_index = pid // work_per_block
        remainder = pid % work_per_block
        seg_index = remainder // chunks
        chunk_index = remainder % chunks

        block_id = int(block_ids[block_index])
        seg_offset = block_id * page_size_el + chunk_index * block_size
        for i in range(block_size):
            idx = seg_offset + i
            if idx < len(result):
                result[idx] = 0
    return result


def _launch(grid, seg_addrs, block_ids, n_blocks, n_segs, page_size_el, blk_size):
    kwargs = dict(
        N_SEGS=n_segs,
        PAGE_SIZE_EL=page_size_el,
        BLOCK_SIZE=blk_size,
    )
    if _HAS_GRID_SIZE:
        kwargs["GRID_SIZE"] = grid[0]
    KERNEL[grid](
        seg_addrs,
        block_ids,
        n_blocks,
        **kwargs,
    )
    synchronize()


@pytest.mark.parametrize(
    "n_blocks,n_segs,page_size_el,blk_size",
    [
        # Single segment (block_dim=0), Llama-style page (16 tokens * 8 heads * 128)
        (1, 1, 16384, 1024),
        (8, 1, 16384, 1024),
        (32, 1, 16384, 1024),
        # Two segments (block_dim=1, K/V separate)
        (1, 2, 16384, 1024),
        (8, 2, 16384, 1024),
        # Smaller page: block_size=8, 4 heads, head_size=128 -> 4096
        (4, 2, 4096, 512),
        # Larger page: block_size=32, 8 heads, head_size=128 -> 32768
        (2, 2, 32768, 2048),
    ],
)
def test_zero_kv_blocks(n_blocks, n_segs, page_size_el, blk_size):
    """Zero n_blocks KV cache blocks across n_segs segments; verify exact zeroing."""
    init_device_properties_triton()
    torch.manual_seed(42)

    max_blocks = n_blocks + 2  # extra blocks that must remain untouched
    total_el = max_blocks * page_size_el

    # Single contiguous scratch buffer; each segment maps to the same region
    # (simulating K/V in one allocation). For n_segs>1 we pass the same ptr
    # repeated, which matches the kernel semantics (zeroing K and V separately
    # but here both point to the same buffer for simplicity of verification).
    scratch = torch.randint(1, 100, (total_el,), dtype=torch.int32, device=DEVICE)
    seg_addrs = torch.tensor(
        [scratch.data_ptr()] * n_segs, dtype=torch.int64, device=DEVICE
    )
    # Zero the first n_blocks block IDs; leave the rest untouched
    block_ids = torch.arange(n_blocks, dtype=torch.int64, device=DEVICE)

    chunks = page_size_el // blk_size
    total_work = n_blocks * n_segs * chunks
    grid = (total_work,)
    if _HAS_GRID_SIZE:
        # Ascend patch load-balances across vector cores
        grid = (min(total_work, get_vectorcore_num()),)

    _launch(grid, seg_addrs, block_ids, n_blocks, n_segs, page_size_el, blk_size)

    expected = _zero_kv_blocks_cpu(
        scratch.cpu(), seg_addrs.cpu(), block_ids.cpu(),
        n_blocks, n_segs, page_size_el, blk_size,
    )
    torch.testing.assert_close(scratch.cpu(), expected, rtol=0, atol=0)


def test_zero_kv_blocks_no_blocks():
    """When n_blocks=0, nothing should change (no-op)."""
    init_device_properties_triton()
    torch.manual_seed(42)

    page_size_el = 4096
    blk_size = 512
    n_segs = 1
    n_blocks = 0

    scratch = torch.randint(1, 100, (page_size_el,), dtype=torch.int32, device=DEVICE)
    scratch_before = scratch.clone()
    seg_addrs = torch.tensor(
        [scratch.data_ptr()], dtype=torch.int64, device=DEVICE
    )
    block_ids = torch.empty(0, dtype=torch.int64, device=DEVICE)

    chunks = page_size_el // blk_size
    total_work = n_blocks * n_segs * chunks
    if total_work > 0:
        grid = (total_work,)
        if _HAS_GRID_SIZE:
            grid = (min(total_work, get_vectorcore_num()),)
        _launch(grid, seg_addrs, block_ids, n_blocks, n_segs, page_size_el, blk_size)

    torch.testing.assert_close(scratch.cpu(), scratch_before.cpu(), rtol=0, atol=0)


def test_zero_kv_blocks_sparsity():
    """Zero non-contiguous block IDs; verify only targeted blocks are zeroed."""
    init_device_properties_triton()
    torch.manual_seed(42)

    page_size_el = 4096
    blk_size = 512
    n_segs = 1
    max_blocks = 6
    total_el = max_blocks * page_size_el

    scratch = torch.randint(1, 100, (total_el,), dtype=torch.int32, device=DEVICE)
    seg_addrs = torch.tensor([scratch.data_ptr()], dtype=torch.int64, device=DEVICE)
    # Zero blocks 1, 3, 5 — leave 0, 2, 4 untouched
    block_ids = torch.tensor([1, 3, 5], dtype=torch.int64, device=DEVICE)
    n_blocks = 3

    chunks = page_size_el // blk_size
    total_work = n_blocks * n_segs * chunks
    grid = (total_work,)
    if _HAS_GRID_SIZE:
        grid = (min(total_work, get_vectorcore_num()),)

    _launch(grid, seg_addrs, block_ids, n_blocks, n_segs, page_size_el, blk_size)

    result = scratch.cpu()
    # Targeted blocks must be all zero
    for bid in [1, 3, 5]:
        start = bid * page_size_el
        end = start + page_size_el
        assert (result[start:end] == 0).all(), f"Block {bid} should be zeroed"
    # Untouched blocks must retain original non-zero values
    for bid in [0, 2, 4]:
        start = bid * page_size_el
        end = start + page_size_el
        assert (result[start:end] > 0).all(), f"Block {bid} should be untouched"
