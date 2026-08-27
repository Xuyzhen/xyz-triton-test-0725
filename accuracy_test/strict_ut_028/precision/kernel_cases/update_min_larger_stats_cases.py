"""update_min_larger_stats: topk/topp helper via a wrapper kernel.

Both sides import the SAME upstream helper
(vllm.v1.sample.ops.topk_topp_triton._update_min_larger_stats), a pure
@triton.jit function inlined into _topk_topp_kernel. We exercise it with the
same wrapper-kernel pattern as the strict UTs: load one tile + running state,
call the helper, store the updated state.

Merge rule under test:
  tile_min <  running_min -> replace min and count
  tile_min == running_min -> accumulate count
  tile_min >  running_min -> keep running values

Outputs: min_larger (fp32, 1e-5) and num_min_larger (int32, exact).
"""

from __future__ import annotations

import torch

import capture_runtime as cr
from capture_runtime import CaseSpec

# Triton is optional at import time so stage-3 (shape_audit / compare_results,
# runnable on any machine) can import this module without vllm installed.
try:
    from vllm.triton_utils import tl, triton
    _HAS_TRITON = True
except Exception:  # noqa: BLE001 - ImportError / missing backend
    _HAS_TRITON = False

if _HAS_TRITON:

    @triton.jit
    def _update_min_larger_wrapper(
        data_ptr,
        above_mask_ptr,
        min_larger_ptr,
        num_min_larger_ptr,
        sentinel: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
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


def build_inputs(params: dict, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    block = params["block_size"]
    pattern = params["pattern"]

    if pattern == "new_min":
        data = torch.randn(block, generator=g, dtype=torch.float32)
        mask = torch.ones(block, dtype=torch.int32)
        init_min, init_cnt = 100.0, 0
    elif pattern == "same_min":
        data = torch.full((block,), 5.0, dtype=torch.float32)
        mask = torch.ones(block, dtype=torch.int32)
        init_min, init_cnt = 5.0, 3
    elif pattern == "larger_min":
        data = torch.full((block,), 10.0, dtype=torch.float32)
        mask = torch.ones(block, dtype=torch.int32)
        init_min, init_cnt = 5.0, 7
    elif pattern == "partial_mask":
        data = torch.randn(block, generator=g, dtype=torch.float32)
        mask = torch.zeros(block, dtype=torch.int32)
        mask[block // 2:] = 1
        init_min, init_cnt = 100.0, 0
    else:  # no_above
        data = torch.randn(block, generator=g, dtype=torch.float32)
        mask = torch.zeros(block, dtype=torch.int32)
        init_min, init_cnt = 3.14, 5

    return {
        "data": data,
        "above_mask": mask,
        "min_larger": torch.tensor([init_min], dtype=torch.float32),
        "num_min_larger": torch.tensor([init_cnt], dtype=torch.int32),
    }


def run(side: str, t: dict[str, torch.Tensor], params: dict) -> dict[str, torch.Tensor]:
    # Inject the upstream helper into this module's globals so the jit
    # wrapper resolves it at compile time. Both sides import the same symbol.
    global _update_min_larger_stats
    from vllm.v1.sample.ops.topk_topp_triton import _update_min_larger_stats as _helper
    _update_min_larger_stats = _helper

    min_larger = t["min_larger"].clone()
    num_min_larger = t["num_min_larger"].clone()
    _update_min_larger_wrapper[(1,)](
        t["data"], t["above_mask"], min_larger, num_min_larger,
        sentinel=float("inf"),
        BLOCK_SIZE=params["block_size"],
    )
    if side == "gpu":
        torch.cuda.synchronize()
    else:
        torch.npu.synchronize()
    return {"min_larger": min_larger, "num_min_larger": num_min_larger}


def _mk(name: str, pattern: str, block: int) -> CaseSpec:
    return CaseSpec(
        kernel="update_min_larger_stats", name=name,
        params={"pattern": pattern, "block_size": block},
        seed=42,
        output_modes={"min_larger": cr.MODE_F32, "num_min_larger": cr.MODE_INT_EXACT},
    )


CASES = [
    _mk("new_min_64", "new_min", 64),
    _mk("same_min_64", "same_min", 64),
    _mk("larger_min_32", "larger_min", 32),
    _mk("partial_mask_128", "partial_mask", 128),
    _mk("no_above_64", "no_above", 64),
]
