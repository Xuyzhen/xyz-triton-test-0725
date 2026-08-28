"""expand_idx_mapping: integer-expansion kernel (exact-match class).

Both sides import the same vanilla vllm module path; on the NPU host it is
compiled by triton-ascend (see npu_ut_shapes.md at the suite root).
"""

from __future__ import annotations

import torch

import capture_runtime as cr
from capture_runtime import CaseSpec


def build_inputs(params: dict, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    if params.get("cu_num_logits") is not None:
        cu = torch.tensor([0] + params["cu_num_logits"], dtype=torch.int64)
    else:
        n, t = params["num_reqs"], params["tokens_per_req"]
        cu = torch.arange(0, n * t + 1, t, dtype=torch.int64)
    num_reqs = cu.numel() - 1
    if params.get("idx_mapping") is not None:
        idx = torch.tensor(params["idx_mapping"], dtype=torch.int32)
    else:
        # non-contiguous state indices are legal; draw deterministically
        idx = torch.randint(0, max(2 * num_reqs, 2), (num_reqs,), generator=g, dtype=torch.int32)
    return {"idx_mapping": idx, "cu_num_logits": cu}


def run(side: str, t: dict[str, torch.Tensor], params: dict) -> dict[str, torch.Tensor]:
    from vllm.triton_utils import triton
    from vllm.v1.worker.gpu.input_batch import _expand_idx_mapping_kernel

    idx, cu = t["idx_mapping"], t["cu_num_logits"]
    num_reqs = idx.shape[0]
    total = int(cu[-1].item())
    tokens = [int(cu[i + 1] - cu[i]) for i in range(num_reqs)]
    out_map = torch.empty(total, dtype=torch.int64, device=idx.device)
    out_pos = torch.empty(total, dtype=torch.int32, device=idx.device)
    _expand_idx_mapping_kernel[(num_reqs,)](
        idx, out_map, out_pos, cu,
        BLOCK_SIZE=triton.next_power_of_2(max(max(tokens), 1)),
    )
    if side == "gpu":
        torch.cuda.synchronize()
    else:
        torch.npu.synchronize()
    return {"expanded_idx_mapping": out_map, "expanded_local_pos": out_pos}


def ref(t: dict[str, torch.Tensor], params: dict) -> dict[str, torch.Tensor]:
    """Golden: out_map[start+j] = idx[r], out_pos[start+j] = j per request."""
    idx, cu = t["idx_mapping"].long(), t["cu_num_logits"].long()
    num_reqs = idx.shape[0]
    total = int(cu[-1].item())
    out_map = torch.empty(total, dtype=torch.int64)
    out_pos = torch.empty(total, dtype=torch.int32)
    for r in range(num_reqs):
        s, e = int(cu[r]), int(cu[r + 1])
        out_map[s:e] = idx[r]
        out_pos[s:e] = torch.arange(e - s, dtype=torch.int32)
    return {"expanded_idx_mapping": out_map, "expanded_local_pos": out_pos}


def _mk(name: str, params: dict) -> CaseSpec:
    return CaseSpec(
        kernel="expand_idx_mapping",
        name=name,
        params=params,
        seed=42,
        output_modes={"expanded_idx_mapping": cr.MODE_INT_EXACT, "expanded_local_pos": cr.MODE_INT_EXACT},
    )


CASES = [
    _mk("basic_1r_1t", {"num_reqs": 1, "tokens_per_req": 1}),
    _mk("basic_2r_3t", {"num_reqs": 2, "tokens_per_req": 3}),
    _mk("basic_4r_8t", {"num_reqs": 4, "tokens_per_req": 8}),
    _mk("uneven_2_5_3", {"cu_num_logits": [2, 7, 10]}),
    _mk("noncontig_mapping", {"cu_num_logits": [2, 4, 6], "idx_mapping": [5, 2, 8]}),
]
