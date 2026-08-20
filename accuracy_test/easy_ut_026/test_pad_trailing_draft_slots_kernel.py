# SPDX-License-Identifier: Apache-2.0
# easy_ut_026 strict UT for _pad_trailing_draft_slots_kernel.
# Source: vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py
# (upstream, no vllm-ascend patch). Category: non-compute scatter/pad (writes
# PAD_SLOT_ID=-1 to [last_token_index+1, query_end) for each group/req).
# All outputs int32 -> bitwise comparison.
#
# Kernel summary (grid = (num_groups, num_reqs)):
#   Per (group_idx, req_idx):
#     start = last_token_indices[req_idx] + 1
#     end = query_start_loc[req_idx + 1]
#     for i in range(start, end, BLOCK_SIZE):
#       offs = i + arange(0, BLOCK_SIZE)
#       mask = offs < end
#       slot_mappings[group, offs] = PAD_ID (masked)

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
        _pad_trailing_draft_slots_kernel as kernel,
    )
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID
except Exception as exc:  # pragma: no cover
    _import_error = exc
    _import_traceback = traceback.format_exc()


_SENTINEL = -12345  # never equal to PAD_SLOT_ID (-1); catches stray writes


def _ref(
    num_groups: int,
    num_reqs: int,
    num_tokens: int,
    slot_mappings_init: torch.Tensor,  # [num_groups, num_tokens]
    query_start_loc: torch.Tensor,    # [num_reqs + 1]
    last_token_indices: torch.Tensor,  # [num_reqs]
    pad_id: int,
) -> torch.Tensor:
    """Independent CPU reference. Returns expected slot_mappings."""
    sm = slot_mappings_init.cpu().clone()
    qsl = query_start_loc.cpu().to(torch.int64).tolist()
    lti = last_token_indices.cpu().to(torch.int64).tolist()
    for g in range(num_groups):
        for r in range(num_reqs):
            start = lti[r] + 1
            end = qsl[r + 1]
            for i in range(start, end):
                sm[g, i] = pad_id
    return sm


def _gen_inputs(
    num_groups: int,
    num_reqs: int,
    num_tokens: int,
    *,
    scenario: str,
    device: str,
    block_size: int = 8,
) -> dict[str, Any]:
    torch.manual_seed(42 + num_groups * 7 + num_reqs)

    idx_mapping = torch.arange(num_reqs, dtype=torch.int64, device=device)

    # Build query_start_loc and last_token_indices
    if scenario == "no_pad":
        # last_token_index = query_end - 1 (no trailing slots to pad)
        query_lens = torch.randint(3, 10, (num_reqs,), dtype=torch.int32,
                                   device=device)
        qsl = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        qsl_cpu = qsl.cpu()
        qsl_cpu[0] = 0
        for r in range(num_reqs):
            qsl_cpu[r + 1] = qsl_cpu[r] + int(query_lens[r].item())
        qsl.copy_(qsl_cpu)
        last_token_indices = (qsl[1:num_reqs + 1] - 1).to(torch.int32)
    elif scenario == "full_pad":
        # last_token_index = query_start (pad entire query except first slot)
        query_lens = torch.randint(3, 10, (num_reqs,), dtype=torch.int32,
                                   device=device)
        qsl = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        qsl_cpu = qsl.cpu()
        qsl_cpu[0] = 0
        for r in range(num_reqs):
            qsl_cpu[r + 1] = qsl_cpu[r] + int(query_lens[r].item())
        qsl.copy_(qsl_cpu)
        last_token_indices = qsl[:num_reqs].clone().to(torch.int32)
    else:  # "partial_pad"
        query_lens = torch.randint(5, 12, (num_reqs,), dtype=torch.int32,
                                   device=device)
        qsl = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        qsl_cpu = qsl.cpu()
        qsl_cpu[0] = 0
        for r in range(num_reqs):
            qsl_cpu[r + 1] = qsl_cpu[r] + int(query_lens[r].item())
        qsl.copy_(qsl_cpu)
        # last_token_index = query_start + random < query_len - 1
        lti_cpu = []
        for r in range(num_reqs):
            qs = int(qsl_cpu[r].item())
            qe = int(qsl_cpu[r + 1].item())
            ql = qe - qs
            lti_cpu.append(qs + int(torch.randint(0, max(1, ql - 1), ()).item()))
        last_token_indices = torch.tensor(lti_cpu, dtype=torch.int32,
                                          device=device)

    # slot_mappings: init with unique sentinel per (group, token)
    slot_mappings = torch.full((num_groups, num_tokens), _SENTINEL,
                               dtype=torch.int32, device=device)

    return {
        "num_groups": num_groups,
        "num_reqs": num_reqs,
        "num_tokens": num_tokens,
        "slot_mappings": slot_mappings,
        "query_start_loc": qsl,
        "last_token_indices": last_token_indices,
        "pad_id": PAD_SLOT_ID if kernel is not None else -1,
        "block_size": block_size,
    }


def _run_kernel(inputs: dict[str, Any], device: str) -> torch.Tensor:
    sm = inputs["slot_mappings"].clone()
    kernel[(inputs["num_groups"], inputs["num_reqs"])](
        sm,
        sm.stride(0),
        inputs["query_start_loc"],
        inputs["last_token_indices"],
        inputs["pad_id"],
        BLOCK_SIZE=inputs["block_size"],
        num_warps=1,
    )
    synchronize()
    return sm


def _assert_bitwise(name: str, expected: torch.Tensor, actual: torch.Tensor) -> None:
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
    "num_groups,num_reqs,num_tokens,scenario,block_size",
    [
        # No padding needed (last_token_index = query_end - 1)
        (1, 1, 32, "no_pad", 8),
        (2, 4, 64, "no_pad", 8),
        # Full pad (last_token_index = query_start)
        (1, 1, 32, "full_pad", 8),
        (2, 4, 64, "full_pad", 8),
        # Partial pad
        (1, 1, 32, "partial_pad", 8),
        (1, 4, 64, "partial_pad", 8),
        (3, 4, 128, "partial_pad", 8),
        # Tile boundary: pad region exactly = block_size
        (1, 1, 32, "full_pad", 8),  # query_len=3..10, may hit boundary
        # Larger batch
        (4, 8, 256, "partial_pad", 8),
        (2, 8, 128, "partial_pad", 4),  # smaller block
        # Realistic block size
        (2, 4, 1024, "partial_pad", 256),
    ],
    ids=lambda v: str(v),
)
def test_pad_trailing_draft_slots(
    num_groups: int,
    num_reqs: int,
    num_tokens: int,
    scenario: str,
    block_size: int,
) -> None:
    """Bitwise comparison: pure scatter/pad write."""
    if _import_error is not None:
        pytest.skip(f"kernel import failed: {_import_error}")
    init_device_properties_triton()

    inputs = _gen_inputs(
        num_groups=num_groups,
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        scenario=scenario,
        device=_STRICT_DEVICE,
        block_size=block_size,
    )

    expected = _ref(
        num_groups=inputs["num_groups"],
        num_reqs=inputs["num_reqs"],
        num_tokens=inputs["num_tokens"],
        slot_mappings_init=inputs["slot_mappings"],
        query_start_loc=inputs["query_start_loc"],
        last_token_indices=inputs["last_token_indices"],
        pad_id=inputs["pad_id"],
    )

    actual = _run_kernel(inputs, _STRICT_DEVICE)
    _assert_bitwise("slot_mappings", expected, actual)


def test_import_error() -> None:
    if _import_error is not None:
        pytest.fail(
            f"Failed to import _pad_trailing_draft_slots_kernel:\n"
            f"{_import_traceback}"
        )
