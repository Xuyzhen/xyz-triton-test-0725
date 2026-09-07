# SPDX-License-Identifier: Apache-2.0
# strict_ut_027 shared dual-side UT for preprocess_mamba_align_fused_kernel.
# Source: vllm/v1/worker/mamba_utils.py (upstream kernel, no vllm-ascend patch).
# Category: integer/index compute (per-req: emit src_col/src_off, advance
# state_idx, conditionally reset num_accepted). All outputs int32 -> bitwise.
#
# Dual-side design: re-exported by gpu/ and npu/ test entry modules; the
# side runtime (CUDA or Ascend NPU) is injected via the ``rt`` pytest fixture.

from __future__ import annotations

import traceback
from typing import Any

import pytest
import torch

kernel = None
_import_error: Exception | None = None
_import_traceback: str | None = None
try:
    from vllm.v1.worker.mamba_utils import (
        preprocess_mamba_align_fused_kernel as kernel,
    )
except Exception as exc:  # pragma: no cover - surfaced as pytest.fail at runtime
    _import_error = exc
    _import_traceback = traceback.format_exc()


# Sentinel for output buffers: -2 is never a legal src_col / src_off /
# state_idx value in production (fresh state_idx is -1, real columns are >=0).
_SENTINEL = -2


def _ref(
    idx_mapping: torch.Tensor,
    state_idx: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    src_col: torch.Tensor,
    src_off: torch.Tensor,
    num_reqs: int,
    block_size: int,
    mamba_block_size: int,
) -> dict[str, torch.Tensor]:
    """Independent CPU reference. Mirrors kernel program/grid semantics.

    Grid = (cdiv(num_reqs, BLOCK_SIZE),). Per program, BLOCK_SIZE lanes; lane
    is active iff offsets < num_reqs. idx_mapping is treated as injective (no
    -1 values): production callers filter skipped reqs before launch because
    this kernel has no req_idx<0 guard.
    """
    idx_np = idx_mapping.cpu().to(torch.int64).numpy()
    state_in = state_idx.cpu().to(torch.int64).numpy()
    ncomputed = num_computed_tokens.cpu().to(torch.int64).numpy()
    qsl = query_start_loc.cpu().to(torch.int64).numpy()
    nacc_in = num_accepted_tokens.cpu().to(torch.int64).numpy()
    src_col_out = src_col.cpu().to(torch.int64).numpy().copy()
    src_off_out = src_off.cpu().to(torch.int64).numpy().copy()
    state_out = state_in.copy()
    nacc_out = nacc_in.copy()

    num_progs = (num_reqs + block_size - 1) // block_size
    for pid in range(num_progs):
        for lane in range(block_size):
            offsets = pid * block_size + lane
            if offsets >= num_reqs:
                continue
            req_idx = int(idx_np[offsets])
            assert req_idx >= 0, (
                f"idx_mapping[{offsets}]={req_idx}<0 not supported by this "
                "kernel (production filters skipped reqs before launch)"
            )
            si = int(state_in[req_idx])
            na = int(nacc_in[req_idx])
            src_off_val = max(na - 1, 0)
            # Stores happen at req_idx (not offsets), matching kernel.
            src_col_out[req_idx] = si
            src_off_out[req_idx] = src_off_val
            nc = int(ncomputed[req_idx])
            qs = int(qsl[offsets])
            qe = int(qsl[offsets + 1])
            computed_after = nc + qe - qs
            new_si = (computed_after + mamba_block_size - 1) // mamba_block_size - 1
            state_out[req_idx] = new_si
            if si >= 0 and si != new_si:
                nacc_out[req_idx] = 1

    return {
        "src_col": torch.from_numpy(src_col_out).to(torch.int32),
        "src_off": torch.from_numpy(src_off_out).to(torch.int32),
        "state_idx": torch.from_numpy(state_out).to(torch.int32),
        "num_accepted_tokens": torch.from_numpy(nacc_out).to(torch.int32),
    }


def _gen_inputs(
    num_reqs: int,
    max_state_slots: int,
    block_size: int,
    mamba_block_size: int,
    *,
    fresh_state: bool,
    reset_path: bool,
    zero_accepted: bool,
    device,
) -> dict[str, Any]:
    """Build inputs that exercise each branch of the kernel decision tree.

    Branches:
      - fresh_state=True:  state_idx starts at -1 for all slots -> src_col=-1,
                            should_reset=False (state_idx>=0 is False).
      - reset_path=True:   state_idx>=0 and != new_state_idx -> num_accepted=1.
      - zero_accepted=True: num_accepted=0 -> src_off=max(-1,0)=0 boundary.
    """
    torch.manual_seed(42 + num_reqs * 7 + block_size * 3 + mamba_block_size)

    # idx_mapping: contiguous [0..num_reqs-1] -> req_state_idx 0..num_reqs-1.
    # No -1 values (contract: this kernel has no req_idx<0 guard).
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)

    # query_start_loc: monotonically non-decreasing, ends at num_reqs.
    # Make per-req query lengths vary across [1, mamba_block_size+3] to hit
    # both within-block and cross-block computed_after values.
    q_lens = torch.randint(1, mamba_block_size + 4, (num_reqs,), dtype=torch.int32, device=device)
    query_start_loc = torch.cat([
        torch.tensor([0], dtype=torch.int32, device=device),
        torch.cumsum(q_lens, dim=0).to(torch.int32),
    ])

    # num_computed_tokens: random in [0, 2*MBS) so computed_after can cross
    # block boundaries for some requests and not for others.
    ncomputed = torch.randint(0, 2 * mamba_block_size, (max_state_slots,), dtype=torch.int32, device=device)

    # state_idx: choose values to exercise the requested branch.
    state_idx = torch.full((max_state_slots,), _SENTINEL, dtype=torch.int32, device=device)
    if fresh_state:
        state_idx[:num_reqs] = -1
    else:
        # Random non-negative running columns in [0, 4*MBS).
        state_idx[:num_reqs] = torch.randint(0, 4 * mamba_block_size, (num_reqs,), dtype=torch.int32, device=device)
        if reset_path and num_reqs > 0:
            # Force at least one request to cross a block boundary so
            # new_state_idx != state_idx and reset fires.
            i = num_reqs // 2
            # Pick a state_idx such that (computed_after + MBS-1)//MBS - 1 != state_idx[i].
            # Use a known computed_after = ncomputed[i] + q_lens[i].
            ca = int(ncomputed[i].item()) + int(q_lens[i].item())
            new_si = (ca + mamba_block_size - 1) // mamba_block_size - 1
            # Set state_idx[i] to something != new_si and >=0.
            state_idx[i] = (new_si + 1) % (4 * mamba_block_size)

    # num_accepted_tokens: choose boundary values.
    nacc = torch.full((max_state_slots,), _SENTINEL, dtype=torch.int32, device=device)
    if zero_accepted:
        nacc[:num_reqs] = 0
    else:
        nacc[:num_reqs] = torch.randint(0, 8, (num_reqs,), dtype=torch.int32, device=device)
        if num_reqs > 0:
            # Ensure at least one request has num_accepted=1 (neutral bias).
            nacc[0] = 1

    # Outputs pre-filled with sentinel so unwritten slots are detectable.
    src_col = torch.full((max_state_slots,), _SENTINEL, dtype=torch.int32, device=device)
    src_off = torch.full((max_state_slots,), _SENTINEL, dtype=torch.int32, device=device)

    return {
        "idx_mapping": idx_mapping,
        "state_idx": state_idx,
        "num_computed_tokens": ncomputed,
        "query_start_loc": query_start_loc,
        "num_accepted_tokens": nacc,
        "src_col": src_col,
        "src_off": src_off,
        "num_reqs": num_reqs,
        "block_size": block_size,
        "mamba_block_size": mamba_block_size,
    }


def _launch(k, inputs: dict[str, Any]) -> None:
    num_reqs = inputs["num_reqs"]
    block_size = inputs["block_size"]
    grid = ((num_reqs + block_size - 1) // block_size,)
    k[grid](
        inputs["idx_mapping"],
        inputs["state_idx"],
        inputs["num_computed_tokens"],
        inputs["query_start_loc"],
        inputs["num_accepted_tokens"],
        inputs["src_col"],
        inputs["src_off"],
        num_reqs,
        BLOCK_SIZE=block_size,
        MAMBA_BLOCK_SIZE=inputs["mamba_block_size"],
    )


# Realistic shapes:
#   - num_reqs in {1, 3, 17, 64, 128} (1=degenerate, 3/17 non-power-of-2)
#   - max_state_slots = num_reqs (no padding) for the basic case
#   - BLOCK_SIZE in {1, 4, 16} (1=per-req programs, 16=typical vectorized)
#   - MAMBA_BLOCK_SIZE in {16, 32, 64, 128} (matches common Mamba models)
SHAPE_PARAMS = [
    # (num_reqs, block_size, mamba_block_size)
    (1, 1, 16),
    (1, 4, 32),
    (3, 1, 64),
    (3, 4, 128),
    (17, 1, 16),
    (17, 16, 32),
    (64, 1, 64),
    (64, 16, 128),
    (128, 1, 16),
    (128, 16, 64),
    # --- strict_ut_027 high-spec additions ---
    # Production-scale concurrency (256/512 running requests).
    (256, 16, 32),
    (256, 64, 64),
    (512, 16, 128),
    # Large BLOCK_SIZE (single-program bulk processing).
    (64, 128, 64),
    (128, 256, 32),
    # Degenerate boundary: BLOCK_SIZE > num_reqs (single partial program).
    (3, 64, 16),
    (17, 128, 128),
]

BRANCH_PARAMS = [
    # (fresh_state, reset_path, zero_accepted)
    (False, False, False),  # nominal: random state_idx, no forced reset
    (False, True, False),   # force one request across block boundary
    (True, False, False),   # all slots fresh (state_idx=-1, no reset)
    (False, False, True),   # num_accepted=0 -> src_off=max(-1,0)=0 boundary
]


@pytest.mark.parametrize(
    "num_reqs,block_size,mamba_block_size", SHAPE_PARAMS
)
@pytest.mark.parametrize(
    "fresh_state,reset_path,zero_accepted", BRANCH_PARAMS
)
def test_preprocess_mamba_align_fused(
    num_reqs: int,
    block_size: int,
    mamba_block_size: int,
    fresh_state: bool,
    reset_path: bool,
    zero_accepted: bool,
    rt,
):
    """Compare kernel (injected-side device) against an independent CPU reference.

    All outputs are int32 with sentinel pre-fill. Pass requires bitwise
    equality on every written slot AND untouched sentinels on every unwritten
    slot (catches over-write / wrong-index bugs).
    """
    if kernel is None:
        pytest.fail(
            "preprocess_mamba_align_fused_kernel import failed; this is not a "
            "precision failure and no kernel was tested.\n"
            f"error={_import_error}\ntraceback:\n{_import_traceback}",
            pytrace=False,
        )

    rt.init_device_properties_triton()
    device = rt.STRICT_DEVICE

    max_state_slots = num_reqs  # no padding for basic case
    inputs = _gen_inputs(
        num_reqs=num_reqs,
        max_state_slots=max_state_slots,
        block_size=block_size,
        mamba_block_size=mamba_block_size,
        fresh_state=fresh_state,
        reset_path=reset_path,
        zero_accepted=zero_accepted,
        device=device,
    )

    # CPU reference: clone tensors that the kernel mutates, so the original
    # inputs stay pristine for the device run.
    ref_inputs = {
        "idx_mapping": inputs["idx_mapping"].cpu().clone(),
        "state_idx": inputs["state_idx"].cpu().clone(),
        "num_computed_tokens": inputs["num_computed_tokens"].cpu().clone(),
        "query_start_loc": inputs["query_start_loc"].cpu().clone(),
        "num_accepted_tokens": inputs["num_accepted_tokens"].cpu().clone(),
        "src_col": inputs["src_col"].cpu().clone(),
        "src_off": inputs["src_off"].cpu().clone(),
        "num_reqs": num_reqs,
        "block_size": block_size,
        "mamba_block_size": mamba_block_size,
    }
    expected = _ref(**ref_inputs)

    # Device run: clone mutated tensors so we can compare against pristine CPU ref.
    dev_inputs = {
        "idx_mapping": inputs["idx_mapping"].clone(),
        "state_idx": inputs["state_idx"].clone(),
        "num_computed_tokens": inputs["num_computed_tokens"].clone(),
        "query_start_loc": inputs["query_start_loc"].clone(),
        "num_accepted_tokens": inputs["num_accepted_tokens"].clone(),
        "src_col": inputs["src_col"].clone(),
        "src_off": inputs["src_off"].clone(),
        "num_reqs": num_reqs,
        "block_size": block_size,
        "mamba_block_size": mamba_block_size,
    }
    _launch(kernel, dev_inputs)
    rt.synchronize()

    mutated = ("src_col", "src_off", "state_idx", "num_accepted_tokens")
    for name in mutated:
        dev_out = dev_inputs[name].cpu()
        ref_out = expected[name]
        assert dev_out.dtype == ref_out.dtype, f"{name}: dtype mismatch"
        assert dev_out.shape == ref_out.shape, f"{name}: shape mismatch"
        if not torch.equal(dev_out, ref_out):
            mismatched = torch.ne(dev_out, ref_out)
            count = int(mismatched.sum().item())
            total = int(mismatched.numel())
            first_idx = torch.nonzero(mismatched, as_tuple=False)
            first_info = ""
            if first_idx.numel() > 0:
                loc = tuple(first_idx[0].tolist())
                first_info = (
                    f"; first mismatch at {loc}: "
                    f"device={dev_out[loc].item()} ref={ref_out[loc].item()}"
                )
            pytest.fail(
                f"preprocess_mamba_align_fused_kernel mismatch for {name}: "
                f"{count}/{total} elements differ{first_info}"
            )


def test_import_error() -> None:
    if _import_error is not None:
        pytest.fail(
            f"Failed to import preprocess_mamba_align_fused_kernel:\n"
            f"{_import_traceback}"
        )
