# SPDX-License-Identifier: Apache-2.0
"""NPU-side accuracy UT for postprocess_mamba_fused_kernel.

Tests the **downstream** (vllm-ascend) implementation directly:
    from vllm_ascend.ops.triton.mamba.postprocess import postprocess_mamba_fused_kernel

The downstream kernel rewrites the copy body for Ascend Triton (inlined
byte-pointer copy, no ``_copy_mamba_state_block`` helper, nested ifs
instead of chained ``and``), but the decision logic and kernel signature
are identical to upstream.  CPU reference and metadata builders are
reused from the shared common impl; only the kernel object differs.
"""
from __future__ import annotations

import traceback

import pytest
import torch

# runtime_npu must be imported before any vllm_ascend import: it installs
# the vllm.triton_utils shim and device-property helpers.
from accuracy_test.acc_ut_260914_f028_main0914.runtime_npu import (  # noqa: F401
    STRICT_DEVICE,
    init_device_properties_triton,
    synchronize,
)

# Reuse shared helpers (CPU reference, input generator, metadata builder,
# launch wrapper, shape/branch parameter sets).
from accuracy_test.acc_ut_260914_f028_main0914.common.postprocess_mamba_fused_impl import (
    BRANCH_PARAMS,
    SHAPE_PARAMS,
    _build_state_metadata,
    _gen_inputs,
    _launch,
    _ref_decision,
)

# --- Import the downstream (vllm-ascend) kernel -----------------------
_npu_import_error: Exception | None = None
_npu_import_traceback: str | None = None
try:
    from vllm_ascend.ops.triton.mamba.postprocess import (
        postprocess_mamba_fused_kernel as npu_kernel,
    )
except Exception as exc:  # pragma: no cover
    npu_kernel = None
    _npu_import_error = exc
    _npu_import_traceback = traceback.format_exc()


# --- Re-define the rt fixture locally (the common impl expects one) ---
@pytest.fixture
def rt():
    import accuracy_test.acc_ut_260914_f028_main0914.runtime_npu as _rt
    return _rt


# =====================================================================
# Test 1: Decision logic (num_accepted_tokens_out)
# =====================================================================
@pytest.mark.parametrize("num_reqs,block_size", SHAPE_PARAMS)
@pytest.mark.parametrize(
    "precomputed,has_idx_mapping,zero_accepted,same_block", BRANCH_PARAMS
)
def test_postprocess_mamba_fused_decision(
    num_reqs: int,
    block_size: int,
    precomputed: bool,
    has_idx_mapping: bool,
    zero_accepted: bool,
    same_block: bool,
    rt,
):
    """Verify the downstream kernel's decision logic (num_accepted_tokens_out).

    Decision logic is pure int32 compute.  Pass requires bitwise equality
    on every written slot AND untouched sentinels on unwritten slots.
    """
    if npu_kernel is None:
        pytest.fail(
            "downstream postprocess_mamba_fused_kernel import failed.\n"
            f"error={_npu_import_error}\ntraceback:\n{_npu_import_traceback}",
            pytrace=False,
        )

    rt.init_device_properties_triton()
    device = rt.STRICT_DEVICE

    max_state_slots = num_reqs
    inputs = _gen_inputs(
        num_reqs=num_reqs,
        max_state_slots=max_state_slots,
        block_size=block_size,
        precomputed=precomputed,
        has_idx_mapping=has_idx_mapping,
        zero_accepted=zero_accepted,
        same_block=same_block,
        device=device,
    )

    # CPU reference
    ref_inputs = {
        "num_accepted_tokens": inputs["num_accepted_tokens"].cpu().clone(),
        "mamba_state_idx": inputs["mamba_state_idx"].cpu().clone(),
        "num_scheduled_tokens": inputs["num_scheduled_tokens"].cpu().clone(),
        "num_computed_tokens": inputs["num_computed_tokens"].cpu().clone(),
        "num_draft_tokens": inputs["num_draft_tokens"].cpu().clone(),
        "num_reqs": num_reqs,
        "block_size": block_size,
        "precomputed_new_computed": precomputed,
        "has_idx_mapping": has_idx_mapping,
        "idx_mapping": inputs["idx_mapping"].cpu().clone() if has_idx_mapping else None,
    }
    expected = _ref_decision(**ref_inputs)

    # Build state metadata for launch (small, 1 state)
    meta = _build_state_metadata(
        num_states=1,
        num_reqs=max_state_slots,
        max_blocks=8,
        block_table_stride_req=8,
        device=device,
    )

    # Device run with downstream kernel
    dev_inputs = {
        **inputs,
        "num_accepted_tokens": inputs["num_accepted_tokens"].clone(),
        "num_accepted_tokens_out": inputs["num_accepted_tokens_out"].clone(),
    }
    _launch(npu_kernel, dev_inputs, meta)
    rt.synchronize()

    # Check num_accepted_tokens_out
    dev_out = dev_inputs["num_accepted_tokens_out"].cpu()
    ref_out = expected["num_accepted_tokens_out"]
    assert dev_out.dtype == ref_out.dtype, "dtype mismatch"
    assert dev_out.shape == ref_out.shape, "shape mismatch"
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
            f"[downstream] postprocess_mamba_fused_kernel num_accepted_tokens_out "
            f"mismatch: {count}/{total} elements differ{first_info}"
        )


# =====================================================================
# Test 2: State block copy
# =====================================================================
def test_postprocess_mamba_fused_state_copy(rt):
    """Verify the downstream kernel performs the actual state block copy.

    Constructs a cross-block temporal copy scenario and checks that the
    destination block matches the source block data after the kernel runs.
    """
    if npu_kernel is None:
        pytest.fail(
            "downstream postprocess_mamba_fused_kernel import failed.\n"
            f"error={_npu_import_error}",
            pytrace=False,
        )

    rt.init_device_properties_triton()
    device = rt.STRICT_DEVICE

    num_reqs = 2
    block_size = 16
    max_blocks = 8
    num_states = 1

    # Build state tensors: [num_reqs, max_blocks, inner_size]
    meta = _build_state_metadata(
        num_states=num_states,
        num_reqs=num_reqs,
        max_blocks=max_blocks,
        block_table_stride_req=max_blocks,
        device=device,
    )

    state_tensor = meta["state_tensors"][0]  # [num_reqs, max_blocks, inner_size]

    # Fill state tensor with recognizable pattern
    for req in range(num_reqs):
        for blk in range(max_blocks):
            state_tensor[req, blk, :] = req * 100 + blk

    # Cross-block copy scenario for req 0 (block_size=16):
    #   num_accepted=17, num_scheduled=1, num_computed=15, num_draft=1
    #   running = 15+1-1 = 15, new_computed = 15+17-1 = 31,
    #   aligned = (31//16)*16 = 16 >= 15 -> copy triggered
    #   dest_block = 16//16-1 = 0, accept_token_bias = 16-15 = 1
    #   src_block(mamba_state_idx) = 2 != dest_block = 0 -> cross-block copy
    # Req 1 stays degenerate (no copy).

    mamba_state_idx = torch.tensor([2, 0], dtype=torch.int32, device=device)
    num_accepted = torch.tensor([17, 1], dtype=torch.int32, device=device)
    num_scheduled = torch.tensor([1, 1], dtype=torch.int32, device=device)
    num_computed = torch.tensor([15, 0], dtype=torch.int32, device=device)
    num_draft = torch.tensor([1, 1], dtype=torch.int32, device=device)
    nacc_out = num_accepted.clone()

    inputs = {
        "num_accepted_tokens": num_accepted,
        "mamba_state_idx": mamba_state_idx,
        "num_scheduled_tokens": num_scheduled,
        "num_computed_tokens": num_computed,
        "num_draft_tokens": num_draft,
        "num_accepted_tokens_out": nacc_out,
        "idx_mapping": None,
        "num_reqs": num_reqs,
        "block_size": block_size,
        "precomputed": False,
        "has_idx_mapping": False,
    }

    _launch(npu_kernel, inputs, meta)
    rt.synchronize()

    # Temporal copy semantics (inlined in downstream kernel):
    #   actual_src_block = bt[bt_row, src_col + token_bias] = bt[0, 2+1] = 3
    #   dst_block       = bt[bt_row, dst_col]              = bt[0, 0]  = 0
    # block_table = arange(num_reqs*max_blocks).view(num_reqs, max_blocks),
    # so block id 3 == state_tensor[0, 3, :] and block id 0 ==
    # state_tensor[0, 0, :].
    actual_dst = state_tensor[0, 0, :]
    src_block_data = state_tensor[0, 3, :]

    assert torch.equal(
        actual_dst, src_block_data
    ), (
        f"[downstream] State copy mismatch: dst block 0 data {actual_dst[:5]} "
        f"!= src block 3 data {src_block_data[:5]}"
    )

    # src_block_idx(2) != dest_block_idx(0) -> num_accepted_out unchanged (17)
    assert nacc_out[0].item() == 17, (
        f"[downstream] num_accepted_out[0] should be 17 (unchanged), "
        f"got {nacc_out[0].item()}"
    )


def test_import_error() -> None:
    if _npu_import_error is not None:
        pytest.fail(
            f"Failed to import downstream postprocess_mamba_fused_kernel:\n"
            f"{_npu_import_traceback}"
        )
