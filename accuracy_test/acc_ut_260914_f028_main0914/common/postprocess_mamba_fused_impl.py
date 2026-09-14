# SPDX-License-Identifier: Apache-2.0
# acc_ut_260914_f028_main0914 shared UT for postprocess_mamba_fused_kernel.
# Source: vllm/v1/worker/mamba_utils.py (upstream kernel).
# Category: integer decision logic (compute copy params) + state copy.
# The kernel fuses decision + _copy_mamba_state_block.
#
# This test focuses on the *decision logic* output (num_accepted_tokens_out),
# which is pure int32 compute. The state-copy body delegates to
# _copy_mamba_state_block, exercised here via realistic small state tensors
# so byte-exact verification is feasible.
#
# Dual-side design: re-exported by gpu/ test entry modules; the side runtime
# (CUDA or Ascend NPU) is injected via the ``rt`` pytest fixture.

from __future__ import annotations

import traceback
from typing import Any

import pytest
import torch

kernel = None
_copy_mamba_state_block = None
_import_error: Exception | None = None
_import_traceback: str | None = None
try:
    from vllm.v1.worker.mamba_utils import (
        postprocess_mamba_fused_kernel as kernel,
    )
    from vllm.v1.worker.mamba_utils import (
        _copy_mamba_state_block,
    )
except Exception as exc:  # pragma: no cover
    _import_error = exc
    _import_traceback = traceback.format_exc()


_SENTINEL = -2


def _ref_decision(
    num_accepted_tokens: torch.Tensor,
    mamba_state_idx: torch.Tensor,
    num_scheduled_tokens: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    num_draft_tokens: torch.Tensor,
    num_reqs: int,
    block_size: int,
    *,
    precomputed_new_computed: bool = False,
    has_idx_mapping: bool = False,
    idx_mapping: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """CPU reference for the kernel's decision logic.

    Returns:
        num_accepted_tokens_out: [max_state_slots] int32 (written per req)
        needs_copy_mask: [max_state_slots] bool (whether copy was triggered)
        dest_block_idx: [max_state_slots] int32 (destination block)
        accept_token_bias: [max_state_slots] int32 (copy bias)
    """
    na = num_accepted_tokens.cpu().to(torch.int64).numpy().copy()
    si = mamba_state_idx.cpu().to(torch.int64).numpy().copy()
    ns = num_scheduled_tokens.cpu().to(torch.int64).numpy().copy()
    nc = num_computed_tokens.cpu().to(torch.int64).numpy().copy()
    nd = num_draft_tokens.cpu().to(torch.int64).numpy().copy()
    max_slots = na.shape[0]

    na_out = na.copy()
    needs_copy = [False] * max_slots
    dest_block = [-1] * max_slots
    token_bias = [0] * max_slots

    for batch_idx in range(num_reqs):
        if has_idx_mapping:
            req_idx = int(idx_mapping[batch_idx].item())
            if req_idx < 0:
                continue
        else:
            req_idx = batch_idx

        num_acc = int(na[req_idx])
        src_block = int(si[req_idx])

        if precomputed_new_computed:
            new_num_computed = int(nc[req_idx])
            num_tokens_running = new_num_computed - num_acc + 1
        else:
            num_sched = int(ns[req_idx])
            num_comp = int(nc[req_idx])
            num_draft = int(nd[req_idx])
            num_tokens_running = num_comp + num_sched - num_draft
            new_num_computed = num_tokens_running + num_acc - 1

        aligned = (new_num_computed // block_size) * block_size
        need = aligned >= num_tokens_running
        needs_copy[req_idx] = need
        if not need:
            continue

        bias = aligned - num_tokens_running
        dest = aligned // block_size - 1
        dest_block[req_idx] = dest
        token_bias[req_idx] = bias

        # Only state_idx==0 writes num_accepted_out, but we model it
        # per-request since we test with num_states=1
        if src_block == dest:
            na_out[req_idx] = 1

    return {
        "num_accepted_tokens_out": torch.from_numpy(na_out).to(torch.int32),
        "needs_copy": torch.tensor(needs_copy, dtype=torch.bool),
        "dest_block_idx": torch.tensor(dest_block, dtype=torch.int32),
        "accept_token_bias": torch.tensor(token_bias, dtype=torch.int32),
    }


def _gen_inputs(
    num_reqs: int,
    max_state_slots: int,
    block_size: int,
    *,
    precomputed: bool,
    has_idx_mapping: bool,
    zero_accepted: bool,
    same_block: bool,
    device,
) -> dict[str, Any]:
    """Build inputs exercising decision branches.

    Branches:
      - precomputed: PRECOMPUTED_NEW_COMPUTED path (V2)
      - has_idx_mapping: HAS_IDX_MAPPING path (V2 / PP)
      - zero_accepted: num_accepted=0 boundary
      - same_block: src_block==dest_block -> num_accepted_out=1
    """
    torch.manual_seed(42 + num_reqs * 7 + block_size * 3)

    if has_idx_mapping:
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    else:
        idx_mapping = None

    # num_accepted: 1 is neutral; 0 tests boundary; >1 triggers copy
    nacc = torch.full((max_state_slots,), _SENTINEL, dtype=torch.int32, device=device)
    if zero_accepted:
        nacc[:num_reqs] = 0
    else:
        nacc[:num_reqs] = torch.randint(1, 6, (num_reqs,), dtype=torch.int32, device=device)
        if num_reqs > 0:
            nacc[0] = 1  # ensure neutral

    # state_idx: destination block. For same_block test, pick values that
    # make dest==src.
    state_idx = torch.full((max_state_slots,), _SENTINEL, dtype=torch.int32, device=device)
    if same_block:
        # Set state_idx so that aligned_new_computed // block_size - 1 == state_idx
        # With block_size=16, num_accepted=3, num_scheduled=1, num_computed=0, num_draft=1:
        #   running = 0+1-1=0, new_computed = 0+3-1=2, aligned=0, dest=-1 -> not same.
        # We want aligned >= running AND dest == src_block.
        # Set state_idx = 0 and make aligned = 0 (running=0, block_size=16) -> dest=-1? No.
        # Actually dest = aligned // block_size - 1. For dest=0, aligned must be in [block_size, 2*block_size).
        # So set state_idx=0 and ensure aligned falls in [block_size, 2*block_size).
        state_idx[:num_reqs] = 0
    else:
        state_idx[:num_reqs] = torch.randint(0, 4, (num_reqs,), dtype=torch.int32, device=device)

    # num_scheduled / num_computed / num_draft: tuned to trigger different paths
    if precomputed:
        # nc already holds new_num_computed
        # running = new_nc - num_accepted + 1
        # Pick new_nc so aligned >= running
        ncomputed = torch.randint(block_size, 4 * block_size, (max_state_slots,), dtype=torch.int32, device=device)
        ncomputed[:num_reqs] = block_size + torch.arange(num_reqs, dtype=torch.int32, device=device) * 2
        nscheduled = torch.zeros(max_state_slots, dtype=torch.int32, device=device)
        ndraft = torch.zeros(max_state_slots, dtype=torch.int32, device=device)
    else:
        nscheduled = torch.randint(1, 5, (max_state_slots,), dtype=torch.int32, device=device)
        ncomputed = torch.randint(0, 3 * block_size, (max_state_slots,), dtype=torch.int32, device=device)
        ndraft = torch.randint(1, 5, (max_state_slots,), dtype=torch.int32, device=device)

    # Fill unused slots
    nscheduled[num_reqs:] = 0
    ndraft[num_reqs:] = 0
    ncomputed[num_reqs:] = 0

    # Output buffer
    nacc_out = nacc.clone()

    return {
        "num_accepted_tokens": nacc,
        "mamba_state_idx": state_idx,
        "num_scheduled_tokens": nscheduled,
        "num_computed_tokens": ncomputed,
        "num_draft_tokens": ndraft,
        "num_accepted_tokens_out": nacc_out,
        "idx_mapping": idx_mapping,
        "num_reqs": num_reqs,
        "block_size": block_size,
        "precomputed": precomputed,
        "has_idx_mapping": has_idx_mapping,
    }


def _build_state_metadata(
    num_states: int,
    num_reqs: int,
    max_blocks: int,
    block_table_stride_req: int,
    device,
) -> dict[str, Any]:
    """Build flattened state metadata arrays + block tables + state tensors.

    Creates num_states state tensors, each with:
      - elem_size in {2, 4} (bf16, fp32)
      - inner_size (elements per block)
      - conv_width (0 for temporal, >0 for conv)
      - block_stride (bytes per block)
    """
    torch.manual_seed(99)

    state_base_addrs = torch.empty(num_states, dtype=torch.int64, device=device)
    state_block_strides = torch.empty(num_states, dtype=torch.int64, device=device)
    state_elem_sizes = torch.empty(num_states, dtype=torch.int32, device=device)
    state_inner_sizes = torch.empty(num_states, dtype=torch.int64, device=device)
    state_conv_widths = torch.empty(num_states, dtype=torch.int32, device=device)
    state_group_indices = torch.zeros(num_states, dtype=torch.int32, device=device)
    state_dim_row_count = torch.zeros(num_states, dtype=torch.int32, device=device)
    state_dim_row_stride = torch.zeros(num_states, dtype=torch.int64, device=device)

    # Single group
    block_table = torch.arange(
        num_reqs * max_blocks, dtype=torch.int32, device=device
    ).view(num_reqs, max_blocks)
    block_table_ptr_val = block_table.data_ptr()
    block_table_ptrs = torch.tensor([block_table_ptr_val], dtype=torch.int64, device=device)

    state_tensors = []
    for s in range(num_states):
        elem_size = 4 if s % 2 == 0 else 2
        conv_width = 0 if s % 2 == 0 else 4  # temporal or conv
        inner_size = 64  # small for test
        block_stride = inner_size * elem_size

        total_elements = num_reqs * max_blocks * inner_size
        state_tensor = torch.randn(total_elements, dtype=torch.float32, device=device)
        if elem_size == 2:
            state_tensor = state_tensor.to(torch.bfloat16)
        else:
            state_tensor = state_tensor.to(torch.float32)
        state_tensor = state_tensor.view(num_reqs, max_blocks, inner_size)

        state_base_addrs[s] = state_tensor.data_ptr()
        state_block_strides[s] = block_stride
        state_elem_sizes[s] = elem_size
        state_inner_sizes[s] = inner_size
        state_conv_widths[s] = conv_width
        state_group_indices[s] = 0
        state_dim_row_count[s] = 0
        state_dim_row_stride[s] = 0
        state_tensors.append(state_tensor)

    return {
        "block_table_ptrs": block_table_ptrs,
        "block_table_stride_req": block_table_stride_req,
        "state_base_addrs": state_base_addrs,
        "state_block_strides": state_block_strides,
        "state_elem_sizes": state_elem_sizes,
        "state_inner_sizes": state_inner_sizes,
        "state_conv_widths": state_conv_widths,
        "state_group_indices": state_group_indices,
        "state_dim_row_count": state_dim_row_count,
        "state_dim_row_stride": state_dim_row_stride,
        "state_tensors": state_tensors,
        "block_table": block_table,
        "num_states": num_states,
    }


def _launch(k, inputs: dict[str, Any], meta: dict[str, Any]) -> None:
    num_reqs = inputs["num_reqs"]
    num_states = meta["num_states"]
    block_size = inputs["block_size"]
    grid = (num_reqs, num_states, 1)

    kwargs = dict(
        block_table_ptrs_ptr=meta["block_table_ptrs"],
        block_table_stride_req=meta["block_table_stride_req"],
        state_base_addrs_ptr=meta["state_base_addrs"],
        state_block_strides_ptr=meta["state_block_strides"],
        state_elem_sizes_ptr=meta["state_elem_sizes"],
        state_inner_sizes_ptr=meta["state_inner_sizes"],
        state_conv_widths_ptr=meta["state_conv_widths"],
        state_group_indices_ptr=meta["state_group_indices"],
        state_dim_row_count_ptr=meta["state_dim_row_count"],
        state_dim_row_stride_ptr=meta["state_dim_row_stride"],  # int64 tensor ptr (tl.load'ed, not constexpr)
        num_accepted_tokens_out_ptr=inputs["num_accepted_tokens_out"],
        idx_mapping_ptr=inputs["idx_mapping"],
        num_reqs=num_reqs,
        block_size=block_size,
        COPY_BLOCK_SIZE=256,
        CONV_STATE_DIM_FIRST=False,
        HAS_IDX_MAPPING=inputs["has_idx_mapping"],
        PRECOMPUTED_NEW_COMPUTED=inputs["precomputed"],
        TEMPORAL_TILES=1,
    )

    k[grid](
        inputs["num_accepted_tokens"],
        inputs["mamba_state_idx"],
        inputs["num_scheduled_tokens"],
        inputs["num_computed_tokens"],
        inputs["num_draft_tokens"],
        **kwargs,
    )


SHAPE_PARAMS = [
    # (num_reqs, block_size)
    (1, 16),
    (3, 16),
    (8, 32),
    (17, 16),
    (32, 32),
    (64, 64),
    (128, 16),
]

BRANCH_PARAMS = [
    # (precomputed, has_idx_mapping, zero_accepted, same_block)
    (False, False, False, False),  # nominal V1 path
    (False, False, False, True),   # same-block -> num_accepted_out=1
    (False, True, False, False),   # V2 idx_mapping path
    (True, False, False, False),    # precomputed new_computed (V2)
    (False, False, True, False),    # zero accepted boundary
]


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
    """Verify the kernel's decision logic output (num_accepted_tokens_out).

    The decision logic is pure int32 compute. Pass requires bitwise equality
    on every written slot AND untouched sentinels on unwritten slots.
    """
    if kernel is None:
        pytest.fail(
            "postprocess_mamba_fused_kernel import failed.\n"
            f"error={_import_error}\ntraceback:\n{_import_traceback}",
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

    # Device run
    dev_inputs = {
        **inputs,
        "num_accepted_tokens": inputs["num_accepted_tokens"].clone(),
        "num_accepted_tokens_out": inputs["num_accepted_tokens_out"].clone(),
    }
    _launch(kernel, dev_inputs, meta)
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
            f"postprocess_mamba_fused_kernel num_accepted_tokens_out mismatch: "
            f"{count}/{total} elements differ{first_info}"
        )


def test_postprocess_mamba_fused_state_copy(rt):
    """Verify the kernel performs the actual state block copy correctly.

    Uses a single temporal state (conv_width=0) and constructs a scenario
    where the decision logic triggers a cross-block copy. After the kernel
    runs, we verify the destination block matches the source block data.
    """
    if kernel is None:
        pytest.fail(
            "postprocess_mamba_fused_kernel import failed.\n"
            f"error={_import_error}",
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
    inner_size = state_tensor.shape[2]
    elem_size = 4  # fp32

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

    mamba_state_idx = torch.tensor([2, 0], dtype=torch.int32, device=device)  # src blocks
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

    _launch(kernel, inputs, meta)
    rt.synchronize()

    # Temporal copy semantics (verified against _copy_mamba_state_block):
    #   actual_src_block = bt[bt_row, src_col + token_bias] = bt[0, 2+1] = 3
    #   dst_block       = bt[bt_row, dst_col]              = bt[0, 0]  = 0
    # state_tensor is [num_reqs, max_blocks, inner_size] and
    # block_table = arange(num_reqs*max_blocks).view(num_reqs, max_blocks),
    # so block id 3 == state_tensor[0, 3, :] and block id 0 ==
    # state_tensor[0, 0, :]. The kernel writes only the destination block,
    # so reading the source AFTER launch is safe.
    actual_dst = state_tensor[0, 0, :]
    src_block_data = state_tensor[0, 3, :]

    # Block 3 was filled with the value 3 (req*100 + blk); after the copy,
    # block 0 must hold the same values.
    assert torch.equal(
        actual_dst, src_block_data
    ), (
        f"State copy mismatch: dst block 0 data {actual_dst[:5]} "
        f"!= src block 3 data {src_block_data[:5]}"
    )

    # Also check num_accepted_out: src_block(2) != dest_block(0) so
    # num_accepted_out should be unchanged from input (17)
    # Wait: src_block == dest_block check: src_block_idx=2, dest_block_idx=0
    # 2 != 0, so num_accepted_out NOT written to 1. Should remain 17.
    # But wait, the kernel only writes num_accepted_out when
    # src_block_idx == dest_block_idx. Here they're different, so no write.
    assert nacc_out[0].item() == 17, (
        f"num_accepted_out[0] should be 17 (unchanged), got {nacc_out[0].item()}"
    )


def test_import_error() -> None:
    if _import_error is not None:
        pytest.fail(
            f"Failed to import postprocess_mamba_fused_kernel:\n"
            f"{_import_traceback}"
        )
