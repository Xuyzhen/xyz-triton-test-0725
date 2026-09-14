# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.triton_utils import triton

from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    _selective_scan_update_kernel,
    selective_state_update,
    try_get_optimal_ssm_config,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _selective_scan_update_cpu(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    z: torch.Tensor | None,
    dt_softplus: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch CPU reference for one-step selective scan update.

    state: (batch, nheads, dim, dstate)
    x:     (batch, nheads, dim)
    dt:    (batch, nheads, dim)
    A:     (nheads, dim, dstate)
    B:     (batch, ngroups, dstate)
    C:     (batch, ngroups, dstate)
    D:     (nheads, dim)
    z:     (batch, nheads, dim)

    Returns:
      - updated state (in-place on a copy)
      - output y = state @ C + D * x  (optionally modulated by z)
    """
    batch, nheads, dim, dstate = state.shape
    ngroups = B.shape[1]
    nheads_ngroups_ratio = nheads // ngroups
    dt_bias_expanded = dt_bias  # (nheads, dim)

    state_out = state.clone()
    y = torch.zeros_like(x)

    for b in range(batch):
        for h in range(nheads):
            g = h // nheads_ngroups_ratio
            s = state_out[b, h]  # (dim, dstate)
            xi = x[b, h]  # (dim,)
            dti = dt[b, h]  # (dim,)

            if dt_softplus:
                dti = torch.where(dti <= 20.0, torch.log(torch.exp(dti) + 1), dti)

            # Check if TIE_HDIM: all dims share same dt and A scalar
            tie_hdim = (A.stride(-1) == 0 and A.stride(-2) == 0
                        and dt.stride(-1) == 0 and dt_bias.stride(-1) == 0)

            if tie_hdim:
                dt_val = dti[0].item()
                if dt_bias is not None:
                    dt_val += dt_bias[h, 0].item()
                if dt_softplus:
                    dt_val = max(dt_val, 0.0)  # softplus approx
                dA = torch.exp(A[h, 0, 0].item() * dt_val)
                dB = B[b, g] * dt_val  # (dstate,)
                for d in range(dim):
                    s[d] = s[d] * dA + dB * xi[d]
                yi = torch.sum(s * C[b, g], dim=-1)  # (dim,)
            else:
                dt_bias_i = dt_bias[h] if dt_bias is not None else 0
                dti = dti + dt_bias_i
                if dt_softplus:
                    dti = torch.where(dti <= 20.0, torch.log(torch.exp(dti) + 1), dti)

                A_h = A[h]  # (dim, dstate)
                B_g = B[b, g]  # (dstate,)
                C_g = C[b, g]  # (dstate,)

                dA = torch.exp(A_h * dti[:, None])  # (dim, dstate)
                dB = B_g[None, :] * dti[:, None]  # (dim, dstate)
                s = s * dA + dB * xi[:, None]

            if D is not None:
                D_h = D[h]
                yi = torch.sum(s * C_g, dim=-1) + xi * D_h
            else:
                yi = torch.sum(s * C_g, dim=-1)

            if z is not None:
                zi = z[b, h]
                yi = yi * zi * torch.sigmoid(zi)

            y[b, h] = yi
            state_out[b, h] = s

    return state_out, y


@pytest.fixture(autouse=True)
def _setup():
    init_device_properties_triton()


def test_selective_scan_update_basic() -> None:
    """Selective scan update: basic single-batch, single-head case.

    Verifies state update and output computation using DT_SOFTPLUS and
    all optional parameters (D, z, dt_bias) enabled.
    """
    torch.manual_seed(42)

    batch = 1
    nheads = 1
    dim = 8
    dstate = 4
    ngroups = 1

    device = torch.device("npu")

    state = torch.randn(batch, nheads, dim, dstate, dtype=torch.float32, device=device)
    x = torch.randn(batch, nheads, dim, dtype=torch.float32, device=device)
    dt = torch.randn(batch, nheads, dim, dtype=torch.float32, device=device)
    dt_bias = torch.randn(nheads, dim, dtype=torch.float32, device=device)
    A = torch.randn(nheads, dim, dstate, dtype=torch.float32, device=device)
    B = torch.randn(batch, ngroups, dstate, dtype=torch.float32, device=device)
    C = torch.randn(batch, ngroups, dstate, dtype=torch.float32, device=device)
    D = torch.randn(nheads, dim, dtype=torch.float32, device=device)
    z = torch.randn(batch, nheads, dim, dtype=torch.float32, device=device)
    out = torch.empty_like(x)

    cu_seqlens = None
    state_batch_indices = None
    dst_state_batch_indices = None
    null_block_id = NULL_BLOCK_ID
    num_accepted_tokens = None

    N = batch
    nheads_ngroups_ratio = nheads // ngroups
    dt_softplus = True
    tie_hdim = False

    BLOCK_SIZE_M, num_warps = try_get_optimal_ssm_config(
        dim, dstate, batch, nheads, "float32", is_blackwell=False
    )

    grid = (
        (dim + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        N,
        nheads,
    )

    _selective_scan_update_kernel[grid](
        state, None, x, dt, dt_bias, A, B, C, D, z, out,
        state_batch_indices, dst_state_batch_indices,
        null_block_id, num_accepted_tokens, cu_seqlens,
        N, nheads, dim, dstate, nheads_ngroups_ratio,
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        x.stride(0), x.stride(1), x.stride(2),
        dt.stride(0), dt.stride(1), dt.stride(2),
        dt_bias.stride(0), dt_bias.stride(1),
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        D.stride(0), D.stride(1),
        z.stride(0) if z is not None else 0,
        z.stride(1) if z is not None else 0,
        z.stride(2) if z is not None else 0,
        out.stride(0), out.stride(1), out.stride(2),
        0, 0,  # state_batch_indices strides (unused)
        0, 0,  # dst_state_batch_indices strides (unused)
        DT_SOFTPLUS=dt_softplus,
        TIE_HDIM=tie_hdim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        HAS_DT_BIAS=True,
        HAS_D=True,
        HAS_Z=True,
        HAS_STATE_BATCH_INDICES=False,
        IS_SPEC_DECODING=False,
        IS_VARLEN=False,
        BLOCK_SIZE_DSTATE=triton.next_power_of_2(dstate),
        USE_RS_ROUNDING=False,
        PHILOX_ROUNDS=0,
        num_warps=num_warps,
    )
    torch.npu.synchronize()

    # CPU reference
    state_cpu = state.cpu()
    x_cpu = x.cpu()
    dt_cpu = dt.cpu()
    dt_bias_cpu = dt_bias.cpu()
    A_cpu = A.cpu()
    B_cpu = B.cpu()
    C_cpu = C.cpu()
    D_cpu = D.cpu()
    z_cpu = z.cpu()

    # Use the non-TIE_HDIM path
    expected_state = state_cpu.clone()
    expected_out = torch.zeros_like(x_cpu)

    for b in range(batch):
        for h in range(nheads):
            g = h // nheads_ngroups_ratio
            s = expected_state[b, h]
            xi = x_cpu[b, h]
            dti = dt_cpu[b, h]
            dti = dti + dt_bias_cpu[h]
            dti = torch.where(dti <= 20.0, torch.log(torch.exp(dti) + 1), dti)

            A_h = A_cpu[h]
            B_g = B_cpu[b, g]
            C_g = C_cpu[b, g]

            dA_mat = torch.exp(A_h * dti[:, None])
            dB_mat = B_g[None, :] * dti[:, None]
            s_new = s * dA_mat + dB_mat * xi[:, None]

            yi = torch.sum(s_new * C_g[None, :], dim=-1) + xi * D_cpu[h]
            zi = z_cpu[b, h]
            yi = yi * zi * torch.sigmoid(zi)

            expected_out[b, h] = yi
            expected_state[b, h] = s_new

    # Compare output
    torch.testing.assert_close(out.cpu(), expected_out, rtol=1e-4, atol=1e-4)

    # Compare state
    torch.testing.assert_close(state.cpu(), expected_state, rtol=1e-4, atol=1e-4)


def test_selective_scan_update_no_z_no_d() -> None:
    """Selective scan: without z and D (HAS_Z=False, HAS_D=False).

    Verifies that the kernel works correctly when z_ptr and D_ptr are None.
    """
    torch.manual_seed(123)

    batch = 1
    nheads = 1
    dim = 4
    dstate = 2
    ngroups = 1

    device = torch.device("npu")

    state = torch.randn(batch, nheads, dim, dstate, dtype=torch.float32, device=device)
    x = torch.randn(batch, nheads, dim, dtype=torch.float32, device=device)
    dt = torch.randn(batch, nheads, dim, dtype=torch.float32, device=device)
    dt_bias = torch.randn(nheads, dim, dtype=torch.float32, device=device)
    A = torch.randn(nheads, dim, dstate, dtype=torch.float32, device=device)
    B = torch.randn(batch, ngroups, dstate, dtype=torch.float32, device=device)
    C = torch.randn(batch, ngroups, dstate, dtype=torch.float32, device=device)
    D = torch.zeros(nheads, dim, dtype=torch.float32, device=device)  # unused for now
    out = torch.empty_like(x)

    BLOCK_SIZE_M, num_warps = try_get_optimal_ssm_config(
        dim, dstate, batch, nheads, "float32", is_blackwell=False
    )

    grid = (
        (dim + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        batch,
        nheads,
    )

    _selective_scan_update_kernel[grid](
        state, None, x, dt, dt_bias, A, B, C, None, None, out,
        None, None, NULL_BLOCK_ID, None, None,
        batch, nheads, dim, dstate, nheads // ngroups,
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        x.stride(0), x.stride(1), x.stride(2),
        dt.stride(0), dt.stride(1), dt.stride(2),
        dt_bias.stride(0), dt_bias.stride(1),
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        0, 0,  # D strides (unused)
        0, 0, 0,  # z strides (unused)
        out.stride(0), out.stride(1), out.stride(2),
        0, 0, 0, 0,
        DT_SOFTPLUS=True,
        TIE_HDIM=False,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        HAS_DT_BIAS=True,
        HAS_D=False,
        HAS_Z=False,
        HAS_STATE_BATCH_INDICES=False,
        IS_SPEC_DECODING=False,
        IS_VARLEN=False,
        BLOCK_SIZE_DSTATE=triton.next_power_of_2(dstate),
        USE_RS_ROUNDING=False,
        PHILOX_ROUNDS=0,
        num_warps=num_warps,
    )
    torch.npu.synchronize()

    # CPU reference (no z, no D)
    state_cpu = state.cpu()
    x_cpu = x.cpu()
    dt_cpu = dt.cpu()
    dt_bias_cpu = dt_bias.cpu()
    A_cpu = A.cpu()
    B_cpu = B.cpu()
    C_cpu = C.cpu()

    expected_state = state_cpu.clone()
    expected_out = torch.zeros_like(x_cpu)

    for b in range(batch):
        for h in range(nheads):
            g = h // (nheads // ngroups)
            s = expected_state[b, h]
            xi = x_cpu[b, h]
            dti = dt_cpu[b, h] + dt_bias_cpu[h]
            dti = torch.where(dti <= 20.0, torch.log(torch.exp(dti) + 1), dti)

            dA_mat = torch.exp(A_cpu[h] * dti[:, None])
            dB_mat = B_cpu[b, g][None, :] * dti[:, None]
            s_new = s * dA_mat + dB_mat * xi[:, None]
            yi = torch.sum(s_new * C_cpu[b, g][None, :], dim=-1)

            expected_out[b, h] = yi
            expected_state[b, h] = s_new

    torch.testing.assert_close(out.cpu(), expected_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state.cpu(), expected_state, rtol=1e-4, atol=1e-4)


def test_selective_scan_update_via_api() -> None:
    """Selective scan via the public ``selective_state_update`` API.

    Tests the end-to-end API wrapper with 4D tensors and optional params.
    """
    torch.manual_seed(77)

    batch = 1
    nheads = 1
    dim = 8
    dstate = 4
    ngroups = 1

    device = torch.device("npu")

    state = torch.randn(batch, nheads, dim, dstate, dtype=torch.float32, device=device)
    x = torch.randn(batch, nheads, dim, dtype=torch.float32, device=device)
    dt = torch.randn(batch, nheads, dim, dtype=torch.float32, device=device)
    A = torch.randn(nheads, dim, dstate, dtype=torch.float32, device=device)
    B = torch.randn(batch, ngroups, dstate, dtype=torch.float32, device=device)
    C = torch.randn(batch, ngroups, dstate, dtype=torch.float32, device=device)
    D = torch.randn(nheads, dim, dtype=torch.float32, device=device)
    dt_bias = torch.randn(nheads, dim, dtype=torch.float32, device=device)

    state_copy = state.clone()
    out = torch.empty_like(x)

    selective_state_update(
        state_copy, x, dt, A, B, C, D=D, dt_bias=dt_bias,
        dt_softplus=True, out=out,
    )
    torch.npu.synchronize()

    # CPU reference
    state_cpu = state.cpu()
    x_cpu = x.cpu()
    dt_cpu = dt.cpu()
    dt_bias_cpu = dt_bias.cpu()
    A_cpu = A.cpu()
    B_cpu = B.cpu()
    C_cpu = C.cpu()
    D_cpu = D.cpu()

    expected_state = state_cpu.clone()
    expected_out = torch.zeros_like(x_cpu)

    for b in range(batch):
        for h in range(nheads):
            g = h // (nheads // ngroups)
            s = expected_state[b, h]
            xi = x_cpu[b, h]
            dti = dt_cpu[b, h] + dt_bias_cpu[h]
            dti = torch.where(dti <= 20.0, torch.log(torch.exp(dti) + 1), dti)

            dA_mat = torch.exp(A_cpu[h] * dti[:, None])
            dB_mat = B_cpu[b, g][None, :] * dti[:, None]
            s_new = s * dA_mat + dB_mat * xi[:, None]
            yi = torch.sum(s_new * C_cpu[b, g][None, :], dim=-1) + xi * D_cpu[h]

            expected_out[b, h] = yi
            expected_state[b, h] = s_new

    torch.testing.assert_close(out.cpu(), expected_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state_copy.cpu(), expected_state, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dstate", [2, 8, 16])
def test_selective_scan_update_varied_dstate(dstate: int) -> None:
    """Selective scan with different dstate sizes.

    Verifies correctness across dstate values that trigger different
    BLOCK_SIZE_DSTATE heuristics.
    """
    torch.manual_seed(99)

    batch = 1
    nheads = 1
    dim = 8
    ngroups = 1

    device = torch.device("npu")

    state = torch.randn(batch, nheads, dim, dstate, dtype=torch.float32, device=device)
    x = torch.randn(batch, nheads, dim, dtype=torch.float32, device=device)
    dt = torch.randn(batch, nheads, dim, dtype=torch.float32, device=device)
    dt_bias = torch.randn(nheads, dim, dtype=torch.float32, device=device)
    A = torch.randn(nheads, dim, dstate, dtype=torch.float32, device=device)
    B = torch.randn(batch, ngroups, dstate, dtype=torch.float32, device=device)
    C = torch.randn(batch, ngroups, dstate, dtype=torch.float32, device=device)
    D = torch.randn(nheads, dim, dtype=torch.float32, device=device)
    out = torch.empty_like(x)

    BLOCK_SIZE_M, num_warps = try_get_optimal_ssm_config(
        dim, dstate, batch, nheads, "float32", is_blackwell=False
    )

    grid = (
        (dim + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        batch,
        nheads,
    )

    _selective_scan_update_kernel[grid](
        state, None, x, dt, dt_bias, A, B, C, D, None, out,
        None, None, NULL_BLOCK_ID, None, None,
        batch, nheads, dim, dstate, nheads // ngroups,
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        x.stride(0), x.stride(1), x.stride(2),
        dt.stride(0), dt.stride(1), dt.stride(2),
        dt_bias.stride(0), dt_bias.stride(1),
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        D.stride(0), D.stride(1),
        0,  # z strides (unused)
        0,
        0,
        out.stride(0), out.stride(1), out.stride(2),
        0, 0, 0, 0,
        DT_SOFTPLUS=True,
        TIE_HDIM=False,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        HAS_DT_BIAS=True,
        HAS_D=True,
        HAS_Z=False,
        HAS_STATE_BATCH_INDICES=False,
        IS_SPEC_DECODING=False,
        IS_VARLEN=False,
        BLOCK_SIZE_DSTATE=triton.next_power_of_2(dstate),
        USE_RS_ROUNDING=False,
        PHILOX_ROUNDS=0,
        num_warps=num_warps,
    )
    torch.npu.synchronize()

    # CPU reference expanded inline
    state_cpu = state.cpu()
    x_cpu = x.cpu()
    dt_cpu = dt.cpu()
    dt_bias_cpu = dt_bias.cpu()
    A_cpu = A.cpu()
    B_cpu = B.cpu()
    C_cpu = C.cpu()
    D_cpu = D.cpu()

    expected_state = state_cpu.clone()
    expected_out = torch.zeros_like(x_cpu)

    for b in range(batch):
        for h in range(nheads):
            g = h // (nheads // ngroups)
            s = expected_state[b, h]
            xi = x_cpu[b, h]
            dti = dt_cpu[b, h] + dt_bias_cpu[h]
            dti = torch.where(dti <= 20.0, torch.log(torch.exp(dti) + 1), dti)

            dA_mat = torch.exp(A_cpu[h] * dti[:, None])
            dB_mat = B_cpu[b, g][None, :] * dti[:, None]
            s_new = s * dA_mat + dB_mat * xi[:, None]
            yi = torch.sum(s_new * C_cpu[b, g][None, :], dim=-1) + xi * D_cpu[h]

            expected_out[b, h] = yi
            expected_state[b, h] = s_new

    torch.testing.assert_close(out.cpu(), expected_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state.cpu(), expected_state, rtol=1e-4, atol=1e-4)
