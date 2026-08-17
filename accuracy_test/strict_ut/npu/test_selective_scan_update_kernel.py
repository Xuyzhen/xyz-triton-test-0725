# GENERATED STRICT UT. Source: accuracy_test/codex/existing_accuracy_tests/from_vllm/test_selective_scan_update_kernel.py
# Do not edit mechanically; update the reviewed Codex source or strict generator.
from accuracy_test.strict_ut.runtime_npu import STRICT_DEVICE as _STRICT_DEVICE
# Standalone Ascend A3 adaptation of an upstream vLLM accuracy path.
# Accuracy UT source: vllm/tests/kernels/mamba/test_mamba_ssm.py
# Kernel source: vllm/vllm/model_executor/layers/mamba/ops/mamba_ssm.py
# Coverage: _selective_scan_update_kernel

# vLLM vanilla kernel: _selective_scan_update_kernel from
# vllm/vllm/model_executor/layers/mamba/ops/mamba_ssm.py

"""
Precision test for _selective_scan_update_kernel.

Mamba state update kernel with heuristics. Implements the selective scan
discretization::

    dA = exp(A * dt)
    dB = B * dt
    state = state * dA + dB * x
    y = state @ C + D * x   (optionally modulated by z)

Uses @triton.heuristics for HAS_DT_BIAS, HAS_D, HAS_Z,
HAS_STATE_BATCH_INDICES, IS_SPEC_DECODING, IS_VARLEN, and BLOCK_SIZE_DSTATE.

Kernel signature:
    _selective_scan_update_kernel[grid](
        state_ptr, rand_seed_ptr, x_ptr, dt_ptr, dt_bias_ptr,
        A_ptr, B_ptr, C_ptr, D_ptr, z_ptr, out_ptr,
        state_batch_indices_ptr, dst_state_batch_indices_ptr,
        null_block_id, num_accepted_tokens_ptr, cu_seqlens_ptr,
        N, nheads, dim, dstate, nheads_ngroups_ratio,
        stride_state_batch, stride_state_head, stride_state_dim, stride_state_dstate,
        stride_x_batch, stride_x_head, stride_x_dim,
        stride_dt_batch, stride_dt_head, stride_dt_dim,
        stride_dt_bias_head, stride_dt_bias_dim,
        stride_A_head, stride_A_dim, stride_A_dstate,
        stride_B_batch, stride_B_group, stride_B_dstate,
        stride_C_batch, stride_C_group, stride_C_dstate,
        stride_D_head, stride_D_dim,
        stride_z_batch, stride_z_head, stride_z_dim,
        stride_out_batch, stride_out_head, stride_out_dim,
        stride_state_indices_batch, stride_state_indices_T,
        stride_dst_state_indices_batch, stride_dst_state_indices_T,
        DT_SOFTPLUS, TIE_HDIM, BLOCK_SIZE_M,
        HAS_DT_BIAS, HAS_D, HAS_Z,
        HAS_STATE_BATCH_INDICES, IS_SPEC_DECODING, IS_VARLEN,
        BLOCK_SIZE_DSTATE, USE_RS_ROUNDING, PHILOX_ROUNDS,
    )

Realistic shapes (Mamba2 / Jamba):
  - dim (head_dim): 64, 128
  - dstate (state_dim): 8, 16, 128
  - nheads: 8, 16, 32
  - ngroups: 1, 8 (nheads // ngroups ratio >= 1)
  - batch: 1 (single-stream), 4 (small batch)
"""

from __future__ import annotations

import pytest
import torch

from accuracy_test.strict_ut.runtime_npu import DEVICE, init_device_properties_triton, synchronize
from vllm.triton_utils import tl, triton
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    _selective_scan_update_kernel,
    try_get_optimal_ssm_config,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

pytestmark = [pytest.mark.npu]


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
    tie_hdim: bool = False,
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

    state_out = state.clone()
    y = torch.zeros_like(x)

    for b in range(batch):
        for h in range(nheads):
            g = h // nheads_ngroups_ratio
            s = state_out[b, h]  # (dim, dstate)
            xi = x[b, h]  # (dim,)
            dti = dt[b, h]  # (dim,)
            B_g = B[b, g]  # (dstate,)
            C_g = C[b, g]  # (dstate,)

            if tie_hdim:
                dt_val = dti[0]
                if dt_bias is not None:
                    dt_val = dt_val + dt_bias[h, 0]
                if dt_softplus:
                    dt_val = torch.nn.functional.softplus(dt_val)
                dA = torch.exp(A[h, 0, 0] * dt_val)
                dB = B_g * dt_val  # (dstate,)
                for d in range(dim):
                    s[d] = s[d] * dA + dB * xi[d]
            else:
                if dt_bias is not None:
                    dti = dti + dt_bias[h]
                if dt_softplus:
                    dti = torch.nn.functional.softplus(dti)

                A_h = A[h]  # (dim, dstate)

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


def _launch(
    state, x, dt, dt_bias, A, B, C, D, z, out,
    batch, nheads, dim, dstate, ngroups,
    dt_softplus, tie_hdim,
):
    N = batch
    nheads_ngroups_ratio = nheads // ngroups

    BLOCK_SIZE_M, num_warps = try_get_optimal_ssm_config(
        dim, dstate, batch, nheads, "float32", is_blackwell=False
    )

    grid = (
        triton.cdiv(dim, BLOCK_SIZE_M),
        N,
        nheads,
    )

    has_d = D is not None
    has_z = z is not None

    _selective_scan_update_kernel[grid](
        state, None, x, dt, dt_bias, A, B, C, D, z, out,
        None, None, NULL_BLOCK_ID, None, None,
        N, nheads, dim, dstate, nheads_ngroups_ratio,
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        x.stride(0), x.stride(1), x.stride(2),
        dt.stride(0), dt.stride(1), dt.stride(2),
        dt_bias.stride(0), dt_bias.stride(1),
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        D.stride(0) if has_d else 0, D.stride(1) if has_d else 0,
        z.stride(0) if has_z else 0, z.stride(1) if has_z else 0, z.stride(2) if has_z else 0,
        out.stride(0), out.stride(1), out.stride(2),
        0, 0,  # state_batch_indices strides (unused)
        0, 0,  # dst_state_batch_indices strides (unused)
        DT_SOFTPLUS=dt_softplus,
        TIE_HDIM=tie_hdim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        HAS_DT_BIAS=True,
        HAS_D=has_d,
        HAS_Z=has_z,
        HAS_STATE_BATCH_INDICES=False,
        IS_SPEC_DECODING=False,
        IS_VARLEN=False,
        BLOCK_SIZE_DSTATE=triton.next_power_of_2(dstate),
        USE_RS_ROUNDING=False,
        PHILOX_ROUNDS=0,
        num_warps=num_warps,
    )
    synchronize()


@pytest.mark.parametrize(
    "batch,nheads,dim,dstate,ngroups",
    [
        # Mamba2-2.7B: dim=64, dstate=128, nheads=32, ngroups=8
        (1, 32, 64, 128, 8),
        (4, 32, 64, 128, 8),
        # Jamba: dim=128, dstate=8, nheads=16, ngroups=1
        (1, 16, 128, 8, 1),
        (4, 16, 128, 8, 1),
        # Mamba2 small: dim=64, dstate=16, nheads=8, ngroups=1
        (1, 8, 64, 16, 1),
        # Mamba2-130M: dim=64, dstate=16, nheads=8, ngroups=8
        (1, 8, 64, 16, 8),
    ],
)
def test_selective_scan_full(batch, nheads, dim, dstate, ngroups):
    """Selective scan with D, z, dt_bias and DT_SOFTPLUS enabled (all options)."""
    init_device_properties_triton()
    torch.manual_seed(42)

    state = torch.randn(batch, nheads, dim, dstate, dtype=torch.float32, device=DEVICE)
    state_before = state.clone()
    x = torch.randn(batch, nheads, dim, dtype=torch.float32, device=DEVICE)
    dt = torch.randn(batch, nheads, dim, dtype=torch.float32, device=DEVICE)
    dt_bias = torch.randn(nheads, dim, dtype=torch.float32, device=DEVICE)
    A = torch.randn(nheads, dim, dstate, dtype=torch.float32, device=DEVICE)
    B = torch.randn(batch, ngroups, dstate, dtype=torch.float32, device=DEVICE)
    C = torch.randn(batch, ngroups, dstate, dtype=torch.float32, device=DEVICE)
    D = torch.randn(nheads, dim, dtype=torch.float32, device=DEVICE)
    z = torch.randn(batch, nheads, dim, dtype=torch.float32, device=DEVICE)
    out = torch.empty_like(x)

    _launch(
        state, x, dt, dt_bias, A, B, C, D, z, out,
        batch, nheads, dim, dstate, ngroups,
        dt_softplus=True, tie_hdim=False,
    )

    expected_state, expected_out = _selective_scan_update_cpu(
        state_before.cpu(), x.cpu(), dt.cpu(), dt_bias.cpu(),
        A.cpu(), B.cpu(), C.cpu(), D=D.cpu(), z=z.cpu(),
        dt_softplus=True, tie_hdim=False,
    )
    torch.testing.assert_close(out.cpu(), expected_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state.cpu(), expected_state, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "batch,nheads,dim,dstate,ngroups",
    [
        (1, 16, 128, 8, 1),
        (1, 32, 64, 128, 8),
        (4, 8, 64, 16, 1),
    ],
)
def test_selective_scan_no_z_no_d(batch, nheads, dim, dstate, ngroups):
    """Selective scan without z and D (HAS_Z=False, HAS_D=False)."""
    init_device_properties_triton()
    torch.manual_seed(123)

    state = torch.randn(batch, nheads, dim, dstate, dtype=torch.float32, device=DEVICE)
    state_before = state.clone()
    x = torch.randn(batch, nheads, dim, dtype=torch.float32, device=DEVICE)
    dt = torch.randn(batch, nheads, dim, dtype=torch.float32, device=DEVICE)
    dt_bias = torch.randn(nheads, dim, dtype=torch.float32, device=DEVICE)
    A = torch.randn(nheads, dim, dstate, dtype=torch.float32, device=DEVICE)
    B = torch.randn(batch, ngroups, dstate, dtype=torch.float32, device=DEVICE)
    C = torch.randn(batch, ngroups, dstate, dtype=torch.float32, device=DEVICE)
    out = torch.empty_like(x)

    _launch(
        state, x, dt, dt_bias, A, B, C, None, None, out,
        batch, nheads, dim, dstate, ngroups,
        dt_softplus=True, tie_hdim=False,
    )

    expected_state, expected_out = _selective_scan_update_cpu(
        state_before.cpu(), x.cpu(), dt.cpu(), dt_bias.cpu(),
        A.cpu(), B.cpu(), C.cpu(), D=None, z=None,
        dt_softplus=True, tie_hdim=False,
    )
    torch.testing.assert_close(out.cpu(), expected_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state.cpu(), expected_state, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dstate", [8, 16, 128])
def test_selective_scan_varied_dstate(dstate):
    """Selective scan with different dstate sizes (triggers different BLOCK_SIZE_DSTATE)."""
    init_device_properties_triton()
    torch.manual_seed(99)

    batch = 1
    nheads = 16
    dim = 128
    ngroups = 1

    state = torch.randn(batch, nheads, dim, dstate, dtype=torch.float32, device=DEVICE)
    state_before = state.clone()
    x = torch.randn(batch, nheads, dim, dtype=torch.float32, device=DEVICE)
    dt = torch.randn(batch, nheads, dim, dtype=torch.float32, device=DEVICE)
    dt_bias = torch.randn(nheads, dim, dtype=torch.float32, device=DEVICE)
    A = torch.randn(nheads, dim, dstate, dtype=torch.float32, device=DEVICE)
    B = torch.randn(batch, ngroups, dstate, dtype=torch.float32, device=DEVICE)
    C = torch.randn(batch, ngroups, dstate, dtype=torch.float32, device=DEVICE)
    D = torch.randn(nheads, dim, dtype=torch.float32, device=DEVICE)
    out = torch.empty_like(x)

    _launch(
        state, x, dt, dt_bias, A, B, C, D, None, out,
        batch, nheads, dim, dstate, ngroups,
        dt_softplus=True, tie_hdim=False,
    )

    expected_state, expected_out = _selective_scan_update_cpu(
        state_before.cpu(), x.cpu(), dt.cpu(), dt_bias.cpu(),
        A.cpu(), B.cpu(), C.cpu(), D=D.cpu(), z=None,
        dt_softplus=True, tie_hdim=False,
    )
    torch.testing.assert_close(out.cpu(), expected_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state.cpu(), expected_state, rtol=1e-4, atol=1e-4)


def test_selective_scan_tie_hdim():
    """TIE_HDIM mode: all dims share same dt and A scalar.

    A has stride(-1) == 0 and stride(-2) == 0, dt has stride(-1) == 0,
    dt_bias has stride(-1) == 0.
    """
    init_device_properties_triton()
    torch.manual_seed(55)

    batch = 1
    nheads = 8
    dim = 64
    dstate = 16
    ngroups = 1

    state = torch.randn(batch, nheads, dim, dstate, dtype=torch.float32, device=DEVICE)
    state_before = state.clone()
    x = torch.randn(batch, nheads, dim, dtype=torch.float32, device=DEVICE)
    # TIE_HDIM requires scalar dt and dt_bias across dim: stride(-1) == 0
    dt_scalar = torch.randn(1, 1, 1, dtype=torch.float32, device=DEVICE)
    dt = dt_scalar.expand(batch, nheads, dim).contiguous()
    dt_bias_scalar = torch.randn(1, 1, dtype=torch.float32, device=DEVICE)
    dt_bias = dt_bias_scalar.expand(nheads, dim).contiguous()
    A_scalar = torch.randn(1, 1, 1, dtype=torch.float32, device=DEVICE)
    A = A_scalar.expand(nheads, dim, dstate).contiguous()
    B = torch.randn(batch, ngroups, dstate, dtype=torch.float32, device=DEVICE)
    C = torch.randn(batch, ngroups, dstate, dtype=torch.float32, device=DEVICE)
    D = torch.randn(nheads, dim, dtype=torch.float32, device=DEVICE)
    out = torch.empty_like(x)

    _launch(
        state, x, dt, dt_bias, A, B, C, D, None, out,
        batch, nheads, dim, dstate, ngroups,
        dt_softplus=True, tie_hdim=True,
    )

    expected_state, expected_out = _selective_scan_update_cpu(
        state_before.cpu(), x.cpu(), dt.cpu(), dt_bias.cpu(),
        A.cpu(), B.cpu(), C.cpu(), D=D.cpu(), z=None,
        dt_softplus=True, tie_hdim=True,
    )
    torch.testing.assert_close(out.cpu(), expected_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state.cpu(), expected_state, rtol=1e-4, atol=1e-4)
