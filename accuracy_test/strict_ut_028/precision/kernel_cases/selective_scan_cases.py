"""selective_scan_update: mamba state update (direct launch, one decode step).

Both sides import the SAME upstream kernel
(vllm.model_executor.layers.mamba.ops.mamba_ssm._selective_scan_update_kernel),
so the launch recipe (strides + constexprs + try_get_optimal_ssm_config) is
identical on both sides; only the device differs.

One-step update:
    dA = exp(A * dt); dB = B * dt
    state = state * dA + dB * x
    y = state @ C + D * x  (optionally * z * sigmoid(z))

Parameter ranges follow the strict UT: A in [-2,-1), dt_bias in [-4,-3),
dt ~ N(0,1) with DT_SOFTPLUS (keeps dA in (0,1)).
Outputs: out (fp32) and state (fp32) -> 1e-5 tolerances.
"""

from __future__ import annotations

import torch

import capture_runtime as cr
from capture_runtime import CaseSpec


def build_inputs(params: dict, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    nheads, dim, dstate, ngroups = params["nheads"], params["dim"], params["dstate"], params["ngroups"]
    batch, has_z = params["batch"], params["has_z"]

    state = torch.randn(batch, nheads, dim, dstate, generator=g, dtype=torch.float32)
    x = torch.randn(batch, nheads, dim, generator=g, dtype=torch.float32)
    A = -(torch.rand(nheads, dim, dstate, generator=g, dtype=torch.float32) + 1.0)
    B = torch.randn(batch, ngroups, dstate, generator=g, dtype=torch.float32)
    C = torch.randn(batch, ngroups, dstate, generator=g, dtype=torch.float32)
    dt = torch.randn(batch, nheads, dim, generator=g, dtype=torch.float32)
    dt_bias = torch.rand(nheads, dim, generator=g, dtype=torch.float32) - 4.0
    D = torch.randn(nheads, dim, generator=g, dtype=torch.float32)
    out = {"state": state, "x": x, "dt": dt, "dt_bias": dt_bias, "A": A,
           "B": B, "C": C, "D": D}
    if has_z:
        out["z"] = torch.randn(batch, nheads, dim, generator=g, dtype=torch.float32)
    return out


def run(side: str, t: dict[str, torch.Tensor], params: dict) -> dict[str, torch.Tensor]:
    from vllm.triton_utils import triton
    from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
        _selective_scan_update_kernel,
        try_get_optimal_ssm_config,
    )
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    nheads, dim, dstate, ngroups = params["nheads"], params["dim"], params["dstate"], params["ngroups"]
    batch = params["batch"]
    N = batch
    nheads_ngroups_ratio = nheads // ngroups

    state = t["state"].clone()
    x, dt, dt_bias = t["x"], t["dt"], t["dt_bias"]
    A, B, C, D = t["A"], t["B"], t["C"], t["D"]
    z = t.get("z")
    out = torch.empty_like(x)

    BLOCK_SIZE_M, num_warps = try_get_optimal_ssm_config(
        dim, dstate, batch, nheads, "float32", is_blackwell=False
    )
    grid = (triton.cdiv(dim, BLOCK_SIZE_M), N, nheads)
    has_d, has_z = D is not None, z is not None

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
        0, 0, 0, 0,
        DT_SOFTPLUS=True,
        TIE_HDIM=params["tie_hdim"],
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
    if side == "gpu":
        torch.cuda.synchronize()
    else:
        torch.npu.synchronize()
    return {"out": out, "state": state}


def ref(t: dict[str, torch.Tensor], params: dict) -> dict[str, torch.Tensor]:
    """fp64 golden for one selective-scan update step.

    Mirrors _selective_scan_update_kernel:
      dt = softplus(dt + dt_bias), softplus(x) = log(exp(x)+1) if x <= 20 else x
      dA = exp(A * dt)                (TIE_HDIM: scalar A[h,0,0] * dt[b,h,0])
      state = state * dA + (B * dt) * x
      out = sum(state * C, -1) + D * x;  HAS_Z: out *= z * sigmoid(z)
    B/C use group pid_h // (nheads // ngroups)  -> repeat_interleave.
    """
    f64 = torch.float64
    state = t["state"].to(f64)
    x = t["x"].to(f64)
    A = t["A"].to(f64)
    B = t["B"].to(f64)
    C = t["C"].to(f64)
    D = t["D"].to(f64)
    dt = t["dt"].to(f64)
    dt_bias = t["dt_bias"].to(f64)
    nheads, ngroups = params["nheads"], params["ngroups"]
    ratio = nheads // ngroups

    if params["tie_hdim"]:
        # scalar per (batch, head): dt[..., 0] and A[h, 0, 0]
        dtv = dt[..., 0] + dt_bias[..., 0]                      # [b, h]
        dtv = torch.where(dtv <= 20.0, torch.log(torch.exp(dtv) + 1.0), dtv)
        dA = torch.exp(A[None, :, 0, 0] * dtv[..., None])       # [b, h, 1]
        dB_scalar = B.repeat_interleave(ratio, dim=1) * dtv[..., None]  # [b, h, n]
        new_state = state * dA.unsqueeze(-1) + dB_scalar.unsqueeze(2) * x.unsqueeze(-1)
    else:
        dtv = dt + dt_bias.unsqueeze(0)                          # [b, h, d]
        dtv = torch.where(dtv <= 20.0, torch.log(torch.exp(dtv) + 1.0), dtv)
        dA = torch.exp(A.unsqueeze(0) * dtv.unsqueeze(-1))       # [b, h, d, n]
        Bh = B.repeat_interleave(ratio, dim=1).unsqueeze(2)      # [b, h, 1, n]
        new_state = state * dA + Bh * x.unsqueeze(-1)

    Ch = C.repeat_interleave(ratio, dim=1).unsqueeze(2)          # [b, h, 1, n]
    out = (new_state * Ch).sum(dim=-1) + D.unsqueeze(0) * x
    if "z" in t:
        z = t["z"].to(f64)
        out = out * z * torch.sigmoid(z)
    return {"out": out, "state": new_state}


def _mk(name: str, batch: int, nheads: int, dim: int, dstate: int,
        ngroups: int, has_z: bool, tie_hdim: bool = False) -> CaseSpec:
    return CaseSpec(
        kernel="selective_scan_update", name=name,
        params={"batch": batch, "nheads": nheads, "dim": dim, "dstate": dstate,
                "ngroups": ngroups, "has_z": has_z, "tie_hdim": tie_hdim},
        seed=42,
        output_modes={"out": cr.MODE_F32, "state": cr.MODE_F32},
    )


CASES = [
    _mk("mamba2_1b_32h_64d_128s_8g", 1, 32, 64, 128, 8, True),
    _mk("jamba_1b_16h_128d_8s_1g_noz", 1, 16, 128, 8, 1, False),
    _mk("mamba2_small_1b_8h_64d_16s_1g", 1, 8, 64, 16, 1, True),
    _mk("tie_hdim_1b_8h_64d_16s_1g", 1, 8, 64, 16, 1, False, tie_hdim=True),
]
