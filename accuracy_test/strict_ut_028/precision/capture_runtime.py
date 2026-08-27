"""Shared capture runtime for the GPU-NPU precision harness.

Design invariants (see README.md):
  * Inputs are ALWAYS built on CPU with a seeded CPU generator, then moved to
    the target device. Device RNG (cuda/npu) is never used for inputs, so both
    sides run on bit-identical inputs.
  * Every case writes a JSON metadata file plus a .pt tensor file under
    ``<root>/results/<side>/cases/<kernel>/<case_id>.{json,pt}``.
  * case_id is a hash of (kernel, sorted params, dtypes) - independent of
    pytest ordering and device - so the compare stage can join on it.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

SCHEMA_VERSION = "026.1"

# Comparison modes per output tensor, declared by each kernel case module.
MODE_INT_EXACT = "int_exact"    # integer tensors: must match bitwise
MODE_F32 = "float32"            # atol/rtol = 1e-5
MODE_F16 = "float16"            # atol/rtol = 1e-3
MODE_BF16 = "bfloat16"          # atol/rtol = 1e-2
MODE_SKIP = "skip"              # stochastic outputs (sampled tokens): not compared

TOLERANCES = {
    MODE_F32: (1e-5, 1e-5),
    MODE_F16: (1e-3, 1e-3),
    MODE_BF16: (1e-2, 1e-2),
}


@dataclass
class CaseSpec:
    """One parameterized capture case for one kernel."""

    kernel: str                       # registry key, e.g. "penalties"
    name: str                         # human readable, e.g. "basic_4req"
    params: dict[str, Any]            # shape/dtype knobs, part of case_id
    seed: int = 42
    stochastic: bool = False          # kernel consumes RNG -> skip token outputs
    # output name -> comparison mode (or "skip")
    output_modes: dict[str, str] = field(default_factory=dict)
    # Optional side-specific normalization before compare:
    #   fn(output_name, side, tensor, inputs, params) -> tensor
    normalize: Callable[..., torch.Tensor] | None = None


def case_id(spec: CaseSpec) -> str:
    """Stable, device-independent id: sha1(kernel|sorted params)."""
    payload = json.dumps(
        {"kernel": spec.kernel, "params": _jsonify(spec.params)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def tensor_digest(t: torch.Tensor) -> str:
    """sha256 over contiguous CPU bytes (shape/dtype are stored separately)."""
    t = t.detach().cpu().contiguous()
    if t.numel() == 0:
        return "empty"
    dtype_label = str(t.dtype)
    if t.dtype == torch.bfloat16:
        # numpy has no bfloat16; hash the raw 16-bit payload bit-exactly
        # (device-independent, so both sides digest identically).
        t = t.view(torch.uint16)
    h = hashlib.sha256()
    h.update(dtype_label.encode())
    h.update(str(tuple(t.shape)).encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Input persistence
# ---------------------------------------------------------------------------

def inputs_path(inputs_root: Path, kernel: str, cid: str) -> Path:
    return inputs_root / kernel / f"{cid}.pt"


def save_inputs(inputs_root: Path, spec: CaseSpec, inputs: dict[str, torch.Tensor]) -> Path:
    path = inputs_path(inputs_root, spec.kernel, case_id(spec))
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu = {k: v.detach().cpu() for k, v in inputs.items()}
    torch.save({"kernel": spec.kernel, "case": spec.name, "params": _jsonify(spec.params), "tensors": cpu}, path)
    return path


def load_inputs(inputs_root: Path, spec: CaseSpec, device: str) -> dict[str, torch.Tensor]:
    path = inputs_path(inputs_root, spec.kernel, case_id(spec))
    if not path.exists():
        raise FileNotFoundError(
            f"inputs for {spec.kernel}/{case_id(spec)} not found under {inputs_root}. "
            "Run the gpu side first (it generates and persists the shared inputs)."
        )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return {k: v.to(device) for k, v in blob["tensors"].items()}


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def collect_env(side: str) -> dict[str, Any]:
    env: dict[str, Any] = {
        "side": side,
        "schema": SCHEMA_VERSION,
        "torch": torch.__version__,
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        import triton
        env["triton"] = triton.__version__
    except Exception:
        pass
    try:
        import vllm
        env["vllm"] = vllm.__version__
    except Exception:
        pass
    if side == "npu":
        try:
            import vllm_ascend
            env["vllm_ascend"] = vllm_ascend.__version__
        except Exception:
            pass
        if hasattr(torch, "npu"):
            env["device"] = torch.npu.get_device_name(0)
    elif side == "gpu" and torch.cuda.is_available():
        env["device"] = torch.cuda.get_device_name(0)
    return env


def save_case_result(
    results_root: Path,
    side: str,
    spec: CaseSpec,
    outputs: dict[str, torch.Tensor],
    inputs: dict[str, torch.Tensor],
) -> Path:
    """Write cases/<kernel>/<case_id>.{json,pt} and refresh the side manifest."""
    case_dir = results_root / side / "cases" / spec.kernel
    case_dir.mkdir(parents=True, exist_ok=True)
    cid = case_id(spec)

    meta: dict[str, Any] = {
        "kernel": spec.kernel,
        "case": spec.name,
        "case_id": cid,
        "params": _jsonify(spec.params),
        "seed": spec.seed,
        "stochastic": spec.stochastic,
        "output_modes": spec.output_modes,
        "inputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "digest": tensor_digest(v)} for k, v in inputs.items()},
        "outputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "digest": tensor_digest(v)} for k, v in outputs.items()},
    }
    (case_dir / f"{cid}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    # Float outputs are stored upcast to fp32 for a stable compare basis;
    # integer outputs keep their dtype.
    stored = {}
    for k, v in outputs.items():
        stored[k] = v.detach().cpu().to(torch.float32) if v.is_floating_point() else v.detach().cpu()
    torch.save(stored, case_dir / f"{cid}.pt")

    _update_manifest(results_root, side, env=None)
    return case_dir / f"{cid}.json"


def _update_manifest(results_root: Path, side: str, env: dict[str, Any] | None) -> None:
    manifest_path = results_root / side / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = {}
    manifest.setdefault("envs", [])
    entry = env or collect_env(side)
    if not manifest["envs"] or manifest["envs"][-1] != entry:
        manifest["envs"].append(entry)
    case_dir = results_root / side / "cases"
    manifest["case_count"] = sum(1 for _ in case_dir.rglob("*.json")) if case_dir.exists() else 0
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def load_case_result(
    results_root: Path, side: str, kernel: str, cid: str
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    case_dir = results_root / side / "cases" / kernel
    meta = json.loads((case_dir / f"{cid}.json").read_text())
    tensors = torch.load(case_dir / f"{cid}.pt", map_location="cpu", weights_only=False)
    return tensors, meta


def list_cases(results_root: Path, side: str) -> list[tuple[str, str]]:
    """All (kernel, case_id) captured for a side."""
    root = results_root / side / "cases"
    if not root.exists():
        return []
    return sorted((p.parent.name, p.stem) for p in root.rglob("*.json"))
