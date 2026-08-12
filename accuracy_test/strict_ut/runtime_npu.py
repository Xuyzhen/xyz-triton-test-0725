"""Lightweight NPU runtime helpers for strict accuracy tests.

Importing ``vllm_ascend.ops.triton.triton_utils`` normally executes
``vllm_ascend.ops.__init__`` first. That package imports the full fused-MoE
stack, so an unrelated vLLM/vLLM-Ascend API mismatch can break collection of
every Triton kernel test. Keep device setup local and register a narrow shim
for Ascend kernel modules that import the historical utility path.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

if not hasattr(torch, "npu") or not torch.npu.is_available():
    pytest.skip(
        "Ascend NPU is required for strict NPU accuracy tests",
        allow_module_level=True,
    )

try:
    from vllm.triton_utils import HAS_TRITON, triton
except (ImportError, ModuleNotFoundError) as exc:
    pytest.fail(f"vLLM Triton runtime is unavailable: {exc}", pytrace=False)

if not HAS_TRITON:
    pytest.fail("vLLM reports HAS_TRITON=False on the NPU test runner", pytrace=False)

DEVICE = "npu"
STRICT_DEVICE = torch.device(DEVICE)

_NUM_AICORE = -1
_NUM_VECTORCORE = -1


def init_device_properties_triton() -> None:
    """Initialize core counts without importing the heavyweight Ascend ops package."""
    global _NUM_AICORE, _NUM_VECTORCORE
    if _NUM_AICORE > 0 and _NUM_VECTORCORE > 0:
        return

    properties: dict[str, Any] = (
        triton.runtime.driver.active.utils.get_device_properties(
            torch.npu.current_device()
        )
    )
    _NUM_AICORE = int(properties.get("num_aicore", -1))
    _NUM_VECTORCORE = int(properties.get("num_vectorcore", -1))
    if _NUM_AICORE <= 0 or _NUM_VECTORCORE <= 0:
        raise RuntimeError(
            "Failed to detect Ascend Triton device properties: "
            f"{properties}"
        )


def get_aicore_num() -> int:
    init_device_properties_triton()
    return _NUM_AICORE


def get_vectorcore_num() -> int:
    init_device_properties_triton()
    return _NUM_VECTORCORE


def synchronize() -> None:
    torch.npu.synchronize()


def _install_triton_utils_shim() -> None:
    """Satisfy legacy utility imports without executing ``ops.__init__``."""
    module_name = "vllm_ascend.ops.triton.triton_utils"
    if module_name in sys.modules:
        return

    import vllm_ascend

    ascend_root = Path(vllm_ascend.__file__).resolve().parent
    packages = {
        "vllm_ascend.ops": ascend_root / "ops",
        "vllm_ascend.ops.triton": ascend_root / "ops" / "triton",
    }
    for package_name, package_path in packages.items():
        if package_name in sys.modules:
            continue
        package = types.ModuleType(package_name)
        package.__package__ = package_name
        package.__path__ = [str(package_path)]
        sys.modules[package_name] = package

    shim = types.ModuleType(module_name)
    shim.__package__ = "vllm_ascend.ops.triton"
    shim.__dict__.update(
        {
            "init_device_properties_triton": init_device_properties_triton,
            "get_aicore_num": get_aicore_num,
            "get_vectorcore_num": get_vectorcore_num,
        }
    )
    sys.modules[module_name] = shim


_install_triton_utils_shim()
