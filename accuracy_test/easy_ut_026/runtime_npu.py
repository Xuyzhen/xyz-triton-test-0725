"""Self-contained NPU runtime helpers for easy_ut_026 accuracy tests.

This module deliberately has NO dependency on any sibling ``accuracy_test.*``
package, so ``easy_ut_026`` can be dropped/copied anywhere and still collect.

Importing ``vllm_ascend.ops.triton.triton_utils`` normally executes
``vllm_ascend.ops.__init__`` first. That package imports the full fused-MoE
stack, so an unrelated vLLM/vLLM-Ascend API mismatch can break collection of
every Triton kernel test. Keep device setup local.
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
        "Ascend NPU is required for easy_ut_026 accuracy tests",
        allow_module_level=True,
    )

try:
    from vllm.triton_utils import HAS_TRITON, tl, triton
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


def _resolve_triton_ascend_op(op_name: str):
    """Resolve an Ascend helper, deferring errors until it is actually used."""
    try:
        import triton.language.extra.cann.extension as cann_extension
    except ImportError:
        cann_extension = None

    if cann_extension is not None:
        extension_op = getattr(cann_extension, op_name, None)
        if extension_op is not None:
            return extension_op

    tl_op = getattr(tl, op_name, None)
    if tl_op is not None:
        return tl_op

    def unavailable_op(*args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            f"Triton op '{op_name}' is unavailable in this Ascend Triton "
            "runtime. Upgrade the runtime before executing a kernel that "
            "requires this op."
        )

    unavailable_op.__name__ = op_name
    return unavailable_op


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
            "insert_slice": _resolve_triton_ascend_op("insert_slice"),
            "extract_slice": _resolve_triton_ascend_op("extract_slice"),
            "get_element": _resolve_triton_ascend_op("get_element"),
        }
    )
    sys.modules[module_name] = shim


_install_triton_utils_shim()