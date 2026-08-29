"""Self-contained NPU runtime helpers for strict_ut_027 accuracy tests.

Copied from easy_ut_026/runtime_npu.py (which has no dependency on any
sibling ``accuracy_test.*`` package, so the suite can be dropped anywhere
and still collect) and renamed for the dual-side strict_ut_027 project.

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
        "Ascend NPU is required for strict_ut_027 NPU-side accuracy tests "
        "(run the GPU side via run_gpu.sh on CUDA hosts)",
        allow_module_level=True,
    )

try:
    import triton
    import triton.language as tl
except (ImportError, ModuleNotFoundError) as exc:
    pytest.fail(f"Triton runtime is unavailable: {exc}", pytrace=False)

HAS_TRITON = True


def _install_vllm_triton_utils_shim() -> types.ModuleType:
    """Provide ``vllm.triton_utils.tl`` without the ``triton.experimental.gluon`` import.

    The installed Triton is 3.2.0, which predates ``triton.experimental.gluon``.
    vLLM's ``triton_utils/__init__.py`` unconditionally does ``from
    triton.experimental import gluon``, so merely importing ``vllm.triton_utils``
    breaks collection on this host.

    Only that one ``__init__`` line needs bypassing (grep confirms no other
    ``vllm.*`` module touches ``triton.experimental``). So instead of replacing
    the whole package, we install a *package-shaped* shim: set ``__path__`` to
    the real ``triton_utils`` directory so submodules like ``allocation`` /
    ``libdevice`` still load from disk, and expose ``tl``/``triton`` from the
    installed triton 3.2. Submodules that do ``from vllm.triton_utils import
    triton`` will pick up the shim's attributes via ``sys.modules``.
    """
    import importlib.util

    if "vllm.triton_utils" in sys.modules:
        existing = sys.modules["vllm.triton_utils"]
        if getattr(existing, "_strict_ut_027_shim", False):
            return existing
        # Already genuinely imported (e.g. by a GPU-side kernel import on a
        # triton>=3.3 host in a dual-backend process): nothing to patch.
        return existing

    # Locate the real package directory without importing it.
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        pytest.fail("Could not locate the vllm package on this host", pytrace=False)
    triton_utils_dir = Path(list(spec.submodule_search_locations)[0]) / "triton_utils"

    def _placeholder_lazy(name: str):
        def _fn(*args, **kwargs):  # pragma: no cover
            del args, kwargs
            raise RuntimeError(
                f"vLLM feature '{name}' (triton.experimental.gluon) is "
                "unavailable under the precision-UT shim for triton==3.2.0"
            )

        _fn.__name__ = name
        return _fn

    shim = types.ModuleType("vllm.triton_utils")
    shim.__package__ = "vllm.triton_utils"
    shim.__path__ = [str(triton_utils_dir)]
    shim._strict_ut_027_shim = True
    shim.HAS_TRITON = True
    shim.triton = triton
    shim.tl = tl
    shim.tldevice = None
    try:
        import triton.language.extra.libdevice as _tldevice
        shim.tldevice = _tldevice
    except ImportError:
        pass
    shim.LOG2E = 1.4426950408889634
    shim.LOGE2 = 0.6931471805599453
    shim.gluon = _placeholder_lazy("gluon")
    shim.gl = _placeholder_lazy("gl")
    shim.aggregate = None
    try:
        from triton.language.core import _aggregate as _aggregate_ref
        shim.aggregate = _aggregate_ref
    except ImportError:
        pass
    shim.use_tensor_descriptor = _placeholder_lazy("use_tensor_descriptor")
    sys.modules["vllm.triton_utils"] = shim
    return shim


_install_vllm_triton_utils_shim()

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


def ensure_default_ascend_config() -> None:
    """Pin vllm-ascend's config singleton to safe UT defaults if uninitialized.

    ``vllm_ascend.sample.sampler._apply_top_k_top_p_pytorch`` reads
    ``get_ascend_config().enable_reduce_sample`` on every call, but a UT
    process never starts the engine, so the singleton is normally
    uninitialized and ``get_ascend_config()`` would raise
    ``RuntimeError: Ascend config is not initialized``. Install a minimal
    default (``enable_reduce_sample=False`` -> single-card sampling path)
    unless a real config is already present. Idempotent.
    """
    try:
        from vllm_ascend import ascend_config
    except ImportError:
        return

    config = getattr(ascend_config, "_ASCEND_CONFIG", None)
    if config is not None and ascend_config._is_ascend_config_initialized(config):
        return

    # ``_is_ascend_config_initialized`` only checks the two attributes below,
    # so this SimpleNamespace passes the guard while keeping the sampling
    # path on the default (non-distributed) branch.
    ascend_config._ASCEND_CONFIG = types.SimpleNamespace(
        enable_reduce_sample=False,
        ascend_compilation_config=None,
        eplb_config=None,
    )


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
