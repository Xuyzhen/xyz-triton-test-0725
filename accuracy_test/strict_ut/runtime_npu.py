"""NPU runtime helpers used by generated NPU tests."""

import pytest
import torch

if not hasattr(torch, "npu") or not torch.npu.is_available():
    pytest.skip("Ascend NPU is required for strict NPU accuracy tests", allow_module_level=True)

try:
    from vllm_ascend.ops.triton.triton_utils import (
        get_vectorcore_num,
        init_device_properties_triton,
    )
except (ImportError, ModuleNotFoundError) as exc:
    pytest.fail(f"NPU Triton runtime is unavailable: {exc}", pytrace=False)

DEVICE = "npu"
STRICT_DEVICE = torch.device(DEVICE)


def synchronize() -> None:
    torch.npu.synchronize()
