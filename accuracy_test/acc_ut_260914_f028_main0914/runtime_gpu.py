# SPDX-License-Identifier: Apache-2.0
"""CUDA runtime helpers used by acc_ut_260914_f028_main0914 GPU-side tests.

Mirrors acc_ut_260907/runtime_gpu.py: device selection plus a no-op Triton
device-property initializer.
"""

import pytest
import torch

if not hasattr(torch.library, "infer_schema"):
    pytest.skip(
        "Installed PyTorch is incompatible with the checked-out vLLM",
        allow_module_level=True,
    )

if not torch.cuda.is_available():
    pytest.skip(
        "CUDA is required for acc_ut_260914_f028_main0914 GPU-side accuracy "
        "tests",
        allow_module_level=True,
    )

DEVICE = "cuda"
STRICT_DEVICE = torch.device(DEVICE)


def init_device_properties_triton() -> None:
    """CUDA Triton initializes device properties through its own runtime."""


def get_vectorcore_num() -> int:
    """Compatibility helper."""
    return 1


def synchronize() -> None:
    torch.cuda.synchronize()
