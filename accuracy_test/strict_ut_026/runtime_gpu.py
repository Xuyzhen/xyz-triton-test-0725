"""CUDA runtime helpers used by generated GPU tests."""

import pytest
import torch

if not hasattr(torch.library, "infer_schema"):
    pytest.skip(
        "Installed PyTorch is incompatible with the checked-out vLLM",
        allow_module_level=True,
    )

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for strict GPU accuracy tests", allow_module_level=True)

DEVICE = "cuda"
STRICT_DEVICE = torch.device(DEVICE)


def init_device_properties_triton() -> None:
    """CUDA Triton initializes device properties through its runtime."""


def get_vectorcore_num() -> int:
    """Compatibility helper for tests whose NPU launch uses vector-core count."""
    return 1


def synchronize() -> None:
    torch.cuda.synchronize()
