# SPDX-License-Identifier: Apache-2.0
"""GPU-side fixtures for acc_ut_260907.

Injects the CUDA runtime module as the ``rt`` fixture consumed by the shared
test implementations in ``acc_ut_260907/common/``.
"""

import pytest

from accuracy_test.acc_ut_260907 import runtime_gpu


@pytest.fixture
def rt():
    """Runtime facade: STRICT_DEVICE / init_device_properties_triton / synchronize."""
    return runtime_gpu
