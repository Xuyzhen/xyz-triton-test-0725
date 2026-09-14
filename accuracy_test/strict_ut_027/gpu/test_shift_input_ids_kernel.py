# SPDX-License-Identifier: Apache-2.0
"""GPU-side entry for _shift_input_ids_kernel (shared implementation).

The ``rt`` fixture from gpu/conftest.py injects the CUDA runtime, so the
shared cases in common/shift_input_ids_impl.py run on the GPU side.
"""
from accuracy_test.strict_ut_027.common.shift_input_ids_impl import (  # noqa: F401
    test_shift_input_ids,
    test_import_error,
)
