# SPDX-License-Identifier: Apache-2.0
"""NPU-side entry for _shift_input_ids_kernel (shared implementation).

The ``rt`` fixture from npu/conftest.py injects the Ascend NPU runtime, so
the shared cases in common/shift_input_ids_impl.py run on the NPU side.
"""
from accuracy_test.strict_ut_027.common.shift_input_ids_impl import (  # noqa: F401
    test_shift_input_ids,
    test_import_error,
)
