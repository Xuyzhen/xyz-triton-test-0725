# SPDX-License-Identifier: Apache-2.0
"""Re-export strict_ut NPU runtime helpers so easy_ut_026 tests stay portable.

All device setup (NPU detection, Triton property init, vllm_ascend shim)
lives in ``accuracy_test.strict_ut.runtime_npu``. Re-export here to keep
import paths local to easy_ut_026 without duplicating logic.
"""

from accuracy_test.strict_ut.runtime_npu import (  # noqa: F401
    DEVICE,
    STRICT_DEVICE,
    get_aicore_num,
    get_vectorcore_num,
    init_device_properties_triton,
    synchronize,
)
