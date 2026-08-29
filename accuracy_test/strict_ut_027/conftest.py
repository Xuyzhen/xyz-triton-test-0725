# SPDX-License-Identifier: Apache-2.0
"""Strict accuracy test policy and backend markers for strict_ut_027.

Dual-side suite (CUDA + Ascend NPU) built from the easy_ut_026 NPU-only
tests with extended, higher-spec shapes, following the strict_ut_028
project layout. Backend compilation failures are not converted to XFAIL:
a collected strict test must either pass or fail visibly.
"""

from pathlib import Path

import pytest


# Level assignment per kernel stem (substring match on the module basename).
# All 9 kernels here are non-compute / copy-only operators verified bitwise
# against an independent CPU reference, except the thinking-budget kernel
# whose float write is verified exactly against the CPU reference (L1).
LEVELS = {
    "shift_input_ids": "accuracy_l0",
    "shift_input_embeds": "accuracy_l0",
    "cache_inputs": "accuracy_l0",
    "pad_trailing_draft_slots": "accuracy_l0",
    "prepare_input_buffers": "accuracy_l0",
    "prepare_input_hidden_states_and_embeddings": "accuracy_l0",
    "preprocess_mamba_align_fused": "accuracy_l0",
    "update_committed_marker_cache": "accuracy_l0",
    "thinking_budget": "accuracy_l1",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        path = Path(str(item.path)).as_posix()
        stem = Path(str(item.path)).stem.removeprefix("test_")

        if "/gpu/" in path:
            item.add_marker(pytest.mark.gpu)
        elif "/npu/" in path:
            item.add_marker(pytest.mark.npu)
            # All 9 kernels are upstream vLLM kernels reused as-is on NPU.
            item.add_marker(pytest.mark.npu_upstream_reuse)

        for key, marker in LEVELS.items():
            if key in stem:
                item.add_marker(getattr(pytest.mark, marker))
                break

        item.add_marker(pytest.mark.deterministic)
