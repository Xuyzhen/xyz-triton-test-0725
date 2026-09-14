# SPDX-License-Identifier: Apache-2.0
"""Accuracy test policy and backend markers for acc_ut_260914_f028_main0914.

Consolidated suite for 5 new Triton kernels (dual-side: GPU + Ascend NPU).
Backend compilation failures are not converted to XFAIL: a collected strict
test must either pass or fail visibly.
"""

from pathlib import Path

import pytest


LEVELS = {
    # --- integer/index compute (bitwise exact) ---
    "fill_num_accepted": "accuracy_l0",
    "pack_sampling_mask": "accuracy_l0",
    # --- memory-copy / state management (bitwise on data, decision logic int32) ---
    "postprocess_mamba_fused": "accuracy_l0",
    # --- sparse->dense scatter (bitwise on scatter indices + float store) ---
    "cache_draft_logits": "accuracy_l1",
    # --- stochastic (Gumbel sampling walk) ---
    "selector_walk": "accuracy_l2",
}


def _normalized_stem(path: Path) -> str:
    stem = path.stem.removeprefix("test_")
    for suffix in ("_upstream", "_downstream"):
        stem = stem.removesuffix(suffix)
    return stem


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        path = Path(str(item.path)).as_posix()
        stem = _normalized_stem(Path(str(item.path)))

        if "/gpu/" in path:
            item.add_marker(pytest.mark.gpu)
        elif "/npu/" in path:
            # NPU side picks the downstream (vllm-ascend) implementation when
            # one exists (postprocess_mamba_fused_kernel,
            # dflash2_greedy_selector_walk_kernel) and falls back to the
            # upstream vLLM Triton kernel sources otherwise
            # (_fill_num_accepted_kernel, _pack_sampling_mask_kernel,
            # _cache_draft_logits_kernel).
            item.add_marker(pytest.mark.npu)
            if "selector_walk" in stem or "postprocess_mamba_fused" in stem:
                item.add_marker(pytest.mark.npu_downstream)
            else:
                item.add_marker(pytest.mark.npu_upstream_reuse)

        for key, marker in LEVELS.items():
            if key in stem:
                item.add_marker(getattr(pytest.mark, marker))
                break

        if any(name in stem for name in ("selector_walk", "cache_draft_logits")):
            item.add_marker(pytest.mark.stochastic)
        else:
            item.add_marker(pytest.mark.deterministic)
