# SPDX-License-Identifier: Apache-2.0
"""Accuracy test policy and backend markers for acc_ut_260907.

Consolidated suite: every NPU operator precision UT under ``accuracy_test/``
keeps only its most recent adaptation here. Operators with both an upstream
(vllm) and a downstream (vllm_ascend) test carry ``_upstream`` /
``_downstream`` filename suffixes. Backend compilation failures are not
converted to XFAIL: a collected strict test must either pass or fail visibly.
"""

from pathlib import Path

import pytest


# Level assignment per kernel stem (substring match on the module basename).
# Merged from strict_ut_027/conftest.py and strict_ut_028/conftest.py, plus
# the codex-only operators.
LEVELS = {
    # --- non-compute / copy-only (bitwise) ---
    "shift_input_ids": "accuracy_l0",
    "shift_input_embeds": "accuracy_l0",
    "cache_inputs": "accuracy_l0",
    "pad_trailing_draft_slots": "accuracy_l0",
    "prepare_input_buffers": "accuracy_l0",
    "prepare_input_hidden_states_and_embeddings": "accuracy_l0",
    "preprocess_mamba_align_fused": "accuracy_l0",
    "update_committed_marker_cache": "accuracy_l0",
    "scatter_num": "accuracy_l0",
    "fill_logprob": "accuracy_l0",
    "prompt_logprobs": "accuracy_l0",
    "flatten": "accuracy_l0",
    "gather_block": "accuracy_l0",
    "apply_write": "accuracy_l0",
    "post_update_num": "accuracy_l0",
    "expand_idx": "accuracy_l0",
    "load_ptr": "accuracy_l0",
    # --- tolerance-based against CPU reference ---
    "thinking_budget": "accuracy_l1",
    "num_nans": "accuracy_l1",
    "prepare_rope": "accuracy_l1",
    "bad_words": "accuracy_l1",
    "temperature": "accuracy_l1",
    "bias": "accuracy_l1",
    "ranks": "accuracy_l1",
    "min_p": "accuracy_l1",
    "penalties": "accuracy_l1",
    "bincount": "accuracy_l1",
    "prepare_prefill": "accuracy_l1",
    "prepare_decode": "accuracy_l1",
    "update_draft": "accuracy_l1",
    "prepare_dflash": "accuracy_l1",
    "insert_resampled": "accuracy_l1",
    "compute_slot": "accuracy_l1",
    "dcp_local": "accuracy_l1",
    "prepare_pos": "accuracy_l1",
    "combine_sampled": "accuracy_l1",
    "get_num_sampled": "accuracy_l1",
    "post_update_kernel": "accuracy_l1",
    "grammar_bitmask": "accuracy_l1",
    "log_softmax": "accuracy_l1",
    # --- statistical (probabilities / sampling / reductions) ---
    "gumbel": "accuracy_l2",
    "topk_log": "accuracy_l2",
    "compute_local_logits": "accuracy_l2",
    "compute_cumulative": "accuracy_l2",
    "compute_local_residual": "accuracy_l2",
    "rejection": "accuracy_l2",
    "resample": "accuracy_l2",
    "compute_block_max": "accuracy_l2",
    "compute_block_stats": "accuracy_l2",
    "compute_global_logsumexp": "accuracy_l2",
    "tl_rand64": "accuracy_l2",
    "probabilistic_rejection": "accuracy_l2",
}

# NPU-side kernels that execute an ascend-adapted (vllm_ascend) variant
# rather than the upstream vLLM kernel. Matched against the file stem after
# stripping the ``test_`` prefix and the ``_upstream``/``_downstream`` /
# ``_npu`` suffixes.
ADAPTED_NPU = {
    "num_nans_kernel",
    "num_nans_kernel_precision_npu",
    "num_nans_kernel_standalone_npu",
    "bad_words",
    "temperature",
    "gumbel_sample",
    "gumbel_sampling",
    "topk_log_softmax",
    "ranks",
    "min_p",
    "penalties",
    "penality",
    "bincount",
    "prepare_dflash_inputs_kernel",
    "rejection_kernel",
    "resample_kernel",
    "post_update_kernel",
    "apply_grammar_bitmask_kernel",
    "compute_slot_mappings_kernel",
    "fill_logprob_token_ids_kernel",
    "topk_topp_kernel",
    "zero_kv_blocks_kernel",
    "log_softmax",
    "npu_gumbel_block_argmax",
    "prepare_prefill_inputs_kernel_speculator",
}

# NPU-side upstream kernels that exist in vLLM but are not wired into the
# NPU production path (kept for coverage documentation).
UPSTREAM_UNWIRED = {
    "compute_cumulative_log_p_kernel",
    "compute_local_residual_mass_kernel",
    "tl_rand64",
    "load_ptr",
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
        is_downstream_file = Path(str(item.path)).stem.endswith("_downstream")
        is_upstream_file = Path(str(item.path)).stem.endswith("_upstream")

        if "/gpu/" in path:
            item.add_marker(pytest.mark.gpu)
        elif "/npu/" in path:
            item.add_marker(pytest.mark.npu)
            if is_downstream_file or stem in ADAPTED_NPU:
                item.add_marker(pytest.mark.npu_ascend_adapted)
            elif stem in UPSTREAM_UNWIRED:
                item.add_marker(pytest.mark.npu_upstream_unwired)
                item.add_marker(pytest.mark.requires_vllm_main)
            else:
                item.add_marker(pytest.mark.npu_upstream_reuse)
            if is_upstream_file:
                item.add_marker(pytest.mark.npu_upstream_reuse)

        for key, marker in LEVELS.items():
            if key in stem:
                item.add_marker(getattr(pytest.mark, marker))
                break

        if any(name in stem for name in ("gumbel", "rejection", "resample")):
            item.add_marker(pytest.mark.stochastic)
        else:
            item.add_marker(pytest.mark.deterministic)
