"""Strict accuracy test policy and backend markers.

Unlike the earlier exploratory suite, backend compilation failures are not
converted to XFAIL: a collected strict test must either pass or fail visibly.
"""

from pathlib import Path

import pytest


LEVELS = {
    "num_nans": "accuracy_l1",
    "prepare_rope": "accuracy_l1",
    "scatter_num": "accuracy_l0",
    "bad_words": "accuracy_l1",
    "temperature": "accuracy_l1",
    "gumbel": "accuracy_l2",
    "bias": "accuracy_l1",
    "topk_log": "accuracy_l2",
    "ranks": "accuracy_l1",
    "fill_logprob": "accuracy_l0",
    "min_p": "accuracy_l1",
    "penalties": "accuracy_l1",
    "bincount": "accuracy_l1",
    "prompt_logprobs": "accuracy_l0",
    "prepare_prefill": "accuracy_l1",
    "prepare_decode": "accuracy_l1",
    "update_draft": "accuracy_l1",
    "prepare_dflash": "accuracy_l1",
    "compute_local_logits": "accuracy_l2",
    "compute_cumulative": "accuracy_l2",
    "compute_local_residual": "accuracy_l2",
    "rejection": "accuracy_l2",
    "resample": "accuracy_l2",
    "insert_resampled": "accuracy_l1",
    "flatten": "accuracy_l0",
    "gather_block": "accuracy_l0",
    "compute_slot": "accuracy_l1",
    "apply_write": "accuracy_l0",
    "dcp_local": "accuracy_l1",
    "prepare_pos": "accuracy_l1",
    "combine_sampled": "accuracy_l1",
    "get_num_sampled": "accuracy_l1",
    "post_update_kernel": "accuracy_l1",
    "post_update_num": "accuracy_l0",
    "expand_idx": "accuracy_l0",
    "grammar_bitmask": "accuracy_l1",
}

ADAPTED_NPU = {
    "bad_words",
    "temperature",
    "gumbel_sample",
    "topk_log_softmax",
    "ranks",
    "min_p",
    "penalties",
    "bincount",
    "prepare_dflash_inputs_kernel",
    "rejection_kernel",
    "resample_kernel",
    "compute_slot_mappings_kernel",
    "post_update_kernel",
    "apply_grammar_bitmask_kernel",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        path = Path(str(item.path)).as_posix()
        stem = Path(str(item.path)).stem.removeprefix("test_")
        if "/gpu/" in path:
            item.add_marker(pytest.mark.gpu)
        elif "/npu/" in path:
            item.add_marker(pytest.mark.npu)
            if stem in ADAPTED_NPU:
                item.add_marker(pytest.mark.npu_ascend_adapted)
            elif stem in {
                "compute_cumulative_log_p_kernel",
                "compute_local_residual_mass_kernel",
            }:
                item.add_marker(pytest.mark.npu_upstream_unwired)
                item.add_marker(pytest.mark.requires_vllm_main)
            else:
                item.add_marker(pytest.mark.npu_upstream_reuse)

        for key, marker in LEVELS.items():
            if key in stem:
                item.add_marker(getattr(pytest.mark, marker))
                break

        if any(name in stem for name in ("gumbel", "rejection", "resample")):
            item.add_marker(pytest.mark.stochastic)
        else:
            item.add_marker(pytest.mark.deterministic)
