"""Generate the strict suite from reviewed accuracy_test/codex tests only.

The generated files are committed artifacts. Re-running this script is useful
when the reviewed Codex tests change,.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX = ROOT / "accuracy_test" / "codex"
OUT = ROOT / "accuracy_test" / "strict_ut"

MISSING = CODEX / "missing_accuracy_tests"
VLLM = CODEX / "existing_accuracy_tests" / "from_vllm"
ASCEND = CODEX / "existing_accuracy_tests" / "from_vllm_ascend"

# Logical filename -> reviewed source. The GPU map always starts from a test
# of the upstream contract. For public wrappers whose only reviewed standalone
# test is the Ascend test, import paths are rewritten to the upstream module.
GPU_SOURCES = {
    "num_nans_kernel": MISSING / "test_num_nans_kernel.py",
    "prepare_rope_positions_kernel": MISSING / "test_prepare_rope_positions_kernel.py",
    "scatter_num_accepted_kernel": VLLM / "test_scatter_num_accepted_kernel.py",
    "bad_words": ASCEND / "test_bad_words.py",
    "temperature": ASCEND / "test_temperature.py",
    "gumbel_sample": ASCEND / "test_gumbel_sampling.py",
    "bias_kernel": VLLM / "test_bias_kernel.py",
    "topk_log_softmax": ASCEND / "test_log_softmax.py",
    "ranks": ASCEND / "test_compute_topk_logprobs.py",
    "fill_logprob_token_ids_kernel": VLLM / "test_fill_logprob_token_ids_kernel.py",
    "min_p": ASCEND / "test_min_p.py",
    "penalties": ASCEND / "test_penality.py",
    "bincount": ASCEND / "test_bincount.py",
    "prompt_logprobs_token_ids_kernel": VLLM / "test_prompt_logprobs_token_ids_kernel.py",
    "ar_prepare_prefill_inputs_kernel": MISSING / "test_prepare_prefill_inputs_kernel_speculator.py",
    "prepare_decode_inputs_kernel": MISSING / "test_prepare_decode_inputs_kernel.py",
    "update_draft_inputs_kernel": MISSING / "test_update_draft_inputs_kernel.py",
    "prepare_dflash_inputs_kernel": VLLM / "test_prepare_dflash_inputs_kernel.py",
    "compute_local_logits_stats_kernel": VLLM / "test_compute_block_max_and_sumexp.py",
    "compute_cumulative_log_p_kernel": VLLM / "test_compute_block_stats_kernel.py",
    "rejection_kernel": VLLM / "test_rejection_kernel.py",
    "resample_kernel": VLLM / "test_resample_kernel.py",
    "insert_resampled_kernel": VLLM / "test_insert_resampled_kernel.py",
    "flatten_sampled_kernel": VLLM / "test_flatten_sampled_kernel.py",
    "gather_block_tables_kernel": VLLM / "test_gather_block_tables_kernel.py",
    "compute_slot_mappings_kernel": ASCEND / "test_compute_slot_mapping.py",
    "apply_write_kernel": VLLM / "test_apply_write_kernel.py",
    "dcp_local_seq_lens_kernel": MISSING / "test_dcp_local_seq_lens_kernel.py",
    "input_batch_prepare_prefill_inputs_kernel": MISSING / "test_prepare_prefill_inputs_kernel.py",
    "prepare_pos_seq_lens_kernel": MISSING / "test_prepare_pos_seq_lens_kernel.py",
    "combine_sampled_and_draft_tokens_kernel": MISSING / "test_combine_sampled_and_draft_tokens_kernel.py",
    "get_num_sampled_and_rejected_kernel": MISSING / "test_get_num_sampled_and_rejected_kernel.py",
    "post_update_kernel": ASCEND / "test_post_update.py",
    "post_update_num_computed_tokens_kernel": MISSING / "test_post_update_num_computed_tokens_kernel.py",
    "expand_idx_mapping_kernel": MISSING / "test_expand_idx_mapping_kernel.py",
    "apply_grammar_bitmask_kernel": MISSING / "test_apply_grammar_bitmask_kernel_patch.py",
}

NPU_SOURCES = {
    **GPU_SOURCES,
    "bad_words": ASCEND / "test_bad_words.py",
    "temperature": ASCEND / "test_temperature.py",
    "gumbel_sample": ASCEND / "test_gumbel_sampling.py",
    "topk_log_softmax": ASCEND / "test_log_softmax.py",
    "ranks": ASCEND / "test_compute_topk_logprobs.py",
    "min_p": ASCEND / "test_min_p.py",
    "penalties": ASCEND / "test_penality.py",
    "bincount": ASCEND / "test_bincount.py",
    "prepare_dflash_inputs_kernel": MISSING / "test_prepare_dflash_inputs_kernel_ascend_patch.py",
    "rejection_kernel": MISSING / "test_probabilistic_rejection_kernel_patch.py",
    "resample_kernel": MISSING / "test_resample_kernel_patch.py",
    "compute_slot_mappings_kernel": ASCEND / "test_compute_slot_mapping.py",
    "post_update_kernel": ASCEND / "test_post_update.py",
    "apply_grammar_bitmask_kernel": MISSING / "test_apply_grammar_bitmask_kernel_patch.py",
}

UPSTREAM_REWRITES = {
    "vllm_ascend.worker.v2.sample.bad_words": "vllm.v1.worker.gpu.sample.bad_words",
    "vllm_ascend.worker.v2.sample.gumbel": "vllm.v1.worker.gpu.sample.gumbel",
    "vllm_ascend.worker.v2.sample.logprob": "vllm.v1.worker.gpu.sample.logprob",
    "vllm_ascend.worker.v2.sample.min_p": "vllm.v1.worker.gpu.sample.min_p",
    "vllm_ascend.worker.v2.sample.penalties": "vllm.v1.worker.gpu.sample.penalties",
    "vllm_ascend.worker.v2.structured_outputs": "vllm.v1.worker.gpu.structured_outputs",
}

HEADER = """# GENERATED STRICT UT. Source: {source}\n# Do not edit mechanically; update the reviewed Codex source or strict generator.\nfrom accuracy_test.strict_ut.runtime_{backend} import STRICT_DEVICE as _STRICT_DEVICE\n"""


def _gpu_transform(text: str, name: str) -> str:
    for old, new in UPSTREAM_REWRITES.items():
        text = text.replace(old, new)

    text = text.replace("torch.npu", "torch.cuda")
    text = text.replace(".npu()", ".cuda()")
    text = text.replace('"npu"', '"cuda"')
    text = text.replace("'npu'", "'cuda'")

    text = re.sub(
        r"from vllm_ascend\.ops\.triton\.triton_utils import "
        r"init_device_properties_triton",
        "from accuracy_test.strict_ut.runtime_gpu import "
        "init_device_properties_triton",
        text,
    )

    # The slot-mapping reviewed test imports both implementations. On GPU both
    # launch sites intentionally use the upstream kernel; its output is still
    # checked against the deterministic mapping assertions in the test.
    if name == "compute_slot_mappings_kernel":
        text = re.sub(
            r"from vllm_ascend\.worker\.v2\.block_table import \(\s*"
            r"_compute_slot_mappings_kernel as ascend_compute_slot_mappings_kernel,\s*\)",
            "ascend_compute_slot_mappings_kernel = ref_compute_slot_mappings_kernel",
            text,
            flags=re.MULTILINE,
        )

    if name == "post_update_kernel":
        start = text.index("post_update_kernel_upstream = None")
        end = text.index("def generate_test_data")
        replacement = (
            "from vllm.v1.worker.gpu.input_batch import (\n"
            "    _post_update_kernel as post_update_kernel_upstream,\n"
            ")\n"
            "post_update_kernel_npu = post_update_kernel_upstream\n"
            "from accuracy_test.strict_ut.runtime_gpu import (\n"
            "    get_vectorcore_num,\n"
            "    init_device_properties_triton,\n"
            ")\n"
            "_post_update_import_error = None\n"
            "_post_update_import_traceback = None\n\n"
        )
        text = text[:start] + replacement + text[end:]

    return text


def _npu_transform(text: str) -> str:
    # Keep the reviewed NPU implementation and launch intact. Only normalize
    # the runtime import so every file gets the same fail-fast device check.
    text = text.replace(
        "from vllm_ascend.ops.triton.triton_utils import "
        "init_device_properties_triton",
        "from accuracy_test.strict_ut.runtime_npu import "
        "init_device_properties_triton",
    )
    return text


def _write_generated(backend: str, name: str, source: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Reviewed source is missing: {source}")
    text = source.read_text(encoding="utf-8")
    text = _gpu_transform(text, name) if backend == "gpu" else _npu_transform(text)
    if name == "compute_slot_mappings_kernel":
        text = text.replace(
            "[t.data_ptr() for t in block_table]",
            "[block_tables[0].data_ptr()]",
        )
        assertion = "        assert torch.equal(slot_mappings, ref_slot_mappings), ("
        expected = (
            "        expected_slots = (\n"
            "            block_tables[0][63, 0].to(torch.int64) * 128\n"
            "            + torch.arange(5, dtype=torch.int64, device=device)\n"
            "        )\n"
            "        assert torch.equal(slot_mappings[0, :5], expected_slots)\n"
            "        assert torch.all(slot_mappings[0, 5:] == -1)\n\n"
        )
        text = text.replace(assertion, expected + assertion)
    if backend == "gpu" and name == "post_update_kernel":
        text = text.replace(
            "launch_post_update_kernel(post_update_kernel_npu, kernel_inputs_npu, ascend=True)",
            "launch_post_update_kernel(post_update_kernel_npu, kernel_inputs_npu, ascend=False)",
        )
    rel = source.relative_to(ROOT).as_posix()
    output = OUT / backend / f"test_{name}.py"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        HEADER.format(source=rel, backend=backend) + text,
        encoding="utf-8",
    )


RESIDUAL_TEMPLATE = r'''# Direct strict test for vLLM-main _compute_local_residual_mass_kernel.
from accuracy_test.strict_ut.runtime_{backend} import DEVICE, synchronize

import pytest
import torch

from vllm.triton_utils import triton
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_local_residual_mass_kernel,
)

from accuracy_test.strict_ut.metrics import assert_float_close
def _local_stats(rows: torch.Tensor, block_size: int):
    num_rows, vocab_size = rows.shape
    num_blocks = triton.cdiv(vocab_size, block_size)
    maxes = torch.empty(num_rows, num_blocks, dtype=torch.float32, device=rows.device)
    sumexp = torch.empty_like(maxes)
    for row in range(num_rows):
        for block in range(num_blocks):
            values = rows[row, block * block_size : min((block + 1) * block_size, vocab_size)].float()
            maximum = values.max()
            maxes[row, block] = maximum
            sumexp[row, block] = torch.exp(values - maximum).sum()
    return maxes, sumexp


@pytest.mark.parametrize("vocab_size", [8191, 8192, 8193])
def test_compute_local_residual_mass(vocab_size):
    torch.manual_seed(17)
    num_reqs = 2
    num_speculative_steps = 3
    rows_per_req = num_speculative_steps + 1
    num_logits = num_reqs * rows_per_req
    block_size = 8192
    num_blocks = triton.cdiv(vocab_size, block_size)

    target = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=DEVICE)
    draft = torch.randn(num_reqs, num_speculative_steps, vocab_size, dtype=torch.float32, device=DEVICE)
    expanded_idx = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE).repeat_interleave(rows_per_req)
    expanded_pos = torch.arange(rows_per_req, dtype=torch.int32, device=DEVICE).repeat(num_reqs)
    temperature = torch.ones(num_reqs, dtype=torch.float32, device=DEVICE)

    target_max, target_sumexp = _local_stats(target, block_size)
    draft_rows = torch.empty_like(target)
    for row in range(num_logits):
        req = int(expanded_idx[row])
        pos = min(int(expanded_pos[row]), num_speculative_steps - 1)
        draft_rows[row] = draft[req, pos]
    draft_max, draft_sumexp = _local_stats(draft_rows, block_size)

    cumulative_log_p = torch.full((num_logits,), torch.log(torch.tensor(0.75)), dtype=torch.float32, device=DEVICE)
    sentinel = -777.0
    output = torch.full((num_logits, num_blocks), sentinel, dtype=torch.float32, device=DEVICE)

    _compute_local_residual_mass_kernel[(num_logits, num_blocks)](
        output,
        output.stride(0),
        cumulative_log_p,
        target,
        target.stride(0),
        target_max,
        target_max.stride(0),
        target_sumexp,
        target_sumexp.stride(0),
        draft,
        draft.stride(0),
        draft.stride(1),
        draft_max,
        draft_max.stride(0),
        draft_sumexp,
        draft_sumexp.stride(0),
        expanded_idx,
        expanded_pos,
        temperature,
        vocab_size,
        num_speculative_steps,
        num_blocks,
        BLOCK_SIZE=block_size,
        PADDED_VOCAB_NUM_BLOCKS=triton.next_power_of_2(num_blocks),
    )
    synchronize()

    expected = torch.full_like(output.cpu(), sentinel)
    for row in range(num_logits):
        pos = int(expanded_pos[row])
        if pos == 0 or pos >= num_speculative_steps:
            continue
        req = int(expanded_idx[row])
        target_prob = torch.softmax(target[row].double().cpu(), dim=-1)
        draft_prob = torch.softmax(draft[req, pos].double().cpu(), dim=-1)
        residual = torch.clamp(0.75 * target_prob - draft_prob, min=0)
        for block in range(num_blocks):
            expected[row, block] = residual[
                block * block_size : min((block + 1) * block_size, vocab_size)
            ].sum().float()

    assert_float_close(output, expected, rtol=2e-5, atol=2e-6)
'''


def main() -> None:
    for backend, mapping in (("gpu", GPU_SOURCES), ("npu", NPU_SOURCES)):
        for name, source in mapping.items():
            _write_generated(backend, name, source)
        residual = OUT / backend / "test_compute_local_residual_mass_kernel.py"
        residual.write_text(RESIDUAL_TEMPLATE.format(backend=backend), encoding="utf-8")

    expected = set(GPU_SOURCES) | {"compute_local_residual_mass_kernel"}
    if len(expected) != 37:
        raise RuntimeError(f"Expected 37 logical operators, got {len(expected)}")
    for backend in ("gpu", "npu"):
        generated = list((OUT / backend).glob("test_*.py"))
        if len(generated) != 37:
            raise RuntimeError(f"Expected 37 {backend} tests, got {len(generated)}")


if __name__ == "__main__":
    main()
