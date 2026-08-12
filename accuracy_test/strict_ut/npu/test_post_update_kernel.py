# GENERATED STRICT UT. Source: accuracy_test/codex/existing_accuracy_tests/from_vllm_ascend/test_post_update.py
# Do not edit mechanically; update the reviewed Codex source or strict generator.
from accuracy_test.strict_ut.runtime_npu import STRICT_DEVICE as _STRICT_DEVICE
# Standalone Ascend A3 accuracy test.
# Accuracy UT source: vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_post_update.py
# Kernel source: vllm-ascend-xyz/vllm_ascend/worker/v2/input_batch.py
# Coverage: direct comparison of upstream and Ascend _post_update_kernel

import importlib
import traceback
from typing import Any

import pytest
import torch

post_update_kernel_upstream = None
post_update_kernel_npu = None
get_vectorcore_num = None
init_device_properties_triton = None
_post_update_import_error = None
_post_update_import_traceback = None
try:
    from vllm.v1.worker.gpu.input_batch import (
        _post_update_kernel as post_update_kernel_upstream,
    )
    from vllm_ascend.ops.triton.triton_utils import (
        get_vectorcore_num,
        init_device_properties_triton,
    )
    from vllm_ascend.worker.v2.input_batch import (
        _post_update_kernel as post_update_kernel_npu,
    )
except Exception as exc:
    _post_update_import_error = exc
    _post_update_import_traceback = traceback.format_exc()


def generate_test_data(
    num_reqs: int, max_num_reqs: int, vocab_size: int, num_speculative_steps: int, device: str
) -> dict[str, Any]:
    """
    Generate random test data.
    Return a dictionary containing all input tensors and the additional field 'expected_query_lens' for validation.
    """

    if num_reqs > max_num_reqs:
        raise ValueError("num_reqs cannot be larger than max_num_reqs")

    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    num_computed_tokens = torch.randint(0, 100, (max_num_reqs,), dtype=torch.int32, device=device)
    last_sampled_tokens = torch.randint(0, vocab_size, (max_num_reqs,), dtype=torch.int32, device=device)
    output_bin_counts = torch.randint(0, 10, (max_num_reqs, vocab_size), dtype=torch.int32, device=device)
    sampled_tokens = torch.randint(
        0, vocab_size, (num_reqs, num_speculative_steps + 1), dtype=torch.int32, device=device
    )
    num_sampled = torch.randint(1, num_speculative_steps + 2, (num_reqs,), dtype=torch.int32, device=device)
    num_rejected = torch.randint(0, num_speculative_steps + 1, (num_reqs,), dtype=torch.int32, device=device)
    num_rejected = torch.min(num_rejected, num_sampled - 1)

    query_lengths = torch.randint(1, 20, (num_reqs,), dtype=torch.int32, device=device)
    query_start_loc = torch.cat(
        [torch.tensor([0], dtype=torch.int32, device=device), torch.cumsum(query_lengths, dim=0)]
    )
    total_len = torch.randint(50, 200, (max_num_reqs,), dtype=torch.int32, device=device)

    max_model_len = 3000  # 或者可以从total_len的最大值获取
    all_token_ids = torch.randint(0, vocab_size, (max_num_reqs, max_model_len), dtype=torch.int32, device=device)

    return {
        "idx_mapping": idx_mapping,
        "num_computed_tokens": num_computed_tokens,
        "last_sampled_tokens": last_sampled_tokens,
        "output_bin_counts": output_bin_counts,
        "sampled_tokens": sampled_tokens,
        "num_sampled": num_sampled,
        "num_rejected": num_rejected,
        "query_start_loc": query_start_loc,
        "all_token_ids": all_token_ids,
        "total_len": total_len,
    }


def post_update_ref(
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    output_bin_counts: torch.Tensor,
    sampled_tokens: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
) -> None:
    """Independent serial CPU reference for Ascend post_update."""
    for row_idx in range(idx_mapping.numel()):
        req_state_idx = int(idx_mapping[row_idx])
        old_total_len = int(total_len[req_state_idx])
        sampled_count = int(num_sampled[row_idx])

        if sampled_count > 0:
            last_sampled_tokens[req_state_idx] = sampled_tokens[
                row_idx, sampled_count - 1
            ]
            total_len[req_state_idx] = old_total_len + sampled_count

        for sample_idx in range(sampled_count):
            token_id = int(sampled_tokens[row_idx, sample_idx])
            output_bin_counts[req_state_idx, token_id] += 1
            all_token_ids[req_state_idx, old_total_len + sample_idx] = token_id

        query_len = int(query_start_loc[row_idx + 1] - query_start_loc[row_idx])
        rejected_count = int(num_rejected[row_idx])
        num_computed_tokens[req_state_idx] += query_len - rejected_count


def launch_post_update_kernel(kernel, inputs: dict[str, torch.Tensor], *, ascend: bool) -> None:
    """Launch an installed kernel using its actual JIT argument names."""
    num_rows = inputs["idx_mapping"].shape[0]
    values = {
        "idx_mapping_ptr": inputs["idx_mapping"],
        "idx_mapping_stride": inputs["idx_mapping"].stride(0),
        "num_computed_tokens_ptr": inputs["num_computed_tokens"],
        "last_sampled_tokens_ptr": inputs["last_sampled_tokens"],
        "output_bin_counts_ptr": inputs["output_bin_counts"],
        "output_bin_counts_stride": inputs["output_bin_counts"].stride(0),
        "sampled_tokens_ptr": inputs["sampled_tokens"],
        "sampled_tokens_stride": inputs["sampled_tokens"].stride(0),
        "num_rows": num_rows,
        "num_sampled_ptr": inputs["num_sampled"],
        "num_rejected_ptr": inputs["num_rejected"],
        "query_start_loc_ptr": inputs["query_start_loc"],
        "all_token_ids_ptr": inputs["all_token_ids"],
        "all_token_ids_stride": inputs["all_token_ids"].stride(0),
        "total_len_ptr": inputs["total_len"],
    }
    arg_names = tuple(kernel.arg_names)
    missing = [name for name in arg_names if name not in values]
    if missing:
        raise RuntimeError(
            "unsupported installed post_update kernel signature; "
            f"missing values for {missing}; arg_names={arg_names}"
        )
    kwargs = {name: values[name] for name in arg_names}
    if ascend:
        grid = (min(num_rows, get_vectorcore_num()),)
        kernel[grid](**kwargs)
    else:
        kernel[(num_rows,)](**kwargs, num_warps=1)


@pytest.mark.parametrize(
    "num_reqs,max_num_reqs,vocab_size,num_speculative_steps",
    [
        (36, 36, 200, 2),
        (48, 48, 32000, 5),
        (128, 128, 32000, 5),
    ],
)
def test_post_update(num_reqs: int, max_num_reqs: int, vocab_size: int, num_speculative_steps: int):
    """Compare upstream and Ascend kernels, with a CPU correctness oracle."""
    if _post_update_import_error is not None:
        pytest.fail(
            "post_update environment compatibility failure; this is not a "
            "precision failure and no kernel was tested.\n"
            f"error={_post_update_import_error}\n"
            f"traceback:\n{_post_update_import_traceback}",
            pytrace=False,
        )

    init_device_properties_triton()
    torch.manual_seed(42)

    post_update_params = [
        "idx_mapping",
        "num_computed_tokens",
        "last_sampled_tokens",
        "output_bin_counts",
        "sampled_tokens",
        "num_sampled",
        "num_rejected",
        "query_start_loc",
        "all_token_ids",
        "total_len",
    ]

    data = generate_test_data(num_reqs, max_num_reqs, vocab_size, num_speculative_steps, device="npu")
    kernel_inputs_upstream = {k: data[k].clone() for k in post_update_params}
    kernel_inputs_npu = {k: data[k].clone() for k in post_update_params}
    reference_inputs = {k: data[k].cpu().clone() for k in post_update_params}

    post_update_ref(**reference_inputs)

    launch_post_update_kernel(
        post_update_kernel_upstream, kernel_inputs_upstream, ascend=False
    )
    torch.npu.synchronize()

    launch_post_update_kernel(post_update_kernel_npu, kernel_inputs_npu, ascend=True)
    torch.npu.synchronize()

    # Every mutated value is int32, so require exact equality with no tolerance.
    mutated_outputs = (
        "num_computed_tokens",
        "last_sampled_tokens",
        "output_bin_counts",
        "all_token_ids",
        "total_len",
    )
    for name in mutated_outputs:
        upstream = kernel_inputs_upstream[name].cpu()
        ascend_output = kernel_inputs_npu[name].cpu()
        reference = reference_inputs[name]
        torch.testing.assert_close(
            upstream,
            reference,
            rtol=0,
            atol=0,
            msg=lambda msg, name=name: (
                f"upstream post_update mismatch against CPU reference for {name}:\n{msg}"
            ),
        )
        torch.testing.assert_close(
            ascend_output,
            upstream,
            rtol=0,
            atol=0,
            msg=lambda msg, name=name: (
                f"Ascend post_update patch mismatch against upstream for {name}:\n{msg}"
            ),
        )
