# Standalone Ascend A3 accuracy test.
# Accuracy UT source: vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_post_update.py
# Kernel source: vllm-ascend-xyz/vllm_ascend/worker/v2/input_batch.py
# Coverage: _post_update_kernel via post_update

from typing import Any
import importlib
import traceback

import pytest
import torch

post_update_gpu = None
post_update_npu = None
_post_update_import_error = None
_post_update_import_traceback = None
try:
    # Bootstrap the parent ops package first. Importing device_op directly can
    # cycle through ops.__init__ -> fused_moe -> experts_selector -> device_op
    # before DeviceOperator is bound. Starting from ops registers the parent
    # package before device_op imports ops.triton children, breaking that cycle.
    importlib.import_module("vllm_ascend.ops")
    device_op = importlib.import_module("vllm_ascend.device.device_op")
    if not hasattr(device_op, "DeviceOperator"):
        raise ImportError(
            "vllm_ascend.device.device_op initialized without DeviceOperator"
        )

    from vllm.v1.worker.gpu.input_batch import post_update as post_update_gpu
    from vllm_ascend.worker.v2.input_batch import post_update as post_update_npu
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


@pytest.mark.parametrize(
    "num_reqs,max_num_reqs,vocab_size,num_speculative_steps",
    [
        (36, 36, 200, 2),
        (48, 48, 32000, 5),
        (128, 128, 32000, 5),
    ],
)
def test_post_update(num_reqs: int, max_num_reqs: int, vocab_size: int, num_speculative_steps: int):
    """Compare upstream and Ascend post_update mutations exactly."""
    if _post_update_import_error is not None:
        pytest.fail(
            "post_update environment compatibility failure; this is not a "
            "precision failure and no kernel was tested.\n"
            f"error={_post_update_import_error}\n"
            f"traceback:\n{_post_update_import_traceback}",
            pytrace=False,
        )

    if post_update_gpu is post_update_npu:
        pytest.fail(
            "post_update reference and Ascend implementation resolve to the "
            "same function after monkey-patching; an independent precision "
            "comparison is impossible.",
            pytrace=False,
        )
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
    kernel_inputs_gpu = {k: data[k].clone() for k in post_update_params}
    kernel_inputs_npu = {k: data[k].clone() for k in post_update_params}

    # Invoke Triton kernel
    post_update_gpu(**kernel_inputs_gpu)
    torch.npu.synchronize()

    post_update_npu(**kernel_inputs_npu)
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
        torch.testing.assert_close(
            kernel_inputs_npu[name],
            kernel_inputs_gpu[name],
            rtol=0,
            atol=0,
            msg=lambda msg, name=name: f"post_update mismatch for {name}:\n{msg}",
        )
