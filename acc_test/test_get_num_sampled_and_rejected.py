# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.gpu.input_batch import (
    _get_num_sampled_and_rejected_kernel,
    get_num_sampled_and_rejected,
)

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _get_num_sampled_and_rejected_cpu(
    num_sampled: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    prefill_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent PyTorch CPU reference implementation."""
    num_reqs = idx_mapping.shape[0]
    output_num_sampled = num_sampled.clone()
    num_rejected = torch.empty(num_reqs, dtype=torch.int32)

    for batch_idx in range(num_reqs):
        req_state_idx = int(idx_mapping[batch_idx])
        seq_len = int(seq_lens[batch_idx])
        prefill_len_val = int(prefill_len[req_state_idx])

        logits_start = int(cu_num_logits[batch_idx])
        logits_end = int(cu_num_logits[batch_idx + 1])
        num_logits = logits_end - logits_start

        is_chunked_prefilling = seq_len < prefill_len_val

        if is_chunked_prefilling:
            output_num_sampled[batch_idx] = 0
            num_rejected[batch_idx] = 0
        else:
            ns = int(num_sampled[batch_idx])
            output_num_sampled[batch_idx] = ns
            num_rejected[batch_idx] = num_logits - ns

    return output_num_sampled, num_rejected


@pytest.mark.parametrize("dtype", [torch.int32])
def test_get_num_sampled_and_rejected_mixed(
    dtype: torch.dtype,
) -> None:
    """Test with a mix of chunked prefill and decode requests."""
    init_device_properties_triton()

    num_reqs = 5
    # idx_mapping: batch_idx -> req_state_idx
    idx_mapping = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)

    # seq_lens indexed by batch_idx
    seq_lens = torch.tensor([5, 3, 10, 8, 15], dtype=torch.int32)

    # prefill_len indexed by req_state_idx
    prefill_len = torch.tensor([100, 50, 10, 20, 15], dtype=torch.int32)

    # Requests 2 and 3 are complete prefills (seq_len >= prefill_len)
    # Requests 0, 1, 4 are chunked prefills (seq_len < prefill_len) -- wait, let me re-check:
    #   req 0: seq=5 < 100 -> chunked
    #   req 1: seq=3 < 50  -> chunked
    #   req 2: seq=10 >= 10 -> full prefill
    #   req 3: seq=8 < 20  -> chunked
    #   req 4: seq=15 >= 15 -> full prefill (seq_len == prefill_len is not chunked)

    # cu_num_logits: cumulative number of logits per batch request
    cu_num_logits = torch.tensor([0, 2, 5, 10, 12, 17], dtype=torch.int32)

    # num_sampled (input, will be modified in-place)
    num_sampled = torch.tensor([0, 0, 2, 0, 3], dtype=torch.int32)

    expected_num_sampled, expected_num_rejected = (
        _get_num_sampled_and_rejected_cpu(
            num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len,
        )
    )

    device = torch.device("npu")
    num_sampled_npu = num_sampled.to(device)
    seq_lens_npu = seq_lens.to(device)
    cu_num_logits_npu = cu_num_logits.to(device)
    idx_mapping_npu = idx_mapping.to(device)
    prefill_len_npu = prefill_len.to(device)

    num_rejected = torch.empty(num_reqs, dtype=torch.int32, device=device)

    _get_num_sampled_and_rejected_kernel[(num_reqs,)](
        num_sampled_npu,
        num_rejected,
        seq_lens_npu,
        cu_num_logits_npu,
        idx_mapping_npu,
        prefill_len_npu,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        num_sampled_npu.cpu(), expected_num_sampled, rtol=0, atol=0
    )
    torch.testing.assert_close(
        num_rejected.cpu(), expected_num_rejected, rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", [torch.int32])
def test_get_num_sampled_and_rejected_all_chunked_prefill(
    dtype: torch.dtype,
) -> None:
    """Test when all requests are chunked prefills."""
    init_device_properties_triton()

    num_reqs = 4
    idx_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    seq_lens = torch.tensor([3, 5, 7, 2], dtype=torch.int32)
    prefill_len = torch.tensor([50, 100, 200, 30], dtype=torch.int32)
    # All seq_len < prefill_len -> chunked

    cu_num_logits = torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32)
    num_sampled = torch.tensor([1, 2, 3, 4], dtype=torch.int32)

    expected_num_sampled, expected_num_rejected = (
        _get_num_sampled_and_rejected_cpu(
            num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len,
        )
    )

    device = torch.device("npu")
    actual_num_sampled, actual_num_rejected = get_num_sampled_and_rejected(
        num_sampled.to(device),
        seq_lens.to(device),
        cu_num_logits.to(device),
        idx_mapping.to(device),
        prefill_len.to(device),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        actual_num_sampled.cpu(), expected_num_sampled, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_num_rejected.cpu(), expected_num_rejected, rtol=0, atol=0
    )


def test_get_num_sampled_and_rejected_through_wrapper() -> None:
    """Test the full pipeline through the public wrapper function."""
    init_device_properties_triton()

    num_reqs = 3
    idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int32)
    seq_lens = torch.tensor([10, 10, 20], dtype=torch.int32)
    prefill_len = torch.tensor([5, 15, 20], dtype=torch.int32)
    # req 0: seq=10 >= 5 -> full prefill
    # req 1: seq=10 < 15 -> chunked
    # req 2: seq=20 >= 20 -> full prefill (seq_len == prefill_len)

    cu_num_logits = torch.tensor([0, 3, 7, 10], dtype=torch.int32)
    num_sampled = torch.tensor([2, 0, 1], dtype=torch.int32)

    device = torch.device("npu")

    expected_num_sampled, expected_num_rejected = (
        _get_num_sampled_and_rejected_cpu(
            num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len,
        )
    )

    actual_num_sampled, actual_num_rejected = get_num_sampled_and_rejected(
        num_sampled.to(device),
        seq_lens.to(device),
        cu_num_logits.to(device),
        idx_mapping.to(device),
        prefill_len.to(device),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        actual_num_sampled.cpu(), expected_num_sampled, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_num_rejected.cpu(), expected_num_rejected, rtol=0, atol=0
    )
