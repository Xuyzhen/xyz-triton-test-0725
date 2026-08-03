# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.gpu.sample.logprob import _fill_logprob_token_ids_kernel

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _fill_logprob_token_ids_cpu(
    out_token_ids: torch.Tensor,
    out_valid_mask: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    topk_indices: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    num_per_req_token_ids: torch.Tensor,
    per_req_token_ids: torch.Tensor,
    num_topk: int,
) -> None:
    """Pure PyTorch CPU reference for fill_logprob_token_ids_kernel.

    Column 0 is always the sampled token (always valid).
    Remaining columns: if the request has custom token IDs (*num_per_req > 0*),
    those override the top-k entries; otherwise top-k indices are used.
    """
    batch_size = sampled_token_ids.shape[0]

    for batch_idx in range(batch_size):
        # Column 0: sampled token, always valid
        out_token_ids[batch_idx, 0] = sampled_token_ids[batch_idx]
        out_valid_mask[batch_idx, 0] = True

        req_state_idx = int(expanded_idx_mapping[batch_idx])
        num_custom = int(num_per_req_token_ids[req_state_idx])

        if num_custom > 0:
            src = per_req_token_ids[req_state_idx]
            n = num_custom
        else:
            src = topk_indices[batch_idx]
            n = num_topk

        for col in range(n):
            out_token_ids[batch_idx, 1 + col] = src[col]
            out_valid_mask[batch_idx, 1 + col] = True

        # Remainder stays zero / False as initialized


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 0 else 1


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("num_topk", [0, 3, 5])
def test_fill_logprob_token_ids_basic(batch_size: int, num_topk: int) -> None:
    """Fill logprob token IDs from top-k and per-request custom tokens.

    Tests a mix where some requests use custom token IDs and others use
    top-k indices. Verifies that column 0 is always the sampled token
    and the remaining columns are filled correctly.
    """
    init_device_properties_triton()
    torch.manual_seed(42)

    max_num_reqs = 8
    max_per_req_ids = 7  # max num_custom across requests
    num_cols = max(num_topk, max_per_req_ids)
    padded_cols = _next_power_of_2(num_cols)

    sampled_token_ids = torch.randint(0, 32000, (batch_size,), dtype=torch.int64)
    expanded_idx_mapping = torch.randint(0, max_num_reqs, (batch_size,),
                                         dtype=torch.int64)

    # Per-request token IDs
    per_req_token_ids = torch.zeros(
        (max_num_reqs, max_per_req_ids), dtype=torch.int32
    )
    num_per_req = torch.zeros(max_num_reqs, dtype=torch.int32)

    for r in range(max_num_reqs):
        # Some requests have custom token IDs, some don't
        if r % 2 == 0:
            n = min(max_per_req_ids, 3 + r % 5)
            num_per_req[r] = n
            per_req_token_ids[r, :n] = torch.randint(0, 32000, (n,))

    topk_indices = torch.randint(0, 32000, (batch_size, num_topk),
                                 dtype=torch.int32) if num_topk > 0 else torch.empty(
        batch_size, 0, dtype=torch.int32
    )

    # CPU reference
    out_token_ids_cpu = torch.zeros(batch_size, 1 + num_cols, dtype=torch.int64)
    out_valid_cpu = torch.zeros(batch_size, 1 + num_cols, dtype=torch.bool)
    _fill_logprob_token_ids_cpu(
        out_token_ids_cpu, out_valid_cpu,
        sampled_token_ids, topk_indices,
        expanded_idx_mapping, num_per_req, per_req_token_ids,
        num_topk,
    )

    # NPU kernel
    device = torch.device("npu")
    out_token_ids_npu = torch.zeros(
        batch_size, 1 + padded_cols, dtype=torch.int64, device=device
    )
    out_valid_npu = torch.zeros(
        batch_size, 1 + padded_cols, dtype=torch.bool, device=device
    )

    _fill_logprob_token_ids_kernel[(batch_size,)](
        out_token_ids_npu,
        out_token_ids_npu.stride(0),
        out_valid_npu,
        out_valid_npu.stride(0),
        sampled_token_ids.to(device),
        topk_indices.to(device),
        topk_indices.stride(0) if num_topk > 0 else 0,
        expanded_idx_mapping.to(device),
        num_per_req.to(device),
        per_req_token_ids.to(device),
        per_req_token_ids.stride(0),
        NUM_TOPK=num_topk,
        PADDED_COLS=padded_cols,
    )
    torch.npu.synchronize()

    # Compare valid columns only (the padded area may differ)
    torch.testing.assert_close(
        out_token_ids_npu.cpu()[:, :1 + num_cols],
        out_token_ids_cpu,
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        out_valid_npu.cpu()[:, :1 + num_cols],
        out_valid_cpu,
    )


def test_fill_logprob_token_ids_all_custom() -> None:
    """All requests use custom per-request token IDs; no top-k fallback."""
    init_device_properties_triton()
    torch.manual_seed(7)

    batch_size = 4
    num_topk = 5
    max_num_reqs = 4
    max_per_req_ids = 6
    num_cols = max(num_topk, max_per_req_ids)
    padded_cols = _next_power_of_2(num_cols)

    expanded_idx_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    sampled_token_ids = torch.tensor([100, 200, 300, 400], dtype=torch.int64)

    per_req_token_ids = torch.zeros(
        (max_num_reqs, max_per_req_ids), dtype=torch.int32
    )
    num_per_req = torch.zeros(max_num_reqs, dtype=torch.int32)
    for r in range(max_num_reqs):
        n = max_per_req_ids
        num_per_req[r] = n
        per_req_token_ids[r] = torch.tensor(
            [10 + r, 20 + r, 30 + r, 40 + r, 50 + r, 60 + r],
        )

    topk_indices = torch.randint(0, 32000, (batch_size, num_topk),
                                 dtype=torch.int32)

    out_token_ids_cpu = torch.zeros(batch_size, 1 + num_cols, dtype=torch.int64)
    out_valid_cpu = torch.zeros(batch_size, 1 + num_cols, dtype=torch.bool)
    _fill_logprob_token_ids_cpu(
        out_token_ids_cpu, out_valid_cpu,
        sampled_token_ids, topk_indices,
        expanded_idx_mapping, num_per_req, per_req_token_ids,
        num_topk,
    )

    device = torch.device("npu")
    out_token_ids_npu = torch.zeros(
        batch_size, 1 + padded_cols, dtype=torch.int64, device=device
    )
    out_valid_npu = torch.zeros(
        batch_size, 1 + padded_cols, dtype=torch.bool, device=device
    )

    _fill_logprob_token_ids_kernel[(batch_size,)](
        out_token_ids_npu,
        out_token_ids_npu.stride(0),
        out_valid_npu,
        out_valid_npu.stride(0),
        sampled_token_ids.to(device),
        topk_indices.to(device),
        topk_indices.stride(0),
        expanded_idx_mapping.to(device),
        num_per_req.to(device),
        per_req_token_ids.to(device),
        per_req_token_ids.stride(0),
        NUM_TOPK=num_topk,
        PADDED_COLS=padded_cols,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        out_token_ids_npu.cpu()[:, :1 + num_cols],
        out_token_ids_cpu,
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        out_valid_npu.cpu()[:, :1 + num_cols],
        out_valid_cpu,
    )


def test_fill_logprob_token_ids_no_topk_no_custom() -> None:
    """NUM_TOPK=0 and no custom per-request tokens: only column 0 is valid."""
    init_device_properties_triton()
    torch.manual_seed(3)

    batch_size = 3
    num_topk = 0
    max_num_reqs = 3
    max_per_req_ids = 0
    num_cols = 0
    padded_cols = 1  # next_power_of_2(0) = 1

    expanded_idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int64)
    sampled_token_ids = torch.tensor([111, 222, 333], dtype=torch.int64)

    per_req_token_ids = torch.zeros(
        (max_num_reqs, max_per_req_ids or 1), dtype=torch.int32
    )
    num_per_req = torch.zeros(max_num_reqs, dtype=torch.int32)

    topk_indices = torch.empty(batch_size, 0, dtype=torch.int32)

    out_token_ids_cpu = torch.zeros(batch_size, 1, dtype=torch.int64)
    out_valid_cpu = torch.zeros(batch_size, 1, dtype=torch.bool)
    _fill_logprob_token_ids_cpu(
        out_token_ids_cpu, out_valid_cpu,
        sampled_token_ids, topk_indices,
        expanded_idx_mapping, num_per_req, per_req_token_ids,
        num_topk,
    )

    device = torch.device("npu")
    out_token_ids_npu = torch.zeros(
        batch_size, 1 + padded_cols, dtype=torch.int64, device=device
    )
    out_valid_npu = torch.zeros(
        batch_size, 1 + padded_cols, dtype=torch.bool, device=device
    )

    _fill_logprob_token_ids_kernel[(batch_size,)](
        out_token_ids_npu,
        out_token_ids_npu.stride(0),
        out_valid_npu,
        out_valid_npu.stride(0),
        sampled_token_ids.to(device),
        topk_indices.to(device),
        0,
        expanded_idx_mapping.to(device),
        num_per_req.to(device),
        per_req_token_ids.to(device),
        per_req_token_ids.stride(0),
        NUM_TOPK=num_topk,
        PADDED_COLS=padded_cols,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        out_token_ids_npu.cpu()[:, :1],
        out_token_ids_cpu,
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        out_valid_npu.cpu()[:, :1],
        out_valid_cpu,
    )
