# GENERATED STRICT UT. Source: accuracy_test/codex/missing_accuracy_tests/test_post_update_num_computed_tokens_kernel.py
# Do not edit mechanically; update the reviewed Codex source or strict generator.
from accuracy_test.strict_ut.runtime_npu import STRICT_DEVICE as _STRICT_DEVICE
# vLLM vanilla kernel: _post_update_num_computed_tokens_kernel from
# vllm/vllm/v1/worker/gpu/input_batch.py

"""
Precision test for _post_update_num_computed_tokens_kernel.

Increments num_computed_tokens for each request by the query length.

Kernel signature:
    _post_update_num_computed_tokens_kernel(
        idx_mapping_ptr,         # [num_reqs] int32
        num_computed_tokens_ptr, # [max_num_reqs] int32
        query_start_loc_ptr,     # [num_reqs + 1] int32
    )
"""

import torch
import pytest

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import _post_update_num_computed_tokens_kernel
from accuracy_test.strict_ut.runtime_npu import init_device_properties_triton


def _post_update_num_computed_tokens_ref(
    idx_mapping,
    num_computed_tokens,
    query_start_loc,
):
    """CPU reference for _post_update_num_computed_tokens_kernel."""
    out = num_computed_tokens.clone()
    num_reqs = idx_mapping.shape[0]
    for batch_id in range(num_reqs):
        qs = int(query_start_loc[batch_id].item())
        qe = int(query_start_loc[batch_id + 1].item())
        query_len = qe - qs
        req_state_idx = int(idx_mapping[batch_id].item())
        out[req_state_idx] += query_len
    return out


class TestPostUpdateNumComputedTokensKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_reqs", [1, 2, 4])
    @pytest.mark.parametrize("query_len", [1, 4, 8])
    def test_basic_increment(self, num_reqs, query_len):
        """Verify num_computed_tokens is incremented by query length."""
        max_num_reqs = max(num_reqs, 4)

        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        num_computed_tokens = torch.full((max_num_reqs,), 10, dtype=torch.int32, device=self.device)
        query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32, device=self.device) * query_len

        _post_update_num_computed_tokens_kernel[(num_reqs,)](
            idx_mapping,
            num_computed_tokens,
            query_start_loc,
        )
        torch.npu.synchronize()

        expected = _post_update_num_computed_tokens_ref(
            idx_mapping.cpu(), torch.full((max_num_reqs,), 10, dtype=torch.int32),
            query_start_loc.cpu(),
        )

        torch.testing.assert_close(num_computed_tokens.cpu(), expected, rtol=0, atol=0)

    def test_non_contiguous_idx_mapping(self):
        """idx_mapping has non-contiguous indices."""
        num_reqs = 3
        max_num_reqs = 6

        idx_mapping = torch.tensor([5, 0, 3], dtype=torch.int32, device=self.device)
        num_computed_tokens = torch.arange(max_num_reqs, dtype=torch.int32, device=self.device)
        query_start_loc = torch.tensor([0, 2, 5, 9], dtype=torch.int32, device=self.device)

        _post_update_num_computed_tokens_kernel[(num_reqs,)](
            idx_mapping,
            num_computed_tokens,
            query_start_loc,
        )
        torch.npu.synchronize()

        expected = _post_update_num_computed_tokens_ref(
            idx_mapping.cpu(), torch.arange(max_num_reqs, dtype=torch.int32),
            query_start_loc.cpu(),
        )

        torch.testing.assert_close(num_computed_tokens.cpu(), expected, rtol=0, atol=0)

    def test_zero_query_len(self):
        """When query_len is 0, num_computed_tokens should not change."""
        num_reqs = 2
        max_num_reqs = 2

        idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
        num_computed_tokens = torch.tensor([5, 10], dtype=torch.int32, device=self.device)
        query_start_loc = torch.tensor([0, 0, 0], dtype=torch.int32, device=self.device)

        before = num_computed_tokens.clone()

        _post_update_num_computed_tokens_kernel[(num_reqs,)](
            idx_mapping,
            num_computed_tokens,
            query_start_loc,
        )
        torch.npu.synchronize()

        torch.testing.assert_close(num_computed_tokens.cpu(), before.cpu(), rtol=0, atol=0)
