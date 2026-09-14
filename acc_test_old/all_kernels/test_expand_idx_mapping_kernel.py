# vLLM vanilla kernel: _expand_idx_mapping_kernel from
# vllm/vllm/v1/worker/gpu/input_batch.py

"""
Precision test for _expand_idx_mapping_kernel.

Expands per-request idx_mapping to per-logit expanded_idx_mapping and
fills expanded_local_pos with positions within each request's logit range.

Kernel signature:
    _expand_idx_mapping_kernel(
        idx_mapping_ptr,         # [num_reqs] int32
        expanded_idx_mapping_ptr, # [total_num_logits] int64
        expanded_local_pos_ptr,   # [total_num_logits] int32
        cu_num_logits_ptr,        # [num_reqs + 1] int64
        BLOCK_SIZE: tl.constexpr,
    )
"""

import torch
import pytest

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import _expand_idx_mapping_kernel
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _expand_idx_mapping_ref(
    idx_mapping,
    cu_num_logits,
    max_expand_len,
):
    """CPU reference for _expand_idx_mapping_kernel."""
    num_reqs = idx_mapping.shape[0]
    total_num_logits = int(cu_num_logits[-1].item())
    out_expanded_idx_mapping = torch.zeros(total_num_logits, dtype=torch.int64)
    out_expanded_local_pos = torch.zeros(total_num_logits, dtype=torch.int32)

    for req_idx in range(num_reqs):
        start = int(cu_num_logits[req_idx].item())
        end = int(cu_num_logits[req_idx + 1].item())
        num_tokens = end - start
        req_state_idx = int(idx_mapping[req_idx].item())
        for j in range(num_tokens):
            out_expanded_idx_mapping[start + j] = req_state_idx
            out_expanded_local_pos[start + j] = j

    return out_expanded_idx_mapping, out_expanded_local_pos


class TestExpandIdxMappingKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_reqs", [1, 2, 4])
    @pytest.mark.parametrize("tokens_per_req", [1, 3, 8])
    def test_basic_expand(self, num_reqs, tokens_per_req):
        """Verify correct expansion of idx_mapping and local positions."""
        total_logits = num_reqs * tokens_per_req

        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        cu_num_logits = torch.arange(0, total_logits + 1, tokens_per_req,
                                     dtype=torch.int64, device=self.device)
        expanded_idx_mapping = torch.empty(total_logits, dtype=torch.int64, device=self.device)
        expanded_local_pos = torch.empty(total_logits, dtype=torch.int32, device=self.device)

        BLOCK_SIZE = triton.next_power_of_2(tokens_per_req)

        _expand_idx_mapping_kernel[(num_reqs,)](
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            cu_num_logits,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.npu.synchronize()

        ref_mapping, ref_pos = _expand_idx_mapping_ref(
            idx_mapping.cpu(), cu_num_logits.cpu(), tokens_per_req,
        )

        torch.testing.assert_close(expanded_idx_mapping.cpu(), ref_mapping, rtol=0, atol=0)
        torch.testing.assert_close(expanded_local_pos.cpu(), ref_pos, rtol=0, atol=0)

    def test_uneven_tokens(self):
        """Different requests have different numbers of tokens."""
        num_reqs = 3
        tokens_per_req = [2, 5, 3]
        total_logits = sum(tokens_per_req)

        idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int32, device=self.device)
        cu_num_logits = torch.tensor([0, 2, 7, 10], dtype=torch.int64, device=self.device)
        expanded_idx_mapping = torch.empty(total_logits, dtype=torch.int64, device=self.device)
        expanded_local_pos = torch.empty(total_logits, dtype=torch.int32, device=self.device)

        BLOCK_SIZE = triton.next_power_of_2(max(tokens_per_req))

        _expand_idx_mapping_kernel[(num_reqs,)](
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            cu_num_logits,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.npu.synchronize()

        ref_mapping, ref_pos = _expand_idx_mapping_ref(
            idx_mapping.cpu(), cu_num_logits.cpu(), max(tokens_per_req),
        )

        torch.testing.assert_close(expanded_idx_mapping.cpu(), ref_mapping, rtol=0, atol=0)
        torch.testing.assert_close(expanded_local_pos.cpu(), ref_pos, rtol=0, atol=0)

    def test_non_contiguous_idx_mapping(self):
        """idx_mapping may not be contiguous (e.g., [5, 2, 8])."""
        num_reqs = 3
        tokens_per_req = 2
        total_logits = num_reqs * tokens_per_req

        idx_mapping = torch.tensor([5, 2, 8], dtype=torch.int32, device=self.device)
        cu_num_logits = torch.tensor([0, 2, 4, 6], dtype=torch.int64, device=self.device)
        expanded_idx_mapping = torch.empty(total_logits, dtype=torch.int64, device=self.device)
        expanded_local_pos = torch.empty(total_logits, dtype=torch.int32, device=self.device)

        BLOCK_SIZE = triton.next_power_of_2(tokens_per_req)

        _expand_idx_mapping_kernel[(num_reqs,)](
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            cu_num_logits,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.npu.synchronize()

        ref_mapping, ref_pos = _expand_idx_mapping_ref(
            idx_mapping.cpu(), cu_num_logits.cpu(), tokens_per_req,
        )

        torch.testing.assert_close(expanded_idx_mapping.cpu(), ref_mapping, rtol=0, atol=0)
        torch.testing.assert_close(expanded_local_pos.cpu(), ref_pos, rtol=0, atol=0)
