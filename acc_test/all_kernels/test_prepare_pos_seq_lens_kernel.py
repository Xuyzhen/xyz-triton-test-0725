# vLLM vanilla kernel: _prepare_pos_seq_lens_kernel from
# vllm/vllm/v1/worker/gpu/input_batch.py

"""
Precision test for _prepare_pos_seq_lens_kernel.

Computes positions and sequence lengths for a batch, padding unused
seq_lens to 0 for CUDA graph support.

Kernel signature:
    _prepare_pos_seq_lens_kernel(
        pos_ptr,                  # [num_tokens] int64
        seq_lens_ptr,             # [max_num_reqs] int32
        idx_mapping_ptr,          # [num_reqs] int32
        query_start_loc_ptr,      # [num_reqs + 1] int32
        num_computed_tokens_ptr,  # [max_num_reqs] int64 (or int32)
        max_num_reqs,
        BLOCK_SIZE: tl.constexpr,
    )
"""

import torch
import pytest

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import _prepare_pos_seq_lens_kernel
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _prepare_pos_seq_lens_ref(
    idx_mapping,           # [num_reqs]
    query_start_loc,       # [num_reqs + 1]
    num_computed_tokens,   # [max_num_reqs]
    max_num_reqs,
    num_tokens,
):
    """CPU reference for pos/seq_lens preparation."""
    num_reqs = idx_mapping.shape[0]
    pos = torch.zeros(num_tokens, dtype=torch.int64)
    seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32)

    for req_id in range(num_reqs):
        req_state_idx = int(idx_mapping[req_id].item())
        nct = int(num_computed_tokens[req_state_idx].item())
        start = int(query_start_loc[req_id].item())
        end = int(query_start_loc[req_id + 1].item())
        query_len = end - start
        seq_len = nct + query_len
        seq_lens[req_id] = seq_len
        for i in range(query_len):
            pos[start + i] = nct + i

    # Padded entries (num_reqs..max_num_reqs) stay 0
    return pos, seq_lens


class TestPreparePosSeqLensKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_reqs", [1, 2, 4, 8])
    @pytest.mark.parametrize("max_num_reqs", [8, 16])
    @pytest.mark.parametrize("tokens_per_req", [1, 4, 8])
    def test_pos_seq_lens(self, num_reqs, max_num_reqs, tokens_per_req):
        """Test position and seq_lens computation."""
        if num_reqs > max_num_reqs:
            pytest.skip("num_reqs must be <= max_num_reqs")

        # Set varying seq_lens each request
        num_tokens = num_reqs * tokens_per_req
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)

        query_start_loc = torch.arange(
            num_reqs + 1, device=self.device, dtype=torch.int32
        ) * tokens_per_req

        num_computed_tokens = torch.randint(
            0, 64, (max_num_reqs,), dtype=torch.int32, device=self.device
        )

        pos = -torch.ones(num_tokens, dtype=torch.int64, device=self.device)
        seq_lens = -torch.ones(max_num_reqs, dtype=torch.int32, device=self.device)

        _prepare_pos_seq_lens_kernel[(num_reqs + 1,)](
            pos,
            seq_lens,
            idx_mapping,
            query_start_loc,
            num_computed_tokens,
            max_num_reqs,
            BLOCK_SIZE=4,
        )
        torch.npu.synchronize()

        ref_pos, ref_seq_lens = _prepare_pos_seq_lens_ref(
            idx_mapping.cpu(),
            query_start_loc.cpu(),
            num_computed_tokens.cpu(),
            max_num_reqs,
            num_tokens,
        )
        torch.testing.assert_close(pos.cpu(), ref_pos, rtol=0, atol=0)
        torch.testing.assert_close(seq_lens.cpu(), ref_seq_lens, rtol=0, atol=0)

    def test_cuda_graph_padding(self):
        """Test that seq_lens entries beyond num_reqs are zeroed."""
        num_reqs = 2
        max_num_reqs = 8
        tokens_per_req = 4
        num_tokens = num_reqs * tokens_per_req

        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        query_start_loc = torch.arange(
            num_reqs + 1, device=self.device, dtype=torch.int32
        ) * tokens_per_req
        num_computed_tokens = torch.full(
            (max_num_reqs,), 10, dtype=torch.int32, device=self.device
        )

        seq_lens = -torch.ones(max_num_reqs, dtype=torch.int32, device=self.device)
        pos = -torch.ones(num_tokens, dtype=torch.int64, device=self.device)

        _prepare_pos_seq_lens_kernel[(num_reqs + 1,)](
            pos,
            seq_lens,
            idx_mapping,
            query_start_loc,
            num_computed_tokens,
            max_num_reqs,
            BLOCK_SIZE=4,
        )
        torch.npu.synchronize()

        # Valid entries should have seq_len = 10 + 4 = 14
        assert torch.all(seq_lens[:num_reqs].cpu() == 14)
        # Padded entries should be 0
        assert torch.all(seq_lens[num_reqs:].cpu() == 0)

    def test_event_driven(self):
        """Zero query lengths preserve active seq_lens and still pad unused rows."""
        num_reqs = 2
        max_num_reqs = 8
        num_tokens = 0  # no tokens at all

        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=self.device)
        num_computed_tokens = torch.full(
            (max_num_reqs,), 10, dtype=torch.int32, device=self.device
        )

        seq_lens = -torch.ones(max_num_reqs, dtype=torch.int32, device=self.device)
        pos = torch.empty(0, dtype=torch.int64, device=self.device)

        _prepare_pos_seq_lens_kernel[(num_reqs + 1,)](
            pos,
            seq_lens,
            idx_mapping,
            query_start_loc,
            num_computed_tokens,
            max_num_reqs,
            BLOCK_SIZE=4,
        )
        torch.npu.synchronize()

        assert pos.numel() == 0
        # Active requests retain their already-computed sequence length.
        assert torch.all(seq_lens[:num_reqs].cpu() == 10)
        # Only unused CUDA graph rows are padding.
        assert torch.all(seq_lens[num_reqs:].cpu() == 0)
