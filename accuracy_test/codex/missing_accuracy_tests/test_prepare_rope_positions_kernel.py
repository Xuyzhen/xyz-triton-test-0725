# vLLM vanilla kernel: _prepare_rope_positions_kernel from
# vllm/vllm/v1/worker/gpu/mm/rope.py

"""
Precision test for _prepare_rope_positions_kernel.

Kernel signature:
    _prepare_rope_positions_kernel(
        positions_ptr,               # [num_dims, max_num_tokens] output positions
        positions_stride,            # stride(0) of positions
        prefill_positions_ptr,       # [max_num_reqs * num_dims, max_model_len] prefill table
        prefill_positions_stride0,   # stride(0) of prefill table  = num_dims * max_model_len
        prefill_positions_stride1,   # stride(1) of prefill table  = max_model_len
        prefill_delta_ptr,           # [max_num_reqs] delta per request
        idx_mapping_ptr,             # [num_reqs] batch_idx -> req_state_idx
        query_start_loc_ptr,         # [num_reqs + 1] query start locations
        prefill_lens_ptr,            # [max_num_reqs] prefill length per request
        num_computed_tokens_ptr,     # [max_num_reqs] tokens already computed
        BLOCK_SIZE: tl.constexpr,    # iteration block size
        NUM_DIMS: tl.constexpr,      # number of RoPE dims (3 for M-RoPE, 3/4 for XD-RoPE)
    )

Computes position IDs for multi-dimensional RoPE:
- Prefill phase: reads from prefill_positions table
- Decode phase: uses orig_pos + delta
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.mm.rope import _prepare_rope_positions_kernel
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

import pytest


def _prepare_rope_positions_ref(
    positions: torch.Tensor,
    positions_stride: int,
    prefill_positions: torch.Tensor,
    prefill_positions_stride0: int,
    prefill_positions_stride1: int,
    prefill_delta: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    prefill_lens: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    BLOCK_SIZE: int,
    NUM_DIMS: int,
):
    """CPU reference for _prepare_rope_positions_kernel."""
    # Triton pointer arithmetic is over flat storage. PyTorch's one-index
    # access on a 2-D tensor selects a row, so use flat views explicitly.
    positions_flat = positions.reshape(-1)
    prefill_positions_flat = prefill_positions.reshape(-1)
    num_reqs = idx_mapping.shape[0]
    for batch_idx in range(num_reqs):
        req_state_idx = idx_mapping[batch_idx].item()
        prefill_len = prefill_lens[req_state_idx].item()
        num_computed = num_computed_tokens[req_state_idx].item()
        is_prefill = num_computed < prefill_len

        query_start = query_start_loc[batch_idx].item()
        query_end = query_start_loc[batch_idx + 1].item()
        query_len = query_end - query_start

        delta = prefill_delta[req_state_idx].item()

        for i in range(0, query_len, BLOCK_SIZE):
            for j in range(min(BLOCK_SIZE, query_len - i)):
                block = i + j
                orig_pos = num_computed + block

                for k in range(NUM_DIMS):
                    if is_prefill:
                        pos = prefill_positions_flat[
                            req_state_idx * prefill_positions_stride0
                            + k * prefill_positions_stride1
                            + orig_pos
                        ].item()
                    else:
                        pos = orig_pos + delta
                    positions_flat[k * positions_stride + query_start + block] = pos


class TestPrepareRopePositionsKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_dims", [3, 4])
    @pytest.mark.parametrize("num_reqs", [1, 4, 8])
    def test_prefill(self, num_dims, num_reqs):
        """Test prefill phase: positions read from prefill_positions table."""
        max_model_len = 512
        max_num_tokens = 256
        BLOCK_SIZE = 1024

        positions = torch.zeros(num_dims, max_num_tokens + 1, dtype=torch.int64, device=self.device)
        prefill_positions = torch.randint(
            0, max_model_len, (num_reqs * num_dims, max_model_len),
            dtype=torch.int32, device=self.device,
        )
        prefill_delta = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32, device=self.device)
        prefill_lens = torch.full((num_reqs,), 20, dtype=torch.int32, device=self.device)
        num_computed_tokens = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        prefill_positions_stride0 = num_dims * max_model_len
        prefill_positions_stride1 = max_model_len

        _prepare_rope_positions_kernel[(num_reqs,)](
            positions,
            positions.stride(0),
            prefill_positions,
            prefill_positions_stride0,
            prefill_positions_stride1,
            prefill_delta,
            idx_mapping,
            query_start_loc,
            prefill_lens,
            num_computed_tokens,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_DIMS=num_dims,
        )
        torch.npu.synchronize()

        # CPU reference
        expected = torch.zeros(num_dims, max_num_tokens + 1, dtype=torch.int64, device=self.device)
        _prepare_rope_positions_ref(
            expected,
            expected.stride(0),
            prefill_positions,
            prefill_positions_stride0,
            prefill_positions_stride1,
            prefill_delta,
            idx_mapping,
            query_start_loc,
            prefill_lens,
            num_computed_tokens,
            BLOCK_SIZE,
            num_dims,
        )

        torch.testing.assert_close(
            positions.cpu(),
            expected.cpu(),
            rtol=0,
            atol=0,
        )

    @pytest.mark.parametrize("num_dims", [3, 4])
    @pytest.mark.parametrize("num_reqs", [1, 4])
    def test_decode(self, num_dims, num_reqs):
        """Test decode phase: positions = orig_pos + delta."""
        max_model_len = 512
        max_num_tokens = 256
        BLOCK_SIZE = 1024

        positions = torch.zeros(num_dims, max_num_tokens + 1, dtype=torch.int64, device=self.device)
        prefill_positions = torch.zeros(
            num_reqs * num_dims, max_model_len,
            dtype=torch.int32, device=self.device,
        )
        delta_vals = torch.randint(10, 100, (num_reqs,), dtype=torch.int32, device=self.device)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32, device=self.device)
        prefill_lens = torch.full((num_reqs,), 20, dtype=torch.int32, device=self.device)
        num_computed_tokens = torch.full((num_reqs,), 25, dtype=torch.int32, device=self.device)
        prefill_positions_stride0 = num_dims * max_model_len
        prefill_positions_stride1 = max_model_len

        _prepare_rope_positions_kernel[(num_reqs,)](
            positions,
            positions.stride(0),
            prefill_positions,
            prefill_positions_stride0,
            prefill_positions_stride1,
            delta_vals,
            idx_mapping,
            query_start_loc,
            prefill_lens,
            num_computed_tokens,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_DIMS=num_dims,
        )
        torch.npu.synchronize()

        expected = torch.zeros(num_dims, max_num_tokens + 1, dtype=torch.int64, device=self.device)
        _prepare_rope_positions_ref(
            expected,
            expected.stride(0),
            prefill_positions,
            prefill_positions_stride0,
            prefill_positions_stride1,
            delta_vals,
            idx_mapping,
            query_start_loc,
            prefill_lens,
            num_computed_tokens,
            BLOCK_SIZE,
            num_dims,
        )

        torch.testing.assert_close(
            positions.cpu(),
            expected.cpu(),
            rtol=0,
            atol=0,
        )

    def test_mixed_prefill_decode(self):
        """Some requests in prefill, some in decode."""
        num_dims = 3
        num_reqs = 4
        max_model_len = 512
        max_num_tokens = 256
        BLOCK_SIZE = 1024

        positions = torch.zeros(num_dims, max_num_tokens + 1, dtype=torch.int64, device=self.device)
        prefill_positions = torch.randint(
            0, max_model_len, (num_reqs * num_dims, max_model_len),
            dtype=torch.int32, device=self.device,
        )
        delta_vals = torch.randint(1, 50, (num_reqs,), dtype=torch.int32, device=self.device)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32, device=self.device)
        # Requests 0,2 in prefill; 1,3 in decode
        prefill_lens = torch.tensor([10, 10, 15, 15], dtype=torch.int32, device=self.device)
        num_computed_tokens = torch.tensor([0, 12, 5, 20], dtype=torch.int32, device=self.device)
        prefill_positions_stride0 = num_dims * max_model_len
        prefill_positions_stride1 = max_model_len

        _prepare_rope_positions_kernel[(num_reqs,)](
            positions,
            positions.stride(0),
            prefill_positions,
            prefill_positions_stride0,
            prefill_positions_stride1,
            delta_vals,
            idx_mapping,
            query_start_loc,
            prefill_lens,
            num_computed_tokens,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_DIMS=num_dims,
        )
        torch.npu.synchronize()

        expected = torch.zeros(num_dims, max_num_tokens + 1, dtype=torch.int64, device=self.device)
        _prepare_rope_positions_ref(
            expected,
            expected.stride(0),
            prefill_positions,
            prefill_positions_stride0,
            prefill_positions_stride1,
            delta_vals,
            idx_mapping,
            query_start_loc,
            prefill_lens,
            num_computed_tokens,
            BLOCK_SIZE,
            num_dims,
        )

        torch.testing.assert_close(
            positions.cpu(),
            expected.cpu(),
            rtol=0,
            atol=0,
        )
