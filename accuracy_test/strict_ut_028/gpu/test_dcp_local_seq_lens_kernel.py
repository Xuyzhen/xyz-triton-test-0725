# GENERATED STRICT UT. Source: accuracy_test/codex/missing_accuracy_tests/test_dcp_local_seq_lens_kernel.py
# Do not edit mechanically; update the reviewed Codex source or strict generator.
from accuracy_test.strict_ut.runtime_gpu import STRICT_DEVICE as _STRICT_DEVICE
# vLLM vanilla kernel: _dcp_local_seq_lens_kernel from
# vllm/vllm/v1/worker/gpu/cp_utils.py

"""
Precision test for _dcp_local_seq_lens_kernel.

Computes per-request local sequence lengths for context parallelism by
distributing the KV cache across ranks in a round-robin fashion.

Kernel signature:
    _dcp_local_seq_lens_kernel(
        out_ptr,        # [max_num_reqs] int32 output
        seq_lens_ptr,   # [max_num_reqs] int32 input
        dcp_size,
        dcp_rank,
        cp_interleave,
        num_reqs,
        max_num_reqs,
        BLOCK_SIZE: tl.constexpr,
    )
"""

import torch
import pytest

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.cp_utils import _dcp_local_seq_lens_kernel
from accuracy_test.strict_ut.runtime_gpu import init_device_properties_triton


def _dcp_local_seq_lens_ref(
    seq_lens,       # [max_num_reqs]
    dcp_size,
    dcp_rank,
    cp_interleave,
    num_reqs,
    max_num_reqs,
):
    """CPU reference for DCP local seq_lens computation."""
    out = torch.zeros(max_num_reqs, dtype=torch.int32)
    for i in range(num_reqs):
        s = int(seq_lens[i].item())
        rounds = s // (dcp_size * cp_interleave)
        remainder = s % (dcp_size * cp_interleave)
        remainder = max(remainder - dcp_rank * cp_interleave, 0)
        remainder = min(remainder, cp_interleave)
        local_len = rounds * cp_interleave + remainder
        out[i] = local_len
    # Padded entries (num_reqs..max_num_reqs) already 0
    return out


class TestDcpLocalSeqLensKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("cuda")

    @pytest.mark.parametrize("num_reqs", [2, 4, 8])
    @pytest.mark.parametrize("max_num_reqs", [8, 16])
    @pytest.mark.parametrize("dcp_size", [2, 4])
    @pytest.mark.parametrize("dcp_rank", [0, 1])
    @pytest.mark.parametrize("cp_interleave", [1, 2])
    def test_dcp_local_seq_lens(self, num_reqs, max_num_reqs, dcp_size, dcp_rank, cp_interleave):
        """Test DCP local seq_lens with various configurations."""
        if dcp_rank >= dcp_size:
            pytest.skip("dcp_rank must be < dcp_size")

        seq_lens = torch.randint(
            1, 128, (max_num_reqs,), dtype=torch.int32, device=self.device
        )
        # Make first num_reqs valid, rest are padding
        out = torch.zeros(max_num_reqs, dtype=torch.int32, device=self.device)
        BLOCK_SIZE = 32

        _dcp_local_seq_lens_kernel[(triton.cdiv(max_num_reqs, BLOCK_SIZE),)](
            out,
            seq_lens,
            dcp_size,
            dcp_rank,
            cp_interleave,
            num_reqs,
            max_num_reqs,
            BLOCK_SIZE,
        )
        torch.cuda.synchronize()

        ref = _dcp_local_seq_lens_ref(
            seq_lens.cpu(), dcp_size, dcp_rank, cp_interleave, num_reqs, max_num_reqs
        )
        torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)

    def test_dcp_rank_highest(self):
        """Test with the highest rank (dcp_rank = dcp_size - 1)."""
        num_reqs = 4
        max_num_reqs = 8
        dcp_size = 4
        dcp_rank = 3
        cp_interleave = 1

        seq_lens = torch.tensor(
            [10, 15, 20, 25, 0, 0, 0, 0], dtype=torch.int32, device=self.device
        )
        out = torch.zeros(max_num_reqs, dtype=torch.int32, device=self.device)
        BLOCK_SIZE = 16

        _dcp_local_seq_lens_kernel[(1,)](
            out,
            seq_lens,
            dcp_size,
            dcp_rank,
            cp_interleave,
            num_reqs,
            max_num_reqs,
            BLOCK_SIZE,
        )
        torch.cuda.synchronize()

        ref = _dcp_local_seq_lens_ref(
            seq_lens.cpu(), dcp_size, dcp_rank, cp_interleave, num_reqs, max_num_reqs
        )
        torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)

    def test_zero_seq_lens(self):
        """Test with zero sequence lengths."""
        num_reqs = 2
        max_num_reqs = 4

        seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=self.device)
        out = -torch.ones(max_num_reqs, dtype=torch.int32, device=self.device)

        _dcp_local_seq_lens_kernel[(1,)](
            out,
            seq_lens,
            dcp_size=2,
            dcp_rank=0,
            cp_interleave=1,
            num_reqs=num_reqs,
            max_num_reqs=max_num_reqs,
            BLOCK_SIZE=16,
        )
        torch.cuda.synchronize()

        # For zero seq_lens, local len is 0
        assert torch.all(out[:num_reqs].cpu() == 0)
        # Padded entries should be 0
        assert torch.all(out[num_reqs:].cpu() == 0)
