import pytest
import torch

from vllm.v1.worker.gpu.cp_utils import _dcp_local_seq_lens_kernel


def _dcp_local_seq_lens_cpu(
    seq_lens: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    cp_interleave: int,
    num_reqs: int,
    max_num_reqs: int,
):
    out = torch.zeros(max_num_reqs, dtype=torch.int32)
    for i in range(num_reqs):
        seq_len = int(seq_lens[i])
        rounds = seq_len // (dcp_size * cp_interleave)
        remainder = seq_len % (dcp_size * cp_interleave)
        remainder = max(remainder - dcp_rank * cp_interleave, 0)
        remainder = min(remainder, cp_interleave)
        local_seq_len = rounds * cp_interleave + remainder
        out[i] = local_seq_len
    for i in range(num_reqs, max_num_reqs):
        out[i] = 0
    return out


@pytest.mark.parametrize(
    "dcp_size,dcp_rank,cp_interleave",
    [(2, 0, 1), (2, 1, 1), (4, 2, 1)],
)
def test_dcp_local_seq_lens_kernel(dcp_size, dcp_rank, cp_interleave):
    torch.manual_seed(42)
    num_reqs = 4
    max_num_reqs = 6

    seq_lens = torch.tensor([10, 20, 5, 33], dtype=torch.int32)

    expected = _dcp_local_seq_lens_cpu(
        seq_lens, dcp_size, dcp_rank, cp_interleave, num_reqs, max_num_reqs
    )

    device = torch.device("npu")
    dcp_local_seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

    BLOCK_SIZE = 128
    num_blocks = (max_num_reqs + BLOCK_SIZE - 1) // BLOCK_SIZE

    _dcp_local_seq_lens_kernel[(num_blocks,)](
        dcp_local_seq_lens,
        seq_lens.to(device),
        dcp_size,
        dcp_rank,
        cp_interleave,
        num_reqs,
        max_num_reqs,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(dcp_local_seq_lens.cpu(), expected, rtol=0, atol=0)


def test_dcp_local_seq_lens_kernel_with_interleave():
    dcp_size = 2
    dcp_rank = 0
    cp_interleave = 4
    num_reqs = 3
    max_num_reqs = 5

    seq_lens = torch.tensor([20, 16, 8], dtype=torch.int32)

    expected = _dcp_local_seq_lens_cpu(
        seq_lens, dcp_size, dcp_rank, cp_interleave, num_reqs, max_num_reqs
    )

    device = torch.device("npu")
    dcp_local_seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

    BLOCK_SIZE = 128
    num_blocks = (max_num_reqs + BLOCK_SIZE - 1) // BLOCK_SIZE

    _dcp_local_seq_lens_kernel[(num_blocks,)](
        dcp_local_seq_lens,
        seq_lens.to(device),
        dcp_size,
        dcp_rank,
        cp_interleave,
        num_reqs,
        max_num_reqs,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(dcp_local_seq_lens.cpu(), expected, rtol=0, atol=0)
