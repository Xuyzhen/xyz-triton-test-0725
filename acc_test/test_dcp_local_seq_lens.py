# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.gpu.cp_utils import _dcp_local_seq_lens_kernel

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _dcp_local_seq_lens_cpu(
    seq_lens: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    cp_interleave: int,
    max_num_reqs: int,
) -> torch.Tensor:
    """Pure PyTorch CPU reference implementation of DCP local seq_lens.

    Distributes KV cache among different ranks in a round-robin manner:
      rounds = seq_len // (dcp_size * cp_interleave)
      remainder = seq_len % (dcp_size * cp_interleave)
      remainder = max(remainder - dcp_rank * cp_interleave, 0)
      remainder = min(remainder, cp_interleave)
      local_seq_len = rounds * cp_interleave + remainder

    Positions beyond num_reqs are padded to 0.
    """
    out = torch.zeros(max_num_reqs, dtype=torch.int32)
    num_reqs = len(seq_lens)

    for i in range(max_num_reqs):
        if i < num_reqs:
            sl = int(seq_lens[i])
            rounds = sl // (dcp_size * cp_interleave)
            remainder = sl % (dcp_size * cp_interleave)
            remainder = max(remainder - dcp_rank * cp_interleave, 0)
            remainder = min(remainder, cp_interleave)
            out[i] = rounds * cp_interleave + remainder
        else:
            out[i] = 0

    return out


@pytest.mark.parametrize("dcp_rank", [0, 1, 2, 3])
def test_dcp_local_seq_lens_basic(dcp_rank: int) -> None:
    """DCP local seq_lens: basic correctness for different ranks.

    Verifies that the kernel correctly distributes KV cache across
    context parallelism ranks.
    """
    init_device_properties_triton()

    dcp_size = 4
    cp_interleave = 2
    num_reqs = 3
    max_num_reqs = 8
    BLOCK_SIZE = 128

    device = torch.device("npu")

    seq_lens = torch.tensor([10, 50, 100], dtype=torch.int32, device=device)
    out = torch.empty(max_num_reqs, dtype=torch.int32, device=device)

    num_blocks = (max_num_reqs + BLOCK_SIZE - 1) // BLOCK_SIZE
    _dcp_local_seq_lens_kernel[(num_blocks,)](
        out,
        seq_lens,
        dcp_size,
        dcp_rank,
        cp_interleave,
        num_reqs,
        max_num_reqs,
        BLOCK_SIZE,
    )
    torch.npu.synchronize()

    expected = _dcp_local_seq_lens_cpu(
        seq_lens.cpu(), dcp_size, dcp_rank, cp_interleave, max_num_reqs
    )

    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_dcp_local_seq_lens_single_rank() -> None:
    """DCP local seq_lens: single rank (no context parallelism).

    With dcp_size=1 and dcp_rank=0, local_seq_lens should equal seq_lens.
    """
    init_device_properties_triton()

    dcp_size = 1
    dcp_rank = 0
    cp_interleave = 1
    num_reqs = 4
    max_num_reqs = 6
    BLOCK_SIZE = 128

    device = torch.device("npu")

    seq_lens = torch.tensor([1, 5, 10, 100], dtype=torch.int32, device=device)
    out = torch.empty(max_num_reqs, dtype=torch.int32, device=device)

    num_blocks = (max_num_reqs + BLOCK_SIZE - 1) // BLOCK_SIZE
    _dcp_local_seq_lens_kernel[(num_blocks,)](
        out,
        seq_lens,
        dcp_size,
        dcp_rank,
        cp_interleave,
        num_reqs,
        max_num_reqs,
        BLOCK_SIZE,
    )
    torch.npu.synchronize()

    expected = _dcp_local_seq_lens_cpu(
        seq_lens.cpu(), dcp_size, dcp_rank, cp_interleave, max_num_reqs
    )

    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_dcp_local_seq_lens_large_interleave() -> None:
    """DCP local seq_lens: larger cp_interleave value.

    Verifies round-robin distribution with larger interleave.
    """
    init_device_properties_triton()

    dcp_size = 2
    dcp_rank = 1
    cp_interleave = 8
    num_reqs = 3
    max_num_reqs = 5
    BLOCK_SIZE = 128

    device = torch.device("npu")

    seq_lens = torch.tensor([16, 32, 64], dtype=torch.int32, device=device)
    out = torch.empty(max_num_reqs, dtype=torch.int32, device=device)

    num_blocks = (max_num_reqs + BLOCK_SIZE - 1) // BLOCK_SIZE
    _dcp_local_seq_lens_kernel[(num_blocks,)](
        out,
        seq_lens,
        dcp_size,
        dcp_rank,
        cp_interleave,
        num_reqs,
        max_num_reqs,
        BLOCK_SIZE,
    )
    torch.npu.synchronize()

    expected = _dcp_local_seq_lens_cpu(
        seq_lens.cpu(), dcp_size, dcp_rank, cp_interleave, max_num_reqs
    )

    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_dcp_local_seq_lens_all_padding() -> None:
    """DCP local seq_lens: all entries are padding (num_reqs=0).

    All output values should be 0.
    """
    init_device_properties_triton()

    dcp_size = 4
    dcp_rank = 0
    cp_interleave = 2
    num_reqs = 0
    max_num_reqs = 4
    BLOCK_SIZE = 128

    device = torch.device("npu")

    seq_lens = torch.zeros(0, dtype=torch.int32, device=device)
    out = torch.full((max_num_reqs,), -1, dtype=torch.int32, device=device)

    num_blocks = (max_num_reqs + BLOCK_SIZE - 1) // BLOCK_SIZE
    _dcp_local_seq_lens_kernel[(num_blocks,)](
        out,
        seq_lens,
        dcp_size,
        dcp_rank,
        cp_interleave,
        num_reqs,
        max_num_reqs,
        BLOCK_SIZE,
    )
    torch.npu.synchronize()

    assert (out.cpu() == 0).all(), "All values should be 0 when num_reqs == 0"
