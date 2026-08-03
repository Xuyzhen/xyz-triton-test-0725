import pytest
import torch

from vllm.v1.worker.gpu.buffer_utils import _apply_write_kernel


def _apply_write_cpu(
    output: torch.Tensor,
    write_indices: list[int],
    write_starts: list[int],
    write_contents: list[int],
    write_cu_lens: list[int],
    row_stride: int,
):
    for pid in range(len(write_indices)):
        row_idx = write_indices[pid]
        start_idx = write_starts[pid]

        cu_start = write_cu_lens[pid - 1] if pid > 0 else 0
        cu_end = write_cu_lens[pid]
        content_len = cu_end - cu_start

        for i in range(content_len):
            output[row_idx * row_stride + start_idx + i] = write_contents[cu_start + i]


def test_apply_write_kernel_single_group():
    torch.manual_seed(42)
    num_rows = 4
    row_len = 16

    output = torch.zeros(num_rows * row_len, dtype=torch.int32)

    write_indices = [0, 2, 1]
    write_starts = [3, 0, 5]
    write_contents = [100, 200, 300, 400, 500, 600]
    write_cu_lens = [2, 4, 6]

    expected = output.clone()
    _apply_write_cpu(
        expected,
        write_indices,
        write_starts,
        write_contents,
        write_cu_lens,
        row_len,
    )

    device = torch.device("npu")
    output_npu = output.to(device)

    output_2d = output_npu.reshape(num_rows, row_len)

    _apply_write_kernel[(3,)](
        output_2d,
        row_len,
        torch.tensor(write_indices, dtype=torch.int32, device=device),
        torch.tensor(write_starts, dtype=torch.int32, device=device),
        torch.tensor(write_contents, dtype=torch.int32, device=device),
        torch.tensor(write_cu_lens, dtype=torch.int32, device=device),
        None,
        BLOCK_SIZE=1024,
        MULTI_GROUP=False,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        output_npu.cpu(), expected, rtol=0, atol=0
    )


def test_apply_write_kernel_overwrite():
    num_rows = 2
    row_len = 8

    output = torch.zeros(num_rows * row_len, dtype=torch.int32)
    output[0] = 99

    write_indices = [0]
    write_starts = [0]
    write_contents = [11, 22]
    write_cu_lens = [2]

    expected = output.clone()
    _apply_write_cpu(
        expected,
        write_indices,
        write_starts,
        write_contents,
        write_cu_lens,
        row_len,
    )

    device = torch.device("npu")
    output_npu = output.to(device)
    output_2d = output_npu.reshape(num_rows, row_len)

    _apply_write_kernel[(1,)](
        output_2d,
        row_len,
        torch.tensor(write_indices, dtype=torch.int32, device=device),
        torch.tensor(write_starts, dtype=torch.int32, device=device),
        torch.tensor(write_contents, dtype=torch.int32, device=device),
        torch.tensor(write_cu_lens, dtype=torch.int32, device=device),
        None,
        BLOCK_SIZE=1024,
        MULTI_GROUP=False,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(output_npu.cpu(), expected, rtol=0, atol=0)
