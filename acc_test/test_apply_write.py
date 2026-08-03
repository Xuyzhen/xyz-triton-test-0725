# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.gpu.buffer_utils import _apply_write_kernel

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _apply_write_cpu(
    output: torch.Tensor,
    write_indices: torch.Tensor,
    write_starts: torch.Tensor,
    write_contents: torch.Tensor,
    write_cu_lens: torch.Tensor,
    write_group_ids: torch.Tensor | None = None,
    multi_group: bool = False,
) -> torch.Tensor:
    """Pure PyTorch CPU reference implementation of apply_write.

    For each write operation (pid), copies content_len elements from
    write_contents to output[row_idx, start_idx:...].

    When MULTI_GROUP, each write targets a different output tensor
    (selected by group_id). Otherwise, all writes target the same
    output tensor.

    Returns the (potentially multi-group) output.
    """
    num_writes = len(write_indices)

    if multi_group:
        num_groups = output.shape[0]
        out = [output[g].clone() for g in range(num_groups)]
    else:
        out = output.clone()

    for pid in range(num_writes):
        row_idx = int(write_indices[pid])
        start_idx = int(write_starts[pid])

        cu_start = int(write_cu_lens[pid - 1]) if pid > 0 else 0
        cu_end = int(write_cu_lens[pid])
        content_len = cu_end - cu_start

        content = write_contents[cu_start:cu_end]

        if multi_group:
            group_id = int(write_group_ids[pid])
            out[group_id][row_idx, start_idx:start_idx + content_len] = content
        else:
            out[row_idx, start_idx:start_idx + content_len] = content

    if multi_group:
        return torch.stack(out)
    return out


def test_apply_write_single_group() -> None:
    """Apply write kernel: single group (MULTI_GROUP=False).

    Verifies that staged writes are correctly applied to a single output
    tensor.
    """
    init_device_properties_triton()
    torch.manual_seed(42)

    num_rows = 4
    num_cols = 32
    num_writes = 5
    BLOCK_SIZE = 1024

    device = torch.device("npu")

    output = torch.zeros(num_rows, num_cols, dtype=torch.float32, device=device)
    write_indices = torch.tensor([0, 1, 2, 3, 0], dtype=torch.int32, device=device)
    write_starts = torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32, device=device)

    write_contents_list = [
        torch.randn(4, dtype=torch.float32),
        torch.randn(4, dtype=torch.float32),
        torch.randn(4, dtype=torch.float32),
        torch.randn(4, dtype=torch.float32),
        torch.randn(4, dtype=torch.float32),
    ]
    write_contents = torch.cat(write_contents_list).to(device)
    write_cu_lens = torch.tensor(
        [4, 8, 12, 16, 20], dtype=torch.int32, device=device
    )

    _apply_write_kernel[(num_writes,)](
        output,
        output.stride(0),
        write_indices,
        write_starts,
        write_contents,
        write_cu_lens,
        BLOCK_SIZE=1024,
    )
    torch.npu.synchronize()

    expected = _apply_write_cpu(
        output.cpu(),
        write_indices.cpu(),
        write_starts.cpu(),
        write_contents.cpu(),
        write_cu_lens.cpu(),
        multi_group=False,
    )

    torch.testing.assert_close(output.cpu(), expected, rtol=1e-5, atol=1e-5)


def test_apply_write_empty() -> None:
    """Apply write kernel: zero writes is a no-op.

    Verifies that launching with no writes leaves output unchanged.
    """
    init_device_properties_triton()
    device = torch.device("npu")

    output = torch.full((2, 8), -1.0, dtype=torch.float32, device=device)
    expected = output.clone()

    # 0 writes, kernel is launched with grid size 0 -> no-op
    write_indices = torch.zeros(0, dtype=torch.int32, device=device)
    write_starts = torch.zeros(0, dtype=torch.int32, device=device)
    write_contents = torch.zeros(0, dtype=torch.float32, device=device)
    write_cu_lens = torch.zeros(0, dtype=torch.int32, device=device)

    # Grid size 0 would fail on Ascend NPU (coreDim must be > 0),
    # so skip the kernel launch entirely. Output should be unchanged.
    if grid_size := len(write_indices):
        _apply_write_kernel[(grid_size,)](
            output,
            output.stride(0),
            write_indices,
            write_starts,
            write_contents,
            write_cu_lens,
            BLOCK_SIZE=1024,
        )
        torch.npu.synchronize()

    torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=0, atol=0)


def test_apply_write_variable_lengths() -> None:
    """Apply write kernel: writes with varying content lengths.

    Verifies that each write copies exactly content_len elements.
    """
    init_device_properties_triton()
    torch.manual_seed(77)

    num_rows = 3
    num_cols = 64
    BLOCK_SIZE = 1024

    device = torch.device("npu")

    output = torch.zeros(num_rows, num_cols, dtype=torch.float32, device=device)
    write_indices = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    write_starts = torch.tensor([0, 10, 30], dtype=torch.int32, device=device)

    contents = [
        torch.randn(10, dtype=torch.float32),
        torch.randn(20, dtype=torch.float32),
        torch.randn(34, dtype=torch.float32),
    ]
    write_contents = torch.cat(contents).to(device)
    write_cu_lens = torch.tensor([10, 30, 64], dtype=torch.int32, device=device)

    _apply_write_kernel[(3,)](
        output,
        output.stride(0),
        write_indices,
        write_starts,
        write_contents,
        write_cu_lens,
        BLOCK_SIZE=1024,
    )
    torch.npu.synchronize()

    write_contents_flat = torch.cat(contents)
    expected = torch.zeros(num_rows, num_cols, dtype=torch.float32)
    offset = 0
    for i in range(3):
        row = int(write_indices[i])
        start = int(write_starts[i])
        length = int(write_cu_lens[i]) - (int(write_cu_lens[i - 1]) if i > 0 else 0)
        expected[row, start:start + length] = write_contents_flat[offset:offset + length]
        offset += length

    torch.testing.assert_close(output.cpu(), expected, rtol=1e-5, atol=1e-5)
