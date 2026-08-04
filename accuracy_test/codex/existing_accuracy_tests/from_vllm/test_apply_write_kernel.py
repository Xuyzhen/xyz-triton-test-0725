# Standalone Ascend A3 adaptation of an upstream vLLM accuracy path.
# Accuracy UT source: vllm/tests/v1/worker/test_gpu_block_table.py
# Kernel source: vllm/vllm/v1/worker/gpu/buffer_utils.py
# Coverage: _apply_write_kernel

# vLLM vanilla kernel: _apply_write_kernel from
# vllm/vllm/v1/worker/gpu/buffer_utils.py

"""
Precision test for _apply_write_kernel.

Applies staged writes to a GPU buffer (or via ptr-to-ptrs indirection for
multi-group fused writes).

Kernel signature:
    _apply_write_kernel(
        output_ptr,            # MULTI_GROUP: ptr-to-ptrs [num_groups]; else data ptr
        output_stride,         # MULTI_GROUP: ptr-to-strides [num_groups]; else row stride
        write_indices_ptr,
        write_starts_ptr,
        write_contents_ptr,
        write_cu_lens_ptr,
        write_group_ids_ptr,   # [num_writes], used only when MULTI_GROUP
        BLOCK_SIZE: tl.constexpr,
        MULTI_GROUP: tl.constexpr,
    )
"""

import torch
import pytest

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import _apply_write_kernel
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


_KERNEL_ARG_NAMES = set(_apply_write_kernel.arg_names)
_HAS_MULTI_GROUP = {
    "write_group_ids_ptr",
    "MULTI_GROUP",
}.issubset(_KERNEL_ARG_NAMES)


def _launch_single_group(
    grid,
    output,
    write_indices,
    write_starts,
    contents,
    cu_lens,
):
    """Launch either the pre-fusion or fused-capable vLLM kernel signature."""
    args = [
        output,
        output.stride(0),
        write_indices,
        write_starts,
        contents,
        cu_lens,
    ]
    kwargs = {"BLOCK_SIZE": 4}
    if _HAS_MULTI_GROUP:
        args.append(None)
        kwargs["MULTI_GROUP"] = False
    _apply_write_kernel[grid](*args, **kwargs)


def _apply_write_ref(
    output,                # 2D tensor [num_rows, row_stride]
    output_stride,         # stride(0) of output
    write_indices,         # [num_writes] row indices
    write_starts,          # [num_writes] column start
    write_contents,        # flat tensor of all values to write
    write_cu_lens,         # [num_writes] cumulative end indices into write_contents
    multi_group=False,
    output_ptrs=None,      # [num_groups] for multi_group mode
    output_strides=None,   # [num_groups]
    group_ids=None,        # [num_writes]
):
    """CPU reference for apply write."""
    n_writes = len(write_indices)

    if multi_group:
        out_refs = output
    else:
        out_refs = [output]

    for pid in range(n_writes):
        if multi_group:
            gid = int(group_ids[pid].item())
            cur_out = out_refs[gid]
        else:
            cur_out = out_refs[0]

        row = int(write_indices[pid].item())
        start = int(write_starts[pid].item())
        cu_start = int(write_cu_lens[pid - 1].item()) if pid > 0 else 0
        cu_end = int(write_cu_lens[pid].item())
        content_len = cu_end - cu_start

        content_slice = write_contents[cu_start:cu_end]
        cur_out[row, start:start + content_len] = content_slice

    return output


class TestApplyWriteKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_rows", [1, 4])
    @pytest.mark.parametrize("num_cols", [16, 32])
    def test_single_group(self, num_rows, num_cols):
        """Test basic single-group write."""
        num_writes = num_rows

        output = torch.zeros(num_rows, num_cols, dtype=torch.int32, device=self.device)

        write_indices = torch.arange(
            num_writes, dtype=torch.int32, device=self.device
        )
        write_starts = torch.zeros(
            num_writes, dtype=torch.int32, device=self.device
        )
        contents = torch.arange(
            num_writes * num_cols, dtype=torch.int32, device=self.device
        )
        cu_lens = (
            torch.arange(
                1, num_writes + 1, dtype=torch.int32, device=self.device
            )
            * num_cols
        )

        _launch_single_group(
            (num_writes,),
            output,
            write_indices,
            write_starts,
            contents,
            cu_lens,
        )
        torch.npu.synchronize()

        ref = _apply_write_ref(
            torch.zeros(num_rows, num_cols, dtype=torch.int32),
            output.stride(0),
            write_indices.cpu(),
            write_starts.cpu(),
            contents.cpu(),
            cu_lens.cpu(),
        )
        torch.testing.assert_close(output.cpu(), ref, rtol=0, atol=0)

    @pytest.mark.parametrize("num_groups", [1, 2, 4])
    @pytest.mark.parametrize("num_writes_per_group", [1, 2])
    def test_multi_group(self, num_groups, num_writes_per_group):
        """Test multi-group fused write."""
        if not _HAS_MULTI_GROUP:
            pytest.skip(
                "installed vLLM predates fused multi-group _apply_write_kernel; "
                "precision was not tested"
            )

        num_rows = num_writes_per_group
        num_cols = 16

        outputs = [
            torch.zeros(num_rows, num_cols, dtype=torch.int32, device=self.device)
            for _ in range(num_groups)
        ]
        output_ptrs = torch.tensor(
            [t.data_ptr() for t in outputs],
            dtype=torch.uint64, device=self.device
        )
        output_strides = torch.full(
            (num_groups,), num_cols, dtype=torch.int64, device=self.device
        )

        total_writes = num_groups * num_writes_per_group
        group_ids = torch.repeat_interleave(
            torch.arange(num_groups, dtype=torch.int32, device=self.device),
            num_writes_per_group,
        )
        write_indices = torch.arange(
            num_writes_per_group, dtype=torch.int32, device=self.device
        ).repeat(num_groups)
        write_starts = torch.zeros(
            total_writes, dtype=torch.int32, device=self.device
        )

        # Each write gets a unique value
        contents_list = []
        for g in range(num_groups):
            for w in range(num_writes_per_group):
                val = g * 100 + w
                contents_list.extend([val] * num_cols)

        contents = torch.tensor(contents_list, dtype=torch.int32, device=self.device)
        cu_lens = (
            torch.arange(
                1, total_writes + 1, dtype=torch.int32, device=self.device
            )
            * num_cols
        )

        _apply_write_kernel[(total_writes,)](
            output_ptrs,
            output_strides,
            write_indices,
            write_starts,
            contents,
            cu_lens,
            group_ids,
            BLOCK_SIZE=4,
            MULTI_GROUP=True,
        )
        torch.npu.synchronize()

        for g in range(num_groups):
            expected = torch.full((num_rows, num_cols), 100 * g, dtype=torch.int32)
            # Adjust for second write overwriting
            if num_writes_per_group > 1:
                expected[1, :] = 100 * g + 1
            torch.testing.assert_close(outputs[g].cpu(), expected, rtol=0, atol=0)
