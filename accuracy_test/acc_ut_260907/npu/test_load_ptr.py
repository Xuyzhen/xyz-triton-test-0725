# Standalone Ascend A3 adaptation of an upstream vLLM accuracy path.
# Accuracy UT source: vllm/tests/v1/worker/test_gpu_block_table.py
# Kernel source: vllm/vllm/v1/worker/gpu/buffer_utils.py
# Coverage: _load_ptr

# vLLM vanilla kernel: _load_ptr from
# vllm/vllm/v1/worker/gpu/buffer_utils.py

"""
Precision test for _load_ptr helper function.

_load_ptr loads a pointer from a pointer-to-pointer tensor and casts it
to the requested element type.  It is a pure Triton-JIT helper (not a
grid-launched kernel), so this test validates its inline behavior by
calling it inside a minimal wrapper kernel.

Helper signature:
    @triton.jit
    def _load_ptr(ptr_to_ptr, elem_dtype):
        ptr = tl.load(ptr_to_ptr)
        ptr = tl.cast(ptr, tl.pointer_type(elem_dtype))
        return tl.multiple_of(ptr, 16)
"""

import torch
import pytest

from vllm.triton_utils import tl, triton
try:
    from vllm.v1.worker.gpu.buffer_utils import _load_ptr
except ImportError as exc:
    pytest.skip(
        f"installed vLLM does not provide _load_ptr; precision was not tested: {exc}",
        allow_module_level=True,
    )
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@triton.jit
def _load_ptr_wrapper_kernel(
    ptr_to_ptr,
    data_ptr,
    output_ptr,
    elem_size: tl.constexpr,
):
    """Minimal wrapper: loads a value through _load_ptr and stores it."""
    loaded_ptr = _load_ptr(ptr_to_ptr, tl.int32)
    val = tl.load(loaded_ptr)
    tl.store(output_ptr, val)


class TestLoadPtr:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    def test_load_ptr_int32(self):
        """Test _load_ptr with int32 data."""
        data = torch.tensor([42, 99, 123], dtype=torch.int32, device=self.device)
        ptr_to_ptr = torch.tensor([data.data_ptr()], dtype=torch.uint64, device=self.device)
        output = torch.zeros(1, dtype=torch.int32, device=self.device)

        _load_ptr_wrapper_kernel[(1,)](
            ptr_to_ptr,
            data.data_ptr(),
            output,
            elem_size=4,
        )
        torch.npu.synchronize()

        assert output.cpu().item() == 42

    def test_load_ptr_float32(self):
        """Test _load_ptr with float32 data, using int32 ptr loading."""

        @triton.jit
        def _load_float_ptr_kernel(
            ptr_to_ptr,
            output_ptr,
        ):
            loaded_ptr = _load_ptr(ptr_to_ptr, tl.float32)
            val = tl.load(loaded_ptr)
            tl.store(output_ptr, val)

        data = torch.tensor([3.14159], dtype=torch.float32, device=self.device)
        ptr_to_ptr = torch.tensor([data.data_ptr()], dtype=torch.uint64, device=self.device)
        output = torch.zeros(1, dtype=torch.float32, device=self.device)

        _load_float_ptr_kernel[(1,)](
            ptr_to_ptr,
            output,
        )
        torch.npu.synchronize()

        torch.testing.assert_close(output.cpu(), torch.tensor([3.14159]), rtol=1e-5, atol=1e-5)

    def test_load_ptr_multiple_values(self):
        """Test that _load_ptr correctly yields the base address for strided access."""
        data = torch.arange(10, dtype=torch.int32, device=self.device)
        ptr_to_ptr = torch.tensor([data.data_ptr()], dtype=torch.uint64, device=self.device)

        @triton.jit
        def _load_and_store_multi_kernel(
            ptr_to_ptr,
            output_ptr,
        ):
            base = _load_ptr(ptr_to_ptr, tl.int32)
            for i in range(10):
                val = tl.load(base + i)
                tl.store(output_ptr + i, val)

        output = torch.zeros(10, dtype=torch.int32, device=self.device)

        _load_and_store_multi_kernel[(1,)](
            ptr_to_ptr,
            output,
        )
        torch.npu.synchronize()

        torch.testing.assert_close(output.cpu(), torch.arange(10, dtype=torch.int32), rtol=0, atol=0)
