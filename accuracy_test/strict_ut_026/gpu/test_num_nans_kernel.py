# SPDX-License-Identifier: Apache-2.0
# Replaces the generated strict UT for _num_nans_kernel in strict_ut_026/gpu.
# Provenance: accuracy_test/nun_nans_npu_gpu/test_num_nans_kernel_precision_gpu.py
# (GPU counterpart of the hand-written extended-precision NPU spec; identical
# shapes, dtypes, layouts, seeds and data generation for GPU-NPU comparison).
# Kernel source: inlined standalone CUDA copy of
# vllm/vllm/v1/worker/gpu/metrics/logits.py::_num_nans_kernel.
"""
Extended precision test for the standalone _num_nans_kernel on CUDA.

The replaced generated file swept only num_reqs x vocab_size x frac_nan for
FLOAT32 inputs with NaNs injected as a contiguous per-row prefix. This
version adds the dimensions that sweep does not cover:

1. dtype dimension (fp32 / bf16 / fp16): the kernel upcasts every loaded
   block via ``.to(tl.float32)`` before ``libdevice.isnan``. Production
   logits are frequently bf16/fp16, so NaN bit patterns must survive the
   low-precision -> fp32 conversion (no NaN flushing, no misclassification,
   no Inf created by overflow rounding being misread as NaN).
2. Scattered NaN layout: NaNs are placed at pseudo-random positions per row
   (a different position set per row) instead of a contiguous prefix, so a
   positional bias in the kernel or in the masked tail load cannot mask a
   misclassification.
3. Special-value interference: +Inf / -Inf / +0.0 / -0.0 / max-finite /
   min-normal / subnormal values tiled across the row must NOT be counted,
   guarding against a ``isnan`` that degenerates into an infinity or
   exponent-class check.

Kernel signature (unchanged):
    _num_nans_kernel(
        logits_ptr,               # logits [num_reqs, vocab_size]
        logits_stride,            # stride(0) of logits
        num_nans_ptr,             # int32 output [num_reqs]
        vocab_size,               # vocab size
        BLOCK_SIZE: tl.constexpr, # block size for iteration
    )

All assertions remain bitwise (rtol=0, atol=0): the kernel returns exact
integer counts with no floating-point accumulation error.
"""

import math

import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

BLOCK_SIZE = 8192
SEED = 42
# (num_reqs, vocab_size): 128 partial tail mask, 5000 non-power-of-two tail
# mask, 8192 exactly one full block, 10000 multi-block + partial tail,
# 16384 two full blocks.
SHAPE_CASES = [(1, 128), (2, 5000), (4, 8192), (8, 10000), (2, 16384)]
DTYPES = [torch.float32, torch.bfloat16, torch.float16]
LAYOUT_PREFIX = "prefix"
LAYOUT_SCATTER = "scatter"


@triton.jit
def _num_nans_kernel(
    logits_ptr,
    logits_stride,
    num_nans_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_nans = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + req_idx * logits_stride + block, mask=mask, other=0
        )
        logits = logits.to(tl.float32)
        is_nan = libdevice.isnan(logits).to(tl.int1)
        num_nans += tl.sum(is_nan).to(tl.int32)
    tl.store(num_nans_ptr + req_idx, num_nans)


def _num_nans_ref(logits: torch.Tensor) -> torch.Tensor:
    """CPU reference: count NaNs row-wise."""
    return torch.isnan(logits).sum(dim=-1).to(torch.int32)


def _inject_nans(
    logits: torch.Tensor, num_nan: int, layout: str, gen: torch.Generator
) -> None:
    """Inject ``num_nan`` NaNs into every row of a CPU tensor, in place.

    ``prefix`` mirrors the base UT (contiguous head of each row);
    ``scatter`` draws a different pseudo-random position set per row.
    """
    if num_nan <= 0:
        return
    vocab_size = logits.shape[-1]
    if layout == LAYOUT_PREFIX:
        logits[:, :num_nan] = float("nan")
    else:
        for row in range(logits.shape[0]):
            idx = torch.randperm(vocab_size, generator=gen)[:num_nan]
            logits[row, idx] = float("nan")


def _run_kernel(logits: torch.Tensor) -> torch.Tensor:
    """Launch _num_nans_kernel on the CUDA tensor and return the CPU counts."""
    num_reqs, vocab_size = logits.shape
    num_nans = torch.empty(num_reqs, dtype=torch.int32, device=logits.device)
    _num_nans_kernel[(num_reqs,)](
        logits,
        logits.stride(0),
        num_nans,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.cuda.synchronize()
    return num_nans.cpu()


class TestNumNansKernelPrecision:
    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(SEED)
        self.device = "cuda"
        # Dedicated generator: scatter positions are deterministic and, being
        # independent of the randn draws, identical across dtype variants of
        # the same shape, isolating the dtype effect. Same SEED as the NPU
        # file, so both backends see bitwise-identical inputs.
        self.gen = torch.Generator().manual_seed(SEED)

    @pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "bf16", "fp16"])
    @pytest.mark.parametrize("layout", [LAYOUT_PREFIX, LAYOUT_SCATTER])
    @pytest.mark.parametrize("num_reqs,vocab_size", SHAPE_CASES)
    @pytest.mark.parametrize("frac_nan", [0.0, 0.1, 0.5, 1.0])
    def test_num_nans_dtype_and_layout(self, num_reqs, vocab_size, layout, dtype, frac_nan):
        """Compare kernel NaN count with the CPU reference per dtype/layout.

        Data is built in fp32 on CPU, NaNs are injected, then the tensor is
        converted to the target dtype and moved to CUDA. The reference is
        computed on the post-conversion tensor, i.e. exactly the values the
        kernel sees, so any NaN loss/creation in the dtype cast would be
        caught instead of silently cancelling out.
        """
        logits = torch.randn(num_reqs, vocab_size, dtype=torch.float32)
        num_nan = int(vocab_size * frac_nan)
        _inject_nans(logits, num_nan, layout, self.gen)
        logits_gpu = logits.to(dtype=dtype, device=self.device)

        num_nans = _run_kernel(logits_gpu)

        expected = _num_nans_ref(logits_gpu.cpu())
        torch.testing.assert_close(num_nans, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "bf16", "fp16"])
    def test_num_nans_special_values_not_counted(self, dtype):
        """Inf / -0.0 / subnormal / max-finite must not be counted as NaN.

        Row 0 tiles only non-NaN special values (expect 0). Row 1 has a
        scattered set of NaNs overwritten on top of the same specials
        (expect exactly that count), proving NaN detection still works when
        the row is dense with other special bit patterns.
        """
        num_reqs, vocab_size = 2, 1024
        num_injected = 37
        finfo = torch.finfo(dtype)
        specials = torch.tensor(
            [
                float("inf"),
                float("-inf"),
                0.0,
                -0.0,
                finfo.max,
                -finfo.max,
                finfo.tiny,
                finfo.tiny / 2,  # subnormal
            ],
            dtype=dtype,
        )
        row = specials.repeat(math.ceil(vocab_size / specials.numel()))[:vocab_size]
        logits = row.unsqueeze(0).repeat(num_reqs, 1)

        idx = torch.randperm(vocab_size, generator=self.gen)[:num_injected]
        logits[1, idx] = float("nan")

        # Sanity: the tiled specials themselves must contain no NaN.
        expected = _num_nans_ref(logits)
        assert expected.tolist() == [0, num_injected]

        logits_gpu = logits.to(device=self.device)
        num_nans = _run_kernel(logits_gpu)
        torch.testing.assert_close(num_nans, expected, rtol=0, atol=0)
