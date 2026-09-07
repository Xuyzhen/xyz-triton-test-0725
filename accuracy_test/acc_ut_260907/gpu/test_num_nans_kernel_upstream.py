# Standalone UT for _num_nans_kernel. Source: accuracy_test/easy_ut_026/test_num_nans_kernel.py
# Fully self-contained: the vllm.triton_utils shim from runtime_npu.py is
# inlined below, so this file has NO dependency on accuracy_test.* modules
# and can be dropped anywhere (run with plain pytest).
# vLLM vanilla kernel: _num_nans_kernel from
# vllm/vllm/v1/worker/gpu/metrics/logits.py

"""
Precision test for _num_nans_kernel.

Kernel signature:
    _num_nans_kernel(
        logits_ptr,               # fp32 logits [num_reqs, vocab_size]
        logits_stride,            # stride(0) of logits
        num_nans_ptr,             # int32 output [num_reqs]
        vocab_size,               # vocab size
        BLOCK_SIZE: tl.constexpr, # block size for iteration
    )

Counts NaN values in logits per request.  Uses libdevice.isnan to detect NaNs
and sums them per row.
"""

import importlib.util
import sys
import types
from pathlib import Path
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice
import pytest
import torch

_NUM_AICORE = 28
_NUM_VECTORCORE = 56

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
        # tl.device_print("is_nan", is_nan)
        # tl.device_print("is_nan",libdevice.isnan(logits).to(tl.int1))
        # tl.device_print("is_nan",libdevice.isnan(logits).tos(tl.int32))
        num_nans += tl.sum(is_nan).to(tl.int32)
    # tl.device_print("num_nans",num_nans)
    tl.store(num_nans_ptr + req_idx, num_nans)


def get_num_nans(logits: torch.Tensor) -> torch.Tensor:
    num_reqs, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    num_nans = torch.empty(num_reqs, dtype=torch.int32, device=logits.device)
    _num_nans_kernel[(num_reqs,)](
        logits,
        logits.stride(0),
        num_nans,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return num_nans


def _num_nans_ref(logits: torch.Tensor) -> torch.Tensor:
    """CPU reference: count NaNs row-wise."""
    num_reqs, vocab_size = logits.shape
    out = torch.empty(num_reqs, dtype=torch.int32)
    for i in range(num_reqs):
        count = 0
        for j in range(vocab_size):
            if torch.isnan(logits[i, j]):
                count += 1
        out[i] = count
    return out


def _gen_logits(num_reqs: int, vocab_size: int, frac_nan: float, device) -> torch.Tensor:
    """Deterministic, device-independent logits for cross-backend alignment.

    A parameter-derived seed drives a CPU generator; the tensor is moved to
    the target device only after generation (and after NaN injection), so the
    CUDA and NPU test files see bitwise-identical inputs for the same
    (num_reqs, vocab_size, frac_nan) combination.
    """
    seed = num_reqs * 1_000_003 + vocab_size * 8_191 + int(frac_nan * 10)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    logits = torch.randn(num_reqs, vocab_size, dtype=torch.float32, generator=g)
    # Inject NaNs at the requested fraction (on CPU, before the device copy).
    num_nan = int(vocab_size * frac_nan)
    if num_nan > 0:
        for i in range(num_reqs):
            logits[i, :num_nan] = float("nan")
    return logits.to(device)


class TestNumNansKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = torch.device("cuda")

    @pytest.mark.parametrize("num_reqs", [1, 2, 4, 8])
    @pytest.mark.parametrize("vocab_size", [128, 1024, 8192, 16384])
    @pytest.mark.parametrize("frac_nan", [0.0, 0.1, 0.5, 1.0])
    def test_num_nans(self, num_reqs, vocab_size, frac_nan):
        """Compare GPU kernel NaN count with CPU reference."""
        logits = _gen_logits(num_reqs, vocab_size, frac_nan, self.device)

        num_nans = torch.empty(num_reqs, dtype=torch.int32, device=self.device)
        _num_nans_kernel[(num_reqs,)](
            logits,
            logits.stride(0),
            num_nans,
            vocab_size,
            BLOCK_SIZE=8192,
        )

        expected = _num_nans_ref(logits.cpu())
        torch.testing.assert_close(num_nans.cpu(), expected, rtol=0, atol=0)

    def test_no_nans(self):
        """When there are no NaNs, all counts should be zero."""
        num_reqs, vocab_size = 4, 4096
        logits = torch.ones(num_reqs, vocab_size, dtype=torch.float32, device=self.device)

        num_nans = torch.empty(num_reqs, dtype=torch.int32, device=self.device)
        _num_nans_kernel[(num_reqs,)](
            logits,
            logits.stride(0),
            num_nans,
            vocab_size,
            BLOCK_SIZE=8192,
        )

        torch.testing.assert_close(num_nans.cpu(), torch.zeros(num_reqs, dtype=torch.int32), rtol=0, atol=0)

    def test_all_nans(self):
        """When all values are NaN, each request should report vocab_size NaN."""
        num_reqs, vocab_size = 3, 512
        logits = torch.full((num_reqs, vocab_size), float("nan"), dtype=torch.float32, device=self.device)

        num_nans = torch.empty(num_reqs, dtype=torch.int32, device=self.device)
        _num_nans_kernel[(num_reqs,)](
            logits,
            logits.stride(0),
            num_nans,
            vocab_size,
            BLOCK_SIZE=8192,
        )

        expected = torch.full((num_reqs,), vocab_size, dtype=torch.int32)
        torch.testing.assert_close(num_nans.cpu(), expected, rtol=0, atol=0)
