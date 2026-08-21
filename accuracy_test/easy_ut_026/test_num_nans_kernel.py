# GENERATED STRICT UT. Source: accuracy_test/codex/missing_accuracy_tests/test_num_nans_kernel.py
# Do not edit mechanically; update the reviewed Codex source or strict generator.
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

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.metrics.logits import _num_nans_kernel
from accuracy_test.easy_ut_026.runtime_npu import init_device_properties_triton

import pytest


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


class TestNumNansKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_reqs", [1, 2, 4, 8])
    @pytest.mark.parametrize("vocab_size", [128, 1024, 8192, 16384])
    @pytest.mark.parametrize("frac_nan", [0.0, 0.1, 0.5, 1.0])
    def test_num_nans(self, num_reqs, vocab_size, frac_nan):
        """Compare GPU kernel NaN count with CPU reference."""
        logits = torch.randn(num_reqs, vocab_size, dtype=torch.float32, device=self.device)
        # Inject NaNs at the requested fraction.
        num_nan = int(vocab_size * frac_nan)
        if num_nan > 0:
            for i in range(num_reqs):
                logits[i, :num_nan] = float("nan")

        num_nans = torch.empty(num_reqs, dtype=torch.int32, device=self.device)
        _num_nans_kernel[(num_reqs,)](
            logits,
            logits.stride(0),
            num_nans,
            vocab_size,
            BLOCK_SIZE=8192,
        )
        torch.npu.synchronize()

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
        torch.npu.synchronize()

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
        torch.npu.synchronize()

        expected = torch.full((num_reqs,), vocab_size, dtype=torch.int32)
        torch.testing.assert_close(num_nans.cpu(), expected, rtol=0, atol=0)
