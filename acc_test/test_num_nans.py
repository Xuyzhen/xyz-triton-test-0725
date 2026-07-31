# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.gpu.metrics.logits import get_num_nans

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab_size", [8191, 8192, 8193, 151936])
def test_get_num_nans(dtype: torch.dtype, vocab_size: int) -> None:
    """Compare the Triton NaN count with the PyTorch reference on NPU."""
    init_device_properties_triton()
    torch.manual_seed(0)

    num_reqs = 4
    # A padded allocation verifies that the kernel honors the row stride.
    logits = torch.randn(
        (num_reqs, vocab_size + 7), device="npu", dtype=dtype
    )[:, :vocab_size]
    assert not logits.is_contiguous()

    logits[1, 0] = torch.nan
    logits[2, -1] = torch.nan
    logits[2, vocab_size // 2] = torch.nan
    logits[3, 0] = torch.nan
    logits[3, -1] = torch.nan
    if vocab_size > 8192:
        logits[3, 8191] = torch.nan
        logits[3, 8192] = torch.nan

    # Infinity must not be classified as NaN.
    logits[0, 0] = torch.inf
    logits[0, -1] = -torch.inf

    expected = torch.isnan(logits).sum(dim=-1, dtype=torch.int32)
    actual = get_num_nans(logits)
    torch.npu.synchronize()

    assert actual.dtype == torch.int32
    assert actual.device.type == "npu"
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
