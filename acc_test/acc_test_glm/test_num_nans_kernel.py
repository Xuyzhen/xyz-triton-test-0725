import pytest
import torch

from vllm.v1.worker.gpu.metrics.logits import _num_nans_kernel


def _num_nans_cpu(logits: torch.Tensor) -> torch.Tensor:
    num_reqs = logits.shape[0]
    num_nans = torch.empty(num_reqs, dtype=torch.int32)
    for i in range(num_reqs):
        num_nans[i] = torch.isnan(logits[i].float()).sum().item()
    return num_nans


@pytest.mark.parametrize(
    "num_reqs,vocab_size",
    [(1, 128), (4, 256), (8, 512)],
)
def test_num_nans_kernel(num_reqs, vocab_size):
    torch.manual_seed(42)
    logits = torch.randn(num_reqs, vocab_size, dtype=torch.float32)
    logits[0, 0] = float("nan")
    logits[0, 3] = float("nan")
    if num_reqs > 2:
        logits[2, 10] = float("nan")

    expected = _num_nans_cpu(logits)

    device = torch.device("npu")
    logits_npu = logits.to(device)
    num_nans = torch.empty(num_reqs, dtype=torch.int32, device=device)

    BLOCK_SIZE = 8192
    _num_nans_kernel[(num_reqs,)](
        logits_npu,
        logits_npu.stride(0),
        num_nans,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(num_nans.cpu(), expected, rtol=0, atol=0)


def test_num_nans_kernel_no_nans():
    num_reqs, vocab_size = 4, 256
    logits = torch.randn(num_reqs, vocab_size, dtype=torch.float32)
    expected = _num_nans_cpu(logits)

    device = torch.device("npu")
    logits_npu = logits.to(device)
    num_nans = torch.empty(num_reqs, dtype=torch.int32, device=device)

    _num_nans_kernel[(num_reqs,)](
        logits_npu,
        logits_npu.stride(0),
        num_nans,
        vocab_size,
        BLOCK_SIZE=8192,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(num_nans.cpu(), expected, rtol=0, atol=0)


def test_num_nans_kernel_all_nans():
    num_reqs, vocab_size = 2, 128
    logits = torch.full((num_reqs, vocab_size), float("nan"), dtype=torch.float32)
    expected = _num_nans_cpu(logits)

    device = torch.device("npu")
    logits_npu = logits.to(device)
    num_nans = torch.empty(num_reqs, dtype=torch.int32, device=device)

    _num_nans_kernel[(num_reqs,)](
        logits_npu,
        logits_npu.stride(0),
        num_nans,
        vocab_size,
        BLOCK_SIZE=8192,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(num_nans.cpu(), expected, rtol=0, atol=0)
