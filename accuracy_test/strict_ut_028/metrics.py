"""Accuracy Standard 2.1 metrics shared by strict tests."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FloatMetrics:
    mare: float
    mere: float
    rmse: float


def float_metrics(actual: torch.Tensor, golden: torch.Tensor) -> FloatMetrics:
    actual64 = actual.detach().cpu().to(torch.float64)
    golden64 = golden.detach().cpu().to(torch.float64)
    finite = torch.isfinite(actual64) & torch.isfinite(golden64)
    if not finite.any():
        return FloatMetrics(0.0, 0.0, 0.0)
    diff = (actual64[finite] - golden64[finite]).abs()
    relative = diff / (golden64[finite].abs() + 1e-7)
    return FloatMetrics(
        mare=relative.max().item(),
        mere=relative.mean().item(),
        rmse=torch.sqrt(torch.mean(diff.square())).item(),
    )


def assert_special_values(actual: torch.Tensor, golden: torch.Tensor) -> None:
    actual = actual.detach().cpu()
    golden = golden.detach().cpu()
    assert torch.equal(torch.isnan(actual), torch.isnan(golden))
    assert torch.equal(torch.isposinf(actual), torch.isposinf(golden))
    assert torch.equal(torch.isneginf(actual), torch.isneginf(golden))


def assert_exact(actual: torch.Tensor, golden: torch.Tensor) -> None:
    torch.testing.assert_close(actual.detach().cpu(), golden.detach().cpu(), rtol=0, atol=0)


def assert_float_close(
    actual: torch.Tensor,
    golden: torch.Tensor,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> None:
    assert_special_values(actual, golden)
    finite = torch.isfinite(golden.detach().cpu())
    torch.testing.assert_close(
        actual.detach().cpu()[finite],
        golden.detach().cpu()[finite],
        rtol=rtol,
        atol=atol,
    )
