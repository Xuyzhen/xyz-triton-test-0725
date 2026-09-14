# Standalone Ascend A3 adaptation of an upstream vLLM accuracy path.
# Accuracy UT source: vllm/tests/v1/worker/test_gpu_gumbel_sample.py
# Kernel source: vllm/vllm/v1/worker/gpu/sample/gumbel.py
# Coverage: tl_rand64's random-uniform contract through the A3 FP32 replacement
# and through gumbel_sample -> _gumbel_sample_kernel

# vLLM vanilla kernel: tl_rand64 from
# vllm/vllm/v1/worker/gpu/sample/gumbel.py

"""
Ascend A3 precision test for the random-uniform contract of tl_rand64.

tl_rand64 is a JIT helper used by gumbel_block_argmax, which is called by
_gumbel_sample_kernel.  It generates a uniform random float in (0, 1] using
tl.randint4x.

Kernel signature (JIT helper, not a standalone kernel):
    tl_rand64(seed, offset, includes_zero: tl.constexpr)

The upstream helper returns float64, which Ascend Triton cannot compile. The
test-only A3 adapter below follows vLLM-Ascend's production choice: tl.rand in
float32, with a small lower clamp when zero must be excluded. It tests the
observable random-uniform contract, not float64 bit-for-bit equivalence.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

import pytest


@triton.jit
def _tl_rand64_a3(seed, offset, includes_zero: tl.constexpr):
    """A3-compatible FP32 replacement used by vLLM-Ascend Gumbel paths."""
    u = tl.rand(seed, offset).to(tl.float32)
    if not includes_zero:
        u = tl.maximum(u, 1.0e-20)
    return u


@triton.jit
def _tl_rand64_wrapper(
    output_ptr,
    seed,
    base_offset,
    NUM_SAMPLES: tl.constexpr,
    INCLUDES_ZERO: tl.constexpr,
):
    """Wrapper to test tl_rand64 by writing results to output."""
    idx = tl.program_id(0)
    offset = base_offset + idx
    u = _tl_rand64_a3(seed, offset, INCLUDES_ZERO)
    tl.store(output_ptr + idx, u)


class TestTlRand64:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")
        self.num_samples = 10000

    @pytest.mark.parametrize("includes_zero", [True, False])
    def test_range(self, includes_zero):
        """Verify random values are in the expected range."""
        output = torch.empty(self.num_samples, dtype=torch.float32, device=self.device)
        grid = (self.num_samples,)
        _tl_rand64_wrapper[grid](
            output,
            seed=42,
            base_offset=0,
            NUM_SAMPLES=self.num_samples,
            INCLUDES_ZERO=includes_zero,
        )
        torch.npu.synchronize()

        output_cpu = output.cpu().numpy()

        if includes_zero:
            assert (output_cpu >= 0.0).all(), f"Min value: {output_cpu.min()}"
        else:
            assert (output_cpu > 0.0).all(), f"Min value: {output_cpu.min()}"
        assert (output_cpu <= 1.0).all(), f"Max value: {output_cpu.max()}"

    def test_statistical_uniformity(self):
        """Crude uniformity check: mean should be ~0.5."""
        output = torch.empty(self.num_samples, dtype=torch.float32, device=self.device)
        grid = (self.num_samples,)
        _tl_rand64_wrapper[grid](
            output,
            seed=42,
            base_offset=0,
            NUM_SAMPLES=self.num_samples,
            INCLUDES_ZERO=False,
        )
        torch.npu.synchronize()

        mean = output.cpu().mean().item()
        # Mean should be close to 0.5 within 5 sigma of std error ~ 0.5/sqrt(10000)/sqrt(12) ≈ 0.014
        assert 0.45 < mean < 0.55, f"Mean {mean} not close to 0.5"

    def test_different_seeds(self):
        """Different seeds should produce different output sequences."""
        output1 = torch.empty(100, dtype=torch.float32, device=self.device)
        output2 = torch.empty(100, dtype=torch.float32, device=self.device)

        _tl_rand64_wrapper[(100,)](output1, seed=42, base_offset=0, NUM_SAMPLES=100, INCLUDES_ZERO=False)
        _tl_rand64_wrapper[(100,)](output2, seed=999, base_offset=0, NUM_SAMPLES=100, INCLUDES_ZERO=False)
        torch.npu.synchronize()

        # Almost certainly different
        assert not torch.allclose(output1.cpu(), output2.cpu(), rtol=0, atol=0)


class TestTlRand64ViaGumbelSample:
    """Exercise the A3 FP32 replacement through the production wrapper."""

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    def _run_dominant_token_case(self) -> torch.Tensor:
        num_tokens = 32
        vocab_size = 128
        dominant_token = 7
        logits = torch.zeros(
            num_tokens, vocab_size, dtype=torch.float32, device=self.device
        )
        logits[:, dominant_token] = 100.0
        expanded_idx_mapping = torch.zeros(
            num_tokens, dtype=torch.int32, device=self.device
        )
        temperature = torch.ones(1, dtype=torch.float32, device=self.device)
        seed = torch.tensor([0xABCD], dtype=torch.int64, device=self.device)
        pos = torch.arange(num_tokens, dtype=torch.int64, device=self.device)

        sampled = gumbel_sample(
            logits,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            apply_temperature=True,
            use_fp64=False,
        )
        torch.npu.synchronize()
        return sampled

    def test_fp32_business_path(self):
        """USE_FP64=False uses tl.rand and must execute successfully on A3."""
        sampled = self._run_dominant_token_case()
        expected = torch.full_like(sampled, 7)
        assert torch.equal(sampled, expected)
