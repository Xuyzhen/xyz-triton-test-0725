# Standalone Ascend A3 adaptation of an upstream vLLM accuracy path.
# Accuracy UT source: vllm/tests/v1/spec_decode/test_rejection_sampler_utils.py
# Kernel source: vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
# Coverage: _compute_global_lse

# vLLM vanilla kernel: _compute_global_logsumexp (helper) from
# vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py

"""
Precision test for _compute_global_logsumexp helper function.

Signature:
    _compute_global_logsumexp(
        local_max_ptr,              # fp32 [num_logits, num_blocks]
        local_max_stride,           # stride(0)
        local_sumexp_ptr,           # fp32 [num_logits, num_blocks]
        local_sumexp_stride,        # stride(0)
        logit_idx,                  # int64 scalar
        vocab_num_blocks,           # int32 scalar
        PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    ) -> fp32

Reduces per-block max and sumexp into a global logsumexp:
    global_max = max(local_max[blocks])
    global_lse = global_max + log(sum(local_sumexp * exp(local_max - global_max)))

This helper is called from _compute_global_logprobs_and_logsumexp, _rejection_kernel,
and other functions in rejection_sampler_utils.py.

We test it indirectly by writing a small wrapper kernel that calls it.
"""

import pytest
import torch

from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

try:
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        _compute_global_lse as _compute_global_lse_helper,
    )
    _HELPER_NAME = "_compute_global_lse"
except ImportError:
    try:
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
            _compute_global_logsumexp as _compute_global_lse_helper,
        )
        _HELPER_NAME = "_compute_global_logsumexp"
    except ImportError as exc:
        pytest.skip(
            "installed vLLM does not provide a global logsumexp helper; "
            f"precision was not tested: {exc}",
            allow_module_level=True,
        )


# Define a minimal wrapper kernel that calls the installed helper name.
@triton.jit
def _global_logsumexp_wrapper(
    local_max_ptr,
    local_max_stride,
    local_sumexp_ptr,
    local_sumexp_stride,
    output_ptr,
    logit_idx,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    """Call the installed vLLM global-LSE helper and store its result."""
    result = _compute_global_lse_helper(
        local_max_ptr,
        local_max_stride,
        local_sumexp_ptr,
        local_sumexp_stride,
        logit_idx,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS,
    )
    tl.store(output_ptr, result)


def _global_logsumexp_ref(
    local_max: torch.Tensor,
    local_sumexp: torch.Tensor,
    logit_idx: int,
    vocab_num_blocks: int,
) -> float:
    """CPU reference for global logsumexp."""
    maxes = local_max[logit_idx, :vocab_num_blocks].float()
    sumexps = local_sumexp[logit_idx, :vocab_num_blocks].float()
    global_max = float(maxes.max().item())
    if global_max > float("-inf"):
        weighted = sumexps * torch.exp(maxes - global_max)
        result = global_max + float(torch.log(torch.sum(weighted)).item())
    else:
        result = global_max  # -inf
    return result


class TestComputeGlobalLogsumexp:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_logits", [1, 4])
    @pytest.mark.parametrize("num_blocks", [1, 2, 4])
    def test_global_logsumexp(self, num_logits, num_blocks):
        """Compare global logsumexp with CPU reference."""
        padded_blocks = triton.next_power_of_2(num_blocks)

        local_max = torch.randn(num_logits, padded_blocks, dtype=torch.float32, device=self.device)
        local_sumexp = torch.rand(num_logits, padded_blocks, dtype=torch.float32, device=self.device) + 0.1
        output = torch.zeros(num_logits, dtype=torch.float32, device=self.device)

        for li in range(num_logits):
            _global_logsumexp_wrapper[(1,)](
                local_max,
                local_max.stride(0),
                local_sumexp,
                local_sumexp.stride(0),
                output[li:li+1],
                li,
                num_blocks,
                PADDED_VOCAB_NUM_BLOCKS=padded_blocks,
            )
        torch.npu.synchronize()

        for li in range(num_logits):
            expected = _global_logsumexp_ref(local_max.cpu(), local_sumexp.cpu(), li, num_blocks)
            torch.testing.assert_close(output[li].item(), expected, rtol=1e-5, atol=1e-5)

    def test_all_neg_inf_blocks(self):
        """When all block maxes are -inf, result should be -inf."""
        num_logits = 1
        num_blocks = 3
        padded_blocks = triton.next_power_of_2(num_blocks)

        local_max = torch.full((num_logits, padded_blocks), float("-inf"), dtype=torch.float32, device=self.device)
        local_sumexp = torch.zeros(num_logits, padded_blocks, dtype=torch.float32, device=self.device)
        output = torch.zeros(num_logits, dtype=torch.float32, device=self.device)

        _global_logsumexp_wrapper[(1,)](
            local_max,
            local_max.stride(0),
            local_sumexp,
            local_sumexp.stride(0),
            output,
            0,
            num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=padded_blocks,
        )
        torch.npu.synchronize()

        assert output[0].item() == float("-inf"), "Global LSE of all -inf should be -inf"

    def test_single_block(self):
        """Single block: global_lse = max + log(sumexp)."""
        num_blocks = 1
        padded_blocks = 1
        num_logits = 1

        local_max = torch.tensor([[1.0]], dtype=torch.float32, device=self.device)
        local_sumexp = torch.tensor([[2.0]], dtype=torch.float32, device=self.device)
        output = torch.zeros(num_logits, dtype=torch.float32, device=self.device)

        _global_logsumexp_wrapper[(1,)](
            local_max,
            local_max.stride(0),
            local_sumexp,
            local_sumexp.stride(0),
            output,
            0,
            num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=padded_blocks,
        )
        torch.npu.synchronize()

        expected = float(1.0 + torch.log(torch.tensor(2.0)).item())
        torch.testing.assert_close(output[0].item(), expected, rtol=1e-5, atol=1e-5)
