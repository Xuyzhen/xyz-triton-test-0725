# vLLM-Ascend patched kernel: _min_p_kernel from
# vllm-ascend/vllm_ascend/worker/v2/sample/min_p.py:27
# PATCH NOTE: This is an Ascend NPU adaptation of the original vLLM Triton kernel

"""
Precision test for patched _min_p_kernel (Ascend NPU version).

Patch differences vs original vllm:
- Uses do_not_specialize=["num_tokens"]
- Uses load-balanced grid based on get_vectorcore_num() instead of (num_tokens,)
- Uses tokens_per_block distribution for better load balancing
- Uses tl.range for loop over tokens (NPU-optimized loop construct)
- Uses tl.minimum for bounds clamping
- Reads logits with float("-inf") other value
- Uses BLOCK_SIZE=min(triton.next_power_of_2(vocab_size), 8192)
- In-place operation: in_logits_ptr == out_logits_ptr

Kernel signature:
    _min_p_kernel(
        in_logits_ptr,              # fp32 logits [num_tokens, vocab_size]
        out_logits_ptr,             # fp32 logits output (same as in_logits_ptr in practice)
        logits_stride,              # stride(0) of logits
        expanded_idx_mapping_ptr,   # [num_tokens] token_idx -> req_state_idx
        min_p_ptr,                  # [max_num_reqs] min_p values
        vocab_size,                 # scalar: vocab size
        num_tokens,                 # scalar: number of tokens
        BLOCK_SIZE: tl.constexpr,   # block size for iteration
    )

Applies min-p sampling: zeros out logits below threshold = max_val + log(min_p).
When min_p == 0.0, the kernel returns early (no-op).
"""

import torch

from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

import pytest


def _min_p_ref(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    min_p: torch.Tensor,
) -> torch.Tensor:
    """CPU reference: apply min-p sampling threshold."""
    out = logits.clone()
    num_tokens, vocab_size = logits.shape
    for token_idx in range(num_tokens):
        req_state_idx = expanded_idx_mapping[token_idx].item()
        mp = min_p[req_state_idx].item()
        if mp == 0.0:
            continue
        max_val = float(out[token_idx].max())
        threshold = max_val + float(torch.log(torch.tensor(mp)))
        out[token_idx][out[token_idx] < threshold] = float("-inf")
    return out


class TestMinPKernelPatch:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_tokens", [1, 2, 4, 8])
    @pytest.mark.parametrize("vocab_size", [128, 1024, 8192, 16384])
    @pytest.mark.parametrize("min_p_val", [0.0, 0.1, 0.5, 0.9, 1.0])
    def test_min_p(self, num_tokens, vocab_size, min_p_val):
        """Compare NPU min-p output with CPU reference."""
        from vllm_ascend.worker.v2.sample.min_p import _min_p_kernel

        max_num_reqs = 4
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        min_p = torch.full((max_num_reqs,), min_p_val, dtype=torch.float32, device=self.device)

        vec_core = get_vectorcore_num()
        core_nums = min(num_tokens, vec_core)

        BLOCK_SIZE = min(triton.next_power_of_2(vocab_size), 8192)

        logits_gpu = logits.clone()
        _min_p_kernel[(core_nums,)](
            logits_gpu,
            logits_gpu,
            logits_gpu.stride(0),
            expanded_idx_mapping,
            min_p,
            vocab_size,
            num_tokens,
            BLOCK_SIZE=BLOCK_SIZE,
            multibuffer=False,
        )
        torch.npu.synchronize()

        expected = _min_p_ref(logits.cpu(), expanded_idx_mapping.cpu(), min_p.cpu())
        torch.testing.assert_close(logits_gpu.cpu(), expected, rtol=0, atol=0)

    def test_multiple_req_states(self):
        """Each token can map to a different request with different min_p values."""
        from vllm_ascend.worker.v2.sample.min_p import _min_p_kernel

        num_tokens, vocab_size = 4, 2048
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=self.device)
        min_p = torch.tensor([0.0, 0.1, 0.5, 1.0], dtype=torch.float32, device=self.device)

        vec_core = get_vectorcore_num()
        core_nums = min(num_tokens, vec_core)
        BLOCK_SIZE = min(triton.next_power_of_2(vocab_size), 8192)

        logits_gpu = logits.clone()
        _min_p_kernel[(core_nums,)](
            logits_gpu,
            logits_gpu,
            logits_gpu.stride(0),
            expanded_idx_mapping,
            min_p,
            vocab_size,
            num_tokens,
            BLOCK_SIZE=BLOCK_SIZE,
            multibuffer=False,
        )
        torch.npu.synchronize()

        expected = _min_p_ref(logits.cpu(), expanded_idx_mapping.cpu(), min_p.cpu())
        torch.testing.assert_close(logits_gpu.cpu(), expected, rtol=0, atol=0)

    def test_min_p_zero_is_noop(self):
        """When min_p == 0, logits must remain unchanged."""
        from vllm_ascend.worker.v2.sample.min_p import _min_p_kernel

        num_tokens, vocab_size = 2, 512
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        min_p = torch.zeros(1, dtype=torch.float32, device=self.device)

        vec_core = get_vectorcore_num()
        core_nums = min(num_tokens, vec_core)
        BLOCK_SIZE = min(triton.next_power_of_2(vocab_size), 8192)

        logits_gpu = logits.clone()
        _min_p_kernel[(core_nums,)](
            logits_gpu,
            logits_gpu,
            logits_gpu.stride(0),
            expanded_idx_mapping,
            min_p,
            vocab_size,
            num_tokens,
            BLOCK_SIZE=BLOCK_SIZE,
            multibuffer=False,
        )
        torch.npu.synchronize()

        torch.testing.assert_close(logits_gpu.cpu(), logits.cpu(), rtol=0, atol=0)
