# GENERATED STRICT UT. Source: accuracy_test/codex/existing_accuracy_tests/from_vllm/test_compute_block_max_and_sumexp.py
# Do not edit mechanically; update the reviewed Codex source or strict generator.
from accuracy_test.strict_ut.runtime_npu import STRICT_DEVICE as _STRICT_DEVICE
# Standalone Ascend A3 adaptation of an upstream vLLM accuracy path.
# Accuracy UT source: vllm/tests/v1/spec_decode/test_rejection_sampler_utils.py
# Kernel source: vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
# Coverage: _compute_block_max_and_sumexp

# vLLM vanilla kernel: _compute_max_and_sumexp / _compute_local_logits_stats_kernel from
# vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py

"""
Precision test for _compute_max_and_sumexp (helper) and the kernel that uses it.

The _compute_max_and_sumexp helper is called from _compute_local_logits_stats_kernel.
This test verifies block-level max and sumexp computed by the kernel against
a CPU reference.

Kernel signature (_compute_local_logits_stats_kernel):
    _compute_local_logits_stats_kernel(
        target_local_argmax_ptr,        # int64 [num_logits, num_blocks]
        target_local_argmax_stride,     # stride(0)
        target_local_max_ptr,           # fp32 [num_logits, num_blocks]
        target_local_max_stride,        # stride(0)
        target_local_sumexp_ptr,        # fp32 [num_logits, num_blocks]
        target_local_sumexp_stride,     # stride(0)
        draft_local_max_ptr,            # fp32 [num_logits, num_blocks]
        draft_local_max_stride,         # stride(0)
        draft_local_sumexp_ptr,         # fp32 [num_logits, num_blocks]
        draft_local_sumexp_stride,      # stride(0)
        target_logits_ptr,              # fp32 [num_logits, V]
        target_logits_stride,           # stride(0)
        draft_logits_ptr,               # fp32 [max_num_reqs, num_spec_steps, V]
        draft_logits_stride_0,          # stride(0)
        draft_logits_stride_1,          # stride(1)
        expanded_idx_mapping_ptr,       # int64 [num_logits]
        expanded_local_pos_ptr,         # int64 [num_logits]
        temp_ptr,                       # fp32 [max_num_reqs]
        vocab_size,
        num_speculative_steps,
        BLOCK_SIZE: tl.constexpr,
        HAS_DRAFT_LOGITS: tl.constexpr,
    )

Computes block-level max, sumexp (and argmax for greedy) of target
and draft logits for each logit position and vocab block.
"""

import torch
import pytest

from vllm.triton_utils import tl, triton
try:
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        _compute_local_logits_stats_kernel,
    )
except ImportError as exc:
    pytest.skip(
        "installed vLLM does not provide _compute_local_logits_stats_kernel; "
        f"precision was not tested: {exc}",
        allow_module_level=True,
    )
from accuracy_test.strict_ut.runtime_npu import init_device_properties_triton


def _compute_max_and_sumexp_ref(logits: torch.Tensor):
    """CPU reference for _compute_max_and_sumexp."""
    max_val = float(logits.max().item())
    if max_val > float("-inf"):
        sumexp = float(torch.sum(torch.exp(logits - max_val)).item())
    else:
        sumexp = 0.0
    return max_val, sumexp


class TestComputeBlockMaxAndSumexp:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_logits", [1, 2, 4])
    @pytest.mark.parametrize("vocab_size", [128, 1024, 8192])
    @pytest.mark.parametrize("num_speculative_steps", [2, 3])
    def test_compute_block_max_and_sumexp(self, num_logits, vocab_size, num_speculative_steps):
        """Compare block-level max and sumexp with CPU reference (greedy and non-greedy)."""
        max_num_reqs = 4
        VOCAB_BLOCK_SIZE = 8192
        vocab_num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
        padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)

        target_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=self.device)
        draft_logits = torch.randn(max_num_reqs, num_speculative_steps, vocab_size, dtype=torch.float32, device=self.device)

        expanded_idx_mapping = torch.arange(num_logits, dtype=torch.int64, device=self.device) % max_num_reqs
        expanded_local_pos = torch.zeros(num_logits, dtype=torch.int64, device=self.device)
        temperature = torch.tensor([0.0, 1.0, 0.8, 0.0], dtype=torch.float32, device=self.device)

        target_local_argmax = torch.empty(num_logits, vocab_num_blocks, dtype=torch.int64, device=self.device)
        target_local_max = torch.empty(num_logits, vocab_num_blocks, dtype=torch.float32, device=self.device)
        target_local_sumexp = torch.empty(num_logits, vocab_num_blocks, dtype=torch.float32, device=self.device)
        draft_local_max = torch.empty(num_logits, vocab_num_blocks, dtype=torch.float32, device=self.device)
        draft_local_sumexp = torch.empty(num_logits, vocab_num_blocks, dtype=torch.float32, device=self.device)

        _compute_local_logits_stats_kernel[(num_logits, vocab_num_blocks)](
            target_local_argmax,
            target_local_argmax.stride(0),
            target_local_max,
            target_local_max.stride(0),
            target_local_sumexp,
            target_local_sumexp.stride(0),
            draft_local_max,
            draft_local_max.stride(0),
            draft_local_sumexp,
            draft_local_sumexp.stride(0),
            target_logits,
            target_logits.stride(0),
            draft_logits,
            draft_logits.stride(0),
            draft_logits.stride(1),
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            vocab_size,
            num_speculative_steps,
            BLOCK_SIZE=VOCAB_BLOCK_SIZE,
            HAS_DRAFT_LOGITS=True,
        )
        torch.npu.synchronize()

        # CPU reference
        tgt_cpu = target_logits.cpu()
        drf_cpu = draft_logits.cpu()
        idx_map_cpu = expanded_idx_mapping.cpu()
        temp_cpu = temperature.cpu()

        for li in range(num_logits):
            rs = idx_map_cpu[li].item()
            temp = temp_cpu[rs].item()
            for bi in range(vocab_num_blocks):
                start = bi * VOCAB_BLOCK_SIZE
                end = min(start + VOCAB_BLOCK_SIZE, vocab_size)
                block_logits = tgt_cpu[li, start:end]

                if temp == 0.0:
                    # Greedy: only target argmax and max
                    expected_max = float(block_logits.max().item())
                    expected_argmax = start + int(block_logits.argmax().item())
                    torch.testing.assert_close(target_local_max[li, bi].item(), expected_max, rtol=1e-5, atol=1e-5)
                    torch.testing.assert_close(target_local_argmax[li, bi].item(), expected_argmax, rtol=0, atol=0)
                else:
                    expected_max, expected_sumexp = _compute_max_and_sumexp_ref(block_logits)
                    torch.testing.assert_close(target_local_max[li, bi].item(), expected_max, rtol=1e-5, atol=1e-5)
                    torch.testing.assert_close(target_local_sumexp[li, bi].item(), expected_sumexp, rtol=1e-5, atol=1e-5)

                    # Draft stats for non-greedy
                    drf_block = drf_cpu[rs, 0, start:end]
                    dmax, dsumexp = _compute_max_and_sumexp_ref(drf_block)
                    torch.testing.assert_close(draft_local_max[li, bi].item(), dmax, rtol=1e-5, atol=1e-5)
                    torch.testing.assert_close(draft_local_sumexp[li, bi].item(), dsumexp, rtol=1e-5, atol=1e-5)

    def test_all_neg_inf(self):
        """Vocab block where all logits are -inf should produce max=-inf, sumexp=0."""
        num_logits = 2
        vocab_size = 8192
        max_num_reqs = 2
        num_speculative_steps = 2
        VOCAB_BLOCK_SIZE = 8192
        vocab_num_blocks = 1

        target_logits = torch.full((num_logits, vocab_size), float("-inf"), dtype=torch.float32, device=self.device)
        draft_logits = torch.full((max_num_reqs, num_speculative_steps, vocab_size), float("-inf"), dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.zeros(num_logits, dtype=torch.int64, device=self.device)
        expanded_local_pos = torch.zeros(num_logits, dtype=torch.int64, device=self.device)
        temperature = torch.ones(max_num_reqs, dtype=torch.float32, device=self.device)

        target_local_sumexp = torch.empty(num_logits, vocab_num_blocks, dtype=torch.float32, device=self.device)
        target_local_max = torch.empty(num_logits, vocab_num_blocks, dtype=torch.float32, device=self.device)
        draft_local_max = torch.empty(num_logits, vocab_num_blocks, dtype=torch.float32, device=self.device)
        draft_local_sumexp = torch.empty(num_logits, vocab_num_blocks, dtype=torch.float32, device=self.device)
        target_local_argmax = torch.empty(num_logits, vocab_num_blocks, dtype=torch.int64, device=self.device)

        _compute_local_logits_stats_kernel[(num_logits, vocab_num_blocks)](
            target_local_argmax,
            target_local_argmax.stride(0),
            target_local_max,
            target_local_max.stride(0),
            target_local_sumexp,
            target_local_sumexp.stride(0),
            draft_local_max,
            draft_local_max.stride(0),
            draft_local_sumexp,
            draft_local_sumexp.stride(0),
            target_logits,
            target_logits.stride(0),
            draft_logits,
            draft_logits.stride(0),
            draft_logits.stride(1),
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            vocab_size,
            num_speculative_steps,
            BLOCK_SIZE=VOCAB_BLOCK_SIZE,
            HAS_DRAFT_LOGITS=True,
        )
        torch.npu.synchronize()

        assert target_local_max[0, 0].item() == float("-inf"), "max should be -inf"
        assert target_local_sumexp[0, 0].item() == 0.0, "sumexp should be 0"
        assert draft_local_max[0, 0].item() == float("-inf"), "draft max should be -inf"
        assert draft_local_sumexp[0, 0].item() == 0.0, "draft sumexp should be 0"
