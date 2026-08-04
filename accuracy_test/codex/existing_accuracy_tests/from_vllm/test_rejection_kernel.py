# Standalone Ascend A3 adaptation of an upstream vLLM accuracy path.
# Accuracy UT source: vllm/tests/v1/spec_decode/test_rejection_sampler_utils.py
# Kernel source: vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
# Coverage: _rejection_kernel

# vLLM vanilla kernel: _rejection_kernel from
# vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py

"""
Precision test for _rejection_kernel.

Kernel signature (_rejection_kernel):
    _rejection_kernel(
        sampled_ptr,                        # int64 [num_reqs, num_spec_steps + 1]
        sampled_stride,                     # stride(0)
        rejected_steps_ptr,                 # int32 [num_reqs]
        target_rejected_logsumexp_ptr,      # fp32 [num_reqs]
        draft_rejected_logsumexp_ptr,       # fp32 [num_reqs]
        target_logits_ptr,                  # fp32 [num_logits, V]
        target_logits_stride,               # stride(0)
        target_local_argmax_ptr,            # int64 [num_logits, num_blocks]
        target_local_argmax_stride,         # stride(0)
        target_local_max_ptr,               # fp32 [num_logits, num_blocks]
        target_local_max_stride,            # stride(0)
        target_local_sumexp_ptr,            # fp32 [num_logits, num_blocks]
        target_local_sumexp_stride,         # stride(0)
        draft_sampled_ptr,                  # int64 [num_logits]
        draft_logits_ptr,                   # fp32 [max_num_reqs, num_spec_steps, V]
        draft_logits_stride_0,              # stride(0)
        draft_logits_stride_1,              # stride(1)
        draft_local_max_ptr,                # fp32 [num_logits, num_blocks]
        draft_local_max_stride,             # stride(0)
        draft_local_sumexp_ptr,             # fp32 [num_logits, num_blocks]
        draft_local_sumexp_stride,          # stride(0)
        cu_num_logits_ptr,                  # int64 [num_reqs + 1]
        idx_mapping_ptr,                    # int32 [num_reqs]
        temp_ptr,                           # fp32 [max_num_reqs]
        seed_ptr,                           # int32 [max_num_reqs]
        pos_ptr,                            # int64 [num_logits]
        synthetic_conditional_rates_ptr,    # fp32 [num_spec_steps] or None
        cumulative_log_p_ptr,               # fp32 [num_logits] or None
        local_residual_mass_ptr,            # fp32 [num_logits, num_blocks] or None
        local_residual_mass_stride,         # stride(0)
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
        HAS_DRAFT_LOGITS: tl.constexpr,
        SYNTHETIC_MODE: tl.constexpr,
        USE_BLOCK_VERIFICATION: tl.constexpr,
    )

Performs rejection sampling for speculative decoding.
Supports greedy, standard speculative decoding, synthetic acceptance rates,
and block verification modes.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_local_logits_stats_kernel,
    _compute_cumulative_log_p_kernel,
    _rejection_kernel,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

import pytest

torch.manual_seed(42)


class TestRejectionKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_reqs", [1, 2])
    @pytest.mark.parametrize("num_draft_tokens", [1, 2])
    @pytest.mark.parametrize("vocab_size", [128, 1024])
    def test_greedy_rejection(self, num_reqs, num_draft_tokens, vocab_size):
        """Greedy (temp=0): accept when draft matches target argmax."""
        max_num_reqs = 4
        num_logits = num_reqs * (num_draft_tokens + 1)
        num_speculative_steps = num_draft_tokens
        VOCAB_BLOCK_SIZE = 8192
        vocab_num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
        padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)

        target_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=self.device)
        draft_logits = torch.zeros(max_num_reqs, num_speculative_steps, vocab_size, dtype=torch.float32, device=self.device)

        cu_num_logits = torch.arange(num_reqs + 1, dtype=torch.int64, device=self.device) * (num_draft_tokens + 1)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)

        draft_sampled = torch.zeros(num_logits + 1, dtype=torch.int64, device=self.device)
        for ri in range(num_reqs):
            for di in range(num_draft_tokens):
                li = ri * (num_draft_tokens + 1) + di
                draft_sampled[li + 1] = target_logits[li].argmax().item()

        temperature = torch.zeros(max_num_reqs, dtype=torch.float32, device=self.device)
        seed = torch.full((max_num_reqs,), 12345, dtype=torch.int32, device=self.device)
        pos = torch.arange(num_logits, dtype=torch.int64, device=self.device)

        expanded_idx_mapping = torch.zeros(num_logits, dtype=torch.int64, device=self.device)
        expanded_local_pos = torch.zeros(num_logits, dtype=torch.int64, device=self.device)
        for ri in range(num_reqs):
            for di in range(num_draft_tokens + 1):
                li = ri * (num_draft_tokens + 1) + di
                expanded_idx_mapping[li] = ri
                expanded_local_pos[li] = di

        # Compute block stats
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

        sampled = torch.empty(num_reqs, num_speculative_steps + 1, dtype=torch.int64, device=self.device)
        rejected_steps = torch.empty(num_reqs, dtype=torch.int32, device=self.device)
        target_rejected_logsumexp = torch.empty(num_reqs, dtype=torch.float32, device=self.device)
        draft_rejected_logsumexp = torch.empty(num_reqs, dtype=torch.float32, device=self.device)

        _rejection_kernel[(num_reqs,)](
            sampled,
            sampled.stride(0),
            rejected_steps,
            target_rejected_logsumexp,
            draft_rejected_logsumexp,
            target_logits,
            target_logits.stride(0),
            target_local_argmax,
            target_local_argmax.stride(0),
            target_local_max,
            target_local_max.stride(0),
            target_local_sumexp,
            target_local_sumexp.stride(0),
            draft_sampled,
            draft_logits,
            draft_logits.stride(0),
            draft_logits.stride(1),
            draft_local_max,
            draft_local_max.stride(0),
            draft_local_sumexp,
            draft_local_sumexp.stride(0),
            cu_num_logits,
            idx_mapping,
            temperature,
            seed,
            pos,
            None,  # synthetic_conditional_rates
            None,  # cumulative_log_p
            None,  # local_residual_mass
            0,  # local_residual_mass_stride
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
            HAS_DRAFT_LOGITS=True,
            SYNTHETIC_MODE=False,
            USE_BLOCK_VERIFICATION=False,
            num_warps=1,
        )
        torch.npu.synchronize()

        # In greedy mode, every draft that matches the target argmax is accepted.
        # Since we set draft_sampled = target_argmax, all should be accepted.
        sampled_cpu = sampled.cpu()
        rejected_steps_cpu = rejected_steps.cpu()

        for ri in range(num_reqs):
            assert rejected_steps_cpu[ri].item() == num_draft_tokens, (
                f"req {ri}: expected all {num_draft_tokens} accepted, "
                f"got {rejected_steps_cpu[ri].item()}"
            )
            for di in range(num_draft_tokens):
                li = ri * (num_draft_tokens + 1) + di
                expected_tok = target_logits[li].argmax().item()
                assert sampled_cpu[ri, di].item() == expected_tok, (
                    f"req {ri}, draft {di}: expected argmax {expected_tok}, "
                    f"got {sampled_cpu[ri, di].item()}"
                )

    @pytest.mark.parametrize("num_reqs", [1, 2])
    @pytest.mark.parametrize("num_draft_tokens", [1, 2])
    @pytest.mark.parametrize("vocab_size", [128])
    def test_non_greedy_rejection(self, num_reqs, num_draft_tokens, vocab_size):
        """Non-greedy (temp=1.0): standard speculative decoding rejection sampling."""
        max_num_reqs = 4
        num_logits = num_reqs * (num_draft_tokens + 1)
        num_speculative_steps = num_draft_tokens
        VOCAB_BLOCK_SIZE = 8192
        vocab_num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
        padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)

        target_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=self.device)
        draft_logits = torch.randn(max_num_reqs, num_speculative_steps, vocab_size, dtype=torch.float32, device=self.device)

        cu_num_logits = torch.arange(num_reqs + 1, dtype=torch.int64, device=self.device) * (num_draft_tokens + 1)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)

        draft_sampled = torch.randint(0, vocab_size, (num_logits + 1,), dtype=torch.int64, device=self.device)

        temperature = torch.full((max_num_reqs,), 1.0, dtype=torch.float32, device=self.device)
        seed = torch.full((max_num_reqs,), 67890, dtype=torch.int32, device=self.device)
        pos = torch.arange(num_logits, dtype=torch.int64, device=self.device)

        expanded_idx_mapping = torch.zeros(num_logits, dtype=torch.int64, device=self.device)
        expanded_local_pos = torch.zeros(num_logits, dtype=torch.int64, device=self.device)
        for ri in range(num_reqs):
            for di in range(num_draft_tokens + 1):
                li = ri * (num_draft_tokens + 1) + di
                expanded_idx_mapping[li] = ri
                expanded_local_pos[li] = di

        # Compute block stats
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

        sampled = torch.empty(num_reqs, num_speculative_steps + 1, dtype=torch.int64, device=self.device)
        rejected_steps = torch.empty(num_reqs, dtype=torch.int32, device=self.device)
        target_rejected_logsumexp = torch.empty(num_reqs, dtype=torch.float32, device=self.device)
        draft_rejected_logsumexp = torch.empty(num_reqs, dtype=torch.float32, device=self.device)

        _rejection_kernel[(num_reqs,)](
            sampled,
            sampled.stride(0),
            rejected_steps,
            target_rejected_logsumexp,
            draft_rejected_logsumexp,
            target_logits,
            target_logits.stride(0),
            target_local_argmax,
            target_local_argmax.stride(0),
            target_local_max,
            target_local_max.stride(0),
            target_local_sumexp,
            target_local_sumexp.stride(0),
            draft_sampled,
            draft_logits,
            draft_logits.stride(0),
            draft_logits.stride(1),
            draft_local_max,
            draft_local_max.stride(0),
            draft_local_sumexp,
            draft_local_sumexp.stride(0),
            cu_num_logits,
            idx_mapping,
            temperature,
            seed,
            pos,
            None,  # synthetic_conditional_rates
            None,  # cumulative_log_p
            None,  # local_residual_mass
            0,  # local_residual_mass_stride
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
            HAS_DRAFT_LOGITS=True,
            SYNTHETIC_MODE=False,
            USE_BLOCK_VERIFICATION=False,
            num_warps=1,
        )
        torch.npu.synchronize()

        # Verify shapes and basic properties
        sampled_cpu = sampled.cpu()
        rejected_steps_cpu = rejected_steps.cpu()

        assert sampled_cpu.shape == (num_reqs, num_speculative_steps + 1)
        assert rejected_steps_cpu.shape == (num_reqs,)
        for ri in range(num_reqs):
            assert 0 <= rejected_steps_cpu[ri].item() <= num_draft_tokens, (
                f"req {ri}: rejected_steps out of range"
            )

        # The draft tokens should have been written to sampled up until rejection
        for ri in range(num_reqs):
            accepted = rejected_steps_cpu[ri].item()
            start = int(cu_num_logits[ri].item())
            for di in range(accepted):
                li = start + di
                expected_draft = draft_sampled[li + 1].item()
                assert sampled_cpu[ri, di].item() == expected_draft, (
                    f"req {ri}, draft {di}: expected {expected_draft}, "
                    f"got {sampled_cpu[ri, di].item()}"
                )

    def test_greedy_rejection_with_rejected(self):
        """When drafts do NOT match the target argmax, they should be rejected."""
        num_reqs = 1
        num_draft_tokens = 2
        vocab_size = 512
        max_num_reqs = 2
        num_logits = num_reqs * (num_draft_tokens + 1)
        num_speculative_steps = num_draft_tokens
        VOCAB_BLOCK_SIZE = 8192
        vocab_num_blocks = 1
        padded_vocab_num_blocks = 1

        target_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=self.device)
        draft_logits = torch.zeros(max_num_reqs, num_speculative_steps, vocab_size, dtype=torch.float32, device=self.device)

        cu_num_logits = torch.arange(num_reqs + 1, dtype=torch.int64, device=self.device) * (num_draft_tokens + 1)
        idx_mapping = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)

        # Deliberately set draft tokens that DON'T match the target argmax
        draft_sampled = torch.zeros(num_logits + 1, dtype=torch.int64, device=self.device)
        for li in range(num_draft_tokens):
            tgt_argmax = target_logits[li].argmax().item()
            wrong_token = (tgt_argmax + 1) % vocab_size
            draft_sampled[li + 1] = wrong_token

        temperature = torch.zeros(max_num_reqs, dtype=torch.float32, device=self.device)
        seed = torch.full((max_num_reqs,), 999, dtype=torch.int32, device=self.device)
        pos = torch.arange(num_logits, dtype=torch.int64, device=self.device)

        expanded_idx_mapping = torch.zeros(num_logits, dtype=torch.int64, device=self.device)
        expanded_local_pos = torch.tensor([0, 1, 2], dtype=torch.int64, device=self.device)

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

        sampled = torch.empty(num_reqs, num_speculative_steps + 1, dtype=torch.int64, device=self.device)
        rejected_steps = torch.empty(num_reqs, dtype=torch.int32, device=self.device)
        target_rejected_logsumexp = torch.empty(num_reqs, dtype=torch.float32, device=self.device)
        draft_rejected_logsumexp = torch.empty(num_reqs, dtype=torch.float32, device=self.device)

        _rejection_kernel[(num_reqs,)](
            sampled,
            sampled.stride(0),
            rejected_steps,
            target_rejected_logsumexp,
            draft_rejected_logsumexp,
            target_logits,
            target_logits.stride(0),
            target_local_argmax,
            target_local_argmax.stride(0),
            target_local_max,
            target_local_max.stride(0),
            target_local_sumexp,
            target_local_sumexp.stride(0),
            draft_sampled,
            draft_logits,
            draft_logits.stride(0),
            draft_logits.stride(1),
            draft_local_max,
            draft_local_max.stride(0),
            draft_local_sumexp,
            draft_local_sumexp.stride(0),
            cu_num_logits,
            idx_mapping,
            temperature,
            seed,
            pos,
            None,  # synthetic_conditional_rates
            None,  # cumulative_log_p
            None,  # local_residual_mass
            0,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
            HAS_DRAFT_LOGITS=True,
            SYNTHETIC_MODE=False,
            USE_BLOCK_VERIFICATION=False,
            num_warps=1,
        )
        torch.npu.synchronize()

        # In greedy mode with mismatched drafts: first draft is rejected at step 0.
        # rejected_steps should be 0 because the first token is rejected.
        # The rejected_stores the accepted length, so it should be 0.
        accepted_len = rejected_steps[0].item()
        assert accepted_len == 0, (
            f"Expected 0 accepted (first draft doesn't match), got {accepted_len}"
        )

    @pytest.mark.parametrize("num_draft_tokens", [1, 2])
    def test_synthetic_mode(self, num_draft_tokens):
        """Synthetic acceptance mode: accept/reject based on provided rates."""
        num_reqs = 1
        vocab_size = 256
        max_num_reqs = 2
        num_logits = num_reqs * (num_draft_tokens + 1)
        num_speculative_steps = num_draft_tokens
        VOCAB_BLOCK_SIZE = 8192
        vocab_num_blocks = 1
        padded_vocab_num_blocks = 1

        target_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=self.device)
        draft_logits = torch.randn(max_num_reqs, num_speculative_steps, vocab_size, dtype=torch.float32, device=self.device)

        cu_num_logits = torch.arange(0, (num_reqs + 1) * (num_draft_tokens + 1), num_draft_tokens + 1, dtype=torch.int64, device=self.device)
        idx_mapping = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)

        draft_sampled = torch.randint(0, vocab_size, (num_logits + 1,), dtype=torch.int64, device=self.device)

        temperature = torch.ones(max_num_reqs, dtype=torch.float32, device=self.device)
        seed = torch.full((max_num_reqs,), 42, dtype=torch.int32, device=self.device)
        pos = torch.arange(num_logits, dtype=torch.int64, device=self.device)

        expanded_idx_mapping = torch.zeros(num_logits, dtype=torch.int64, device=self.device)
        expanded_local_pos = torch.arange(num_logits, dtype=torch.int64, device=self.device)

        # Set synthetic rates: 0.0 (reject all), 0.5, 1.0 (accept all)
        synthetic_rates = torch.tensor([0.0, 1.0][:num_draft_tokens], dtype=torch.float32, device=self.device)

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

        sampled = torch.empty(num_reqs, num_speculative_steps + 1, dtype=torch.int64, device=self.device)
        rejected_steps = torch.empty(num_reqs, dtype=torch.int32, device=self.device)
        target_rejected_logsumexp = torch.empty(num_reqs, dtype=torch.float32, device=self.device)
        draft_rejected_logsumexp = torch.empty(num_reqs, dtype=torch.float32, device=self.device)

        _rejection_kernel[(num_reqs,)](
            sampled,
            sampled.stride(0),
            rejected_steps,
            target_rejected_logsumexp,
            draft_rejected_logsumexp,
            target_logits,
            target_logits.stride(0),
            target_local_argmax,
            target_local_argmax.stride(0),
            target_local_max,
            target_local_max.stride(0),
            target_local_sumexp,
            target_local_sumexp.stride(0),
            draft_sampled,
            draft_logits,
            draft_logits.stride(0),
            draft_logits.stride(1),
            draft_local_max,
            draft_local_max.stride(0),
            draft_local_sumexp,
            draft_local_sumexp.stride(0),
            cu_num_logits,
            idx_mapping,
            temperature,
            seed,
            pos,
            synthetic_rates,
            None,  # cumulative_log_p
            None,  # local_residual_mass
            0,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
            HAS_DRAFT_LOGITS=True,
            SYNTHETIC_MODE=True,
            USE_BLOCK_VERIFICATION=False,
            num_warps=1,
        )
        torch.npu.synchronize()

        rejected_steps_cpu = rejected_steps.cpu()
        # Rate for step 0 is 0.0 --> first token should be rejected (accepted_length=0)
        assert rejected_steps_cpu[0].item() == 0, (
            f"Synthetic mode with rate[0]=0.0 should reject at step 0, "
            f"got accepted_length={rejected_steps_cpu[0].item()}"
        )
