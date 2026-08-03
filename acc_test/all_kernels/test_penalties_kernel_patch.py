# vLLM-Ascend patched kernel: _penalties_kernel from
# vllm-ascend/vllm_ascend/worker/v2/sample/penalties.py:26
# PATCH NOTE: This is an Ascend NPU adaptation of the original vLLM Triton kernel

"""
Precision test for patched _penalties_kernel (Ascend NPU version).

Patch differences vs original vllm:
- NPU-compatible chain of 'or' operations (chained or not supported in NPU Triton):
  use_penalty = use_rep_penalty or use_freq_penalty; use_penalty = use_penalty or use_pres_penalty
- Uses tl.range for loop over previous positions (instead of range())
- BLOCK_SIZE set to 4096 (different from original)
- Uses tl.load with packed bitmask representation for prompt_bin_mask
- Simplified prompt_bin_mask loading (avoids degradation to scalar on NPU)
- Uses tl.cdiv for packed_block bounds

Kernel signature:
    _penalties_kernel(
        logits_ptr,                 # fp32 logits [num_tokens, vocab_size]
        logits_stride,              # stride(0) of logits
        expanded_idx_mapping_ptr,   # [num_tokens] token_idx -> req_state_idx
        token_ids_ptr,              # [num_tokens] token IDs for context
        expanded_local_pos_ptr,     # [num_tokens] local position within request
        repetition_penalty_ptr,     # [max_num_reqs] repetition penalty
        frequency_penalty_ptr,      # [max_num_reqs] frequency penalty
        presence_penalty_ptr,       # [max_num_reqs] presence penalty
        prompt_bin_mask_ptr,        # [max_num_reqs, padded_vocab//32] packed bitmask
        prompt_bin_mask_stride,     # stride(0) of prompt_bin_mask
        output_bin_counts_ptr,      # [max_num_reqs, vocab_size] output bin counts
        output_bin_counts_stride,   # stride(0) of output_bin_counts
        vocab_size,                 # scalar: vocab size
        BLOCK_SIZE: tl.constexpr,   # block size (4096)
    )

Applies repetition, frequency, and presence penalties to logits.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

import pytest


def _penalties_ref(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    token_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
) -> torch.Tensor:
    """CPU reference for _penalties_kernel."""
    out = logits.clone()
    num_tokens, vocab_size = logits.shape

    for token_idx in range(num_tokens):
        req_state_idx = expanded_idx_mapping[token_idx].item()
        rep = repetition_penalty[req_state_idx].item()
        freq = frequency_penalty[req_state_idx].item()
        pres = presence_penalty[req_state_idx].item()

        if rep == 1.0 and freq == 0.0 and pres == 0.0:
            continue

        pos = expanded_local_pos[token_idx].item()
        start_idx = token_idx - pos

        # Build output bin counts (count tokens in output context)
        counts = output_bin_counts[req_state_idx].clone()
        for prev_pos in range(pos):
            prev_token = token_ids[start_idx + prev_pos + 1].item()
            counts[prev_token] += 1

        output_bin_mask = counts != 0

        # Apply repetition penalty
        if rep != 1.0:
            prompt_mask = prompt_bin_mask[req_state_idx]
            combined_mask = prompt_mask | output_bin_mask
            for v in range(vocab_size):
                if combined_mask[v]:
                    scale = rep
                else:
                    scale = 1.0
                if out[token_idx, v] > 0:
                    out[token_idx, v] /= scale
                else:
                    out[token_idx, v] *= scale

        # Apply frequency penalty
        out[token_idx] -= freq * counts
        # Apply presence penalty
        out[token_idx] -= pres * output_bin_mask

    return out


class TestPenaltiesKernelPatch:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")
        self.BLOCK_SIZE = 4096

    def _run_kernel(
        self,
        logits,
        expanded_idx_mapping,
        token_ids,
        expanded_local_pos,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        prompt_bin_mask,
        output_bin_counts,
    ):
        from vllm_ascend.worker.v2.sample.penalties import _penalties_kernel

        num_tokens, vocab_size = logits.shape
        num_blocks = triton.cdiv(vocab_size, self.BLOCK_SIZE)

        _penalties_kernel[(num_tokens, num_blocks)](
            logits,
            logits.stride(0),
            expanded_idx_mapping,
            token_ids,
            expanded_local_pos,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            prompt_bin_mask,
            prompt_bin_mask.stride(0),
            output_bin_counts,
            output_bin_counts.stride(0),
            vocab_size,
            BLOCK_SIZE=self.BLOCK_SIZE,
        )
        torch.npu.synchronize()

    def test_no_penalties(self):
        """When all penalties are identity values, logits must remain unchanged."""
        num_tokens, vocab_size = 2, 128
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        logits_copy = logits.clone().cpu()
        expanded_idx_mapping = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        token_ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int32, device=self.device)
        expanded_local_pos = torch.tensor([0, 1], dtype=torch.int32, device=self.device)

        repetition_penalty = torch.ones(1, dtype=torch.float32, device=self.device)
        frequency_penalty = torch.zeros(1, dtype=torch.float32, device=self.device)
        presence_penalty = torch.zeros(1, dtype=torch.float32, device=self.device)

        padded_vocab = triton.cdiv(vocab_size, 32)
        prompt_bin_mask = torch.zeros(1, padded_vocab, dtype=torch.int32, device=self.device)
        output_bin_counts = torch.zeros(1, vocab_size, dtype=torch.int32, device=self.device)

        self._run_kernel(
            logits, expanded_idx_mapping, token_ids, expanded_local_pos,
            repetition_penalty, frequency_penalty, presence_penalty,
            prompt_bin_mask, output_bin_counts,
        )

        torch.testing.assert_close(logits.cpu(), logits_copy, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("vocab_size", [128, 512, 2048])
    def test_repetition_penalty(self, vocab_size):
        """Repetition penalty should scale logits based on prompt + output presence."""
        num_tokens = 2
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.tensor([0, 0], dtype=torch.int32, device=self.device)
        token_ids = torch.tensor([10, 20], dtype=torch.int32, device=self.device)
        expanded_local_pos = torch.tensor([0, 1], dtype=torch.int32, device=self.device)

        repetition_penalty = torch.tensor([1.5], dtype=torch.float32, device=self.device)
        frequency_penalty = torch.zeros(1, dtype=torch.float32, device=self.device)
        presence_penalty = torch.zeros(1, dtype=torch.float32, device=self.device)

        padded_vocab = triton.cdiv(vocab_size, 32)
        prompt_bin_mask = torch.zeros(1, padded_vocab, dtype=torch.int32, device=self.device)
        output_bin_counts = torch.zeros(1, vocab_size, dtype=torch.int32, device=self.device)

        logits_copy = logits.clone().cpu()

        self._run_kernel(
            logits, expanded_idx_mapping, token_ids, expanded_local_pos,
            repetition_penalty, frequency_penalty, presence_penalty,
            prompt_bin_mask, output_bin_counts,
        )

        expected = _penalties_ref(
            logits_copy, expanded_idx_mapping.cpu(), token_ids.cpu(),
            expanded_local_pos.cpu(), repetition_penalty.cpu(),
            frequency_penalty.cpu(), presence_penalty.cpu(),
            prompt_bin_mask.cpu(), output_bin_counts.cpu(),
        )

        torch.testing.assert_close(logits.cpu(), expected, rtol=1e-5, atol=1e-5)

    def test_frequency_penalty(self):
        """Frequency penalty should subtract penalty * count from logits."""
        num_tokens, vocab_size = 2, 64
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        token_ids = torch.tensor([5, 5], dtype=torch.int32, device=self.device)
        expanded_local_pos = torch.tensor([0, 1], dtype=torch.int32, device=self.device)

        repetition_penalty = torch.ones(1, dtype=torch.float32, device=self.device)
        frequency_penalty = torch.tensor([0.5], dtype=torch.float32, device=self.device)
        presence_penalty = torch.zeros(1, dtype=torch.float32, device=self.device)

        padded_vocab = triton.cdiv(vocab_size, 32)
        prompt_bin_mask = torch.zeros(1, padded_vocab, dtype=torch.int32, device=self.device)
        output_bin_counts = torch.zeros(1, vocab_size, dtype=torch.int32, device=self.device)

        logits_copy = logits.clone().cpu()

        self._run_kernel(
            logits, expanded_idx_mapping, token_ids, expanded_local_pos,
            repetition_penalty, frequency_penalty, presence_penalty,
            prompt_bin_mask, output_bin_counts,
        )

        expected = _penalties_ref(
            logits_copy, expanded_idx_mapping.cpu(), token_ids.cpu(),
            expanded_local_pos.cpu(), repetition_penalty.cpu(),
            frequency_penalty.cpu(), presence_penalty.cpu(),
            prompt_bin_mask.cpu(), output_bin_counts.cpu(),
        )

        torch.testing.assert_close(logits.cpu(), expected, rtol=1e-5, atol=1e-5)

    def test_presence_penalty(self):
        """Presence penalty should subtract penalty once per seen token."""
        num_tokens, vocab_size = 2, 64
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.tensor([0, 0], dtype=torch.int32, device=self.device)
        token_ids = torch.tensor([10, 20], dtype=torch.int32, device=self.device)
        expanded_local_pos = torch.tensor([0, 1], dtype=torch.int32, device=self.device)

        repetition_penalty = torch.ones(1, dtype=torch.float32, device=self.device)
        frequency_penalty = torch.zeros(1, dtype=torch.float32, device=self.device)
        presence_penalty = torch.tensor([0.3], dtype=torch.float32, device=self.device)

        padded_vocab = triton.cdiv(vocab_size, 32)
        prompt_bin_mask = torch.zeros(1, padded_vocab, dtype=torch.int32, device=self.device)
        output_bin_counts = torch.zeros(1, vocab_size, dtype=torch.int32, device=self.device)

        logits_copy = logits.clone().cpu()

        self._run_kernel(
            logits, expanded_idx_mapping, token_ids, expanded_local_pos,
            repetition_penalty, frequency_penalty, presence_penalty,
            prompt_bin_mask, output_bin_counts,
        )

        expected = _penalties_ref(
            logits_copy, expanded_idx_mapping.cpu(), token_ids.cpu(),
            expanded_local_pos.cpu(), repetition_penalty.cpu(),
            frequency_penalty.cpu(), presence_penalty.cpu(),
            prompt_bin_mask.cpu(), output_bin_counts.cpu(),
        )

        torch.testing.assert_close(logits.cpu(), expected, rtol=1e-5, atol=1e-5)
