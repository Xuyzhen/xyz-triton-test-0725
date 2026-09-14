# vLLM vanilla kernel: _penalties_kernel from vllm/vllm/v1/worker/gpu/sample/penalties.py

"""
Precision test for _penalties_kernel.

Kernel signature:
    _penalties_kernel(
        logits_ptr,                    # fp32 logits [num_tokens, vocab_size]
        logits_stride,                 # stride(0) of logits
        expanded_idx_mapping_ptr,      # int32 [num_tokens] token_idx -> req_state_idx
        token_ids_ptr,                 # int32 [total_num_logits] draft token IDs
        expanded_local_pos_ptr,        # int32 [num_tokens] position within request
        repetition_penalty_ptr,        # fp32 [max_num_reqs]
        frequency_penalty_ptr,         # fp32 [max_num_reqs]
        presence_penalty_ptr,          # fp32 [max_num_reqs]
        prompt_bin_mask_ptr,           # int32 [max_num_reqs, cdiv(vocab_size, 32)]
        prompt_bin_mask_stride,        # stride(0) of prompt_bin_mask
        output_bin_counts_ptr,         # int32 [max_num_reqs, vocab_size]
        output_bin_counts_stride,      # stride(0) of output_bin_counts
        vocab_size,                    # vocab size
        BLOCK_SIZE: tl.constexpr,      # block size for iteration
    )

Applies repetition, frequency, and presence penalties to logits.
Uses a 2D grid (num_tokens, num_blocks).
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.penalties import _penalties_kernel
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
    """CPU reference: apply all three penalties."""
    out = logits.clone()
    num_tokens, vocab_size = logits.shape
    max_num_reqs = repetition_penalty.shape[0]

    # Unpack packed prompt_bin_mask
    prompt_bin_mask_unpacked = torch.zeros(max_num_reqs, vocab_size, dtype=torch.bool)
    for req in range(max_num_reqs):
        for vb in range(vocab_size):
            packed_idx = vb // 32
            bit_idx = vb % 32
            if prompt_bin_mask.shape[-1] > packed_idx:
                val = prompt_bin_mask[req, packed_idx].item()
                prompt_bin_mask_unpacked[req, vb] = bool((val >> bit_idx) & 1)

    for token_idx in range(num_tokens):
        req_state_idx = expanded_idx_mapping[token_idx].item()
        rep_pen = repetition_penalty[req_state_idx].item()
        freq_pen = frequency_penalty[req_state_idx].item()
        pres_pen = presence_penalty[req_state_idx].item()

        use_rep = rep_pen != 1.0
        use_freq = freq_pen != 0.0
        use_pres = pres_pen != 0.0
        if not (use_rep or use_freq or use_pres):
            continue

        pos = expanded_local_pos[token_idx].item()
        start_idx = token_idx - pos

        # Build output_bin_mask counting draft tokens
        bin_counts = output_bin_counts[req_state_idx].clone()
        for prev in range(pos):
            prev_token = token_ids[start_idx + prev + 1].item()
            if 0 <= prev_token < vocab_size:
                bin_counts[prev_token] += 1
        bin_mask = bin_counts > 0

        for vb in range(vocab_size):
            val = out[token_idx, vb].item()
            if use_rep:
                in_prompt_or_output = prompt_bin_mask_unpacked[req_state_idx, vb] or bin_mask[vb].item()
                scale = rep_pen if in_prompt_or_output else 1.0
                if val > 0:
                    val /= scale
                else:
                    val *= scale
            if use_freq:
                val -= freq_pen * bin_counts[vb].item()
            if use_pres:
                val -= pres_pen * (1 if bin_mask[vb].item() else 0)
            out[token_idx, vb] = val
    return out


class TestPenaltiesKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_tokens", [1, 2, 4])
    @pytest.mark.parametrize("vocab_size", [128, 1024, 4096])
    def test_repetition_penalty_only(self, num_tokens, vocab_size):
        """Apply repetition penalty only."""
        max_num_reqs = 4
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        token_ids = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        expanded_local_pos = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        repetition_penalty = torch.full((max_num_reqs,), 1.2, dtype=torch.float32, device=self.device)
        frequency_penalty = torch.zeros(max_num_reqs, dtype=torch.float32, device=self.device)
        presence_penalty = torch.zeros(max_num_reqs, dtype=torch.float32, device=self.device)
        num_bins = triton.cdiv(vocab_size, 32)
        prompt_bin_mask = torch.zeros(max_num_reqs, num_bins, dtype=torch.int32, device=self.device)
        output_bin_counts = torch.zeros(max_num_reqs, vocab_size, dtype=torch.int32, device=self.device)

        logits_gpu = logits.clone()
        num_blocks = triton.cdiv(vocab_size, 8192)
        _penalties_kernel[(num_tokens, num_blocks)](
            logits_gpu,
            logits_gpu.stride(0),
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
            BLOCK_SIZE=8192,
        )
        torch.npu.synchronize()

        expected = _penalties_ref(
            logits.cpu(), expanded_idx_mapping.cpu(), token_ids.cpu(),
            expanded_local_pos.cpu(), repetition_penalty.cpu(),
            frequency_penalty.cpu(), presence_penalty.cpu(),
            prompt_bin_mask.cpu(), output_bin_counts.cpu(),
        )
        torch.testing.assert_close(logits_gpu.cpu(), expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("vocab_size", [128, 2048])
    def test_frequency_penalty(self, vocab_size):
        """Apply frequency penalty with pre-populated output bin counts."""
        num_tokens = 2
        max_num_reqs = 2
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
        token_ids = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        expanded_local_pos = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        repetition_penalty = torch.ones(max_num_reqs, dtype=torch.float32, device=self.device)
        frequency_penalty = torch.tensor([0.1, 0.5], dtype=torch.float32, device=self.device)
        presence_penalty = torch.zeros(max_num_reqs, dtype=torch.float32, device=self.device)
        num_bins = triton.cdiv(vocab_size, 32)
        prompt_bin_mask = torch.zeros(max_num_reqs, num_bins, dtype=torch.int32, device=self.device)
        output_bin_counts = torch.zeros(max_num_reqs, vocab_size, dtype=torch.int32, device=self.device)
        output_bin_counts[0, 10] = 3
        output_bin_counts[1, 100] = 5

        logits_gpu = logits.clone()
        num_blocks = triton.cdiv(vocab_size, 8192)
        _penalties_kernel[(num_tokens, num_blocks)](
            logits_gpu,
            logits_gpu.stride(0),
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
            BLOCK_SIZE=8192,
        )
        torch.npu.synchronize()

        expected = _penalties_ref(
            logits.cpu(), expanded_idx_mapping.cpu(), token_ids.cpu(),
            expanded_local_pos.cpu(), repetition_penalty.cpu(),
            frequency_penalty.cpu(), presence_penalty.cpu(),
            prompt_bin_mask.cpu(), output_bin_counts.cpu(),
        )
        torch.testing.assert_close(logits_gpu.cpu(), expected, rtol=1e-5, atol=1e-5)

    def test_all_penalties(self):
        """All three penalties active with draft token positions."""
        num_tokens, vocab_size = 2, 512
        max_num_reqs = 2
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        token_ids = torch.tensor([0, 10], dtype=torch.int32, device=self.device)
        expanded_local_pos = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
        repetition_penalty = torch.tensor([1.5], dtype=torch.float32, device=self.device)
        frequency_penalty = torch.tensor([0.2], dtype=torch.float32, device=self.device)
        presence_penalty = torch.tensor([0.3], dtype=torch.float32, device=self.device)
        num_bins = triton.cdiv(vocab_size, 32)
        prompt_bin_mask = torch.zeros(max_num_reqs, num_bins, dtype=torch.int32, device=self.device)
        # Mark token 10 and 20 in prompt for req 0
        prompt_bin_mask[0, 10 // 32] |= (1 << (10 % 32))
        prompt_bin_mask[0, 20 // 32] |= (1 << (20 % 32))
        output_bin_counts = torch.zeros(max_num_reqs, vocab_size, dtype=torch.int32, device=self.device)

        logits_gpu = logits.clone()
        num_blocks = triton.cdiv(vocab_size, 8192)
        _penalties_kernel[(num_tokens, num_blocks)](
            logits_gpu,
            logits_gpu.stride(0),
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
            BLOCK_SIZE=8192,
        )
        torch.npu.synchronize()

        expected = _penalties_ref(
            logits.cpu(), expanded_idx_mapping.cpu(), token_ids.cpu(),
            expanded_local_pos.cpu(), repetition_penalty.cpu(),
            frequency_penalty.cpu(), presence_penalty.cpu(),
            prompt_bin_mask.cpu(), output_bin_counts.cpu(),
        )
        torch.testing.assert_close(logits_gpu.cpu(), expected, rtol=1e-5, atol=1e-5)

    def test_no_penalty_noop(self):
        """When no penalties are active (rep=1, freq=0, pres=0), logits are unchanged."""
        num_tokens, vocab_size = 2, 256
        max_num_reqs = 1
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        token_ids = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        expanded_local_pos = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        repetition_penalty = torch.ones(max_num_reqs, dtype=torch.float32, device=self.device)
        frequency_penalty = torch.zeros(max_num_reqs, dtype=torch.float32, device=self.device)
        presence_penalty = torch.zeros(max_num_reqs, dtype=torch.float32, device=self.device)
        num_bins = triton.cdiv(vocab_size, 32)
        prompt_bin_mask = torch.zeros(max_num_reqs, num_bins, dtype=torch.int32, device=self.device)
        output_bin_counts = torch.zeros(max_num_reqs, vocab_size, dtype=torch.int32, device=self.device)

        logits_gpu = logits.clone()
        num_blocks = triton.cdiv(vocab_size, 8192)
        _penalties_kernel[(num_tokens, num_blocks)](
            logits_gpu,
            logits_gpu.stride(0),
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
            BLOCK_SIZE=8192,
        )
        torch.npu.synchronize()

        torch.testing.assert_close(logits_gpu.cpu(), logits.cpu(), rtol=1e-5, atol=1e-5)
