import pytest
import torch

from vllm.v1.worker.gpu.sample.penalties import _penalties_kernel


def _penalties_cpu(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
    penalty_dict_ptr: torch.Tensor,
    prompt_lens_ptr: torch.Tensor,
    repeat_penalty: float,
) -> torch.Tensor:
    out = logits.clone()
    num_tokens, vocab_size = out.shape
    for token_idx in range(num_tokens):
        req_state_idx = int(expanded_idx_mapping[token_idx])
        p_len = int(prompt_lens_ptr[req_state_idx])

        for vocab_idx in range(vocab_size):
            bin_idx = vocab_idx // 32
            bit_idx = vocab_idx % 32
            in_prompt = bool(prompt_bin_mask[req_state_idx, bin_idx] & (1 << bit_idx))
            count = int(output_bin_counts[req_state_idx, vocab_idx])

            logit = float(out[token_idx, vocab_idx])
            if count > 0 or in_prompt:
                if logit > 0:
                    logit = logit / repeat_penalty
                else:
                    logit = logit * repeat_penalty
            out[token_idx, vocab_idx] = logit

    return out


def test_penalties_kernel():
    torch.manual_seed(42)
    num_tokens, vocab_size, num_requests = 2, 64, 1
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32)

    expanded_idx_mapping = torch.zeros(num_tokens, dtype=torch.int32)
    prompt_bin_mask = torch.zeros(num_requests, 2, dtype=torch.int32)
    prompt_bin_mask[0, 0] = (1 << 3) | (1 << 7)
    output_bin_counts = torch.zeros(num_requests, vocab_size, dtype=torch.int32)
    output_bin_counts[0, 10] = 2
    output_bin_counts[0, 20] = 1

    prompt_lens = torch.tensor([5], dtype=torch.int32)
    repeat_penalty = 1.2

    expected = _penalties_cpu(
        logits,
        expanded_idx_mapping,
        prompt_bin_mask,
        output_bin_counts,
        None,
        prompt_lens,
        repeat_penalty,
    )

    device = torch.device("npu")

    from vllm.triton_utils import triton

    BLOCK_SIZE = 8192
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)

    _penalties_kernel[(num_tokens, num_blocks)](
        logits.to(device),
        logits.stride(0),
        vocab_size,
        expanded_idx_mapping.to(device),
        prompt_bin_mask.to(device),
        output_bin_counts.to(device),
        prompt_lens.to(device),
        repeat_penalty,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(logits.cpu(), expected, atol=1e-4, rtol=1e-4)
