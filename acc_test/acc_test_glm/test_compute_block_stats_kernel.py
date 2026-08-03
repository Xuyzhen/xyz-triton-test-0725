import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_block_max_and_sumexp,
    _compute_block_stats_kernel,
)


def _compute_block_max_and_sumexp_cpu(logits_block: torch.Tensor):
    block_max = logits_block.max().item()
    if block_max > float("-inf"):
        block_sumexp = torch.exp(logits_block - block_max).sum().item()
    else:
        block_sumexp = 0.0
    return block_max, block_sumexp


def test_compute_block_max_and_sumexp():
    torch.manual_seed(42)
    logits = torch.randn(1024, dtype=torch.float32)

    expected_max, expected_sumexp = _compute_block_max_and_sumexp_cpu(logits)

    from vllm.triton_utils import tl, triton

    @triton.jit
    def _test_kernel(
        logits_ptr,
        out_max_ptr,
        out_sumexp_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        block = tl.arange(0, BLOCK_SIZE)
        logits = tl.load(logits_ptr + block)
        m, s = _compute_block_max_and_sumexp(logits)
        tl.store(out_max_ptr, m)
        tl.store(out_sumexp_ptr, s)

    device = torch.device("npu")
    logits_npu = logits.to(device)
    out_max = torch.empty(1, dtype=torch.float32, device=device)
    out_sumexp = torch.empty(1, dtype=torch.float32, device=device)

    _test_kernel[(1,)](
        logits_npu,
        out_max,
        out_sumexp,
        BLOCK_SIZE=1024,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(out_max.cpu().item(), expected_max, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(
        out_sumexp.cpu().item(), expected_sumexp, atol=1e-2, rtol=1e-4
    )


def test_compute_block_stats_kernel_greedy():
    torch.manual_seed(42)
    num_logits = 2
    vocab_size = 64
    VOCAB_BLOCK_SIZE = 64
    vocab_num_blocks = 1

    target_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32)
    expanded_idx_mapping = torch.arange(num_logits, dtype=torch.int32)
    expanded_local_pos = torch.tensor([0, 0], dtype=torch.int32)
    temp = torch.zeros(2, dtype=torch.float32)

    target_local_argmax = torch.zeros(num_logits, vocab_num_blocks, dtype=torch.int64)
    target_local_max = torch.zeros(num_logits, vocab_num_blocks, dtype=torch.float32)
    target_local_sumexp = torch.zeros(num_logits, vocab_num_blocks, dtype=torch.float32)
    draft_local_max = torch.zeros(num_logits, vocab_num_blocks, dtype=torch.float32)
    draft_local_sumexp = torch.zeros(num_logits, vocab_num_blocks, dtype=torch.float32)

    draft_logits_dummy = target_logits.new_empty(1, 1, 1)

    device = torch.device("npu")

    target_local_argmax_npu = target_local_argmax.to(device)
    target_local_max_npu = target_local_max.to(device)
    target_local_sumexp_npu = target_local_sumexp.to(device)
    draft_local_max_npu = draft_local_max.to(device)
    draft_local_sumexp_npu = draft_local_sumexp.to(device)

    _compute_block_stats_kernel[(num_logits, vocab_num_blocks)](
        target_local_argmax_npu,
        target_local_argmax_npu.stride(0),
        target_local_max_npu,
        target_local_max_npu.stride(0),
        target_local_sumexp_npu,
        target_local_sumexp_npu.stride(0),
        draft_local_max_npu,
        draft_local_max_npu.stride(0),
        draft_local_sumexp_npu,
        draft_local_sumexp_npu.stride(0),
        target_logits.to(device),
        target_logits.stride(0),
        draft_logits_dummy.to(device),
        draft_logits_dummy.stride(0),
        draft_logits_dummy.stride(1),
        expanded_idx_mapping.to(device),
        expanded_local_pos.to(device),
        temp.to(device),
        vocab_size,
        num_speculative_steps=2,
        BLOCK_SIZE=VOCAB_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=False,
    )
    torch.npu.synchronize()

    for i in range(num_logits):
        expected_argmax = target_logits[i].argmax().item()
        expected_max = target_logits[i].max().item()
        actual_argmax = target_local_argmax_npu[i, 0].cpu().item()
        actual_max = target_local_max_npu[i, 0].cpu().item()
        assert actual_argmax == expected_argmax, (
            f"argmax mismatch at logit {i}: got {actual_argmax}, expected {expected_argmax}"
        )
        torch.testing.assert_close(
            torch.tensor(actual_max), torch.tensor(expected_max), atol=1e-5, rtol=1e-5
        )
