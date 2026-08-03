import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import _compute_global_lse


def _compute_global_lse_cpu(
    local_max: torch.Tensor,
    local_sumexp: torch.Tensor,
    logit_idx: int,
    vocab_num_blocks: int,
):
    maxes = local_max[logit_idx, :vocab_num_blocks]
    sumexps = local_sumexp[logit_idx, :vocab_num_blocks]
    global_max = maxes.max().item()
    global_lse = global_max + torch.log(
        (sumexps * torch.exp(maxes - global_max)).sum()
    ).item()
    return global_lse


def test_compute_global_lse():
    torch.manual_seed(42)
    num_logits = 2
    vocab_num_blocks = 3
    PADDED_VOCAB_NUM_BLOCKS = 4

    local_max = torch.randn(num_logits, PADDED_VOCAB_NUM_BLOCKS, dtype=torch.float32)
    local_sumexp = torch.abs(torch.randn(num_logits, PADDED_VOCAB_NUM_BLOCKS, dtype=torch.float32))

    from vllm.triton_utils import tl, triton

    @triton.jit
    def _test_global_lse_kernel(
        local_max_ptr,
        local_max_stride,
        local_sumexp_ptr,
        local_sumexp_stride,
        out_ptr,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    ):
        lse = _compute_global_lse(
            local_max_ptr,
            local_max_stride,
            local_sumexp_ptr,
            local_sumexp_stride,
            0,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
        )
        tl.store(out_ptr, lse)

    device = torch.device("npu")
    out = torch.empty(1, dtype=torch.float32, device=device)

    _test_global_lse_kernel[(1,)](
        local_max.to(device),
        local_max.stride(0),
        local_sumexp.to(device),
        local_sumexp.stride(0),
        out,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=PADDED_VOCAB_NUM_BLOCKS,
    )
    torch.npu.synchronize()

    expected = _compute_global_lse_cpu(local_max, local_sumexp, 0, vocab_num_blocks)
    torch.testing.assert_close(out.cpu().item(), expected, atol=1e-3, rtol=1e-3)
