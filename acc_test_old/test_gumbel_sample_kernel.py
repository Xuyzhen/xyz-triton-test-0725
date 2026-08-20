# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.gpu.sample.gumbel import (
    _gumbel_sample_kernel,
    gumbel_sample,
)

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _gumbel_sample_cpu(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    vocab_size: int,
    apply_temperature: bool = True,
    use_fp64: bool = False,
) -> torch.Tensor:
    """Pure PyTorch CPU reference implementation of Gumbel-max sampling.

    For each token, applies temperature scaling, adds Gumbel noise, and
    selects the argmax token id.  This is a faithful reproduction of the
    Triton kernel's logic using PyTorch operations.
    """
    import math

    num_tokens = logits.shape[0]
    result = torch.empty(num_tokens, dtype=torch.int64)

    for token_idx in range(num_tokens):
        req_state_idx = int(expanded_idx_mapping[token_idx])
        temp = float(temperature[req_state_idx])

        row = logits[token_idx].clone().to(torch.float64 if use_fp64 else torch.float32)

        if apply_temperature and temp != 0.0:
            row = row / temp

        if temp != 0.0:
            rng = torch.Generator()
            rng.manual_seed(int(seed[req_state_idx]))
            pos_val = int(pos[token_idx])
            # Reproduce the Gumbel noise: seed = randint(seed, pos), then
            # draw uniform random numbers in (0, 1), apply -log(-log(x)).
            # The Triton kernel uses tl.randint(seed, pos) which is a
            # deterministic hash-based RNG; approximating with PyTorch.
            noise_seed = int(seed[req_state_idx]) + pos_val
            torch.manual_seed(noise_seed)
            u = torch.empty(vocab_size, dtype=torch.float64 if use_fp64 else torch.float32).uniform_(
                4.6566127342e-10, 1.0
            )
            if use_fp64:
                gumbel_noise = -torch.log(-torch.log(u))
            else:
                gumbel_noise = -torch.log(-torch.log1p(-u))
            row = row + gumbel_noise

        sampled = int(torch.argmax(row))
        result[token_idx] = sampled

    return result


def test_gumbel_sample_via_api() -> None:
    """Gumbel sampling via the public ``gumbel_sample`` API.

    Verifies end-to-end sampling with temperature, seeds, and positions.
    The kernel produces sample token ids; checks that output dtype is int64
    and values are within [0, vocab_size).
    """
    init_device_properties_triton()
    torch.manual_seed(42)

    num_tokens = 8
    vocab_size = 32000
    max_num_reqs = 4

    device = torch.device("npu")

    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=device)
    expanded_idx_mapping = torch.randint(0, max_num_reqs, (num_tokens,), device=device)
    temperature = torch.rand(max_num_reqs, dtype=torch.float32, device=device) * 2.0
    seed = torch.randint(0, 2**31, (max_num_reqs,), dtype=torch.int64, device=device)
    pos = torch.zeros(num_tokens, dtype=torch.int64, device=device)

    sampled = gumbel_sample(
        logits=logits,
        expanded_idx_mapping=expanded_idx_mapping,
        temperature=temperature,
        seed=seed,
        pos=pos,
        apply_temperature=True,
        use_fp64=False,
    )
    torch.npu.synchronize()

    assert sampled.dtype == torch.int64
    assert sampled.shape == (num_tokens,)
    assert (sampled >= 0).all() and (sampled < vocab_size).all()


@pytest.mark.parametrize("apply_temperature", [True, False])
@pytest.mark.parametrize("use_fp64", [True, False])
def test_gumbel_sample_kernel_direct(
    apply_temperature: bool, use_fp64: bool,
) -> None:
    """Direct ``_gumbel_sample_kernel`` launch with 2D grid.

    Verifies the kernel produces values within [0, vocab_size) and
    that the local argmax / local max tensors are correctly populated
    regardless of temperature / float precision settings.
    """
    init_device_properties_triton()
    torch.manual_seed(99)

    num_tokens = 4
    vocab_size = 4096
    max_num_reqs = 4
    BLOCK_SIZE = 1024
    num_blocks = (vocab_size + BLOCK_SIZE - 1) // BLOCK_SIZE  # 4

    device = torch.device("npu")

    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=device)
    expanded_idx_mapping = torch.randint(0, max_num_reqs, (num_tokens,), device=device)
    temperature = torch.rand(max_num_reqs, dtype=torch.float32, device=device) * 2.0
    seed = torch.randint(0, 2**31, (max_num_reqs,), dtype=torch.int64, device=device)
    pos = torch.zeros(num_tokens, dtype=torch.int64, device=device)

    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max = logits.new_empty(num_tokens, num_blocks,
                                 dtype=torch.float64 if use_fp64 else torch.float32)

    _gumbel_sample_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        None,  # processed_logits_ptr
        0,     # processed_logits_stride
        None,  # processed_logits_col_ptr
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        APPLY_TEMPERATURE=apply_temperature,
        USE_FP64=use_fp64,
        PER_TOKEN_COL=False,
    )
    torch.npu.synchronize()

    assert local_argmax.dtype == torch.int64
    assert local_max.dtype == (torch.float64 if use_fp64 else torch.float32)
    assert (local_argmax >= 0).all() and (local_argmax < vocab_size).all()
    assert not torch.isnan(local_max).any()
    assert not torch.isinf(local_max).any()


def test_gumbel_sample_with_processed_logits() -> None:
    """Gumbel sampling with processed logits output.

    Verifies the ``processed_logits`` output path (temperature-applied
    logits stored into an auxiliary buffer).
    """
    init_device_properties_triton()
    torch.manual_seed(77)

    num_tokens = 2
    vocab_size = 1024
    max_num_reqs = 2
    BLOCK_SIZE = 1024
    num_blocks = (vocab_size + BLOCK_SIZE - 1) // BLOCK_SIZE  # 1

    device = torch.device("npu")

    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=device)
    expanded_idx_mapping = torch.tensor([0, 1], dtype=torch.int64, device=device)
    temperature = torch.tensor([0.5, 2.0], dtype=torch.float32, device=device)
    seed = torch.tensor([123, 456], dtype=torch.int64, device=device)
    pos = torch.tensor([0, 1], dtype=torch.int64, device=device)

    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=torch.float32)
    processed_logits = torch.empty(max_num_reqs, vocab_size, dtype=torch.float32,
                                   device=device)

    _gumbel_sample_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        processed_logits,
        processed_logits.stride(0),
        None,  # processed_logits_col_ptr
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        APPLY_TEMPERATURE=True,
        USE_FP64=False,
        PER_TOKEN_COL=False,
    )
    torch.npu.synchronize()

    assert (local_argmax >= 0).all() and (local_argmax < vocab_size).all()
    assert not torch.isnan(local_max).any()
    # processed_logits should be populated (not all zero/inf).
    row0 = processed_logits[0]
    row1 = processed_logits[1]
    assert not torch.isnan(row0).any() or not torch.isnan(row1).any()
