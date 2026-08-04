# Ascend A3 indirect patch accuracy test.
# Requested operator: vLLM _compute_block_max_and_sumexp (legacy: _compute_max_and_sumexp).
# Ascend parent kernel import: vllm-ascend-xyz/vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py
# Parent kernel source UT path: vllm/tests/v1/spec_decode/test_rejection_sampler_utils.py
# Coverage: helper through vLLM-Ascend's _compute_block_stats_kernel alias.

"""Validate block max and stable sumexp through the Ascend execution path.

vLLM-Ascend does not export the inline helper itself. It imports the legacy
_compute_local_logits_stats_kernel as _compute_block_stats_kernel; that parent
kernel calls _compute_max_and_sumexp for every non-greedy target/draft block.
"""

import pytest
import torch

from vllm.triton_utils import triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

try:
    from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import (
        _compute_block_stats_kernel,
    )
except (ImportError, ModuleNotFoundError) as exc:
    pytest.skip(
        "installed vLLM-Ascend does not expose _compute_block_stats_kernel; "
        f"precision was not tested: {exc}",
        allow_module_level=True,
    )


@pytest.mark.parametrize("vocab_size", [15, 31, 65])
def test_compute_block_max_and_sumexp_via_ascend_parent(vocab_size):
    init_device_properties_triton()
    device = torch.device("npu")
    block_size = 16
    num_blocks = triton.cdiv(vocab_size, block_size)

    target_logits = torch.linspace(
        -4.0, 4.0, steps=vocab_size, dtype=torch.float32, device=device
    ).reshape(1, vocab_size)
    draft_logits = target_logits.new_empty(1, 1, 1)
    expanded_idx_mapping = torch.zeros(1, dtype=torch.int32, device=device)
    expanded_local_pos = torch.zeros(1, dtype=torch.int32, device=device)
    temperature = torch.ones(1, dtype=torch.float32, device=device)

    target_local_argmax = torch.full(
        (1, num_blocks), -1, dtype=torch.int64, device=device
    )
    target_local_max = torch.full(
        (1, num_blocks), float("nan"), dtype=torch.float32, device=device
    )
    target_local_sumexp = torch.full_like(target_local_max, float("nan"))
    draft_local_max = torch.full_like(target_local_max, float("nan"))
    draft_local_sumexp = torch.full_like(target_local_max, float("nan"))

    _compute_block_stats_kernel[(1, num_blocks)](
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
        1,
        BLOCK_SIZE=block_size,
        HAS_DRAFT_LOGITS=False,
    )
    torch.npu.synchronize()

    logits_cpu = target_logits[0].cpu()
    expected_max = []
    expected_sumexp = []
    for block_idx in range(num_blocks):
        block = logits_cpu[
            block_idx * block_size : min((block_idx + 1) * block_size, vocab_size)
        ]
        block_max = block.max()
        expected_max.append(block_max)
        expected_sumexp.append(torch.exp(block - block_max).sum())

    torch.testing.assert_close(
        target_local_max.cpu(), torch.stack(expected_max).reshape(1, -1),
        rtol=1e-5, atol=1e-5,
    )
    torch.testing.assert_close(
        target_local_sumexp.cpu(), torch.stack(expected_sumexp).reshape(1, -1),
        rtol=1e-5, atol=1e-5,
    )
