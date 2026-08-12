# Direct strict test for vLLM-main _compute_local_residual_mass_kernel.
from accuracy_test.strict_ut.runtime_gpu import DEVICE, synchronize

import pytest
import torch

from vllm.triton_utils import triton
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_local_residual_mass_kernel,
)

from accuracy_test.strict_ut.metrics import assert_float_close
def _local_stats(rows: torch.Tensor, block_size: int):
    num_rows, vocab_size = rows.shape
    num_blocks = triton.cdiv(vocab_size, block_size)
    maxes = torch.empty(num_rows, num_blocks, dtype=torch.float32, device=rows.device)
    sumexp = torch.empty_like(maxes)
    for row in range(num_rows):
        for block in range(num_blocks):
            values = rows[row, block * block_size : min((block + 1) * block_size, vocab_size)].float()
            maximum = values.max()
            maxes[row, block] = maximum
            sumexp[row, block] = torch.exp(values - maximum).sum()
    return maxes, sumexp


@pytest.mark.parametrize("vocab_size", [8191, 8192, 8193])
def test_compute_local_residual_mass(vocab_size):
    torch.manual_seed(17)
    num_reqs = 2
    num_speculative_steps = 3
    rows_per_req = num_speculative_steps + 1
    num_logits = num_reqs * rows_per_req
    block_size = 8192
    num_blocks = triton.cdiv(vocab_size, block_size)

    target = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=DEVICE)
    draft = torch.randn(num_reqs, num_speculative_steps, vocab_size, dtype=torch.float32, device=DEVICE)
    expanded_idx = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE).repeat_interleave(rows_per_req)
    expanded_pos = torch.arange(rows_per_req, dtype=torch.int32, device=DEVICE).repeat(num_reqs)
    temperature = torch.ones(num_reqs, dtype=torch.float32, device=DEVICE)

    target_max, target_sumexp = _local_stats(target, block_size)
    draft_rows = torch.empty_like(target)
    for row in range(num_logits):
        req = int(expanded_idx[row])
        pos = min(int(expanded_pos[row]), num_speculative_steps - 1)
        draft_rows[row] = draft[req, pos]
    draft_max, draft_sumexp = _local_stats(draft_rows, block_size)

    cumulative_log_p = torch.full((num_logits,), torch.log(torch.tensor(0.75)), dtype=torch.float32, device=DEVICE)
    sentinel = -777.0
    output = torch.full((num_logits, num_blocks), sentinel, dtype=torch.float32, device=DEVICE)

    _compute_local_residual_mass_kernel[(num_logits, num_blocks)](
        output,
        output.stride(0),
        cumulative_log_p,
        target,
        target.stride(0),
        target_max,
        target_max.stride(0),
        target_sumexp,
        target_sumexp.stride(0),
        draft,
        draft.stride(0),
        draft.stride(1),
        draft_max,
        draft_max.stride(0),
        draft_sumexp,
        draft_sumexp.stride(0),
        expanded_idx,
        expanded_pos,
        temperature,
        vocab_size,
        num_speculative_steps,
        num_blocks,
        BLOCK_SIZE=block_size,
        PADDED_VOCAB_NUM_BLOCKS=triton.next_power_of_2(num_blocks),
    )
    synchronize()

    expected = torch.full_like(output.cpu(), sentinel)
    for row in range(num_logits):
        pos = int(expanded_pos[row])
        if pos == 0 or pos >= num_speculative_steps:
            continue
        req = int(expanded_idx[row])
        target_prob = torch.softmax(target[row].double().cpu(), dim=-1)
        draft_prob = torch.softmax(draft[req, pos].double().cpu(), dim=-1)
        residual = torch.clamp(0.75 * target_prob - draft_prob, min=0)
        for block in range(num_blocks):
            expected[row, block] = residual[
                block * block_size : min((block + 1) * block_size, vocab_size)
            ].sum().float()

    assert_float_close(output, expected, rtol=2e-5, atol=2e-6)
