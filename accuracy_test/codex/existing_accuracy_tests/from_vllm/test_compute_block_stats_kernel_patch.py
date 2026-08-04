# Ascend A3 alias-path accuracy patch.
# Requested operator: vLLM _compute_block_stats_kernel (legacy: _compute_local_logits_stats_kernel).
# Ascend import source: vllm-ascend-xyz/vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py
# Source UT path: vllm/tests/v1/spec_decode/test_rejection_sampler_utils.py
# Coverage: direct launch through vLLM-Ascend's _compute_block_stats_kernel alias.

"""Direct accuracy tests for the block-stats kernel exposed by vLLM-Ascend."""

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


def _expected_block_stats(row, block_size):
    maxima = []
    sumexps = []
    argmax_ids = []
    for start in range(0, row.numel(), block_size):
        block = row[start : start + block_size]
        block_max, local_idx = block.max(dim=0)
        maxima.append(block_max)
        sumexps.append(torch.exp(block - block_max).sum())
        argmax_ids.append(local_idx.to(torch.int64) + start)
    return torch.stack(maxima), torch.stack(sumexps), torch.stack(argmax_ids)


class TestComputeBlockStatsKernelPatch:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")
        self.block_size = 16
        self.vocab_size = 37
        self.num_blocks = triton.cdiv(self.vocab_size, self.block_size)

    def _allocate_outputs(self, num_logits):
        argmax = torch.full(
            (num_logits, self.num_blocks), -1, dtype=torch.int64, device=self.device
        )
        local_max = torch.full(
            (num_logits, self.num_blocks), float("nan"),
            dtype=torch.float32, device=self.device,
        )
        local_sumexp = torch.full_like(local_max, float("nan"))
        draft_max = torch.full_like(local_max, float("nan"))
        draft_sumexp = torch.full_like(local_max, float("nan"))
        return argmax, local_max, local_sumexp, draft_max, draft_sumexp

    def test_non_greedy_target_and_draft_stats(self):
        num_logits = 2
        num_speculative_steps = 2
        target_logits = torch.arange(
            num_logits * self.vocab_size, dtype=torch.float32, device=self.device
        ).reshape(num_logits, self.vocab_size) / 11.0 - 3.0
        draft_logits = torch.flip(target_logits, dims=[1]).reshape(
            1, num_speculative_steps, self.vocab_size
        )
        expanded_idx_mapping = torch.zeros(
            num_logits, dtype=torch.int32, device=self.device
        )
        expanded_local_pos = torch.arange(
            num_logits, dtype=torch.int32, device=self.device
        )
        temperature = torch.ones(1, dtype=torch.float32, device=self.device)
        outputs = self._allocate_outputs(num_logits)
        target_argmax, target_max, target_sumexp, draft_max, draft_sumexp = outputs

        _compute_block_stats_kernel[(num_logits, self.num_blocks)](
            target_argmax, target_argmax.stride(0),
            target_max, target_max.stride(0),
            target_sumexp, target_sumexp.stride(0),
            draft_max, draft_max.stride(0),
            draft_sumexp, draft_sumexp.stride(0),
            target_logits, target_logits.stride(0),
            draft_logits, draft_logits.stride(0), draft_logits.stride(1),
            expanded_idx_mapping, expanded_local_pos, temperature,
            self.vocab_size, num_speculative_steps,
            BLOCK_SIZE=self.block_size, HAS_DRAFT_LOGITS=True,
        )
        torch.npu.synchronize()

        for row_idx in range(num_logits):
            expected_tmax, expected_tsum, _ = _expected_block_stats(
                target_logits[row_idx].cpu(), self.block_size
            )
            expected_dmax, expected_dsum, _ = _expected_block_stats(
                draft_logits[0, row_idx].cpu(), self.block_size
            )
            torch.testing.assert_close(
                target_max[row_idx].cpu(), expected_tmax, rtol=1e-5, atol=1e-5
            )
            torch.testing.assert_close(
                target_sumexp[row_idx].cpu(), expected_tsum, rtol=1e-5, atol=1e-5
            )
            torch.testing.assert_close(
                draft_max[row_idx].cpu(), expected_dmax, rtol=1e-5, atol=1e-5
            )
            torch.testing.assert_close(
                draft_sumexp[row_idx].cpu(), expected_dsum, rtol=1e-5, atol=1e-5
            )

    def test_greedy_argmax_and_max(self):
        target_logits = torch.arange(
            self.vocab_size, dtype=torch.float32, device=self.device
        ).reshape(1, self.vocab_size)
        draft_logits = target_logits.new_empty(1, 1, 1)
        expanded_idx_mapping = torch.zeros(1, dtype=torch.int32, device=self.device)
        expanded_local_pos = torch.zeros(1, dtype=torch.int32, device=self.device)
        temperature = torch.zeros(1, dtype=torch.float32, device=self.device)
        outputs = self._allocate_outputs(1)
        target_argmax, target_max, target_sumexp, draft_max, draft_sumexp = outputs

        _compute_block_stats_kernel[(1, self.num_blocks)](
            target_argmax, target_argmax.stride(0),
            target_max, target_max.stride(0),
            target_sumexp, target_sumexp.stride(0),
            draft_max, draft_max.stride(0),
            draft_sumexp, draft_sumexp.stride(0),
            target_logits, target_logits.stride(0),
            draft_logits, draft_logits.stride(0), draft_logits.stride(1),
            expanded_idx_mapping, expanded_local_pos, temperature,
            self.vocab_size, 1,
            BLOCK_SIZE=self.block_size, HAS_DRAFT_LOGITS=False,
        )
        torch.npu.synchronize()

        expected_max, _, expected_argmax = _expected_block_stats(
            target_logits[0].cpu(), self.block_size
        )
        torch.testing.assert_close(
            target_max[0].cpu(), expected_max, rtol=0, atol=0
        )
        assert torch.equal(target_argmax[0].cpu(), expected_argmax)
