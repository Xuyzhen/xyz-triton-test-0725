# GENERATED STRICT UT. Source: accuracy_test/codex/existing_accuracy_tests/from_vllm/test_insert_resampled_kernel.py
# Do not edit mechanically; update the reviewed Codex source or strict generator.
from accuracy_test.strict_ut.runtime_npu import STRICT_DEVICE as _STRICT_DEVICE
# Standalone Ascend A3 adaptation of an upstream vLLM accuracy path.
# Accuracy UT source: vllm/tests/v1/spec_decode/test_rejection_sampler_utils.py
# Kernel source: vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
# Coverage: _insert_resampled_kernel

# vLLM vanilla kernel: _insert_resampled_kernel from
# vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py

"""
Precision test for _insert_resampled_kernel.

Inserts the resampled token into the sampled output tensor.

Kernel signature:
    _insert_resampled_kernel(
        sampled_ptr,                    # [num_reqs, num_spec_steps+1] int64
        sampled_stride,
        num_sampled_ptr,                # [num_reqs] int64
        resampled_local_argmax_ptr,     # [num_reqs, num_blocks] int64
        resampled_local_argmax_stride,
        resampled_local_max_ptr,        # [num_reqs, num_blocks] fp32/fp64
        resampled_local_max_stride,
        resample_num_blocks,
        cu_num_logits_ptr,              # [num_reqs+1] int64
        expanded_idx_mapping_ptr,       # [num_logits] int64
        temp_ptr,                       # [max_num_reqs] fp32
        PADDED_RESAMPLE_NUM_BLOCKS: tl.constexpr,
    )
"""

import torch
import pytest

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _insert_resampled_kernel,
)
from accuracy_test.strict_ut.runtime_npu import init_device_properties_triton


def _insert_resampled_ref(
    sampled,                  # [num_reqs, num_spec_steps+1] int64
    num_sampled,              # [num_reqs] int64
    resampled_local_argmax,   # [num_reqs, num_blocks] int64
    resampled_local_max,      # [num_reqs, num_blocks] fp32
    resample_num_blocks,
    cu_num_logits,            # [num_reqs+1]
    expanded_idx_mapping,     # [num_logits]
    temp,                     # [max_num_reqs]
):
    """
    CPU reference for _insert_resampled_kernel.

    For each request, finds the block with maximum value in
    resampled_local_max and inserts its argmax token into sampled.
    """
    out_sampled = sampled.clone()
    out_num_sampled = num_sampled.clone()

    for req_idx in range(sampled.shape[0]):
        n_sampled = int(num_sampled[req_idx].item())
        start_idx = int(cu_num_logits[req_idx].item())
        end_idx = int(cu_num_logits[req_idx + 1].item())
        resample_token_idx = start_idx + n_sampled
        req_state_idx = int(expanded_idx_mapping[resample_token_idx].item())

        # Increment num_sampled
        out_num_sampled[req_idx] = n_sampled + 1

        t = float(temp[req_state_idx].item())
        is_bonus = (resample_token_idx == end_idx - 1)

        if t == 0.0 and not is_bonus:
            continue

        # Find block with max value
        max_val = float("-inf")
        max_block = 0
        for block_idx in range(resample_num_blocks):
            v = float(resampled_local_max[req_idx, block_idx].item())
            if v > max_val:
                max_val = v
                max_block = block_idx

        token_id = int(resampled_local_argmax[req_idx, max_block].item())
        out_sampled[req_idx, n_sampled] = token_id

    return out_sampled, out_num_sampled


class TestInsertResampledKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_reqs", [1, 2, 4])
    @pytest.mark.parametrize("num_spec_steps", [1, 3])
    @pytest.mark.parametrize("vocab_size", [4096, 129280, 163840, 248320])
    def test_insert_resampled_basic(self, num_reqs, num_spec_steps, vocab_size):
        """Test the insert kernel inserts the correct resampled token."""
        RESAMPLE_BLOCK_SIZE = 1024
        resample_num_blocks = triton.cdiv(vocab_size, RESAMPLE_BLOCK_SIZE)
        padded_resample_num_blocks = triton.next_power_of_2(resample_num_blocks)
        num_logits = num_reqs * (num_spec_steps + 1)
        max_num_reqs = num_reqs

        sampled = torch.zeros(num_reqs, num_spec_steps + 1, dtype=torch.int64, device=self.device)
        num_sampled = torch.zeros(num_reqs, dtype=torch.int64, device=self.device)
        cu_num_logits = (
            torch.arange(num_reqs + 1, device=self.device, dtype=torch.int32)
            * (num_spec_steps + 1)
        )
        expanded_idx_mapping = torch.arange(
            num_reqs, device=self.device, dtype=torch.int32
        ).repeat_interleave(num_spec_steps + 1)
        temp = torch.full(
            (max_num_reqs,), 1.0, dtype=torch.float32, device=self.device
        )
        sampled_before = sampled.clone()
        num_sampled_before = num_sampled.clone()

        # Create resampled outputs: each block has a token id and a max value
        resampled_local_argmax = torch.randint(0, vocab_size, (num_reqs, resample_num_blocks), dtype=torch.int64, device=self.device)
        resampled_local_max = torch.randn(num_reqs, resample_num_blocks, dtype=torch.float32, device=self.device)

        _insert_resampled_kernel[(num_reqs,)](
            sampled,
            sampled.stride(0),
            num_sampled,
            resampled_local_argmax,
            resampled_local_argmax.stride(0),
            resampled_local_max,
            resampled_local_max.stride(0),
            resample_num_blocks,
            cu_num_logits,
            expanded_idx_mapping,
            temp,
            PADDED_RESAMPLE_NUM_BLOCKS=padded_resample_num_blocks,
        )
        torch.npu.synchronize()

        ref_sampled, ref_num_sampled = _insert_resampled_ref(
            sampled_before.cpu(),
            num_sampled_before.cpu(),
            resampled_local_argmax.cpu(),
            resampled_local_max.cpu(),
            resample_num_blocks,
            cu_num_logits.cpu(),
            expanded_idx_mapping.cpu(),
            temp.cpu(),
        )

        torch.testing.assert_close(sampled.cpu(), ref_sampled, rtol=0, atol=0)
        torch.testing.assert_close(num_sampled.cpu(), ref_num_sampled, rtol=0, atol=0)

    def test_greedy_non_bonus_skip(self):
        """When temp==0 and not bonus, kernel should skip (early return)."""
        num_reqs = 2
        num_spec_steps = 2
        vocab_size = 1024
        resample_num_blocks = 2
        padded_resample_num_blocks = 2
        num_logits = num_reqs * (num_spec_steps + 1)

        sampled = -torch.ones(num_reqs, num_spec_steps + 1, dtype=torch.int64, device=self.device)
        num_sampled = torch.zeros(num_reqs, dtype=torch.int64, device=self.device)
        cu_num_logits = (
            torch.arange(num_reqs + 1, device=self.device, dtype=torch.int32)
            * (num_spec_steps + 1)
        )
        expanded_idx_mapping = torch.arange(
            num_reqs, device=self.device, dtype=torch.int32
        ).repeat_interleave(num_spec_steps + 1)
        temp = torch.zeros(
            num_reqs, dtype=torch.float32, device=self.device
        )  # greedy

        # Triton compiles the loads after the runtime early return, so valid
        # typed pointers are required even though this case must not read them.
        dummy_argmax = torch.zeros(
            num_reqs,
            resample_num_blocks,
            dtype=torch.int64,
            device=self.device,
        )
        dummy_max = torch.zeros(
            num_reqs,
            resample_num_blocks,
            dtype=torch.float32,
            device=self.device,
        )

        _insert_resampled_kernel[(num_reqs,)](
            sampled,
            sampled.stride(0),
            num_sampled,
            dummy_argmax,
            dummy_argmax.stride(0),
            dummy_max,
            dummy_max.stride(0),
            resample_num_blocks,
            cu_num_logits,
            expanded_idx_mapping,
            temp,
            PADDED_RESAMPLE_NUM_BLOCKS=padded_resample_num_blocks,
        )
        torch.npu.synchronize()

        # Greedy, non-bonus: num_sampled should be incremented but sampled unchanged
        assert torch.all(num_sampled.cpu() == 1)
        assert torch.all(sampled.cpu() == -1)

    def test_bonus_token_greedy(self):
        """Bonus token with greedy temp should still write resampled value."""
        num_reqs = 1
        num_spec_steps = 2
        vocab_size = 512
        resample_num_blocks = 2
        padded_resample_num_blocks = 2
        num_logits = num_reqs * (num_spec_steps + 1)

        sampled = -torch.ones(num_reqs, num_spec_steps + 1, dtype=torch.int64, device=self.device)
        num_sampled = torch.tensor([num_spec_steps], dtype=torch.int64, device=self.device)  # points to bonus token
        cu_num_logits = torch.tensor(
            [0, num_logits], device=self.device, dtype=torch.int32
        )
        expanded_idx_mapping = torch.zeros(
            num_logits, device=self.device, dtype=torch.int32
        )
        temp = torch.zeros(num_reqs, dtype=torch.float32, device=self.device)  # greedy

        resampled_local_argmax = torch.tensor([[42, 99]], dtype=torch.int64, device=self.device)
        resampled_local_max = torch.tensor([[10.0, 5.0]], dtype=torch.float32, device=self.device)

        _insert_resampled_kernel[(num_reqs,)](
            sampled,
            sampled.stride(0),
            num_sampled,
            resampled_local_argmax,
            resampled_local_argmax.stride(0),
            resampled_local_max,
            resampled_local_max.stride(0),
            resample_num_blocks,
            cu_num_logits,
            expanded_idx_mapping,
            temp,
            PADDED_RESAMPLE_NUM_BLOCKS=padded_resample_num_blocks,
        )
        torch.npu.synchronize()

        # Bonus token: resampled value should be written to the bonus position
        # Best block is block 0 with max 10.0, so argmax = 42
        assert num_sampled.cpu().item() == num_spec_steps + 1
        assert sampled[0, num_spec_steps].cpu().item() == 42
