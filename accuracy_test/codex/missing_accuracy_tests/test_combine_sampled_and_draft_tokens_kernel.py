# vLLM vanilla kernel: _combine_sampled_and_draft_tokens_kernel from
# vllm/vllm/v1/worker/gpu/input_batch.py

"""
Precision test for _combine_sampled_and_draft_tokens_kernel.

Combines sampled and draft tokens into the input_ids buffer for the next
step of speculative decoding. Also populates logits_indices.

Kernel signature:
    _combine_sampled_and_draft_tokens_kernel(
        input_ids_ptr,            # [num_tokens] int32
        idx_mapping_ptr,          # [num_reqs] int32
        last_sampled_tokens_ptr,  # [max_num_reqs] int32
        query_start_loc_ptr,      # [num_reqs + 1] int32
        seq_lens_ptr,             # [num_reqs] int32
        prefill_len_ptr,          # [max_num_reqs] int32
        draft_tokens_ptr,         # [max_num_reqs, num_spec_steps] int32
        draft_tokens_stride,
        cu_num_logits_ptr,        # [num_reqs + 1] int64
        logits_indices_ptr,       # [total_num_logits] int64
        BLOCK_SIZE: tl.constexpr,
        NUM_NEW_SAMPLED_TOKENS: tl.constexpr = 1,
    )
"""

import torch
import pytest

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import _combine_sampled_and_draft_tokens_kernel
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _combine_sampled_and_draft_tokens_ref(
    input_ids,                # [num_tokens] int32
    idx_mapping,              # [num_reqs]
    last_sampled_tokens,      # [max_num_reqs]
    query_start_loc,          # [num_reqs + 1]
    seq_lens,                 # [num_reqs]
    prefill_len,              # [max_num_reqs]
    draft_tokens,             # [max_num_reqs, num_spec_steps]
    cu_num_logits,            # [num_reqs + 1]
    num_new_sampled_tokens=1,
):
    """CPU reference for combine sampled and draft tokens."""
    num_reqs = idx_mapping.shape[0]
    out_input_ids = input_ids.clone()
    total_logits = int(cu_num_logits[-1].item())
    out_logits_indices = torch.zeros(total_logits, dtype=torch.int64)

    for batch_idx in range(num_reqs):
        req_state_idx = int(idx_mapping[batch_idx].item())

        cu_start = int(cu_num_logits[batch_idx].item())
        cu_end = int(cu_num_logits[batch_idx + 1].item())
        num_logits = cu_end - cu_start
        num_draft_tokens = num_logits - num_new_sampled_tokens

        query_end = int(query_start_loc[batch_idx + 1].item())
        logits_start = query_end - num_logits

        # Store logits indices
        for j in range(num_logits):
            out_logits_indices[cu_start + j] = logits_start + j

        seq_len = int(seq_lens[batch_idx].item())
        plen = int(prefill_len[req_state_idx].item())

        if seq_len <= plen:
            continue

        if num_new_sampled_tokens > 0:
            last_token = int(last_sampled_tokens[req_state_idx].item())
            out_input_ids[logits_start] = last_token

        if num_draft_tokens > 0:
            for j in range(num_draft_tokens):
                dt = int(draft_tokens[req_state_idx, j].item())
                out_input_ids[query_end - num_draft_tokens + j] = dt

    return out_input_ids, out_logits_indices


class TestCombineSampledAndDraftTokensKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_reqs", [1, 2, 4])
    @pytest.mark.parametrize("num_spec_steps", [1, 3])
    @pytest.mark.parametrize("num_new_sampled_tokens", [0, 1])
    def test_combine_basic(self, num_reqs, num_spec_steps, num_new_sampled_tokens):
        """Test basic combine of sampled and draft tokens."""
        vocab_size = 100
        max_num_reqs = num_reqs
        num_draft_tokens = num_spec_steps
        total_logits = num_reqs * (num_new_sampled_tokens + num_draft_tokens)
        num_tokens = total_logits  # simplificiation

        input_ids = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        last_sampled_tokens = torch.randint(
            0, vocab_size, (max_num_reqs,), dtype=torch.int32, device=self.device
        )
        seq_lens = torch.full((num_reqs,), 20, dtype=torch.int32, device=self.device)
        prefill_len = torch.full(
            (max_num_reqs,), 5, dtype=torch.int32, device=self.device
        )
        draft_tokens = torch.randint(
            0, vocab_size, (max_num_reqs, num_spec_steps), dtype=torch.int32, device=self.device
        )
        cu_num_logits = torch.arange(
            0, total_logits + 1, num_new_sampled_tokens + num_draft_tokens,
            dtype=torch.int64, device=self.device
        )
        query_start_loc = torch.arange(
            0,
            total_logits + 1,
            num_new_sampled_tokens + num_spec_steps,
            dtype=torch.int32,
            device=self.device,
        )
        logits_indices = torch.zeros(total_logits, dtype=torch.int64, device=self.device)

        BLOCK_SIZE = triton.next_power_of_2(num_spec_steps + num_new_sampled_tokens)

        _combine_sampled_and_draft_tokens_kernel[(num_reqs,)](
            input_ids,
            idx_mapping,
            last_sampled_tokens,
            query_start_loc,
            seq_lens,
            prefill_len,
            draft_tokens,
            draft_tokens.stride(0),
            cu_num_logits,
            logits_indices,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_NEW_SAMPLED_TOKENS=num_new_sampled_tokens,
        )
        torch.npu.synchronize()

        ref_input_ids, ref_logits_indices = _combine_sampled_and_draft_tokens_ref(
            torch.zeros(num_tokens, dtype=torch.int32),
            idx_mapping.cpu(),
            last_sampled_tokens.cpu(),
            query_start_loc.cpu(),
            seq_lens.cpu(),
            prefill_len.cpu(),
            draft_tokens.cpu(),
            cu_num_logits.cpu(),
            num_new_sampled_tokens=num_new_sampled_tokens,
        )
        torch.testing.assert_close(input_ids.cpu(), ref_input_ids, rtol=0, atol=0)
        torch.testing.assert_close(logits_indices.cpu(), ref_logits_indices, rtol=0, atol=0)

    def test_prefill_only(self):
        """When seq_len <= prefill_len, kernel should only write logits_indices."""
        num_reqs = 2
        num_spec_steps = 2
        num_new_sampled_tokens = 1
        total_logits = num_reqs * (num_new_sampled_tokens + num_spec_steps)
        num_tokens = total_logits

        input_ids = -torch.ones(num_tokens, dtype=torch.int32, device=self.device)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        last_sampled_tokens = torch.randint(
            0, 100, (num_reqs,), dtype=torch.int32, device=self.device
        )
        seq_lens = torch.full((num_reqs,), 3, dtype=torch.int32, device=self.device)
        prefill_len = torch.full((num_reqs,), 10, dtype=torch.int32, device=self.device)
        draft_tokens = torch.randint(
            0, 100, (num_reqs, num_spec_steps), dtype=torch.int32, device=self.device
        )
        cu_num_logits = torch.arange(
            0, total_logits + 1, num_new_sampled_tokens + num_spec_steps,
            dtype=torch.int64, device=self.device
        )
        query_start_loc = torch.arange(
            0,
            total_logits + 1,
            num_new_sampled_tokens + num_spec_steps,
            dtype=torch.int32,
            device=self.device,
        )
        logits_indices = torch.zeros(total_logits, dtype=torch.int64, device=self.device)

        BLOCK_SIZE = triton.next_power_of_2(num_spec_steps + num_new_sampled_tokens)

        _combine_sampled_and_draft_tokens_kernel[(num_reqs,)](
            input_ids,
            idx_mapping,
            last_sampled_tokens,
            query_start_loc,
            seq_lens,
            prefill_len,
            draft_tokens,
            draft_tokens.stride(0),
            cu_num_logits,
            logits_indices,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_NEW_SAMPLED_TOKENS=num_new_sampled_tokens,
        )
        torch.npu.synchronize()

        ref_input_ids, ref_logits_indices = _combine_sampled_and_draft_tokens_ref(
            -torch.ones(num_tokens, dtype=torch.int32),
            idx_mapping.cpu(),
            last_sampled_tokens.cpu(),
            query_start_loc.cpu(),
            seq_lens.cpu(),
            prefill_len.cpu(),
            draft_tokens.cpu(),
            cu_num_logits.cpu(),
            num_new_sampled_tokens=num_new_sampled_tokens,
        )
        torch.testing.assert_close(input_ids.cpu(), ref_input_ids, rtol=0, atol=0)
        torch.testing.assert_close(
            logits_indices.cpu(), ref_logits_indices, rtol=0, atol=0
        )
