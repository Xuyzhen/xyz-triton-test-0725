# GENERATED STRICT UT. Source: accuracy_test/codex/missing_accuracy_tests/test_combine_sampled_and_draft_tokens_kernel.py
# Do not edit mechanically; update the reviewed Codex source or strict generator.
from accuracy_test.strict_ut.runtime_npu import STRICT_DEVICE as _STRICT_DEVICE
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
from accuracy_test.strict_ut.runtime_npu import init_device_properties_triton


@triton.jit
def _combine_sampled_and_draft_tokens_required_constexpr(
    input_ids_ptr,
    idx_mapping_ptr,
    last_sampled_tokens_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    prefill_len_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    cu_num_logits_ptr,
    logits_indices_ptr,
    BLOCK_SIZE: tl.constexpr,
    NUM_NEW_SAMPLED_TOKENS: tl.constexpr,
):
    """Test-only copy of the upstream kernel with no constexpr default."""
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    cu_num_logits_start = tl.load(cu_num_logits_ptr + batch_idx)
    cu_num_logits_end = tl.load(cu_num_logits_ptr + batch_idx + 1)
    num_logits = cu_num_logits_end - cu_num_logits_start
    num_draft_tokens = num_logits - NUM_NEW_SAMPLED_TOKENS

    block = tl.arange(0, BLOCK_SIZE)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    logits_start = query_end - num_logits
    tl.store(
        logits_indices_ptr + cu_num_logits_start + block,
        logits_start + block,
        mask=block < num_logits,
    )

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    if seq_len <= prefill_len:
        return

    if NUM_NEW_SAMPLED_TOKENS > 0:
        last_token_id = tl.load(last_sampled_tokens_ptr + req_state_idx)
        tl.store(input_ids_ptr + query_end - num_logits, last_token_id)

    if num_draft_tokens > 0:
        mask = block < num_draft_tokens
        draft_tokens = tl.load(
            draft_tokens_ptr + req_state_idx * draft_tokens_stride + block,
            mask=mask,
        )
        tl.store(
            input_ids_ptr + query_end - num_draft_tokens + block,
            draft_tokens,
            mask=mask,
        )


def _launch_combine_kernel(grid, args, block_size, num_new_sampled_tokens):
    if num_new_sampled_tokens == 1:
        _combine_sampled_and_draft_tokens_kernel[grid](
            *args,
            BLOCK_SIZE=block_size,
        )
    else:
        _combine_sampled_and_draft_tokens_required_constexpr[grid](
            *args,
            BLOCK_SIZE=block_size,
            NUM_NEW_SAMPLED_TOKENS=num_new_sampled_tokens,
        )


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

        _launch_combine_kernel(
            (num_reqs,),
            (
                input_ids, idx_mapping, last_sampled_tokens, query_start_loc,
                seq_lens, prefill_len, draft_tokens, draft_tokens.stride(0),
                cu_num_logits, logits_indices,
            ),
            BLOCK_SIZE,
            num_new_sampled_tokens,
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

        _launch_combine_kernel(
            (num_reqs,),
            (
                input_ids, idx_mapping, last_sampled_tokens, query_start_loc,
                seq_lens, prefill_len, draft_tokens, draft_tokens.stride(0),
                cu_num_logits, logits_indices,
            ),
            BLOCK_SIZE,
            num_new_sampled_tokens,
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
