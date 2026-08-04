# vLLM vanilla kernel: _prepare_prefill_inputs_kernel from
# vllm/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py

"""
Precision test for the speculator variant of _prepare_prefill_inputs_kernel.

This kernel is part of the speculative decoding (speculator) flow and has a
different signature from the input_batch.py version. It handles:
  - Shifting target_input_ids by one into draft_input_ids
  - Copying positions from target to draft
  - Writing next_token (from last_sampled or next_prefill_tokens)
  - Handling rejected tokens (num_rejected decreases query_len)
  - Setting last_token_indices
  - Padding buffers for CUDA graph compatibility

Kernel signature:
    _prepare_prefill_inputs_kernel(
        last_token_indices_ptr,         # int32 [num_reqs]
        draft_current_step_ptr,         # int32 scalar
        draft_input_ids_ptr,            # int32 [max_num_tokens]
        draft_positions_ptr,            # int32 [max_num_tokens]
        draft_query_start_loc_ptr,      # int32 [max_num_reqs + 1]
        draft_seq_lens_ptr,             # int32 [max_num_reqs]
        target_input_ids_ptr,           # int32 [max_num_tokens]
        target_positions_ptr,           # int32 [max_num_tokens]
        idx_mapping_ptr,                # int32 [num_reqs]
        last_sampled_ptr,               # int32 [max_num_reqs]
        next_prefill_tokens_ptr,        # int32 [max_num_reqs]
        num_sampled_ptr,                # int32 [num_reqs]
        num_rejected_ptr,               # int32 [num_reqs]
        query_start_loc_ptr,            # int32 [num_reqs + 1]
        seq_lens_ptr,                   # int32 [num_reqs]
        max_num_reqs,
        BLOCK_SIZE: tl.constexpr,
    )
"""

import torch
import pytest

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    _prepare_prefill_inputs_kernel,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _prepare_prefill_inputs_speculator_ref(
    last_token_indices: torch.Tensor,
    draft_input_ids: torch.Tensor,
    draft_positions: torch.Tensor,
    draft_query_start_loc: torch.Tensor,
    draft_seq_lens: torch.Tensor,
    target_input_ids: torch.Tensor,
    target_positions: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_num_reqs: int,
) -> tuple:
    """CPU reference for speculator _prepare_prefill_inputs_kernel.

    Returns (last_token_indices, draft_input_ids, draft_positions,
             draft_query_start_loc, draft_seq_lens, draft_current_step).
    """
    draft_input_ids_out = draft_input_ids.clone()
    draft_positions_out = draft_positions.clone()
    draft_query_start_loc_out = draft_query_start_loc.clone()
    draft_seq_lens_out = draft_seq_lens.clone()
    last_token_indices_out = last_token_indices.clone()
    draft_current_step_out = torch.tensor([0], dtype=torch.int32)

    num_reqs = idx_mapping.shape[0]

    for req_idx in range(num_reqs):
        req_state_idx = idx_mapping[req_idx].item()

        query_start = query_start_loc[req_idx].item()
        query_end = query_start_loc[req_idx + 1].item()
        query_len = query_end - query_start
        seq_len = seq_lens[req_idx].item()

        # Adjust for rejected tokens
        n_rejected = num_rejected[req_idx].item()
        query_len -= n_rejected

        n_sampled = num_sampled[req_idx].item()
        if n_sampled > 0:
            next_token = last_sampled[req_state_idx].item()
        else:
            next_token = next_prefill_tokens[req_state_idx].item()

        # Shift target_input_ids by one into draft_input_ids
        for i in range(1, query_len):
            draft_input_ids_out[query_start + i - 1] = target_input_ids[query_start + i].item()

        last_token_index = query_start + query_len - 1
        last_token_indices_out[req_idx] = last_token_index
        draft_input_ids_out[last_token_index] = next_token

        # Copy positions
        for i in range(query_len):
            draft_positions_out[query_start + i] = target_positions[query_start + i].item()

        # Copy query start locations and sequence lengths
        draft_query_start_loc_out[req_idx] = query_start
        draft_seq_lens_out[req_idx] = seq_len

    # Padding (only last req does this in the kernel)
    draft_query_start_loc_out[num_reqs] = query_end
    for i in range(num_reqs + 1, max_num_reqs + 1):
        draft_query_start_loc_out[i] = query_end
    for i in range(num_reqs, max_num_reqs):
        draft_seq_lens_out[i] = 0
    for i in range(num_reqs, max_num_reqs):
        last_token_indices_out[i] = 0

    return (last_token_indices_out, draft_input_ids_out, draft_positions_out,
            draft_query_start_loc_out, draft_seq_lens_out,
            draft_current_step_out)


class TestPreparePrefillInputsKernelSpeculator:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_reqs", [1, 2, 4])
    @pytest.mark.parametrize("query_len", [4, 16])
    def test_basic_prefill(self, num_reqs, query_len):
        """Basic prefill: shift input ids, copy positions, set next token."""
        max_num_reqs = 8
        max_num_tokens = num_reqs * (query_len + 4)

        target_input_ids = torch.arange(max_num_tokens, dtype=torch.int32,
                                        device=self.device)
        target_positions = torch.arange(max_num_tokens, dtype=torch.int32,
                                        device=self.device)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32,
                                   device=self.device)
        query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32,
                                       device=self.device) * query_len
        seq_lens = torch.full((num_reqs,), 128, dtype=torch.int32,
                              device=self.device)

        num_sampled = torch.ones(num_reqs, dtype=torch.int32,
                                 device=self.device)
        num_rejected = torch.zeros(num_reqs, dtype=torch.int32,
                                   device=self.device)
        last_sampled = torch.full((max_num_reqs,), 999, dtype=torch.int32,
                                  device=self.device)
        next_prefill_tokens = torch.full((max_num_reqs,), 888, dtype=torch.int32,
                                         device=self.device)

        last_token_indices = torch.zeros(num_reqs, dtype=torch.int32,
                                         device=self.device)
        draft_current_step = torch.zeros(1, dtype=torch.int32,
                                         device=self.device)
        draft_input_ids = torch.zeros(max_num_tokens, dtype=torch.int32,
                                      device=self.device)
        draft_positions = torch.zeros(max_num_tokens, dtype=torch.int32,
                                      device=self.device)
        draft_query_start_loc = torch.zeros(max_num_reqs + 1, dtype=torch.int32,
                                            device=self.device)
        draft_seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32,
                                     device=self.device)

        _prepare_prefill_inputs_kernel[(num_reqs,)](
            last_token_indices,
            draft_current_step,
            draft_input_ids,
            draft_positions,
            draft_query_start_loc,
            draft_seq_lens,
            target_input_ids,
            target_positions,
            idx_mapping,
            last_sampled,
            next_prefill_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
            seq_lens,
            max_num_reqs,
            BLOCK_SIZE=1024,
        )
        torch.npu.synchronize()

        (lte_ref, di_ref, dp_ref, dql_ref, dsl_ref, dcs_ref
         ) = _prepare_prefill_inputs_speculator_ref(
            last_token_indices.cpu(),
            torch.zeros(max_num_tokens, dtype=torch.int32),
            torch.zeros(max_num_tokens, dtype=torch.int32),
            torch.zeros(max_num_reqs + 1, dtype=torch.int32),
            torch.zeros(max_num_reqs, dtype=torch.int32),
            target_input_ids.cpu(),
            target_positions.cpu(),
            idx_mapping.cpu(),
            last_sampled.cpu(),
            next_prefill_tokens.cpu(),
            num_sampled.cpu(),
            num_rejected.cpu(),
            query_start_loc.cpu(),
            seq_lens.cpu(),
            max_num_reqs,
        )

        torch.testing.assert_close(last_token_indices.cpu(), lte_ref,
                                   rtol=0, atol=0)
        torch.testing.assert_close(draft_input_ids.cpu(), di_ref,
                                   rtol=0, atol=0)
        torch.testing.assert_close(draft_positions.cpu(), dp_ref,
                                   rtol=0, atol=0)
        torch.testing.assert_close(draft_query_start_loc.cpu(), dql_ref,
                                   rtol=0, atol=0)
        torch.testing.assert_close(draft_seq_lens.cpu(), dsl_ref,
                                   rtol=0, atol=0)
        assert draft_current_step.cpu().item() == 0, (
            "draft_current_step should be reset to 0"
        )

    @pytest.mark.parametrize("num_reqs", [1, 2])
    def test_chunked_prefill_path(self, num_reqs):
        """When num_sampled == 0, use next_prefill_tokens instead of last_sampled."""
        max_num_reqs = 4
        query_len = 4
        max_num_tokens = num_reqs * (query_len + 2)

        target_input_ids = torch.arange(max_num_tokens, dtype=torch.int32,
                                        device=self.device)
        target_positions = torch.arange(max_num_tokens, dtype=torch.int32,
                                        device=self.device)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32,
                                   device=self.device)
        query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32,
                                       device=self.device) * query_len
        seq_lens = torch.full((num_reqs,), 64, dtype=torch.int32,
                              device=self.device)

        num_sampled = torch.zeros(num_reqs, dtype=torch.int32,
                                  device=self.device)
        num_rejected = torch.zeros(num_reqs, dtype=torch.int32,
                                   device=self.device)
        last_sampled = torch.full((max_num_reqs,), 777, dtype=torch.int32,
                                  device=self.device)
        next_prefill_tokens = torch.tensor([100, 200, 300, 400],
                                           dtype=torch.int32,
                                           device=self.device)

        last_token_indices = torch.zeros(num_reqs, dtype=torch.int32,
                                         device=self.device)
        draft_current_step = torch.zeros(1, dtype=torch.int32,
                                         device=self.device)
        draft_input_ids = torch.zeros(max_num_tokens, dtype=torch.int32,
                                      device=self.device)
        draft_positions = torch.zeros(max_num_tokens, dtype=torch.int32,
                                      device=self.device)
        draft_query_start_loc = torch.zeros(max_num_reqs + 1, dtype=torch.int32,
                                            device=self.device)
        draft_seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32,
                                     device=self.device)

        _prepare_prefill_inputs_kernel[(num_reqs,)](
            last_token_indices,
            draft_current_step,
            draft_input_ids,
            draft_positions,
            draft_query_start_loc,
            draft_seq_lens,
            target_input_ids,
            target_positions,
            idx_mapping,
            last_sampled,
            next_prefill_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
            seq_lens,
            max_num_reqs,
            BLOCK_SIZE=1024,
        )
        torch.npu.synchronize()

        # Verify last token gets the next_prefill_token value, not last_sampled
        for req_idx in range(num_reqs):
            query_start = query_start_loc[req_idx].item()
            query_end = query_start_loc[req_idx + 1].item()
            qlen = query_end - query_start
            last_idx = int(query_start + qlen - 1)
            expected_next = next_prefill_tokens[req_idx].item()
            got_next = draft_input_ids[last_idx].item()
            assert got_next == expected_next, (
                f"req {req_idx}: expected next_prefill_token {expected_next}, "
                f"got {got_next}"
            )

        # draft_current_step should be reset to 0
        assert draft_current_step.cpu().item() == 0, (
            "draft_current_step should be reset to 0"
        )

    @pytest.mark.parametrize("num_reqs", [1, 2])
    def test_rejected_tokens_adjustment(self, num_reqs):
        """Test that num_rejected reduces query_len and shift still works."""
        max_num_reqs = 4
        query_len = 8
        max_num_tokens = num_reqs * (query_len + 2)

        target_input_ids = torch.arange(max_num_tokens, dtype=torch.int32,
                                        device=self.device)
        target_positions = torch.arange(max_num_tokens, dtype=torch.int32,
                                        device=self.device)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32,
                                   device=self.device)
        query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32,
                                       device=self.device) * query_len
        seq_lens = torch.full((num_reqs,), 64, dtype=torch.int32,
                              device=self.device)

        num_sampled = torch.ones(num_reqs, dtype=torch.int32,
                                 device=self.device)
        num_rejected = torch.full((num_reqs,), 2, dtype=torch.int32,
                                  device=self.device)
        last_sampled = torch.full((max_num_reqs,), 555, dtype=torch.int32,
                                  device=self.device)
        next_prefill_tokens = torch.full((max_num_reqs,), 444, dtype=torch.int32,
                                         device=self.device)

        last_token_indices = torch.zeros(num_reqs, dtype=torch.int32,
                                         device=self.device)
        draft_current_step = torch.zeros(1, dtype=torch.int32,
                                         device=self.device)
        draft_input_ids = torch.zeros(max_num_tokens, dtype=torch.int32,
                                      device=self.device)
        draft_positions = torch.zeros(max_num_tokens, dtype=torch.int32,
                                      device=self.device)
        draft_query_start_loc = torch.zeros(max_num_reqs + 1, dtype=torch.int32,
                                            device=self.device)
        draft_seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32,
                                     device=self.device)

        _prepare_prefill_inputs_kernel[(num_reqs,)](
            last_token_indices,
            draft_current_step,
            draft_input_ids,
            draft_positions,
            draft_query_start_loc,
            draft_seq_lens,
            target_input_ids,
            target_positions,
            idx_mapping,
            last_sampled,
            next_prefill_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
            seq_lens,
            max_num_reqs,
            BLOCK_SIZE=1024,
        )
        torch.npu.synchronize()

        (lte_ref, di_ref, dp_ref, dql_ref, dsl_ref, dcs_ref
         ) = _prepare_prefill_inputs_speculator_ref(
            torch.zeros(num_reqs, dtype=torch.int32),
            torch.zeros(max_num_tokens, dtype=torch.int32),
            torch.zeros(max_num_tokens, dtype=torch.int32),
            torch.zeros(max_num_reqs + 1, dtype=torch.int32),
            torch.zeros(max_num_reqs, dtype=torch.int32),
            target_input_ids.cpu(),
            target_positions.cpu(),
            idx_mapping.cpu(),
            last_sampled.cpu(),
            next_prefill_tokens.cpu(),
            num_sampled.cpu(),
            num_rejected.cpu(),
            query_start_loc.cpu(),
            seq_lens.cpu(),
            max_num_reqs,
        )

        torch.testing.assert_close(last_token_indices.cpu(), lte_ref,
                                   rtol=0, atol=0)
        torch.testing.assert_close(draft_input_ids.cpu(), di_ref,
                                   rtol=0, atol=0)
        torch.testing.assert_close(draft_positions.cpu(), dp_ref,
                                   rtol=0, atol=0)
        torch.testing.assert_close(draft_query_start_loc.cpu(), dql_ref,
                                   rtol=0, atol=0)
        torch.testing.assert_close(draft_seq_lens.cpu(), dsl_ref,
                                   rtol=0, atol=0)

    def test_padding_for_cuda_graphs(self):
        """Verify padding of seq_lens, query_start_loc, and last_token_indices."""
        num_reqs = 1
        max_num_reqs = 4
        query_len = 4
        max_num_tokens = 16

        target_input_ids = torch.arange(max_num_tokens, dtype=torch.int32,
                                        device=self.device)
        target_positions = torch.arange(max_num_tokens, dtype=torch.int32,
                                        device=self.device)
        idx_mapping = torch.zeros(num_reqs, dtype=torch.int32,
                                  device=self.device)
        query_start_loc = torch.tensor([0, 4], dtype=torch.int32,
                                       device=self.device)
        seq_lens = torch.tensor([50], dtype=torch.int32, device=self.device)

        num_sampled = torch.ones(num_reqs, dtype=torch.int32,
                                 device=self.device)
        num_rejected = torch.zeros(num_reqs, dtype=torch.int32,
                                   device=self.device)
        last_sampled = torch.full((max_num_reqs,), 333, dtype=torch.int32,
                                  device=self.device)
        next_prefill_tokens = torch.full((max_num_reqs,), 222, dtype=torch.int32,
                                         device=self.device)

        last_token_indices = torch.zeros(num_reqs, dtype=torch.int32,
                                         device=self.device)
        draft_current_step = torch.ones(1, dtype=torch.int32,
                                        device=self.device)
        draft_input_ids = torch.full((max_num_tokens,), -1, dtype=torch.int32,
                                     device=self.device)
        draft_positions = torch.full((max_num_tokens,), -1, dtype=torch.int32,
                                     device=self.device)
        draft_query_start_loc = torch.full((max_num_reqs + 1,), -1,
                                           dtype=torch.int32, device=self.device)
        draft_seq_lens = torch.full((max_num_reqs,), -1, dtype=torch.int32,
                                    device=self.device)

        _prepare_prefill_inputs_kernel[(num_reqs,)](
            last_token_indices,
            draft_current_step,
            draft_input_ids,
            draft_positions,
            draft_query_start_loc,
            draft_seq_lens,
            target_input_ids,
            target_positions,
            idx_mapping,
            last_sampled,
            next_prefill_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
            seq_lens,
            max_num_reqs,
            BLOCK_SIZE=1024,
        )
        torch.npu.synchronize()

        # query_end = 4
        # Padding: positions [num_reqs .. max_num_reqs] should be query_end
        for i in range(num_reqs + 1, max_num_reqs + 1):
            assert draft_query_start_loc[i].item() == 4, (
                f"draft_query_start_loc[{i}] should be padded to 4, "
                f"got {draft_query_start_loc[i].item()}"
            )
        # Padding: positions [num_reqs .. max_num_reqs) should be 0
        for i in range(num_reqs, max_num_reqs):
            assert draft_seq_lens[i].item() == 0, (
                f"draft_seq_lens[{i}] should be padded to 0, "
                f"got {draft_seq_lens[i].item()}"
            )
            assert last_token_indices[i].item() == 0, (
                f"last_token_indices[{i}] should be padded to 0, "
                f"got {last_token_indices[i].item()}"
            )
        # draft_current_step should be reset to 0
        assert draft_current_step.cpu().item() == 0, (
            "draft_current_step should be reset to 0"
        )
