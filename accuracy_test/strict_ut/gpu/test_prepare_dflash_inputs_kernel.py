# GENERATED STRICT UT. Source: accuracy_test/codex/existing_accuracy_tests/from_vllm/test_prepare_dflash_inputs_kernel.py
# Do not edit mechanically; update the reviewed Codex source or strict generator.
from accuracy_test.strict_ut.runtime_gpu import STRICT_DEVICE as _STRICT_DEVICE
# Standalone Ascend A3 adaptation of an upstream vLLM accuracy path.
# Accuracy UT source: vllm/tests/v1/spec_decode/test_dflash_lookahead.py
# Kernel source: vllm/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
# Coverage: _prepare_dflash_inputs_kernel

# vLLM vanilla kernel: _prepare_dflash_inputs_kernel from
# vllm/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py

"""
Precision test for _prepare_dflash_inputs_kernel.

Kernel signature:
    _prepare_dflash_inputs_kernel(
        out_input_ids_ptr,               # int32 output
        out_query_positions_ptr,         # int64 output
        out_query_start_loc_ptr,         # int32 output [max_num_reqs + 1]
        out_seq_lens_ptr,                # int32 output [max_num_reqs]
        out_query_slot_mapping_ptr,      # int32 output
        out_context_positions_ptr,       # int64 output
        out_context_slot_mapping_ptr,    # int32 output
        out_sample_indices_ptr,          # int32 output
        out_sample_pos_ptr,              # int64 output
        out_sample_idx_mapping_ptr,      # int32 output
        out_temperature_ptr,             # float32 output [max_num_reqs]
        out_seeds_ptr,                   # int64 output [max_num_reqs]
        target_positions_ptr,            # int64 input [num_tokens]
        target_query_start_loc_ptr,      # int32 input [num_reqs + 1]
        idx_mapping_ptr,                 # int32 input [num_reqs]
        last_sampled_ptr,                # int32 input [max_num_reqs]
        next_prefill_tokens_ptr,         # int32 input [max_num_reqs]
        num_sampled_ptr,                 # int32 input [num_reqs]
        num_rejected_ptr,                # int32 input [num_reqs]
        temperature_ptr,                 # float32 input [max_num_reqs]
        seeds_ptr,                       # int64 input [max_num_reqs]
        block_table_ptr,                 # int64 input [max_num_reqs, max_num_blocks]
        block_table_stride,              # stride(0) of block_table
        parallel_drafting_token_id,      # scalar int32
        block_size,                      # scalar
        num_query_per_req,               # scalar
        num_speculative_steps,           # scalar
        max_num_reqs,                    # scalar
        max_num_tokens,                  # scalar
        max_model_len,                   # scalar
        cp_rank,                         # scalar (CP rank, 0 when CP unused)
        SAMPLE_FROM_ANCHOR: tl.constexpr,
        PAD_SLOT_ID: tl.constexpr,
        CP_SIZE: tl.constexpr,           # 1 when CP unused
        CP_INTERLEAVE: tl.constexpr,     # False when CP unused
        BLOCK_SIZE: tl.constexpr,
    )

Prepares DFlash speculative decoding inputs: context positions/slots, query
positions/slots/input_ids, and sample indices/positions/idx_mapping.
"""

import torch
import pytest

from vllm.triton_utils import tl, triton
try:
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
        _prepare_dflash_inputs_kernel,
    )
except (ImportError, ModuleNotFoundError) as exc:
    pytest.skip(
        "installed vLLM does not provide the legacy dflash speculator module; "
        f"precision was not tested: {exc}",
        allow_module_level=True,
    )
from vllm.v1.attention.backends.utils import PAD_SLOT_ID as PAD_SLOT_ID_CONST
from accuracy_test.strict_ut.runtime_gpu import init_device_properties_triton


class TestPrepareDFlashInputsKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("cuda")

    @pytest.mark.parametrize("num_reqs", [1, 2])
    @pytest.mark.parametrize("num_ctx", [2, 4])
    @pytest.mark.parametrize("num_query_per_req_val", [2, 3, 5])
    def test_prepare_dflash_inputs(self, num_reqs, num_ctx, num_query_per_req_val):
        """Compare kernel output with CPU reference for basic fields."""
        num_speculative_steps = num_query_per_req_val  # typically same
        max_num_reqs = 8
        max_num_tokens = 32
        max_model_len = 128
        block_size = 16
        parallel_drafting_token_id = 0
        sample_from_anchor = False
        max_blocks = 8
        num_tokens = num_reqs * num_ctx  # target tokens

        # Input tensors
        target_positions = torch.arange(num_tokens, dtype=torch.int64, device=self.device)
        target_query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32, device=self.device) * num_ctx
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        last_sampled = torch.tensor([100, 200] + [0] * (max_num_reqs - 2), dtype=torch.int32, device=self.device)
        next_prefill_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=self.device)
        num_sampled = torch.tensor([1, 0], dtype=torch.int32, device=self.device)  # req0 has sampled, req1 is prefill
        num_rejected = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        block_table = torch.randint(0, 100, (max_num_reqs, max_blocks), dtype=torch.int64, device=self.device)

        # Sampling state copied per request (pass-through at req_state_idx).
        temperature = torch.rand(max_num_reqs, dtype=torch.float32, device=self.device) * 1.5 + 0.5
        seeds = torch.randint(0, 2**31, (max_num_reqs,), dtype=torch.int64, device=self.device)

        total_context = num_tokens
        total_query = num_reqs * num_query_per_req_val

        # Output tensors
        out_input_ids = torch.full((total_query,), -1, dtype=torch.int32, device=self.device)
        out_query_positions = torch.full((total_query,), -1, dtype=torch.int64, device=self.device)
        out_query_start_loc = torch.full((max_num_reqs + 1,), -1, dtype=torch.int32, device=self.device)
        out_seq_lens = torch.full((max_num_reqs,), -1, dtype=torch.int32, device=self.device)
        out_query_slot_mapping = torch.full((max_num_tokens,), -2, dtype=torch.int32, device=self.device)
        out_context_positions = torch.full((num_tokens,), -1, dtype=torch.int64, device=self.device)
        out_context_slot_mapping = torch.full((num_tokens,), -1, dtype=torch.int32, device=self.device)
        out_sample_indices = torch.full((num_reqs * num_speculative_steps,), -1, dtype=torch.int32, device=self.device)
        out_sample_pos = torch.full((num_reqs * num_speculative_steps,), -1, dtype=torch.int64, device=self.device)
        out_sample_idx_mapping = torch.full((num_reqs * num_speculative_steps,), -1, dtype=torch.int32, device=self.device)
        out_temperature = torch.full((max_num_reqs,), -1.0, dtype=torch.float32, device=self.device)
        out_seeds = torch.full((max_num_reqs,), -1, dtype=torch.int64, device=self.device)

        max_tokens_per_req = num_ctx + num_query_per_req_val
        blk_size = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))
        num_blocks = triton.cdiv(max_tokens_per_req, blk_size)

        _prepare_dflash_inputs_kernel[(num_reqs, num_blocks)](
            out_input_ids,
            out_query_positions,
            out_query_start_loc,
            out_seq_lens,
            out_query_slot_mapping,
            out_context_positions,
            out_context_slot_mapping,
            out_sample_indices,
            out_sample_pos,
            out_sample_idx_mapping,
            out_temperature,
            out_seeds,
            target_positions,
            target_query_start_loc,
            idx_mapping,
            last_sampled,
            next_prefill_tokens,
            num_sampled,
            num_rejected,
            temperature,
            seeds,
            block_table,
            block_table.stride(0),
            parallel_drafting_token_id,
            block_size,
            num_query_per_req_val,
            num_speculative_steps,
            max_num_reqs,
            max_num_tokens,
            max_model_len,
            cp_rank=0,
            SAMPLE_FROM_ANCHOR=sample_from_anchor,
            PAD_SLOT_ID=PAD_SLOT_ID_CONST,
            CP_SIZE=1,
            CP_INTERLEAVE=False,
            BLOCK_SIZE=blk_size,
        )
        torch.cuda.synchronize()

        # CPU reference checks
        for req_idx in range(num_reqs):
            rs_idx = idx_mapping[req_idx].item()
            qs = target_query_start_loc[req_idx].item()
            qe = target_query_start_loc[req_idx + 1].item()
            ctx_len = qe - qs

            # Query start_loc
            assert out_query_start_loc[req_idx].item() == req_idx * num_query_per_req_val

            # Bonus token from last_sampled or next_prefill_tokens
            if num_sampled[req_idx].item() > 0:
                bonus = last_sampled[rs_idx].item()
            else:
                bonus = next_prefill_tokens[rs_idx].item()
            bonus_idx = req_idx * num_query_per_req_val
            assert out_input_ids[bonus_idx].item() == bonus, (
                f"req {req_idx}: bonus token mismatch"
            )

            # Non-bonus query tokens use parallel_drafting_token_id
            for qoff in range(1, num_query_per_req_val):
                qi = req_idx * num_query_per_req_val + qoff
                assert out_input_ids[qi].item() == parallel_drafting_token_id

            # seq_lens check (full context + query)
            last_valid_pos = target_positions[qe + (-1) - num_rejected[req_idx].item()].item()
            expected_seq_len = last_valid_pos + 1 + num_query_per_req_val
            assert out_seq_lens[req_idx].item() == expected_seq_len, (
                f"req {req_idx}: expected seq_len {expected_seq_len}, "
                f"got {out_seq_lens[req_idx].item()}"
            )

            # Context positions should match target positions
            for j in range(ctx_len):
                assert out_context_positions[qs + j].item() == target_positions[qs + j].item()

            # Sampling state pass-through at req_state_idx
            assert out_temperature[rs_idx].item() == temperature[rs_idx].item(), (
                f"req {req_idx}: temperature pass-through mismatch"
            )
            assert out_seeds[rs_idx].item() == seeds[rs_idx].item(), (
                f"req {req_idx}: seed pass-through mismatch"
            )

            # Sample indices/pos/idx_mapping
            # When not SAMPLE_FROM_ANCHOR: sample from query_off >= 1
            for qoff in range(1, num_query_per_req_val):
                qi = req_idx * num_query_per_req_val + qoff
                sidx = req_idx * num_speculative_steps + (qoff - 1)
                expected_sample_pos = last_valid_pos + 1 + qoff
                assert out_sample_indices[sidx].item() == qi, (
                    f"req {req_idx}, qoff {qoff}: sample index mismatch"
                )
                assert out_sample_pos[sidx].item() == expected_sample_pos, (
                    f"req {req_idx}, qoff {qoff}: sample pos mismatch"
                )
                assert out_sample_idx_mapping[sidx].item() == rs_idx

    def test_cuda_graph_padding(self):
        """Verify padding in the last thread block (block_idx == 0, req_idx == num_reqs - 1)."""
        num_reqs = 1
        num_ctx = 2
        num_query_per_req_val = 2
        num_speculative_steps = 2
        max_num_reqs = 4
        max_num_tokens = 16
        max_model_len = 128
        block_size = 16
        parallel_drafting_token_id = 0
        sample_from_anchor = False
        max_blocks = 4

        target_positions = torch.arange(num_ctx, dtype=torch.int64, device=self.device)
        target_query_start_loc = torch.tensor([0, num_ctx], dtype=torch.int32, device=self.device)
        idx_mapping = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        last_sampled = torch.zeros(max_num_reqs, dtype=torch.int32, device=self.device)
        next_prefill_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=self.device)
        num_sampled = torch.ones(num_reqs, dtype=torch.int32, device=self.device)
        num_rejected = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        block_table = torch.zeros((max_num_reqs, max_blocks), dtype=torch.int64, device=self.device)

        # Sampling state (pass-through at req_state_idx)
        temperature = torch.full((max_num_reqs,), 0.7, dtype=torch.float32, device=self.device)
        seeds = torch.arange(max_num_reqs, dtype=torch.int64, device=self.device) + 1000

        total_query = num_reqs * num_query_per_req_val

        out_input_ids = torch.full((total_query,), -1, dtype=torch.int32, device=self.device)
        out_query_positions = torch.full((total_query,), -1, dtype=torch.int64, device=self.device)
        out_query_start_loc = torch.full((max_num_reqs + 1,), -1, dtype=torch.int32, device=self.device)
        out_seq_lens = torch.full((max_num_reqs,), -1, dtype=torch.int32, device=self.device)
        out_query_slot_mapping = torch.full((max_num_tokens,), -2, dtype=torch.int32, device=self.device)
        out_context_positions = torch.full((num_ctx,), -1, dtype=torch.int64, device=self.device)
        out_context_slot_mapping = torch.full((num_ctx,), -1, dtype=torch.int32, device=self.device)
        out_sample_indices = torch.full((num_reqs * num_speculative_steps,), -1, dtype=torch.int32, device=self.device)
        out_sample_pos = torch.full((num_reqs * num_speculative_steps,), -1, dtype=torch.int64, device=self.device)
        out_sample_idx_mapping = torch.full((num_reqs * num_speculative_steps,), -1, dtype=torch.int32, device=self.device)
        out_temperature = torch.full((max_num_reqs,), -1.0, dtype=torch.float32, device=self.device)
        out_seeds = torch.full((max_num_reqs,), -1, dtype=torch.int64, device=self.device)

        max_tokens_per_req = num_ctx + num_query_per_req_val
        blk_size = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))
        num_blocks = triton.cdiv(max_tokens_per_req, blk_size)

        _prepare_dflash_inputs_kernel[(num_reqs, num_blocks)](
            out_input_ids,
            out_query_positions,
            out_query_start_loc,
            out_seq_lens,
            out_query_slot_mapping,
            out_context_positions,
            out_context_slot_mapping,
            out_sample_indices,
            out_sample_pos,
            out_sample_idx_mapping,
            out_temperature,
            out_seeds,
            target_positions,
            target_query_start_loc,
            idx_mapping,
            last_sampled,
            next_prefill_tokens,
            num_sampled,
            num_rejected,
            temperature,
            seeds,
            block_table,
            block_table.stride(0),
            parallel_drafting_token_id,
            block_size,
            num_query_per_req_val,
            num_speculative_steps,
            max_num_reqs,
            max_num_tokens,
            max_model_len,
            cp_rank=0,
            SAMPLE_FROM_ANCHOR=sample_from_anchor,
            PAD_SLOT_ID=PAD_SLOT_ID_CONST,
            CP_SIZE=1,
            CP_INTERLEAVE=False,
            BLOCK_SIZE=blk_size,
        )
        torch.cuda.synchronize()

        # Verify padding for query_start_loc
        last_query_end = num_reqs * num_query_per_req_val  # 2
        for i in range(num_reqs, max_num_reqs + 1):
            assert out_query_start_loc[i].item() == last_query_end, (
                f"Padded query_start_loc[{i}] should be {last_query_end}"
            )
        # Padded seq_lens should be 0
        for i in range(num_reqs, max_num_reqs):
            assert out_seq_lens[i].item() == 0, f"Padded seq_lens[{i}] should be 0"
        # Padded sampling state must be untouched (still sentinel)
        assert out_temperature[0].item() == 0.7
        assert out_seeds[0].item() == 1000
        for i in range(num_reqs, max_num_reqs):
            assert out_temperature[i].item() == -1.0, f"Padded temperature[{i}] untouched"
            assert out_seeds[i].item() == -1, f"Padded seeds[{i}] untouched"
        # Padded sample slots
        pad_start = num_reqs * num_speculative_steps
        for i in range(pad_start, max_num_reqs * num_speculative_steps):
            assert out_sample_indices[i].item() == 0
            assert out_sample_pos[i].item() == 0
            assert out_sample_idx_mapping[i].item() == -1
        # Padded query slot mappings
        q_pad_start = num_reqs * num_query_per_req_val
        for i in range(q_pad_start, max_num_tokens):
            assert out_query_slot_mapping[i].item() == PAD_SLOT_ID_CONST

    def test_sample_from_anchor(self):
        """When SAMPLE_FROM_ANCHOR, sample at every query position (query_off >= 0)."""
        num_reqs = 1
        num_ctx = 2
        num_query_per_req_val = 3
        num_speculative_steps = 3
        max_num_reqs = 4
        max_num_tokens = 16
        max_model_len = 128
        block_size = 16
        parallel_drafting_token_id = 0
        sample_from_anchor = True
        max_blocks = 4

        target_positions = torch.arange(num_ctx, dtype=torch.int64, device=self.device)
        target_query_start_loc = torch.tensor([0, num_ctx], dtype=torch.int32, device=self.device)
        idx_mapping = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        last_sampled = torch.tensor([42], dtype=torch.int32, device=self.device)
        next_prefill_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=self.device)
        num_sampled = torch.ones(num_reqs, dtype=torch.int32, device=self.device)
        num_rejected = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        block_table = torch.zeros((max_num_reqs, max_blocks), dtype=torch.int64, device=self.device)

        # Sampling state (pass-through at req_state_idx)
        temperature = torch.full((max_num_reqs,), 1.1, dtype=torch.float32, device=self.device)
        seeds = torch.full((max_num_reqs,), 77, dtype=torch.int64, device=self.device)

        total_query = num_reqs * num_query_per_req_val

        out_input_ids = torch.full((total_query,), -1, dtype=torch.int32, device=self.device)
        out_query_positions = torch.full((total_query,), -1, dtype=torch.int64, device=self.device)
        out_query_start_loc = torch.full((max_num_reqs + 1,), -1, dtype=torch.int32, device=self.device)
        out_seq_lens = torch.full((max_num_reqs,), -1, dtype=torch.int32, device=self.device)
        out_query_slot_mapping = torch.full((max_num_tokens,), -2, dtype=torch.int32, device=self.device)
        out_context_positions = torch.full((num_ctx,), -1, dtype=torch.int64, device=self.device)
        out_context_slot_mapping = torch.full((num_ctx,), -1, dtype=torch.int32, device=self.device)
        out_sample_indices = torch.full((num_reqs * num_speculative_steps,), -1, dtype=torch.int32, device=self.device)
        out_sample_pos = torch.full((num_reqs * num_speculative_steps,), -1, dtype=torch.int64, device=self.device)
        out_sample_idx_mapping = torch.full((num_reqs * num_speculative_steps,), -1, dtype=torch.int32, device=self.device)
        out_temperature = torch.full((max_num_reqs,), -1.0, dtype=torch.float32, device=self.device)
        out_seeds = torch.full((max_num_reqs,), -1, dtype=torch.int64, device=self.device)

        max_tokens_per_req = num_ctx + num_query_per_req_val
        blk_size = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))
        num_blocks = triton.cdiv(max_tokens_per_req, blk_size)

        _prepare_dflash_inputs_kernel[(num_reqs, num_blocks)](
            out_input_ids,
            out_query_positions,
            out_query_start_loc,
            out_seq_lens,
            out_query_slot_mapping,
            out_context_positions,
            out_context_slot_mapping,
            out_sample_indices,
            out_sample_pos,
            out_sample_idx_mapping,
            out_temperature,
            out_seeds,
            target_positions,
            target_query_start_loc,
            idx_mapping,
            last_sampled,
            next_prefill_tokens,
            num_sampled,
            num_rejected,
            temperature,
            seeds,
            block_table,
            block_table.stride(0),
            parallel_drafting_token_id,
            block_size,
            num_query_per_req_val,
            num_speculative_steps,
            max_num_reqs,
            max_num_tokens,
            max_model_len,
            cp_rank=0,
            SAMPLE_FROM_ANCHOR=sample_from_anchor,
            PAD_SLOT_ID=PAD_SLOT_ID_CONST,
            CP_SIZE=1,
            CP_INTERLEAVE=False,
            BLOCK_SIZE=blk_size,
        )
        torch.cuda.synchronize()

        # With SAMPLE_FROM_ANCHOR: sample_off=0, so all query positions get sample entries
        last_valid_pos = target_positions[num_ctx - 1].item()
        for qoff in range(num_query_per_req_val):
            qi = qoff
            sidx = qoff  # req_idx=0, sample_off=0
            expected_sample_pos = last_valid_pos + 1 + qoff + 1  # query_pos + 1
            assert out_sample_indices[sidx].item() == qi
            assert out_sample_pos[sidx].item() == expected_sample_pos, (
                f"qoff {qoff}: expected sample_pos {expected_sample_pos}, "
                f"got {out_sample_pos[sidx].item()}"
            )
