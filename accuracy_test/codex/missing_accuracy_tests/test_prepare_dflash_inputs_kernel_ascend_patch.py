# vLLM-Ascend patched kernel: _prepare_dflash_inputs_kernel_ascend from
# vllm-ascend/vllm_ascend/worker/v2/spec_decode/dflash/speculator.py:140
# PATCH NOTE: This is an Ascend NPU adaptation of the original vLLM Triton kernel

"""
Precision test for patched _prepare_dflash_inputs_kernel_ascend (Ascend NPU version).

Patch differences vs original vllm _prepare_dflash_inputs_kernel:
- Renamed from _prepare_dflash_inputs_kernel to _prepare_dflash_inputs_kernel_ascend
- Uses tl.minimum for bounds clamping instead of min()
- Uses tl.int64 for block_id cast
- Uses clamp of query_pos to max_model_len - 1: clamped_query_pos = tl.minimum(query_pos, max_model_len - 1)
- Uses tl.minimum(block_num, block_table_stride - 1) for block_num safety
- Simplified loop structure using tl.range
- Returns -1 for padded sample idx mapping (OOB guard)

Kernel signature:
    _prepare_dflash_inputs_kernel_ascend(
        # Outputs
        out_input_ids_ptr,              # [max_num_tokens] input token IDs
        out_query_positions_ptr,        # [max_num_tokens] query positions
        out_query_start_loc_ptr,        # [max_num_reqs + 1] query start locations
        out_seq_lens_ptr,               # [max_num_reqs] sequence lengths
        out_query_slot_mapping_ptr,     # [max_num_tokens] query slot mappings
        out_context_positions_ptr,      # [max_num_tokens] context positions
        out_context_slot_mapping_ptr,   # [max_num_tokens] context slot mappings
        out_sample_indices_ptr,         # [max_num_reqs * num_spec_steps] sample indices
        out_sample_pos_ptr,             # [max_num_reqs * num_spec_steps] sample positions
        out_sample_idx_mapping_ptr,     # [max_num_reqs * num_spec_steps] sample idx mapping
        # Inputs from target batch
        target_positions_ptr,           # [num_tokens] target positions
        target_query_start_loc_ptr,     # [num_reqs + 1] target query start
        idx_mapping_ptr,                # [num_reqs] request index mapping
        last_sampled_ptr,               # [max_num_reqs] last sampled token ID
        next_prefill_tokens_ptr,        # [max_num_reqs] next prefill tokens
        num_sampled_ptr,                # [num_reqs] num sampled
        num_rejected_ptr,               # [num_reqs] num rejected
        # Block table
        block_table_ptr,                # [num_reqs, max_blocks] block table
        block_table_stride,             # stride(0) of block_table
        # Scalars
        parallel_drafting_token_id,     # token ID for parallel drafting
        block_size,                     # KV cache block size
        num_query_per_req,              # queries per request
        num_speculative_steps,          # speculative steps
        max_num_reqs,                   # max requests
        max_num_tokens,                 # max tokens
        max_model_len,                  # max model length
        SAMPLE_FROM_ANCHOR: tl.constexpr,   # whether to sample from anchor
        PAD_SLOT_ID: tl.constexpr,      # padding slot ID
        BLOCK_SIZE: tl.constexpr,       # triton block size
    )

Prepares draft inputs for DFlash speculative decoding on Ascend NPU.
Monkey-patches the original _prepare_dflash_inputs_kernel.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

import pytest

_prepare_dflash_inputs_kernel_ascend = None
_modern_dflash_inputs_kernel = None
_dflash_import_errors = []
try:
    from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
        _prepare_dflash_inputs_kernel_ascend,
    )
except (ImportError, ModuleNotFoundError) as exc:
    _dflash_import_errors.append(f"legacy worker-v2: {exc}")

if _prepare_dflash_inputs_kernel_ascend is None:
    try:
        from vllm_ascend.ops.triton.spec_decode.utils import (
            copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid
            as _modern_dflash_inputs_kernel,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        _dflash_import_errors.append(f"modern spec-decode utils: {exc}")

PAD_SLOT_ID = -1


def _dflash_environment_diagnostics():
    """Describe installed packages without importing unrelated modules."""
    import importlib.metadata
    from pathlib import Path

    import vllm
    import vllm_ascend

    def package_version(distribution, module):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return getattr(module, "__version__", "unknown")

    ascend_root = Path(vllm_ascend.__file__).resolve().parent
    candidates = []
    for path in ascend_root.rglob("*.py"):
        relative = path.relative_to(ascend_root)
        relative_text = str(relative).replace("\\", "/")
        if "dflash" in relative_text.lower() or (
            "spec_decode" in relative.parts and path.name == "utils.py"
        ):
            candidates.append(relative_text)

    return "\n".join(
        [
            f"vllm={package_version('vllm', vllm)} ({vllm.__file__})",
            "vllm-ascend="
            f"{package_version('vllm-ascend', vllm_ascend)} "
            f"({vllm_ascend.__file__})",
            "candidate files=" + (", ".join(sorted(candidates)) or "<none>"),
            "import errors=" + "; ".join(_dflash_import_errors),
        ]
    )


def _prepare_dflash_inputs_ref(
    # Outputs
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
    # Inputs
    target_positions,
    target_query_start_loc,
    idx_mapping,
    last_sampled,
    next_prefill_tokens,
    num_sampled,
    num_rejected,
    block_table,
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_steps,
    max_num_reqs,
    max_num_tokens,
    max_model_len,
    SAMPLE_FROM_ANCHOR=False,
):
    """CPU reference for _prepare_dflash_inputs_kernel_ascend."""
    num_reqs = idx_mapping.shape[0]

    for req_idx in range(num_reqs):
        req_state_idx = idx_mapping[req_idx].item()
        ctx_start = target_query_start_loc[req_idx].item()
        ctx_end = target_query_start_loc[req_idx + 1].item()
        num_ctx = ctx_end - ctx_start
        nrejected = num_rejected[req_idx].item()
        valid_ctx_end = ctx_end - nrejected
        nsampled = num_sampled[req_idx].item()

        if nsampled > 0:
            bonus_token = last_sampled[req_state_idx].item()
        else:
            bonus_token = next_prefill_tokens[req_state_idx].item()

        last_valid_pos = target_positions[valid_ctx_end - 1].item()
        query_base = req_idx * num_query_per_req

        # Context
        for j in range(num_ctx):
            ctx_pos_idx = ctx_start + j
            ctx_pos = target_positions[ctx_pos_idx].item()
            ctx_block_num = ctx_pos // block_size
            ctx_block_num = min(ctx_block_num, block_table.shape[1] - 1)
            ctx_block_id = block_table[req_idx, ctx_block_num]
            ctx_slot = ctx_block_id * block_size + (ctx_pos % block_size)
            out_context_positions[ctx_pos_idx] = ctx_pos
            out_context_slot_mapping[ctx_pos_idx] = ctx_slot

        # Queries
        for q_off in range(num_query_per_req):
            query_pos = last_valid_pos + 1 + q_off
            query_idx = query_base + q_off
            if q_off == 0:
                input_id = bonus_token
            else:
                input_id = parallel_drafting_token_id

            q_block_num = query_pos // block_size
            q_block_num = min(q_block_num, block_table.shape[1] - 1)
            q_block_id = block_table[req_idx, q_block_num]
            q_slot = q_block_id * block_size + (query_pos % block_size)

            out_input_ids[query_idx] = input_id
            clamped_query_pos = min(query_pos, max_model_len - 1)
            out_query_positions[query_idx] = clamped_query_pos
            out_query_slot_mapping[query_idx] = q_slot

        sample_off = 0 if SAMPLE_FROM_ANCHOR else 1
        for s_off in range(sample_off, num_query_per_req):
            sample_idx = req_idx * num_speculative_steps + (s_off - sample_off)
            query_idx = query_base + s_off
            query_pos = last_valid_pos + 1 + s_off
            sample_pos = query_pos + 1 if SAMPLE_FROM_ANCHOR else query_pos
            out_sample_indices[sample_idx] = query_idx
            out_sample_pos[sample_idx] = sample_pos
            out_sample_idx_mapping[sample_idx] = req_state_idx

        out_query_start_loc[req_idx] = query_base
        out_seq_lens[req_idx] = last_valid_pos + 1 + num_query_per_req

    # Padding
    if num_reqs > 0:
        last_query_end = num_reqs * num_query_per_req
        for i in range(num_reqs, max_num_reqs + 1):
            out_query_start_loc[i] = last_query_end
        for i in range(num_reqs, max_num_reqs):
            out_seq_lens[i] = 0
        pad_start = num_reqs * num_speculative_steps
        pad_end = max_num_reqs * num_speculative_steps
        for i in range(pad_start, pad_end):
            out_sample_indices[i] = 0
            out_sample_pos[i] = 0
            out_sample_idx_mapping[i] = -1
        q_pad_start = num_reqs * num_query_per_req
        for i in range(q_pad_start, max_num_tokens):
            out_query_slot_mapping[i] = PAD_SLOT_ID


def _modern_dflash_inputs_ref(
    next_token_ids,
    target_positions,
    context_slot_mapping,
    block_table,
    query_start_loc,
    seq_lens,
    num_rejected,
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_steps,
    sample_from_anchor,
):
    """Independent CPU reference for the modern Ascend DFlash kernel."""
    num_reqs = next_token_ids.shape[0]
    num_context = target_positions.numel()
    num_queries = num_reqs * num_query_per_req
    out_input_ids = torch.full((num_queries,), -999, dtype=torch.int32)
    out_context_positions = torch.full((num_context,), -999, dtype=torch.int32)
    out_query_positions = torch.full((num_queries,), -999, dtype=torch.int32)
    out_context_slots = torch.full((num_context,), -999, dtype=torch.int64)
    out_query_slots = torch.full((num_queries,), -999, dtype=torch.int64)
    out_sample_indices = torch.full(
        (num_reqs * num_speculative_steps,), -999, dtype=torch.int32
    )

    for req_idx in range(num_reqs):
        ctx_start = int(query_start_loc[req_idx])
        ctx_end = int(query_start_loc[req_idx + 1])
        rejected = int(num_rejected[req_idx])
        valid_ctx_end = ctx_end - rejected
        effective_seq_len = int(seq_lens[req_idx]) - rejected

        out_context_positions[ctx_start:ctx_end] = target_positions[
            ctx_start:ctx_end
        ]
        out_context_slots[ctx_start:ctx_end] = context_slot_mapping[
            ctx_start:ctx_end
        ]
        last_pos = int(target_positions[valid_ctx_end - 1])

        for query_offset in range(num_query_per_req):
            query_idx = req_idx * num_query_per_req + query_offset
            query_pos = last_pos + 1 + query_offset
            out_query_positions[query_idx] = query_pos
            out_input_ids[query_idx] = (
                int(next_token_ids[req_idx])
                if query_offset == 0
                else parallel_drafting_token_id
            )

            cache_pos = effective_seq_len + query_offset
            block_num = cache_pos // block_size
            block_id = int(block_table[req_idx, block_num])
            out_query_slots[query_idx] = (
                block_id * block_size + cache_pos % block_size
            )

            if sample_from_anchor:
                sample_idx = req_idx * num_speculative_steps + query_offset
                out_sample_indices[sample_idx] = query_idx
            elif query_offset > 0:
                sample_idx = req_idx * num_speculative_steps + query_offset - 1
                out_sample_indices[sample_idx] = query_idx

    return (
        out_input_ids,
        out_context_positions,
        out_query_positions,
        out_context_slots,
        out_query_slots,
        out_sample_indices,
    )


class TestPrepareDFlashInputsKernelAscendPatch:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")
        self.BLOCK_SIZE = 1024

    def _test_modern_dflash_inputs(self, num_reqs, sample_from_anchor):
        """Test the installed modern Ascend DFlash input kernel."""
        num_speculative_steps = 3
        num_query_per_req = (
            num_speculative_steps
            if sample_from_anchor
            else num_speculative_steps + 1
        )
        context_per_req = 4
        num_context = num_reqs * context_per_req
        block_size = 64
        parallel_drafting_token_id = 0

        next_token_ids_cpu = torch.arange(5, 5 + num_reqs, dtype=torch.int32)
        target_positions_cpu = torch.arange(num_context, dtype=torch.int32)
        context_slots_cpu = torch.arange(
            1000, 1000 + num_context, dtype=torch.int64
        )
        query_start_loc_cpu = (
            torch.arange(num_reqs + 1, dtype=torch.int32) * context_per_req
        )
        seq_lens_cpu = query_start_loc_cpu[1:].clone()
        num_rejected_cpu = torch.zeros(num_reqs, dtype=torch.int32)
        block_table_cpu = torch.arange(
            100, 100 + num_reqs * 16, dtype=torch.int32
        ).reshape(num_reqs, 16)

        expected = _modern_dflash_inputs_ref(
            next_token_ids_cpu, target_positions_cpu, context_slots_cpu,
            block_table_cpu, query_start_loc_cpu, seq_lens_cpu,
            num_rejected_cpu, parallel_drafting_token_id, block_size,
            num_query_per_req, num_speculative_steps, sample_from_anchor,
        )

        device = self.device
        next_token_ids = next_token_ids_cpu.to(device)
        target_positions = target_positions_cpu.to(device)
        context_slots = context_slots_cpu.to(device)
        block_table = block_table_cpu.to(device)
        query_start_loc = query_start_loc_cpu.to(device)
        seq_lens = seq_lens_cpu.to(device)
        num_rejected = num_rejected_cpu.to(device)
        outputs = tuple(t.to(device) for t in expected)
        for output in outputs:
            output.fill_(-999)

        launch_kwargs = {"HAS_NUM_REJECTED": True}
        if sample_from_anchor:
            launch_kwargs["SAMPLE_FROM_ANCHOR"] = True

        _modern_dflash_inputs_kernel[(1,)](
            next_token_ids, target_positions, context_slots,
            outputs[0], outputs[1], outputs[2], outputs[3], outputs[4],
            outputs[5], block_table, block_table.stride(0),
            query_start_loc, seq_lens, num_rejected,
            parallel_drafting_token_id, block_size, num_query_per_req,
            num_speculative_steps, num_context, num_reqs, **launch_kwargs,
        )
        torch.npu.synchronize()

        for actual, reference in zip(outputs, expected):
            torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)

    @pytest.mark.parametrize("num_reqs", [1, 2])
    @pytest.mark.parametrize("SAMPLE_FROM_ANCHOR", [False, True])
    def test_prepare_dflash_inputs(self, num_reqs, SAMPLE_FROM_ANCHOR):
        """Compare the installed legacy or modern DFlash kernel with CPU."""

        if _prepare_dflash_inputs_kernel_ascend is None:
            if _modern_dflash_inputs_kernel is None:
                pytest.fail(
                    "DFlash environment compatibility failure; this is not "
                    "a precision failure and no kernel was tested.\n"
                    + _dflash_environment_diagnostics(),
                    pytrace=False,
                )
            self._test_modern_dflash_inputs(num_reqs, SAMPLE_FROM_ANCHOR)
            return

        max_num_reqs = 4
        num_query_per_req = 3
        num_speculative_steps = 3
        block_size = 64
        parallel_drafting_token_id = 0
        max_model_len = 1024
        max_num_tokens = max_num_reqs * num_query_per_req
        total_tokens = num_reqs * 4  # 4 tokens each in context
        num_blocks_in_table = 16

        # --- Input tensors ---
        target_positions = torch.arange(total_tokens, dtype=torch.int32, device=self.device)
        target_query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32, device=self.device) * 4
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        last_sampled = torch.arange(max_num_reqs, dtype=torch.int32, device=self.device) * 10 + 5
        next_prefill_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=self.device)
        num_sampled = torch.full((num_reqs,), 2, dtype=torch.int32, device=self.device)
        num_rejected = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)

        block_table = torch.randint(100, 200, (num_reqs, num_blocks_in_table), dtype=torch.int32, device=self.device)

        # --- Output tensors (initialized with sentinel values) ---
        out_input_ids = torch.full((max_num_tokens,), -999, dtype=torch.int32, device=self.device)
        out_query_positions = torch.full((max_num_tokens,), -999, dtype=torch.int32, device=self.device)
        out_query_start_loc = torch.full((max_num_reqs + 1,), -999, dtype=torch.int32, device=self.device)
        out_seq_lens = torch.full((max_num_reqs,), -999, dtype=torch.int32, device=self.device)
        out_query_slot_mapping = torch.full((max_num_tokens,), -999, dtype=torch.int32, device=self.device)
        out_context_positions = torch.full((total_tokens,), -999, dtype=torch.int32, device=self.device)
        out_context_slot_mapping = torch.full((total_tokens,), -999, dtype=torch.int32, device=self.device)
        out_sample_indices = torch.full((max_num_reqs * num_speculative_steps,), -999, dtype=torch.int32, device=self.device)
        out_sample_pos = torch.full((max_num_reqs * num_speculative_steps,), -999, dtype=torch.int32, device=self.device)
        out_sample_idx_mapping = torch.full((max_num_reqs * num_speculative_steps,), -999, dtype=torch.int32, device=self.device)

        # --- CPU reference outputs ---
        ref_kwargs = dict(
            out_input_ids=torch.full((max_num_tokens,), -999, dtype=torch.int32),
            out_query_positions=torch.full((max_num_tokens,), -999, dtype=torch.int32),
            out_query_start_loc=torch.full((max_num_reqs + 1,), -999, dtype=torch.int32),
            out_seq_lens=torch.full((max_num_reqs,), -999, dtype=torch.int32),
            out_query_slot_mapping=torch.full((max_num_tokens,), -999, dtype=torch.int32),
            out_context_positions=torch.full((total_tokens,), -999, dtype=torch.int32),
            out_context_slot_mapping=torch.full((total_tokens,), -999, dtype=torch.int32),
            out_sample_indices=torch.full((max_num_reqs * num_speculative_steps,), -999, dtype=torch.int32),
            out_sample_pos=torch.full((max_num_reqs * num_speculative_steps,), -999, dtype=torch.int32),
            out_sample_idx_mapping=torch.full((max_num_reqs * num_speculative_steps,), -999, dtype=torch.int32),
        )
        _prepare_dflash_inputs_ref(
            **ref_kwargs,
            target_positions=target_positions.cpu(),
            target_query_start_loc=target_query_start_loc.cpu(),
            idx_mapping=idx_mapping.cpu(),
            last_sampled=last_sampled.cpu(),
            next_prefill_tokens=next_prefill_tokens.cpu(),
            num_sampled=num_sampled.cpu(),
            num_rejected=num_rejected.cpu(),
            block_table=block_table.cpu(),
            parallel_drafting_token_id=parallel_drafting_token_id,
            block_size=block_size,
            num_query_per_req=num_query_per_req,
            num_speculative_steps=num_speculative_steps,
            max_num_reqs=max_num_reqs,
            max_num_tokens=max_num_tokens,
            max_model_len=max_model_len,
            SAMPLE_FROM_ANCHOR=SAMPLE_FROM_ANCHOR,
        )

        # --- Launch kernel ---
        grid = (num_reqs, 1)
        _prepare_dflash_inputs_kernel_ascend[grid](
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
            target_positions,
            target_query_start_loc,
            idx_mapping,
            last_sampled,
            next_prefill_tokens,
            num_sampled,
            num_rejected,
            block_table,
            block_table.stride(0),
            parallel_drafting_token_id,
            block_size,
            num_query_per_req,
            num_speculative_steps,
            max_num_reqs,
            max_num_tokens,
            max_model_len,
            SAMPLE_FROM_ANCHOR=SAMPLE_FROM_ANCHOR,
            PAD_SLOT_ID=PAD_SLOT_ID,
            BLOCK_SIZE=1024,
        )
        torch.npu.synchronize()

        # --- Verify ---
        torch.testing.assert_close(out_input_ids.cpu(), ref_kwargs["out_input_ids"], rtol=0, atol=0)
        torch.testing.assert_close(out_query_positions.cpu(), ref_kwargs["out_query_positions"], rtol=0, atol=0)
        torch.testing.assert_close(out_query_start_loc.cpu(), ref_kwargs["out_query_start_loc"], rtol=0, atol=0)
        torch.testing.assert_close(out_seq_lens.cpu(), ref_kwargs["out_seq_lens"], rtol=0, atol=0)
        torch.testing.assert_close(out_query_slot_mapping.cpu(), ref_kwargs["out_query_slot_mapping"], rtol=0, atol=0)
        torch.testing.assert_close(out_context_positions.cpu(), ref_kwargs["out_context_positions"], rtol=0, atol=0)
        torch.testing.assert_close(out_context_slot_mapping.cpu(), ref_kwargs["out_context_slot_mapping"], rtol=0, atol=0)
        torch.testing.assert_close(out_sample_indices.cpu(), ref_kwargs["out_sample_indices"], rtol=0, atol=0)
        torch.testing.assert_close(out_sample_pos.cpu(), ref_kwargs["out_sample_pos"], rtol=0, atol=0)
        torch.testing.assert_close(out_sample_idx_mapping.cpu(), ref_kwargs["out_sample_idx_mapping"], rtol=0, atol=0)
