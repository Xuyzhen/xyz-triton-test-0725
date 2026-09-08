# vLLM vanilla kernel: _prepare_prefill_inputs_kernel from vllm/vllm/v1/worker/gpu/input_batch.py

"""
Precision test for _prepare_prefill_inputs_kernel.

Kernel signature (vllm >= dec13a33b7, #48892 multi-layer MTP):
    _prepare_prefill_inputs_kernel(
        input_ids_ptr,               # int32 output [max_num_tokens]
        next_prefill_tokens_ptr,     # int32 output [num_lookahead, max_num_reqs]
        next_prefill_tokens_stride,  # stride(0) of next_prefill_tokens
        num_lookahead,               # number of lookahead tokens to fetch
        idx_mapping_ptr,             # int32 [num_reqs] batch_idx -> req_state_idx
        query_start_loc_ptr,         # int32 [num_reqs + 1]
        all_token_ids_ptr,           # int32 [max_num_reqs, max_model_len]
        all_token_ids_stride,        # stride(0) of all_token_ids
        prefill_lens_ptr,            # int32 [max_num_reqs]
        num_computed_tokens_ptr,     # int32 [max_num_reqs]
        BLOCK_SIZE: tl.constexpr,    # block size for iteration
        LOOKAHEAD_BLOCK: tl.constexpr,  # power-of-2 bound covering num_lookahead
    )

Legacy signature (pre-#48892) omits next_prefill_tokens_stride / num_lookahead
/ LOOKAHEAD_BLOCK and uses a flat [max_num_reqs] next_prefill_tokens buffer.

Copies prefill token IDs from all_token_ids to input_ids.
Stores the next num_lookahead prefill tokens into next_prefill_tokens
[num_lookahead, max_num_reqs]. Lookahead slots whose position falls beyond
prefill_len are stored as 0 (load mask other=0; store mask only checks
in_lookahead). When num_computed >= prefill_len, returns early (no writes).
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import _prepare_prefill_inputs_kernel
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

import pytest

# vllm dec13a33b7 (#48892, multi-layer MTP speculator) added the lookahead
# args. Probe so the same UT runs against both kernel generations.
_HAS_LOOKAHEAD_ARGS = "num_lookahead" in tuple(
    _prepare_prefill_inputs_kernel.arg_names
)


def _prepare_prefill_inputs_ref(
    input_ids: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    prefill_lens: torch.Tensor,
    num_computed_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference mirroring both kernel generations.

    New kernels: next_prefill_tokens is [num_lookahead, max_num_reqs]; every
    in-lookahead slot is written, 0 for positions beyond prefill_len.
    Legacy kernels: next_prefill_tokens is [max_num_reqs]; the single slot is
    written only when the next position is still within prefill_len.
    """
    input_ids_out = input_ids.clone()
    next_prefill_tokens_out = next_prefill_tokens.clone()
    num_reqs = idx_mapping.shape[0]
    num_lookahead = (
        next_prefill_tokens.shape[0] if _HAS_LOOKAHEAD_ARGS else 1
    )

    for batch_idx in range(num_reqs):
        rs_idx = idx_mapping[batch_idx].item()
        prefill_len = prefill_lens[rs_idx].item()
        num_computed = num_computed_tokens[rs_idx].item()
        if num_computed >= prefill_len:
            continue

        qs = query_start_loc[batch_idx].item()
        qe = query_start_loc[batch_idx + 1].item()
        qlen = qe - qs

        for i in range(qlen):
            tok = all_token_ids[rs_idx, num_computed + i].item()
            input_ids_out[qs + i] = tok

        for j in range(num_lookahead):
            next_pos = num_computed + qlen + j
            if _HAS_LOOKAHEAD_ARGS:
                # New kernels always store the slot: the loaded token when
                # next_pos < prefill_len, otherwise the load's other=0.
                tok = (
                    all_token_ids[rs_idx, next_pos].item()
                    if next_pos < prefill_len
                    else 0
                )
                next_prefill_tokens_out[j, rs_idx] = tok
            elif next_pos < prefill_len:
                next_prefill_tokens_out[rs_idx] = all_token_ids[
                    rs_idx, next_pos
                ].item()

    return input_ids_out, next_prefill_tokens_out


class TestPreparePrefillInputsKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    def _launch_kernel(
        self,
        input_ids,
        next_prefill_tokens,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        prefill_lens,
        num_computed_tokens,
    ):
        """Launch the kernel with the signature matching the installed vllm.

        New kernels (#48892) take [num_lookahead, max_num_reqs]
        next_prefill_tokens plus its stride and num_lookahead, mirroring
        prepare_prefill_inputs() in vllm/v1/worker/gpu/input_batch.py.
        """
        num_reqs = idx_mapping.shape[0]
        if _HAS_LOOKAHEAD_ARGS:
            _prepare_prefill_inputs_kernel[(num_reqs,)](
                input_ids,
                next_prefill_tokens,
                next_prefill_tokens.stride(0),
                next_prefill_tokens.shape[0],
                idx_mapping,
                query_start_loc,
                all_token_ids,
                all_token_ids.stride(0),
                prefill_lens,
                num_computed_tokens,
                BLOCK_SIZE=1024,
                LOOKAHEAD_BLOCK=triton.next_power_of_2(
                    next_prefill_tokens.shape[0]
                ),
            )
        else:
            _prepare_prefill_inputs_kernel[(num_reqs,)](
                input_ids,
                next_prefill_tokens,
                idx_mapping,
                query_start_loc,
                all_token_ids,
                all_token_ids.stride(0),
                prefill_lens,
                num_computed_tokens,
                BLOCK_SIZE=1024,
            )
        torch.npu.synchronize()

    def _make_next_prefill_tokens(self, fill, max_num_reqs, num_lookahead=1):
        """[num_lookahead, max_num_reqs] on new kernels, flat otherwise."""
        if _HAS_LOOKAHEAD_ARGS:
            return torch.full(
                (num_lookahead, max_num_reqs), fill, dtype=torch.int32,
                device=self.device,
            )
        return torch.full(
            (max_num_reqs,), fill, dtype=torch.int32, device=self.device
        )

    @pytest.mark.parametrize("num_reqs", [1, 2, 4])
    @pytest.mark.parametrize("query_len", [1, 4, 16])
    def test_prepare_prefill_inputs(self, num_reqs, query_len):
        """Compare kernel output with CPU reference."""
        max_model_len = 128
        max_num_reqs = 8
        max_num_tokens = num_reqs * query_len

        all_token_ids = torch.arange(max_model_len, dtype=torch.int32, device=self.device).unsqueeze(0).repeat(max_num_reqs, 1)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32, device=self.device) * query_len
        num_computed_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=self.device)
        prefill_lens = torch.full((max_num_reqs,), 64, dtype=torch.int32, device=self.device)

        input_ids = torch.zeros(max_num_tokens, dtype=torch.int32, device=self.device)
        next_prefill_tokens = self._make_next_prefill_tokens(0, max_num_reqs)

        self._launch_kernel(
            input_ids,
            next_prefill_tokens,
            idx_mapping,
            query_start_loc,
            all_token_ids,
            prefill_lens,
            num_computed_tokens,
        )

        input_ids_exp, next_prefill_exp = _prepare_prefill_inputs_ref(
            torch.zeros(max_num_tokens, dtype=torch.int32),
            self._make_next_prefill_tokens(0, max_num_reqs).cpu(),
            idx_mapping.cpu(), query_start_loc.cpu(),
            all_token_ids.cpu(), prefill_lens.cpu(), num_computed_tokens.cpu(),
        )
        torch.testing.assert_close(input_ids.cpu(), input_ids_exp, rtol=0, atol=0)
        torch.testing.assert_close(next_prefill_tokens.cpu(), next_prefill_exp, rtol=0, atol=0)

    def test_early_return_when_prefill_done(self):
        """When num_computed >= prefill_len, kernel should be a no-op."""
        num_reqs = 2
        query_len = 4
        max_num_tokens = num_reqs * query_len
        max_num_reqs = 4
        max_model_len = 32

        all_token_ids = torch.randint(0, 100, (max_num_reqs, max_model_len), dtype=torch.int32, device=self.device)
        idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
        query_start_loc = torch.tensor([0, 4, 8], dtype=torch.int32, device=self.device)
        num_computed_tokens = torch.tensor([20, 15], dtype=torch.int32, device=self.device)
        prefill_lens = torch.tensor([10, 20], dtype=torch.int32, device=self.device)

        input_ids = torch.full((max_num_tokens,), -1, dtype=torch.int32, device=self.device)
        next_prefill_tokens = self._make_next_prefill_tokens(-1, max_num_reqs)
        expected_input_ids = input_ids.clone().cpu()
        expected_next_prefill = next_prefill_tokens.clone().cpu()

        self._launch_kernel(
            input_ids,
            next_prefill_tokens,
            idx_mapping,
            query_start_loc,
            all_token_ids,
            prefill_lens,
            num_computed_tokens,
        )

        # req 0: prefill_lens[0]=10, num_computed[0]=20 -> done -> early return
        # req 1: prefill_lens[1]=20, num_computed[1]=15 -> still prefilling
        token_ids_cpu = all_token_ids.cpu()
        for i in range(query_len):
            expected_input_ids[query_len + i] = token_ids_cpu[1, 15 + i]
        # num_computed + query_len = 19, which is < 20
        if _HAS_LOOKAHEAD_ARGS:
            expected_next_prefill[0, 1] = token_ids_cpu[1, 15 + 4]
        else:
            expected_next_prefill[1] = token_ids_cpu[1, 15 + 4]

        torch.testing.assert_close(input_ids.cpu(), expected_input_ids, rtol=0, atol=0)
        torch.testing.assert_close(next_prefill_tokens.cpu(), expected_next_prefill, rtol=0, atol=0)

    def test_exact_prefill_boundary(self):
        """When num_computed + query_len == prefill_len, the first lookahead
        slot falls beyond prefill_len: new kernels store the load's other=0,
        legacy kernels leave the buffer untouched."""
        num_reqs = 1
        query_len = 5
        max_num_tokens = query_len
        max_num_reqs = 2
        max_model_len = 16

        all_token_ids = torch.arange(max_model_len, dtype=torch.int32, device=self.device).unsqueeze(0).repeat(max_num_reqs, 1)
        idx_mapping = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        query_start_loc = torch.tensor([0, 5], dtype=torch.int32, device=self.device)
        num_computed_tokens = torch.tensor([5], dtype=torch.int32, device=self.device)
        prefill_lens = torch.tensor([10, 10], dtype=torch.int32, device=self.device)

        input_ids = torch.full((max_num_tokens,), -1, dtype=torch.int32, device=self.device)
        next_prefill_tokens = self._make_next_prefill_tokens(-1, max_num_reqs)

        self._launch_kernel(
            input_ids,
            next_prefill_tokens,
            idx_mapping,
            query_start_loc,
            all_token_ids,
            prefill_lens,
            num_computed_tokens,
        )

        # num_computed=5, query_len=5, prefill_len=10 => next_pos=10 == prefill_len
        expected_input_ids = torch.tensor([5, 6, 7, 8, 9], dtype=torch.int32)
        if _HAS_LOOKAHEAD_ARGS:
            # Store mask is in_lookahead only: the active req's out-of-range
            # slot is written with the load's other=0; the inactive req's
            # slot is never written and keeps its -1 sentinel (mirrors the
            # codex version's expected_next[:, 0] = 0 pattern).
            expected_next = torch.full((1, max_num_reqs), -1, dtype=torch.int32)
            expected_next[0, 0] = 0
        else:
            expected_next = torch.full((max_num_reqs,), -1, dtype=torch.int32)

        torch.testing.assert_close(input_ids.cpu(), expected_input_ids, rtol=0, atol=0)
        torch.testing.assert_close(next_prefill_tokens.cpu(), expected_next, rtol=0, atol=0)

    @pytest.mark.skipif(
        not _HAS_LOOKAHEAD_ARGS,
        reason="lookahead args require vllm dec13a33b7+ (#48892)",
    )
    @pytest.mark.parametrize("num_lookahead", [3, 4])
    def test_multi_lookahead_tokens(self, num_lookahead):
        """Store num_lookahead>1 tokens; slots beyond prefill_len become 0.

        req 0: prefill_len leaves all lookahead positions valid.
        req 1: prefill_len cuts the lookahead window -> trailing slots = 0.
        """
        num_reqs = 2
        query_len = 4
        max_num_reqs = 4
        max_model_len = 32

        all_token_ids = torch.randint(
            0, 100, (max_num_reqs, max_model_len),
            dtype=torch.int32, device=self.device,
        )
        idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
        query_start_loc = torch.tensor([0, 4, 8], dtype=torch.int32, device=self.device)
        num_computed_tokens = torch.tensor([0, 0], dtype=torch.int32, device=self.device)
        # req0: lookahead positions 4..4+num_lookahead-1 all < 16 (valid).
        # req1: positions 4..4+num_lookahead-1 vs prefill_len=6 -> only the
        # first two slots are in range, the rest store 0.
        prefill_lens = torch.tensor([16, 6], dtype=torch.int32, device=self.device)

        input_ids = torch.full((num_reqs * query_len,), -1, dtype=torch.int32, device=self.device)
        next_prefill_tokens = self._make_next_prefill_tokens(
            -1, max_num_reqs, num_lookahead=num_lookahead
        )

        self._launch_kernel(
            input_ids,
            next_prefill_tokens,
            idx_mapping,
            query_start_loc,
            all_token_ids,
            prefill_lens,
            num_computed_tokens,
        )

        input_ids_exp, next_exp = _prepare_prefill_inputs_ref(
            torch.full((num_reqs * query_len,), -1, dtype=torch.int32),
            torch.full((num_lookahead, max_num_reqs), -1, dtype=torch.int32),
            idx_mapping.cpu(), query_start_loc.cpu(),
            all_token_ids.cpu(), prefill_lens.cpu(), num_computed_tokens.cpu(),
        )
        torch.testing.assert_close(input_ids.cpu(), input_ids_exp, rtol=0, atol=0)
        torch.testing.assert_close(next_prefill_tokens.cpu(), next_exp, rtol=0, atol=0)
        # Explicit spot checks for the zero-fill semantics.
        assert next_prefill_tokens.cpu()[num_lookahead - 1, 0].item() == \
            int(all_token_ids.cpu()[0, 4 + num_lookahead - 1])
        assert next_prefill_tokens.cpu()[num_lookahead - 1, 1].item() == 0
