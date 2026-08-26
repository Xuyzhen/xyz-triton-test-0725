# GENERATED STRICT UT. Source: accuracy_test/codex/existing_accuracy_tests/from_vllm/test_prompt_logprobs_token_ids_kernel.py
# Do not edit mechanically; update the reviewed Codex source or strict generator.
from accuracy_test.strict_ut.runtime_npu import STRICT_DEVICE as _STRICT_DEVICE
# Standalone Ascend A3 adaptation of an upstream vLLM accuracy path.
# Accuracy UT source: vllm/tests/v1/sample/test_logprobs.py
# Kernel source: vllm/vllm/v1/worker/gpu/sample/prompt_logprob.py
# Coverage: _prompt_logprobs_token_ids_kernel

# vLLM vanilla kernel: _prompt_logprobs_token_ids_kernel from vllm/vllm/v1/worker/gpu/sample/prompt_logprob.py

"""
Precision test for _prompt_logprobs_token_ids_kernel.

Kernel signature:
    _prompt_logprobs_token_ids_kernel(
        prompt_logprobs_token_ids_ptr,  # int64 output [num_tokens]
        query_start_loc_ptr,            # int32 [num_reqs + 1]
        idx_mapping_ptr,               # int32 [num_reqs] batch_idx -> req_state_idx
        num_computed_tokens_ptr,        # int32 [max_num_reqs]
        all_token_ids_ptr,              # int32 [max_num_reqs, max_model_len]
        all_token_ids_stride,           # stride(0) of all_token_ids
        BLOCK_SIZE: tl.constexpr,       # block size for iteration
    )

Copies token IDs from all_token_ids (shifted by one) into the output tensor
for prompt logprob computation.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.prompt_logprob import _prompt_logprobs_token_ids_kernel
from accuracy_test.strict_ut.runtime_npu import init_device_properties_triton

import pytest


def _prompt_logprobs_token_ids_ref(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    all_token_ids: torch.Tensor,
) -> torch.Tensor:
    """CPU reference for prompt logprobs token IDs."""
    out = torch.empty(num_tokens, dtype=torch.int64)
    num_reqs = idx_mapping.shape[0]
    for batch_idx in range(num_reqs):
        rs_idx = idx_mapping[batch_idx].item()
        qs = query_start_loc[batch_idx].item()
        qe = query_start_loc[batch_idx + 1].item()
        qlen = qe - qs
        nct = num_computed_tokens[rs_idx].item()
        for i in range(qlen):
            target_pos = nct + 1 + i
            out[qs + i] = all_token_ids[rs_idx, target_pos].item()
    return out


class TestPromptLogprobsTokenIdsKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_reqs", [1, 2, 4])
    @pytest.mark.parametrize("query_len", [1, 4, 16])
    def test_prompt_logprobs_token_ids(self, num_reqs, query_len):
        """Compare kernel output with CPU reference."""
        max_model_len = 128
        max_num_reqs = 8
        num_tokens = num_reqs * query_len
        vocab_size = 64  # not directly used by kernel

        all_token_ids = torch.randint(0, vocab_size, (max_num_reqs, max_model_len), dtype=torch.int32, device=self.device)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32, device=self.device) * query_len
        num_computed_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=self.device)

        token_ids = torch.empty(num_tokens, dtype=torch.int64, device=self.device)
        _prompt_logprobs_token_ids_kernel[(num_reqs,)](
            token_ids,
            query_start_loc,
            idx_mapping,
            num_computed_tokens,
            all_token_ids,
            all_token_ids.stride(0),
            BLOCK_SIZE=1024,
        )
        torch.npu.synchronize()

        expected = _prompt_logprobs_token_ids_ref(
            num_tokens, query_start_loc.cpu(), idx_mapping.cpu(),
            num_computed_tokens.cpu(), all_token_ids.cpu(),
        )
        torch.testing.assert_close(token_ids.cpu(), expected, rtol=0, atol=0)

    def test_nonzero_num_computed_tokens(self):
        """Shift by num_computed_tokens + 1."""
        num_reqs = 2
        query_len = 3
        num_tokens = num_reqs * query_len
        max_model_len = 32
        max_num_reqs = 4

        all_token_ids = torch.arange(max_model_len, dtype=torch.int32, device=self.device).unsqueeze(0).repeat(max_num_reqs, 1)
        idx_mapping = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        idx_mapping[1] = 1
        query_start_loc = torch.tensor([0, 3, 6], dtype=torch.int32, device=self.device)
        num_computed_tokens = torch.tensor([5, 10], dtype=torch.int32, device=self.device)

        token_ids = torch.empty(num_tokens, dtype=torch.int64, device=self.device)
        _prompt_logprobs_token_ids_kernel[(num_reqs,)](
            token_ids,
            query_start_loc,
            idx_mapping,
            num_computed_tokens,
            all_token_ids,
            all_token_ids.stride(0),
            BLOCK_SIZE=1024,
        )
        torch.npu.synchronize()

        expected = _prompt_logprobs_token_ids_ref(
            num_tokens, query_start_loc.cpu(), idx_mapping.cpu(),
            num_computed_tokens.cpu(), all_token_ids.cpu(),
        )
        torch.testing.assert_close(token_ids.cpu(), expected, rtol=0, atol=0)
