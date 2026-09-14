# GENERATED STRICT UT. Source: accuracy_test/codex/existing_accuracy_tests/from_vllm/test_bias_kernel.py
# Do not edit mechanically; update the reviewed Codex source or strict generator.
from accuracy_test.strict_ut.runtime_gpu import STRICT_DEVICE as _STRICT_DEVICE
# Standalone Ascend A3 adaptation of an upstream vLLM accuracy path.
# Accuracy UT source: vllm/tests/v1/sample/test_sampler.py
# Kernel source: vllm/vllm/v1/worker/gpu/sample/logit_bias.py
# Coverage: _bias_kernel

# vLLM vanilla kernel: _bias_kernel from
# vllm/vllm/v1/worker/gpu/sample/logit_bias.py

"""
Precision test for _bias_kernel.

Kernel signature:
    _bias_kernel(
        logits_ptr,                     # fp32 logits [num_tokens, vocab_size]
        logits_stride,                  # stride(0) of logits
        vocab_size,
        expanded_idx_mapping_ptr,       # [num_tokens] token_idx -> req_state_idx
        # Allowed token IDs
        num_allowed_token_ids_ptr,      # [max_num_reqs]
        allowed_token_ids_ptr,          # [max_num_reqs, MAX_NUM_ALLOWED_TOKEN_IDS]
        allowed_token_ids_stride,
        # Logit bias
        num_logit_bias_ptr,             # [max_num_reqs]
        bias_token_ids_ptr,             # [max_num_reqs, MAX_NUM_LOGIT_BIAS_TOKENS]
        bias_token_ids_stride,
        bias_ptr,                       # [max_num_reqs, MAX_NUM_LOGIT_BIAS_TOKENS]
        bias_stride,
        # Min tokens
        pos_ptr,                        # [num_tokens]
        min_lens_ptr,                   # [max_num_reqs]
        num_stop_token_ids_ptr,         # [max_num_reqs]
        stop_token_ids_ptr,             # [max_num_reqs, MAX_NUM_STOP_TOKEN_IDS]
        stop_token_ids_stride,
        BLOCK_SIZE: tl.constexpr,
        LOGITS_BLOCK_SIZE: tl.constexpr,
    )

Applies allowed token IDs, logit bias, and min-token suppression to logits:
1. If allowed_token_ids are set: save logits for allowed tokens, set all
   logits to -inf, then restore saved logits.
2. If logit_bias is set: add bias values to the specified token positions.
3. If min_tokens active and pos+1 < min_len: set stop token logits to -inf.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.logit_bias import _bias_kernel
from accuracy_test.strict_ut.runtime_gpu import init_device_properties_triton

import pytest


MAX_NUM_ALLOWED_TOKEN_IDS = 1024
MAX_NUM_LOGIT_BIAS_TOKENS = 1024
MAX_NUM_STOP_TOKEN_IDS = 128


def _bias_ref(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    num_allowed_token_ids: torch.Tensor,
    allowed_token_ids: torch.Tensor,
    num_logit_bias: torch.Tensor,
    bias_token_ids: torch.Tensor,
    bias: torch.Tensor,
    pos: torch.Tensor,
    min_lens: torch.Tensor,
    num_stop_token_ids: torch.Tensor,
    stop_token_ids: torch.Tensor,
):
    """CPU reference for _bias_kernel."""
    num_tokens, vocab_size = logits.shape
    for token_idx in range(num_tokens):
        req_state_idx = expanded_idx_mapping[token_idx].item()

        # 1. Allowed token IDs
        n_allowed = num_allowed_token_ids[req_state_idx].item()
        if n_allowed > 0:
            saved = {}
            for j in range(n_allowed):
                tid = allowed_token_ids[req_state_idx, j].item()
                saved[tid] = logits[token_idx, tid].clone()
            logits[token_idx, :] = float("-inf")
            for tid, val in saved.items():
                logits[token_idx, tid] = val

        # 2. Logit bias
        n_bias = num_logit_bias[req_state_idx].item()
        if n_bias > 0:
            for j in range(n_bias):
                tid = bias_token_ids[req_state_idx, j].item()
                bias_val = bias[req_state_idx, j].item()
                logits[token_idx, tid] = logits[token_idx, tid] + bias_val

        # 3. Min tokens
        n_stop = num_stop_token_ids[req_state_idx].item()
        pos_val = pos[token_idx].item()
        min_len = min_lens[req_state_idx].item()
        if n_stop > 0 and pos_val + 1 < min_len:
            for j in range(n_stop):
                tid = stop_token_ids[req_state_idx, j].item()
                logits[token_idx, tid] = float("-inf")


class TestBiasKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("cuda")

    def _run_kernel(
        self,
        logits,
        expanded_idx_mapping,
        num_allowed_token_ids,
        allowed_token_ids,
        num_logit_bias,
        bias_token_ids,
        bias,
        pos,
        min_lens,
        num_stop_token_ids,
        stop_token_ids,
    ):
        num_tokens, vocab_size = logits.shape
        BLOCK_SIZE = triton.next_power_of_2(
            max(
                allowed_token_ids.shape[-1],
                bias_token_ids.shape[-1],
                stop_token_ids.shape[-1],
            )
        )
        LOGITS_BLOCK_SIZE = 8192
        _bias_kernel[(num_tokens,)](
            logits,
            logits.stride(0),
            vocab_size,
            expanded_idx_mapping,
            num_allowed_token_ids,
            allowed_token_ids,
            allowed_token_ids.stride(0),
            num_logit_bias,
            bias_token_ids,
            bias_token_ids.stride(0),
            bias,
            bias.stride(0),
            pos,
            min_lens,
            num_stop_token_ids,
            stop_token_ids,
            stop_token_ids.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            LOGITS_BLOCK_SIZE=LOGITS_BLOCK_SIZE,
        )
        torch.cuda.synchronize()

    @pytest.mark.parametrize("num_tokens", [1, 4, 8])
    @pytest.mark.parametrize("vocab_size", [128, 1024, 129280, 163840, 248320])
    def test_allowed_token_ids(self, num_tokens, vocab_size):
        """Only allowed token IDs should have non -inf logits."""
        num_reqs = 4
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.randint(0, num_reqs, (num_tokens,), dtype=torch.int32, device=self.device)
        pos = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)

        # Set up allowed tokens
        num_allowed_token_ids = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        allowed_token_ids = torch.zeros(num_reqs, MAX_NUM_ALLOWED_TOKEN_IDS, dtype=torch.int32, device=self.device)
        allowed_tokens = [5, 10, 15]
        for i in range(num_reqs):
            num_allowed_token_ids[i] = len(allowed_tokens)
            for j, tid in enumerate(allowed_tokens):
                allowed_token_ids[i, j] = tid

        # No bias, no stop tokens
        num_logit_bias = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        bias_token_ids = torch.zeros(num_reqs, MAX_NUM_LOGIT_BIAS_TOKENS, dtype=torch.int32, device=self.device)
        bias = torch.zeros(num_reqs, MAX_NUM_LOGIT_BIAS_TOKENS, dtype=torch.float32, device=self.device)
        min_lens = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        num_stop_token_ids = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        stop_token_ids = torch.zeros(num_reqs, MAX_NUM_STOP_TOKEN_IDS, dtype=torch.int32, device=self.device)

        logits_copy = logits.clone().cpu()

        self._run_kernel(
            logits, expanded_idx_mapping,
            num_allowed_token_ids, allowed_token_ids,
            num_logit_bias, bias_token_ids, bias,
            pos, min_lens, num_stop_token_ids, stop_token_ids,
        )

        expected = logits_copy.clone()
        _bias_ref(
            expected, expanded_idx_mapping.cpu(),
            num_allowed_token_ids.cpu(), allowed_token_ids.cpu(),
            num_logit_bias.cpu(), bias_token_ids.cpu(), bias.cpu(),
            pos.cpu(), min_lens.cpu(), num_stop_token_ids.cpu(), stop_token_ids.cpu(),
        )

        torch.testing.assert_close(logits.cpu(), expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("num_tokens", [1, 4])
    def test_logit_bias(self, num_tokens):
        """Logit bias should be added to specified token positions."""
        num_reqs = 2
        vocab_size = 64
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.randint(0, num_reqs, (num_tokens,), dtype=torch.int32, device=self.device)
        pos = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)

        # No allowed tokens
        num_allowed_token_ids = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        allowed_token_ids = torch.zeros(num_reqs, MAX_NUM_ALLOWED_TOKEN_IDS, dtype=torch.int32, device=self.device)

        # Set up logit bias
        num_logit_bias = torch.tensor([2, 1], dtype=torch.int32, device=self.device)
        bias_token_ids = torch.zeros(num_reqs, MAX_NUM_LOGIT_BIAS_TOKENS, dtype=torch.int32, device=self.device)
        bias_token_ids[0, 0] = 10
        bias_token_ids[0, 1] = 20
        bias_token_ids[1, 0] = 5
        bias = torch.zeros(num_reqs, MAX_NUM_LOGIT_BIAS_TOKENS, dtype=torch.float32, device=self.device)
        bias[0, 0] = 2.0
        bias[0, 1] = -1.0
        bias[1, 0] = 0.5

        min_lens = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        num_stop_token_ids = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        stop_token_ids = torch.zeros(num_reqs, MAX_NUM_STOP_TOKEN_IDS, dtype=torch.int32, device=self.device)

        logits_copy = logits.clone().cpu()

        self._run_kernel(
            logits, expanded_idx_mapping,
            num_allowed_token_ids, allowed_token_ids,
            num_logit_bias, bias_token_ids, bias,
            pos, min_lens, num_stop_token_ids, stop_token_ids,
        )

        expected = logits_copy.clone()
        _bias_ref(
            expected, expanded_idx_mapping.cpu(),
            num_allowed_token_ids.cpu(), allowed_token_ids.cpu(),
            num_logit_bias.cpu(), bias_token_ids.cpu(), bias.cpu(),
            pos.cpu(), min_lens.cpu(), num_stop_token_ids.cpu(), stop_token_ids.cpu(),
        )

        torch.testing.assert_close(logits.cpu(), expected, rtol=1e-5, atol=1e-5)

    def test_combined(self):
        """Combined allowed tokens + logit bias + min tokens."""
        num_tokens = 2
        num_reqs = 2
        vocab_size = 64
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)
        expanded_idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
        pos = torch.tensor([0, 5], dtype=torch.int32, device=self.device)

        # Req 0: allowed tokens only
        num_allowed_token_ids = torch.tensor([3, 0], dtype=torch.int32, device=self.device)
        allowed_token_ids = torch.zeros(num_reqs, MAX_NUM_ALLOWED_TOKEN_IDS, dtype=torch.int32, device=self.device)
        allowed_token_ids[0, 0] = 10
        allowed_token_ids[0, 1] = 20
        allowed_token_ids[0, 2] = 30

        # Req 1: logit bias only
        num_logit_bias = torch.tensor([0, 2], dtype=torch.int32, device=self.device)
        bias_token_ids = torch.zeros(num_reqs, MAX_NUM_LOGIT_BIAS_TOKENS, dtype=torch.int32, device=self.device)
        bias_token_ids[1, 0] = 5
        bias_token_ids[1, 1] = 15
        bias = torch.zeros(num_reqs, MAX_NUM_LOGIT_BIAS_TOKENS, dtype=torch.float32, device=self.device)
        bias[1, 0] = 3.0
        bias[1, 1] = -2.0

        # Req 1 also has min tokens active
        min_lens = torch.tensor([0, 10], dtype=torch.int32, device=self.device)
        num_stop_token_ids = torch.tensor([0, 2], dtype=torch.int32, device=self.device)
        stop_token_ids = torch.zeros(num_reqs, MAX_NUM_STOP_TOKEN_IDS, dtype=torch.int32, device=self.device)
        stop_token_ids[1, 0] = 1
        stop_token_ids[1, 1] = 2

        logits_copy = logits.clone().cpu()

        self._run_kernel(
            logits, expanded_idx_mapping,
            num_allowed_token_ids, allowed_token_ids,
            num_logit_bias, bias_token_ids, bias,
            pos, min_lens, num_stop_token_ids, stop_token_ids,
        )

        expected = logits_copy.clone()
        _bias_ref(
            expected, expanded_idx_mapping.cpu(),
            num_allowed_token_ids.cpu(), allowed_token_ids.cpu(),
            num_logit_bias.cpu(), bias_token_ids.cpu(), bias.cpu(),
            pos.cpu(), min_lens.cpu(), num_stop_token_ids.cpu(), stop_token_ids.cpu(),
        )

        torch.testing.assert_close(logits.cpu(), expected, rtol=1e-5, atol=1e-5)
