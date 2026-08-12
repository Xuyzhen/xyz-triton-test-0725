# GENERATED STRICT UT. Source: accuracy_test/codex/existing_accuracy_tests/from_vllm_ascend/test_log_softmax.py
# Do not edit mechanically; update the reviewed Codex source or strict generator.
from accuracy_test.strict_ut.runtime_gpu import STRICT_DEVICE as _STRICT_DEVICE
# Standalone Ascend A3 accuracy test.
# Accuracy UT source: vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_log_softmax.py
# Kernel source: vllm-ascend-xyz/vllm_ascend/worker/v2/sample/logprob.py
# Coverage: _topk_log_softmax_kernel (direct)

import pytest
import torch
try:
    from vllm.triton_utils import triton
    from vllm.v1.worker.gpu.sample.logprob import _topk_log_softmax_kernel
except (ImportError, ModuleNotFoundError) as exc:
    pytest.skip(
        f"installed stack does not provide _topk_log_softmax_kernel; precision was not tested: {exc}",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    "batch_size,vocab_size,num_logprobs",
    [
        (48, 102400, 50),
        (96, 102400, 1),
        (24, 151936, 8),
    ],
)
def test_topk_log_softmax_kernel(batch_size, vocab_size, num_logprobs):
    """Test _topk_log_softmax_kernel for computing log probabilities
    Args:
        batch_size: Number of sequences in the batch
        vocab_size: Size of the vocabulary
        num_logprobs: Number of tokens to compute log probabilities for
    """
    # ========== Setup test data ==========
    torch.manual_seed(42)

    # Generate random logits
    logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=torch.float32)

    # Generate token_ids for which to compute logprobs
    token_ids = torch.randint(0, vocab_size, (batch_size, num_logprobs), device="cuda", dtype=torch.int64)

    # ========== Execute test ==========
    # Prepare output tensor
    triton_output = torch.empty(batch_size, num_logprobs, dtype=torch.float32, device="cuda")

    # Invoke Triton kernel
    _topk_log_softmax_kernel[(batch_size,)](
        triton_output,
        logits,
        logits.stride(0),
        token_ids,
        num_logprobs,
        vocab_size,
        BLOCK_SIZE=1024,
        PADDED_TOPK=max(triton.next_power_of_2(num_logprobs), 2),
    )
    torch.cuda.synchronize()

    # Compute reference values using PyTorch
    torch_logprobs = torch.log_softmax(logits, dim=-1)

    # Avoid inheriting the Triton output's NPU internal format. That inheritance
    # warns when torch_npu is configured with allow_internal_format=False.
    ref_output = torch.gather(torch_logprobs, dim=1, index=token_ids)

    # ========== Verify results ==========
    assert torch.allclose(triton_output, ref_output, rtol=1e-3, atol=1e-3), (
        f"Triton output differs from PyTorch reference.\n"
        f"Max diff: {torch.max(torch.abs(triton_output - ref_output))}\n"
        f"Mean diff: {torch.mean(torch.abs(triton_output - ref_output))}"
    )
