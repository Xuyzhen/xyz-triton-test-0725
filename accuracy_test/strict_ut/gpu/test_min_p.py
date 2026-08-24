# GENERATED STRICT UT. Source: accuracy_test/codex/existing_accuracy_tests/from_vllm_ascend/test_min_p.py
# Do not edit mechanically; update the reviewed Codex source or strict generator.
from accuracy_test.strict_ut.runtime_gpu import STRICT_DEVICE as _STRICT_DEVICE
# Standalone Ascend A3 accuracy test.
# Accuracy UT source: vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_min_p.py
# Kernel source: vllm-ascend-xyz/vllm_ascend/worker/v2/sample/min_p.py
# Coverage: _min_p_kernel via apply_min_p

import pytest
import torch

try:
    from accuracy_test.strict_ut.runtime_gpu import init_device_properties_triton
    from vllm.v1.worker.gpu.sample.min_p import apply_min_p
except (ImportError, ModuleNotFoundError) as exc:
    pytest.skip(
        f"installed vLLM does not provide apply_min_p; precision was not tested: {exc}",
        allow_module_level=True,
    )


def torch_min_p_torch(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    min_p: torch.Tensor,
):
    num_tokens, _ = logits.shape
    out = logits.clone()

    for token_idx in range(num_tokens):
        req_state_idx = expanded_idx_mapping[token_idx].item()

        min_p_val = min_p[req_state_idx].item()
        if min_p_val == 0.0:
            continue

        token_logits = out[token_idx]
        max_val = token_logits.max()
        threshold = max_val + torch.log(torch.tensor(min_p_val, device=logits.device))
        token_logits = torch.where(token_logits < threshold, -torch.inf, token_logits)
        out[token_idx] = token_logits

    return out


@pytest.mark.parametrize(
    "num_reqs,vocab_size",
    [
        (48, 102400),
        (96, 102400),
        (24, 151936),
        (1, 32000),
        (4, 129280),  # DeepSeek V4
        (4, 163840),  # Kimi K3
        (4, 248320),  # Qwen 3.8 2.4T
    ],
)
def test_apply_min_p_kernel(num_reqs, vocab_size):
    """Test apply_min_p for computing Min-P sampling mask
    Args:
        num_reqs: Number of sequences in the batch
        vocab_size: Size of the vocabulary
    """

    init_device_properties_triton()
    # ========== Setup test data ==========
    torch.manual_seed(42)

    # Generate random logits (using float32 as specified in your kernel)
    device = "cuda"

    original_logits = torch.randn(num_reqs, vocab_size, device=device, dtype=torch.float32)

    triton_logits = original_logits.clone()
    ref_logits = original_logits.clone()

    expanded_idx_mapping = torch.arange(num_reqs - 1, -1, -1, device=device, dtype=torch.int32)

    # Generate random min_p values (valid range is typically (0, 1.0])
    min_p = torch.empty(num_reqs, device=device, dtype=torch.float32).uniform_(0.01, 0.5)

    # ========== Execute test ==========
    # 1. Invoke your Triton kernel wrapper
    apply_min_p(triton_logits, expanded_idx_mapping, min_p)
    torch.cuda.synchronize()

    # 2. Compute reference values using PyTorch
    ref_logits = torch_min_p_torch(
        ref_logits,
        expanded_idx_mapping,
        min_p,
    )

    # ========== Verify results ==========
    triton_inf_mask = torch.isinf(triton_logits)
    ref_inf_mask = torch.isinf(ref_logits)

    assert torch.equal(triton_inf_mask, ref_inf_mask), (
        "Masked positions (where logits == -inf) do not match between Triton and PyTorch."
    )

    valid_triton_logits = triton_logits[~triton_inf_mask]
    valid_ref_logits = ref_logits[~ref_inf_mask]

    assert torch.allclose(valid_triton_logits, valid_ref_logits, rtol=1e-4, atol=1e-4), (
        f"Logits values differ between Triton and PyTorch reference.\n"
        f"Max diff: {torch.max(torch.abs(valid_triton_logits - valid_ref_logits))}"
    )
