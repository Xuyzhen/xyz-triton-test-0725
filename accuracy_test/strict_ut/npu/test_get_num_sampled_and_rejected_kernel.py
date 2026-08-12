# GENERATED STRICT UT. Source: accuracy_test/codex/missing_accuracy_tests/test_get_num_sampled_and_rejected_kernel.py
# Do not edit mechanically; update the reviewed Codex source or strict generator.
from accuracy_test.strict_ut.runtime_npu import STRICT_DEVICE as _STRICT_DEVICE
# vLLM vanilla kernel: _get_num_sampled_and_rejected_kernel from
# vllm/vllm/v1/worker/gpu/input_batch.py

"""
Precision test for _get_num_sampled_and_rejected_kernel.

Computes the number of rejected tokens as:
    num_rejected = num_logits - num_sampled
For chunked-prefilling requests (seq_len < prefill_len), both
num_sampled and num_rejected are set to 0.

Kernel signature:
    _get_num_sampled_and_rejected_kernel(
        num_sampled_ptr,    # [num_reqs] int32 - updated in-place
        num_rejected_ptr,   # [num_reqs] int32 - output
        seq_lens_ptr,       # [num_reqs] int32
        cu_num_logits_ptr,  # [num_reqs + 1] int64
        idx_mapping_ptr,    # [num_reqs] int32
        prefill_len_ptr,    # [max_num_reqs] int32
    )
"""

import torch
import pytest

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import _get_num_sampled_and_rejected_kernel
from accuracy_test.strict_ut.runtime_npu import init_device_properties_triton


def _get_num_sampled_and_rejected_ref(
    num_sampled,
    seq_lens,
    cu_num_logits,
    idx_mapping,
    prefill_len,
):
    """CPU reference for _get_num_sampled_and_rejected_kernel."""
    out_num_sampled = num_sampled.clone()
    num_reqs = idx_mapping.shape[0]
    out_num_rejected = torch.zeros(num_reqs, dtype=torch.int32)

    for batch_idx in range(num_reqs):
        req_state_idx = int(idx_mapping[batch_idx].item())
        seq_len = int(seq_lens[batch_idx].item())
        prefill_len_ = int(prefill_len[req_state_idx].item())
        is_chunked_prefilling = seq_len < prefill_len_

        n_sampled = int(num_sampled[batch_idx].item())
        if is_chunked_prefilling:
            out_num_sampled[batch_idx] = 0
        else:
            out_num_sampled[batch_idx] = n_sampled

        logits_start = int(cu_num_logits[batch_idx].item())
        logits_end = int(cu_num_logits[batch_idx + 1].item())
        num_logits = logits_end - logits_start

        num_rejected = num_logits - n_sampled
        if is_chunked_prefilling:
            out_num_rejected[batch_idx] = 0
        else:
            out_num_rejected[batch_idx] = num_rejected

    return out_num_sampled, out_num_rejected


class TestGetNumSampledAndRejectedKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_reqs", [1, 2, 4])
    @pytest.mark.parametrize("num_logits_per_req", [1, 3, 5])
    def test_basic(self, num_reqs, num_logits_per_req):
        """Basic test: computing rejected count from sampled and logits."""
        max_num_reqs = max(num_reqs, 4)

        num_sampled = torch.randint(0, num_logits_per_req, (num_reqs,), dtype=torch.int32, device=self.device)
        seq_lens = torch.full((num_reqs,), 20, dtype=torch.int32, device=self.device)
        cu_num_logits = torch.arange(0, num_reqs * num_logits_per_req + 1,
                                     num_logits_per_req, dtype=torch.int64, device=self.device)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        prefill_len = torch.full((max_num_reqs,), 10, dtype=torch.int32, device=self.device)

        num_rejected = torch.empty(num_reqs, dtype=torch.int32, device=self.device)

        ref_ns, ref_nr = _get_num_sampled_and_rejected_ref(
            num_sampled.cpu(), seq_lens.cpu(), cu_num_logits.cpu(),
            idx_mapping.cpu(), prefill_len.cpu(),
        )

        _get_num_sampled_and_rejected_kernel[(num_reqs,)](
            num_sampled,
            num_rejected,
            seq_lens,
            cu_num_logits,
            idx_mapping,
            prefill_len,
        )
        torch.npu.synchronize()

        torch.testing.assert_close(num_sampled.cpu(), ref_ns, rtol=0, atol=0)
        torch.testing.assert_close(num_rejected.cpu(), ref_nr, rtol=0, atol=0)

    def test_chunked_prefilling(self):
        """When seq_len < prefill_len, both num_sampled and num_rejected should be 0."""
        num_reqs = 2
        max_num_reqs = 2

        num_sampled = torch.tensor([3, 5], dtype=torch.int32, device=self.device)
        seq_lens = torch.tensor([3, 8], dtype=torch.int32, device=self.device)
        cu_num_logits = torch.tensor([0, 4, 10], dtype=torch.int64, device=self.device)
        idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
        # Both requests: seq_len (3, 8) < prefill_len (10, 10) => chunked prefill
        prefill_len = torch.full((max_num_reqs,), 10, dtype=torch.int32, device=self.device)

        num_rejected = torch.empty(num_reqs, dtype=torch.int32, device=self.device)

        _get_num_sampled_and_rejected_kernel[(num_reqs,)](
            num_sampled,
            num_rejected,
            seq_lens,
            cu_num_logits,
            idx_mapping,
            prefill_len,
        )
        torch.npu.synchronize()

        assert torch.all(num_sampled.cpu() == 0), "num_sampled should be 0 for chunked prefilling"
        assert torch.all(num_rejected.cpu() == 0), "num_rejected should be 0 for chunked prefilling"

    @pytest.mark.parametrize("num_sampled_val, expected_rejected", [(0, 3), (2, 1), (3, 0)])
    def test_various_sampled_counts(self, num_sampled_val, expected_rejected):
        """Test various sampled counts produce correct rejected counts."""
        num_reqs = 1
        max_num_reqs = 2
        num_logits = 3

        num_sampled = torch.tensor([num_sampled_val], dtype=torch.int32, device=self.device)
        seq_lens = torch.tensor([20], dtype=torch.int32, device=self.device)
        cu_num_logits = torch.tensor([0, num_logits], dtype=torch.int64, device=self.device)
        idx_mapping = torch.tensor([0], dtype=torch.int32, device=self.device)
        prefill_len = torch.full((max_num_reqs,), 10, dtype=torch.int32, device=self.device)

        num_rejected = torch.empty(num_reqs, dtype=torch.int32, device=self.device)

        _get_num_sampled_and_rejected_kernel[(num_reqs,)](
            num_sampled,
            num_rejected,
            seq_lens,
            cu_num_logits,
            idx_mapping,
            prefill_len,
        )
        torch.npu.synchronize()

        assert num_sampled[0].item() == num_sampled_val
        assert num_rejected[0].item() == expected_rejected, \
            f"Expected rejected={expected_rejected}, got {num_rejected[0].item()}"
