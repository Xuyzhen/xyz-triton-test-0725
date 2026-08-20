# GENERATED STRICT UT. Source: vllm/v1/worker/gpu/input_batch.py::_post_update_kernel
# Do not edit mechanically; update the reviewed Codex source or strict generator.
from accuracy_test.strict_ut.runtime_npu import STRICT_DEVICE as _STRICT_DEVICE
# Standalone Ascend NPU accuracy test.
# Accuracy UT source: vllm/v1/worker/gpu/input_batch.py (upstream kernel, no NPU patch).
# Kernel source: vllm/vllm/v1/worker/gpu/input_batch.py
# Coverage: precision of upstream _post_update_kernel on Ascend NPU against an
#           independent CPU reference. vllm-ascend previously shipped a patched
#           _post_update_kernel, but it was removed in commit 3281a5fc4 ("Support
#           pipeline parallelism on Ascend", 2026-08-01). Only the upstream
#           kernel is tested here.

import traceback
from typing import Any

import pytest
import torch

post_update_kernel = None
_post_update_import_error = None
_post_update_import_traceback = None
try:
    from vllm.v1.worker.gpu.input_batch import _post_update_kernel as post_update_kernel
except Exception as exc:
    _post_update_import_error = exc
    _post_update_import_traceback = traceback.format_exc()


# ---------------------------------------------------------------------------
# CPU reference implementation (independent of the kernel under test).
# Mirrors the semantics of the upstream kernel, including the optional-argument
# branches introduced for pipeline parallelism (output_bin_counts=None,
# query_start_loc=None, negative idx_mapping entries).
# ---------------------------------------------------------------------------
def post_update_ref(
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    output_bin_counts: torch.Tensor | None,
    sampled_tokens: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
) -> None:
    for row_idx in range(idx_mapping.shape[0]):
        req_state_idx = int(idx_mapping[row_idx].item())
        if req_state_idx < 0:
            continue

        old_total_len = int(total_len[req_state_idx].item())
        sampled_count = int(num_sampled[row_idx].item())

        if sampled_count > 0:
            last_sampled_tokens[req_state_idx] = sampled_tokens[
                row_idx, sampled_count - 1
            ].item()
            total_len[req_state_idx] = old_total_len + sampled_count

        for i in range(sampled_count):
            token_id = int(sampled_tokens[row_idx, i].item())
            all_token_ids[req_state_idx, old_total_len + i] = token_id
            if output_bin_counts is not None:
                output_bin_counts[req_state_idx, token_id] += 1

        if query_start_loc is None:
            query_len = 0
        else:
            query_len = int(query_start_loc[row_idx + 1].item()) - int(
                query_start_loc[row_idx].item()
            )
        rejected_count = int(num_rejected[row_idx].item())
        delta = query_len - rejected_count
        if delta != 0:
            num_computed_tokens[req_state_idx] = (
                int(num_computed_tokens[req_state_idx].item()) + delta
            )


def _gen_inputs(
    num_reqs: int,
    max_num_reqs: int,
    vocab_size: int,
    num_speculative_steps: int,
    max_model_len: int,
    *,
    with_output_bin_counts: bool,
    with_query_start_loc: bool,
    with_negative_idx: bool,
    device: str,
) -> dict[str, Any]:
    """Build a batch that mirrors a real MRV2 post_update call.

    Shapes follow vllm's GPUModelRunner.post_update:
      idx_mapping       [num_reqs]                       int32
      num_computed_tokens [max_num_reqs]                 int32
      last_sampled_tokens [max_num_reqs]                 int32
      output_bin_counts [max_num_reqs, vocab_size]       int32 (or None)
      sampled_tokens    [num_reqs, num_spec_steps+1]     int32
      num_sampled       [num_reqs]                       int32  in [0, spec_steps+1]
      num_rejected      [num_reqs]                       int32  in [0, num_sampled]
      query_start_loc   [num_reqs+1]                     int32 (or None)
      all_token_ids     [max_num_reqs, max_model_len]    int32
      total_len         [max_num_reqs]                   int32  in [0, max_model_len - spec_steps - 1]
    """
    if num_reqs > max_num_reqs:
        raise ValueError("num_reqs cannot be larger than max_num_reqs")

    # idx_mapping: contiguous [0..num_reqs-1]; optionally mark one row as -1 (PP skip).
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    if with_negative_idx and num_reqs >= 2:
        # Mark the middle request as skipped (mirrors non-last PP rank replay).
        idx_mapping[num_reqs // 2] = -1

    num_computed_tokens = torch.randint(
        0, 100, (max_num_reqs,), dtype=torch.int32, device=device
    )
    last_sampled_tokens = torch.randint(
        0, vocab_size, (max_num_reqs,), dtype=torch.int32, device=device
    )

    # sampled_tokens in valid token-id range
    sampled_tokens = torch.randint(
        0, vocab_size, (num_reqs, num_speculative_steps + 1), dtype=torch.int32, device=device
    )

    # num_sampled: 0 ~ num_speculative_steps+1
    num_sampled = torch.randint(
        1, num_speculative_steps + 2, (num_reqs,), dtype=torch.int32, device=device
    )
    # num_rejected must be in [0, num_sampled-1] (cannot reject more than sampled-1).
    num_rejected = torch.randint(
        0, num_speculative_steps + 1, (num_reqs,), dtype=torch.int32, device=device
    )
    num_rejected = torch.min(num_rejected, num_sampled - 1)

    # Build a contiguous query_start_loc whose per-row query_len >= num_rejected
    # so the num_computed_tokens delta is non-negative (matches real execution).
    query_lengths = num_rejected + torch.randint(
        0, 5, (num_reqs,), dtype=torch.int32, device=device
    )
    query_start_loc = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.cumsum(query_lengths, dim=0).to(torch.int32),
        ]
    )

    # total_len must leave enough room for sampled_count new tokens.
    max_initial_total = max(1, max_model_len - (num_speculative_steps + 1))
    total_len = torch.randint(
        0, max_initial_total, (max_num_reqs,), dtype=torch.int32, device=device
    )

    # all_token_ids: sentinel so we can verify unwritten slots stay untouched.
    sentinel = -1
    all_token_ids = torch.full(
        (max_num_reqs, max_model_len), sentinel, dtype=torch.int32, device=device
    )

    # output_bin_counts starts at zero so increments are observable.
    output_bin_counts = (
        torch.zeros((max_num_reqs, vocab_size), dtype=torch.int32, device=device)
        if with_output_bin_counts
        else None
    )

    return {
        "idx_mapping": idx_mapping,
        "num_computed_tokens": num_computed_tokens,
        "last_sampled_tokens": last_sampled_tokens,
        "output_bin_counts": output_bin_counts,
        "sampled_tokens": sampled_tokens,
        "num_sampled": num_sampled,
        "num_rejected": num_rejected,
        "query_start_loc": query_start_loc if with_query_start_loc else None,
        "all_token_ids": all_token_ids,
        "total_len": total_len,
    }


def _launch(kernel, inputs: dict[str, Any]) -> None:
    """Invoke the upstream _post_update_kernel using its real JIT signature."""
    output_bin_counts = inputs["output_bin_counts"]
    query_start_loc = inputs["query_start_loc"]
    num_reqs = inputs["idx_mapping"].shape[0]
    kernel[(num_reqs,)](
        inputs["idx_mapping"],
        inputs["num_computed_tokens"],
        inputs["last_sampled_tokens"],
        output_bin_counts,
        output_bin_counts.stride(0) if output_bin_counts is not None else 0,
        inputs["sampled_tokens"],
        inputs["sampled_tokens"].stride(0),
        inputs["num_sampled"],
        inputs["num_rejected"],
        query_start_loc,
        inputs["all_token_ids"],
        inputs["all_token_ids"].stride(0),
        inputs["total_len"],
        num_warps=1,
    )


# ---------------------------------------------------------------------------
# Test matrix.
#
# Realistic shapes (mirrors vllm default MRV2 configuration):
#   - max_num_reqs in {32, 128}      (small / large batch)
#   - vocab_size in {32000, 128256}  (Llama-3 8B / Qwen2 7B)
#   - num_speculative_steps in {2,5} (typical MTP/EAGLE-2 settings)
#   - max_model_len = 4096           (short-context decode workload)
#
# Branch coverage:
#   - last PP rank:    with_output_bin_counts=True, with_query_start_loc=True
#   - non-last PP rank: with_output_bin_counts=False, with_query_start_loc=False
#   - PP skip rows:    with_negative_idx=True
# ---------------------------------------------------------------------------
PARAMS = [
    # (num_reqs, max_num_reqs, vocab_size, num_speculative_steps, max_model_len)
    (32, 32, 32000, 2, 4096),
    (128, 128, 128256, 5, 4096),
]


@pytest.mark.parametrize(
    "num_reqs,max_num_reqs,vocab_size,num_speculative_steps,max_model_len",
    PARAMS,
)
@pytest.mark.parametrize(
    "with_output_bin_counts,with_query_start_loc,with_negative_idx",
    [
        # last PP rank: full bookkeeping
        (True, True, False),
        # last PP rank with a skipped row (e.g. PP replay)
        (True, True, True),
        # non-last PP rank: no output_bin_counts, no query_start_loc
        (False, False, False),
        # non-last PP rank with a skipped row
        (False, False, True),
        # mixed: output_bin_counts present, query_start_loc absent (defensive)
        (True, False, False),
    ],
)
def test_post_update(
    num_reqs: int,
    max_num_reqs: int,
    vocab_size: int,
    num_speculative_steps: int,
    max_model_len: int,
    with_output_bin_counts: bool,
    with_query_start_loc: bool,
    with_negative_idx: bool,
):
    """Compare upstream _post_update_kernel (NPU) against an independent CPU ref."""
    if post_update_kernel is None:
        pytest.fail(
            "post_update environment compatibility failure; this is not a "
            "precision failure and no kernel was tested.\n"
            f"error={_post_update_import_error}\n"
            f"traceback:\n{_post_update_import_traceback}",
            pytrace=False,
        )

    torch.manual_seed(42)
    device = str(_STRICT_DEVICE)

    inputs = _gen_inputs(
        num_reqs=num_reqs,
        max_num_reqs=max_num_reqs,
        vocab_size=vocab_size,
        num_speculative_steps=num_speculative_steps,
        max_model_len=max_model_len,
        with_output_bin_counts=with_output_bin_counts,
        with_query_start_loc=with_query_start_loc,
        with_negative_idx=with_negative_idx,
        device=device,
    )

    # CPU reference: clone everything to CPU (int32 -> exact equality later).
    param_names = [
        "idx_mapping",
        "num_computed_tokens",
        "last_sampled_tokens",
        "output_bin_counts",
        "sampled_tokens",
        "num_sampled",
        "num_rejected",
        "query_start_loc",
        "all_token_ids",
        "total_len",
    ]
    ref_inputs: dict[str, Any] = {}
    for k in param_names:
        v = inputs[k]
        ref_inputs[k] = v.cpu().clone() if v is not None else None
    post_update_ref(**ref_inputs)

    # NPU run: clone inputs so the originals stay pristine.
    npu_inputs: dict[str, Any] = {}
    for k in param_names:
        v = inputs[k]
        npu_inputs[k] = v.clone() if v is not None else None
    _launch(post_update_kernel, npu_inputs)
    torch.npu.synchronize()

    # Compare every mutated tensor with rtol=0/atol=0 (int32 semantics).
    mutated_outputs = (
        "num_computed_tokens",
        "last_sampled_tokens",
        "output_bin_counts",
        "all_token_ids",
        "total_len",
    )
    for name in mutated_outputs:
        if not with_output_bin_counts and name == "output_bin_counts":
            assert npu_inputs[name] is None
            assert ref_inputs[name] is None
            continue

        npu_out = npu_inputs[name].cpu()
        ref_out = ref_inputs[name]
        assert npu_out.dtype == ref_out.dtype, f"{name}: dtype mismatch"
        assert npu_out.shape == ref_out.shape, f"{name}: shape mismatch"
        if not torch.equal(npu_out, ref_out):
            mismatched = torch.ne(npu_out, ref_out)
            count = int(mismatched.sum().item())
            total = int(mismatched.numel())
            first_idx = torch.nonzero(mismatched, as_tuple=False)
            first_info = ""
            if first_idx.numel() > 0:
                loc = tuple(first_idx[0].tolist())
                first_info = (
                    f"; first mismatch at {loc}: "
                    f"npu={npu_out[loc].item()} ref={ref_out[loc].item()}"
                )
            pytest.fail(
                f"_post_update_kernel mismatch for {name}: "
                f"{count}/{total} elements differ{first_info}"
            )
