# SPDX-License-Identifier: Apache-2.0
"""Shape constraint validator for acc_ut_260914_f028_main0914.

Verifies that all test parametrizations use shapes consistent with the real
production constraints of each kernel. Run this script standalone after the
test suite passes:

    python -m accuracy_test.acc_ut_260914_f028_main0914.check_shape_constraints

Checks:
  1. postprocess_mamba_fused_kernel: num_reqs in [1,512], block_size in {16,32,64,128}
  2. _fill_num_accepted_kernel: max_num_reqs >= num_reqs, num_sampled >= 0
  3. _pack_sampling_mask_kernel: BLOCK_SIZE >= 8 and power-of-2, vocab_size > 0
  4. _selector_walk_kernel: top_k >= 1, num_steps >= 1, num_reqs >= 1
  5. _cache_draft_logits_kernel: max_num_reqs >= num_reqs, vocab_size >= top_k
"""

from __future__ import annotations

import sys


# Import the parametrization lists from each impl module
def _check_postprocess():
    from accuracy_test.acc_ut_260914_f028_main0914.common.postprocess_mamba_fused_impl import (
        SHAPE_PARAMS as SP,
        BRANCH_PARAMS as BP,
    )
    errors = []
    for num_reqs, block_size in SP:
        if not (1 <= num_reqs <= 512):
            errors.append(f"postprocess: num_reqs={num_reqs} out of [1,512]")
        if block_size not in (16, 32, 64, 128):
            errors.append(f"postprocess: block_size={block_size} not in {{16,32,64,128}}")
    for precomputed, has_idx, zero_acc, same_block in BP:
        if not isinstance(precomputed, bool):
            errors.append(f"postprocess: precomputed must be bool, got {type(precomputed)}")
    return errors


def _check_fill_num_accepted():
    from accuracy_test.acc_ut_260914_f028_main0914.common.fill_num_accepted_impl import (
        SHAPE_PARAMS as SP,
    )
    errors = []
    for num_reqs, max_num_reqs, num_sampled in SP:
        if max_num_reqs < num_reqs:
            errors.append(f"fill_num_accepted: max_num_reqs={max_num_reqs} < num_reqs={num_reqs}")
        if num_sampled < 0:
            errors.append(f"fill_num_accepted: num_sampled={num_sampled} < 0")
    return errors


def _check_pack_sampling_mask():
    from accuracy_test.acc_ut_260914_f028_main0914.common.pack_sampling_mask_impl import (
        SHAPE_PARAMS as SP,
    )
    errors = []
    for num_reqs, vocab_size, block_size in SP:
        if vocab_size <= 0:
            errors.append(f"pack_sampling_mask: vocab_size={vocab_size} <= 0")
        if block_size < 8:
            errors.append(f"pack_sampling_mask: BLOCK_SIZE={block_size} < 8")
        if block_size & (block_size - 1) != 0:
            errors.append(f"pack_sampling_mask: BLOCK_SIZE={block_size} not power-of-2")
        if num_reqs < 1:
            errors.append(f"pack_sampling_mask: num_reqs={num_reqs} < 1")
    return errors


def _check_selector_walk():
    from accuracy_test.acc_ut_260914_f028_main0914.common.selector_walk_impl import (
        SHAPE_PARAMS as SP,
    )
    errors = []
    for num_reqs, num_steps, top_k in SP:
        if top_k < 1:
            errors.append(f"selector_walk: top_k={top_k} < 1")
        if num_steps < 1:
            errors.append(f"selector_walk: num_steps={num_steps} < 1")
        if num_reqs < 1:
            errors.append(f"selector_walk: num_reqs={num_reqs} < 1")
        # top_k must be <= 64 (triton.next_power_of_2 constraint)
        if top_k > 64:
            errors.append(f"selector_walk: top_k={top_k} > 64 (BLOCK_K too large)")
    return errors


def _check_cache_draft_logits():
    from accuracy_test.acc_ut_260914_f028_main0914.common.cache_draft_logits_impl import (
        SHAPE_PARAMS as SP,
    )
    errors = []
    for num_reqs, num_steps, top_k, max_num_reqs, vocab_size in SP:
        if max_num_reqs < num_reqs:
            errors.append(
                f"cache_draft_logits: max_num_reqs={max_num_reqs} < num_reqs={num_reqs}"
            )
        if vocab_size < top_k:
            errors.append(
                f"cache_draft_logits: vocab_size={vocab_size} < top_k={top_k}"
            )
        if top_k < 1:
            errors.append(f"cache_draft_logits: top_k={top_k} < 1")
        if num_steps < 1:
            errors.append(f"cache_draft_logits: num_steps={num_steps} < 1")
    return errors


def main():
    all_errors = []
    for name, checker in [
        ("postprocess_mamba_fused", _check_postprocess),
        ("_fill_num_accepted", _check_fill_num_accepted),
        ("_pack_sampling_mask", _check_pack_sampling_mask),
        ("_selector_walk", _check_selector_walk),
        ("_cache_draft_logits", _check_cache_draft_logits),
    ]:
        errs = checker()
        if errs:
            all_errors.extend(f"[{name}] {e}" for e in errs)
            print(f"FAIL: {name}: {len(errs)} constraint violations")
            for e in errs:
                print(f"  - {e}")
        else:
            print(f"PASS: {name}: all shapes within production constraints")

    if all_errors:
        print(f"\nTOTAL: {len(all_errors)} shape constraint violations")
        sys.exit(1)
    else:
        print("\nAll shape constraints satisfied.")
        sys.exit(0)


if __name__ == "__main__":
    main()
