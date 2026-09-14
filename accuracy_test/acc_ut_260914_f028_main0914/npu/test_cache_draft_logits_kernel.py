# SPDX-License-Identifier: Apache-2.0
"""NPU-side entry for _cache_draft_logits_kernel (shared implementation)."""
from accuracy_test.acc_ut_260914_f028_main0914.common.cache_draft_logits_impl import (  # noqa: F401
    test_cache_draft_logits,
    test_cache_draft_logits_first_round,
    test_cache_draft_logits_incremental_cleanup,
    test_cache_draft_logits_skip_invalid,
    test_import_error,
)
