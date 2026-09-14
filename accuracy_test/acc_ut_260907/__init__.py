# SPDX-License-Identifier: Apache-2.0
"""acc_ut_260907: consolidated NPU operator accuracy UT suite.

Aggregates the latest version of every NPU operator precision UT scattered
under ``accuracy_test/`` (strict_ut_028 + strict_ut_027 + strict_ut-only +
codex-only operators + num_nans specials). Each operator keeps only its
most recent adaptation; operators tested both upstream (vllm) and
downstream (vllm_ascend) carry ``_upstream`` / ``_downstream`` filename
suffixes.

Standalone by design: the only imports allowed are vllm, vllm_ascend and
their dependency chain (torch / triton / pytest). No sibling
``accuracy_test.*`` package is referenced.
"""
