"""strict_ut_026: standalone strict accuracy suite + GPU-NPU precision harness.

This package is a complete, self-contained snapshot of ``strict_ut`` (pytest
suite under gpu/ and npu/ plus all runtime scaffolding) extended with the
``precision/`` capture-and-compare toolkit. It runs on its own when copied
anywhere - nothing here imports from sibling suites.
"""
