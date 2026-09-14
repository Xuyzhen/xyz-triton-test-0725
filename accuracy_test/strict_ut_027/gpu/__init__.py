# SPDX-License-Identifier: Apache-2.0
"""GPU-side (CUDA) test entries for strict_ut_027.

Each ``test_*.py`` here re-exports the shared implementation from
``strict_ut_027/common/*_impl.py``. The ``rt`` fixture (defined in this
directory's conftest.py) injects the CUDA runtime, so the exact same
cases, references and bitwise assertions run on the GPU side.
"""
