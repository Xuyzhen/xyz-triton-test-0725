# SPDX-License-Identifier: Apache-2.0
"""Shared test implementations for acc_ut_260914_f028_main0914.

Each ``*_impl.py`` module contains the complete test implementation
(kernel import, input generation, CPU reference, launcher, assertions and
the pytest entry points) for one operator. The device-specific runtime is
injected through the ``rt`` fixture defined in gpu/conftest.py, so the exact
same cases run on CUDA (and could be reused on Ascend NPU).
"""
