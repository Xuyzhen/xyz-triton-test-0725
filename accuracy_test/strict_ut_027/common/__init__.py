# SPDX-License-Identifier: Apache-2.0
"""Shared dual-side test implementations for strict_ut_027.

Each ``*_impl.py`` module contains the complete test implementation
(kernel import, input generation, CPU reference, launcher, assertions and
the pytest entry points) for one operator. The device-specific runtime is
injected through the ``rt`` fixture defined in gpu/conftest.py and
npu/conftest.py, so the exact same cases run on CUDA and Ascend NPU.
"""
