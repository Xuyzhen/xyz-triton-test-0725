# SPDX-License-Identifier: Apache-2.0
"""strict_ut_027: dual-side (CUDA GPU + Ascend NPU) strict operator
accuracy suite.

Built from the easy_ut_026 NPU-only tests (excluding test_num_nans_kernel,
test_topk_topp_kernel_a2a3 and test_topk_topp_kernel_a5) with extended,
higher-spec shapes, following the strict_ut_028 project layout. No dual
reference-standard (双标杆) logic is included: every test compares the
device kernel output against an independent CPU reference.
"""
