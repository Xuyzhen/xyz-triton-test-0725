# SPDX-License-Identifier: Apache-2.0
"""acc_ut_260914_f028_main0914: dual-side (GPU + Ascend NPU) accuracy UT for 5 new vLLM Triton kernels.

Standalone by design: the only imports allowed are vllm / vllm-ascend and
their dependency chain (torch / torch_npu / triton / pytest). No sibling
``accuracy_test.*`` package is referenced by the runtime shims.

Layout (mirrors acc_ut_260907):
  - common/     shared implementations (CPU reference + device launch),
                backend injected via the ``rt`` pytest fixture
  - gpu/        CUDA-side test entries (run via run_gpu.sh)
  - npu/        Ascend NPU-side test entries (run via run_npu.sh, which
                isolates each test file in a fresh process because an
                Ascend vector-core exception poisons the process device
                context)

Kernels covered:
  1. postprocess_mamba_fused_kernel  (mamba_utils.py)
  2. _fill_num_accepted_kernel       (mamba_hybrid.py)
  3. _pack_sampling_mask_kernel       (sample/output.py, base commit 50ba4bc6b2)
  4. _selector_walk_kernel            (dflash2/speculator.py)
  5. _cache_draft_logits_kernel       (dflash2/speculator.py)
"""
