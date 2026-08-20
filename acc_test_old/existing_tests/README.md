# vLLM + vLLM-Ascend 已有算子精度测试记录

## 说明

本文档记录 vLLM 和 vLLM-Ascend 项目中**已经存在**的算子精度 UT 测试。每个测试文件记录其所在仓库路径、测试了什么算子和核函数、测试函数名，以及与 `acc_test/all_kernels/` 中我们手写测试的对应关系。

目的：
1. 了解哪些算子已有官方/第三方测试，可以直接复用
2. 发现哪些算子缺失测试（既无项目内测试，也无我们手写的测试）
3. 为后续测试维护提供对照索引

---

## 一、vLLM-Ascend 项目中已有的单算子精度测试

位置：`vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/`

### 覆盖的算子对照表

| 序号 | 目标算子 | 已有测试文件 | 直接核函数测试? | 备注 |
|------|---------|-------------|:--------------:|------|
| P1 | `_bad_words_kernel` (patch) | [test_bad_words.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_bad_words.py) | ⚠️ 间接 | 通过 `apply_bad_words` 封装调用，3种规模+边界，`@pytest.mark.skip` 缺失 |
| P2 | `_temperature_kernel` (patch) | [test_temperature.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_temperature.py) | ⚠️ 间接 | 通过 `apply_temperature` 调用，1个参考对比测试 |
| P3 | `_gumbel_sample_kernel` (patch) | [test_gumbel_sampling.py](vllm-ascend-xyz/tests/ut/sample/a2/test_gumbel_sampling.py) | ⚠️ 间接 | 通过 `gumbel_sample` 调用，14个测试用例，较完整 |
| P4 | `_topk_log_softmax_kernel` (patch) | [test_log_softmax.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_log_softmax.py) | ✅ 直接 | 直接调 `_topk_log_softmax_kernel[(batch_size,)]`，3种参数 |
| P5 | `_ranks_kernel` (patch) | ⛔ 无直接测试 | - | 已被 `test_compute_topk_logprobs` 间接覆盖 |
| P6 | `_min_p_kernel` (patch) | [test_min_p.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_min_p.py) | ⚠️ 间接 | 通过 `apply_min_p` 调用，4种参数 |
| P7 | `_penalties_kernel` (patch) | [test_penality.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_penality.py) | ⚠️ 间接 | 通过 `apply_penalties` 调用，含 PyTorch 参考实现 |
| P8 | `_bincount_kernel` (patch) | [test_bincount.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_bincount.py) | ✅ 直接 | 直接调 `_bincount_kernel`，含 PyTorch 参考，标注 `@pytest.mark.skip` |
| P9 | `_resample_kernel` (patch) | [test_rejection_sample.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rejection_sample.py) | ✅ 部分 | 测试了 `rejection_random_sample_kernel` + `rejection_random_sample_block_verify_kernel`，2个函数含 skip |
| P10 | `_compute_slot_mappings_kernel` (patch) | [test_compute_slot_mapping.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_slot_mapping.py) | ✅ 直接 | 直接调 ascend 版本并与 GPU 参考对比，1个测试 |
| P11 | `_post_update_kernel` (patch) | [test_post_update.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_post_update.py) | ⚠️ 间接 | 通过 `post_update` 函数对比 GPU v1 版本，3种参数 |
| P12 | `_apply_grammar_bitmask_kernel` (patch) | ⛔ 无测试 | - | - |
| P13 | `_zero_kv_blocks_kernel` (patch) | ⛔ 无测试 | - | - |
| P14 | `_prepare_dflash_inputs_kernel_ascend` (patch) | ⛔ 无测试 | - | - |
| P15 | `_npu_gumbel_block_argmax` (patch) | ⛔ 无直接测试 | - | 被 `test_gumbel_sampling.py` 间接覆盖 |
| P16 | `_probabilistic_rejection_kernel` (patch) | ⛔ 无测试 | - | `test_rejection_sample.py` 测试的是不同的 kernel |

### 其他已有测试的算子（不在我们60个目标内的 Ascend 算子）

| 文件 | 测试的算子 | 备注 |
|------|-----------|------|
| [test_batch_memcpy.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_batch_memcpy.py) | `batch_memcpy_kernel` | Mamba state copy，参数化 dtype |
| [test_causal_conv1d.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_causal_conv1d.py) | `causal_conv1d` | - |
| [test_fused_gdn_gating.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_fused_gdn_gating.py) | `fused_gdn_gating` | - |
| [test_fused_qkvzba_split_reshape_cat.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_fused_qkvzba_split_reshape_cat.py) | `fused_qkvzba_split_reshape_cat` | - |
| [test_l2norm.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_l2norm.py) | `l2norm_fwd` | L2 Normalization |
| [test_lightning_attn.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_lightning_attn.py) | 4个 lightning attention Triton 核 | `_fwd_diag_kernel`, `_fwd_kv_parallel`, `_fwd_kv_reduce`, `_fwd_none_diag_kernel` |
| [test_mrope.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mrope.py) | `triton_mrope` | M-RoPE 嵌入 |
| [test_muls_add.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_muls_add.py) | `muls_add_triton` | - |
| [test_apply_penalties_triton.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_apply_penalties_triton.py) | `apply_all_penalties`（组合 penalty 函数） | 对比 v1 GPU 版本 |
| [test_compute_token_logprobs.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_token_logprobs.py) | `compute_token_logprobs`（封装 2 个核函数） | 含边界测试，全部 skip |
| [test_compute_topk_logprobs.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_topk_logprobs.py) | `compute_topk_logprobs`（封装多个核函数） | 含 ranks/logprobs/token_ids 对比 |
| [test_postprocess_mamba.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_postprocess_mamba.py) | `postprocess_mamba` | - |
| [test_prepare_inputs_padded.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_prepare_inputs_padded.py) | `prepare_inputs_padded_kernel` | 对比 PyTorch 参考 |
| [test_rope.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rope.py) | `rope_forward_triton`, `rope_forward_triton_siso` | 3个测试函数，参数量大 |
| [test_split_qkv_rmsnorm_mrope.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_split_qkv_rmsnorm_mrope.py) | `split_qkv_rmsnorm_mrope` | - |
| [test_split_qkv_rmsnorm_rope.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_split_qkv_rmsnorm_rope.py) | `split_qkv_rmsnorm_rope` | - |
| [test_split_qkv_tp_rmsnorm_rope.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_split_qkv_tp_rmsnorm_rope.py) | `split_qkv_tp_rmsnorm_rope` | - |
| [test_swiglustep.py](vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_swiglustep.py) | `swiglustep` | - |

---

## 二、vLLM 项目中已有的单算子精度测试

位置：`vllm/tests/`

### 与 60 个目标算子相关的测试

| 目标算子 | 已有测试文件 | 测试方式 | 备注 |
|---------|-------------|---------|------|
| `_rejection_kernel` (#24) | [test_rejection_sampler_utils.py](vllm/tests/v1/spec_decode/test_rejection_sampler_utils.py) | `rejection_sample` 函数（封装 kernel） | 基于概率分布检验，需 CUDA |
| `_flatten_sampled_kernel` (#27) | [test_rejection_sampler_utils.py](vllm/tests/v1/spec_decode/test_rejection_sampler_utils.py) | 间接测试 | 通过 batch rejection sampler 测试 |
| `_resample_kernel` (#25) | [test_rejection_sampler_utils.py](vllm/tests/v1/spec_decode/test_rejection_sampler_utils.py) | 间接测试 | 同上 |
| `_insert_resampled_kernel` (#26) | [test_rejection_sampler_utils.py](vllm/tests/v1/spec_decode/test_rejection_sampler_utils.py) | 间接测试 | 同上 |
| `_gather_block_tables_kernel` (#28) | [test_gpu_block_table.py](vllm/tests/v1/worker/test_gpu_block_table.py) | 通过 `BlockTables` 封装测试 | 需要 CUDA |
| `_compute_slot_mappings_kernel` (#29) | [test_gpu_block_table.py](vllm/tests/v1/worker/test_gpu_block_table.py) | 间接测试 | 同上 |
| `_apply_write_kernel` (#30) | [test_gpu_block_table.py](vllm/tests/v1/worker/test_gpu_block_table.py) | 通过 `apply_staged_writes` | 同上 |
| `_topk_topp_kernel` (#41) | [test_topk_topp_sampler.py](vllm/tests/v1/sample/test_topk_topp_sampler.py) | 通过 `Sampler` 集成测试 | 不是单算子测试 |
| `_bad_words_kernel` (#4) | [test_sampler.py](vllm/tests/v1/sample/test_sampler.py) | 通过 `LogitsProcessors` 集成测试 | 不是单算子测试 |

### 其他无关的测试分类

| 测试目录 | 内容 | 无关原因 |
|---------|------|---------|
| `tests/kernels/attention/` | attention 相关 Triton 测试 | 测试的是 attention 核函数（flash attention 等），不在我们 60 个目标内 |
| `tests/kernels/moe/` | MoE 相关 Triton 测试 | 不在目标内 |
| `tests/kernels/quantization/` | 量化相关 Triton 测试 | 不在目标内 |
| `tests/kernels/test_cache_kernels.py` | KV cache 操作 | 不在目标内 |
| `tests/kernels/test_mhc_kernels.py` | MHC 核函数 | 不在目标内 |
| `tests/samplers/test_logprobs.py` | logprobs 采样器集成测试 | 集成测试，非单算子 |
| `tests/samplers/test_no_bad_words.py` | bad words 集成测试 | 集成测试，非单算子 |

---

## 三、测试覆盖状态汇总

### 60 个算子测试覆盖矩阵

| 状态 | 含义 | 数量 |
|------|------|:----:|
| ✅ **已有** + 有 `all_kernels/` 测试 | 项目内有测试，我们也写了 | 约 20 个 |
| ⚠️ **已有**但 `all_kernels/` 可能缺失新内容 | 项目内已有但我们的测试不匹配 | - |
| ❌ **缺失项目内测试**但有 `all_kernels/` 测试 | 项目内无现成测试，我们手写了 | 约 36 个 |
| 🚫 两方都缺失 | 项目内和我们都未覆盖 | 约 4 个 |

### 具体覆盖明细

**Patch 算子（16个）：**
- 项目内有直接测试：P4(_topk_log_softmax)、P8(_bincount)、P10(_compute_slot_mappings) — 3个
- 项目内有间接测试（通过封装）：P1、P2、P3、P6、P7、P9、P11 — 7个
- 项目内完全无测试：P12(_apply_grammar_bitmask)、P13(_zero_kv_blocks)、P14(_prepare_dflash)、P15(_npu_gumbel_block_argmax)、P16(_probabilistic_rejection) — 5个

**vLLM vanilla 算子（44个）：**
- 项目内有相关测试（集成测试非单算子）：#4、#24、#25、#26、#27、#28、#29、#30、#41 — 9个
- 项目内完全无单算子精度测试：其余 35 个

---

## 四、关键发现

### 1. 官方测试以集成测试为主，缺少单算子精度测试

vLLM 项目的 `tests/` 目录下几乎所有核函数测试都是通过上层封装（Sampler、BlockTables、RejectionSampler 等）间接测试的，很少有直接调 `kernel[(grid,)](...)` 的精确定位测试。

### 2. vLLM-Ascend 的测试更偏向直接调 kernel

vLLM-Ascend 项目中的 `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/` 目录包含大量直接调用 Triton kernel 的精度测试，是我们 `acc_test/all_kernels/` 测试风格的主要参考来源。

### 3. 多个测试被 `@pytest.mark.skip` 跳过

以下测试由于已知问题已被跳过，在 Ascend NPU 上可能无法通过：
- `test_bincount.py` — "atomic_or operator hangs in current npu_ir version"
- `test_rejection_sample.py` — 2 个测试 "Probabilistic failure"
- `test_compute_token_logprobs.py` — 全部 "UB overflow"
- `test_apply_penalties_triton.py` — "Probabilistic failure"

### 4. 许多 Ascend 特有的 Triton 算子不在我们 60 个目标内

vLLM-Ascend 测试了大量 Nu 特有的 Triton 核（lightning attention、Mamba、fused gating、split_qkv 等），这些不在我们原始的 60 个目标清单中，但代码质量高，值得参考。
