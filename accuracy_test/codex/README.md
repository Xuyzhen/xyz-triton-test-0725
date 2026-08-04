# Triton 算子精度测试汇总

本文件汇总当前工作区 `vllm` 与 `vllm-ascend-xyz` 中目标 Triton 算子的已有精度测试。路径均相对于工作区根目录 `va024triton/`。

判定：**直接**表示测试直接 launch kernel 并与参考结果比较；**间接**表示测试调用会 launch kernel 的 wrapper，并检查数值或统计分布；**未找到**表示没有满足上述条件的项目内测试；**上游复用**表示 Ascend 没有本地重定义，不代表有 Ascend 专属 UT。

## 汇总

| 算子/辅助函数 | vLLM 已有精度测试 | vLLM-Ascend 已有精度测试或适配 | 结论 |
| --- | --- | --- | --- |
| `_num_nans_kernel`（wrapper） | 未找到 | 未找到本地实现/测试，上游复用 | 无针对性 UT |
| `_prepare_rope_positions_kernel`（wrapper） | 未找到 | 未找到同名适配；`test_mrope.py` 测试另一套 `triton_mrope` | 不应混记 |
| `_scatter_num_accepted_kernel`（launch） | 间接：`vllm/tests/kernels/mamba/test_mamba_ssm.py:861`，经 `selective_state_update` | 未找到本地同名 UT | vLLM 有 wrapper 覆盖 |
| `_bad_words_kernel` | 间接：`vllm/tests/v1/sample/test_sampler.py` | 间接：`vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_bad_words.py:93`，经 `apply_bad_words` | Ascend 测试更有针对性 |
| `_temperature_kernel` | 间接：`vllm/tests/v1/worker/test_gpu_gumbel_sample.py:114` | 间接：`vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_temperature.py:43`，经 `apply_temperature` | 两边均有 wrapper 覆盖 |
| `tl_rand64` | 间接：`vllm/tests/v1/worker/test_gpu_gumbel_sample.py:114` | Ascend 改用 `tl.rand` | 无独立 helper UT |
| `tl_rand32`（bench helper） | 当前源码未找到该名称 | 未找到 | 可能是旧版名称 |
| `gumbel_block_argmax`（bench helper） | 间接：`vllm/tests/v1/worker/test_gpu_gumbel_sample.py:114` | 改名 `_npu_gumbel_block_argmax`：`vllm-ascend-xyz/vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py:34`；无独立 UT | 存在改名适配 |
| `_gumbel_sample_kernel` | 间接：`vllm/tests/v1/worker/test_gpu_gumbel_sample.py:114` 等 | 间接：`vllm-ascend-xyz/tests/ut/sample/a2/test_gumbel_sampling.py`，经 `gumbel_sample` | 两边均有较完整覆盖 |
| `_bias_kernel` | 间接：`vllm/tests/v1/sample/test_sampler.py` 的 logit-bias 用例 | 未找到本地重定义/UT | 仅 vLLM 集成覆盖 |
| `_topk_log_softmax_kernel` | 间接：`vllm/tests/v1/sample/test_logprobs.py` | **直接**：`vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_log_softmax.py:16`；另有 `test_compute_token_logprobs.py:51`（skip） | Ascend 有直接 UT |
| `_ranks_kernel` | 间接：`vllm/tests/v1/sample/test_logprobs.py` | 间接：`vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_topk_logprobs.py:17` | Ascend wrapper 覆盖 rank |
| `_fill_logprob_token_ids_kernel`（launch） | 间接：`vllm/tests/v1/sample/test_logprobs.py` | 当前 Ascend `compute_topk_logprobs` 未调用该 kernel | 仅 vLLM 集成覆盖 |
| `_min_p_kernel` | 间接：`vllm/tests/v1/sample/test_sampler.py` | 间接：`vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_min_p.py:41` | Ascend 有针对性 wrapper UT |
| `_penalties_kernel` | 间接：`vllm/tests/v1/sample/test_sampler.py`；`test_logprobs.py:1075` | 间接：`vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_penality.py` | Ascend 有针对性 wrapper UT |
| `_bincount_kernel` | 未找到直接 UT | **直接**：`vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_bincount.py:40`；当前 skip | Ascend 已有但默认不执行 |
| `_prompt_logprobs_token_ids_kernel` | 间接：`vllm/tests/v1/sample/test_logprobs.py:1192` 等 | 未找到本地同名实现；patch 使用 Ascend `compute_topk_logprobs` | 无直接 kernel UT |
| `_prepare_prefill_inputs_kernel`（input batch/AR） | 未找到 | 未找到本地同名 UT，上游复用 | 无针对性 UT |
| `_prepare_decode_inputs_kernel`（wrapper） | 未找到 | 未找到同名适配/UT | 无针对性 UT |
| `_update_draft_inputs_kernel`（wrapper） | 未找到 | 未找到同名适配/UT | 无针对性 UT |
| `_prepare_dflash_inputs_kernel`（wrapper） | `vllm/tests/v1/spec_decode/test_dflash_lookahead.py` 有集成覆盖，非 kernel UT | 改名 `_prepare_dflash_inputs_kernel_ascend`：`vllm-ascend-xyz/vllm_ascend/worker/v2/spec_decode/dflash/speculator.py:153`；无针对性 UT | 存在改名 patch |
| `_compute_block_max_and_sumexp` / `_compute_max_and_sumexp`（helper） | 间接：`vllm/tests/v1/spec_decode/test_rejection_sampler_utils.py:141` | 无独立 UT | 当前 vLLM 名称为前者 |
| `_compute_global_lse` / `_compute_global_logsumexp`（helper） | 间接：同上 | Ascend 以旧名导入后别名为 `_compute_global_lse`；无独立 UT | 两仓版本名称错位 |
| `_compute_block_stats_kernel` / `_compute_local_logits_stats_kernel`（launch） | 间接：同上 | Ascend 以旧名导入后别名为 `_compute_block_stats_kernel`；无独立 UT | 两仓版本名称错位 |
| `_compute_global_residual_mass`（helper） | 当前源码未找到该名称 | 未找到 | 可能来自另一版本 |
| `_compute_global_target_argmax`（helper） | 当前源码未找到该名称 | 未找到 | 可能来自另一版本 |
| `_compute_global_logprobs_and_logsumexp`（helper） | 当前源码未找到该名称 | 未找到 | 可能来自另一版本 |
| `_compute_cumulative_log_p_kernel`（launch） | 当前源码未找到该名称 | 未找到 | 可能来自另一版本 |
| `_compute_local_residual_mass_kernel`（launch） | 当前源码未找到该名称 | 未找到 | 可能来自另一版本 |
| `_rejection_kernel`（launch） | 间接：`vllm/tests/v1/spec_decode/test_rejection_sampler_utils.py:141`、`:183`、`:231` | 改名/重写 `_probabilistic_rejection_kernel`：`vllm-ascend-xyz/vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py:192`；无专属 UT | vLLM 有 wrapper 覆盖 |
| `_resample_kernel`（launch） | 间接：同一 rejection sampler 测试 | Ascend 同名适配：`vllm-ascend-xyz/vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py:82`；`test_rejection_sample.py` 测另一组 kernel | 不应混淆两套实现 |
| `_insert_resampled_kernel`（launch） | 间接：同一 rejection sampler 测试 | 从 vLLM 直接导入；无 Ascend 专属 UT | 上游复用 |
| `_flatten_sampled_kernel` | 间接：同一 rejection sampler 测试 | 未找到本地重定义/UT | vLLM 有间接覆盖 |
| `_gather_block_tables_kernel` | 间接：`vllm/tests/v1/worker/test_gpu_block_table.py:16`、`:110` | 未找到本地同名 UT | vLLM 有 wrapper 覆盖 |
| `_compute_slot_mappings_kernel` | 间接：`vllm/tests/v1/worker/test_gpu_block_table.py` | **直接**：`vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_slot_mapping.py`，与上游输出对比 | Ascend 有直接 UT |
| `_apply_write_kernel` | 间接：`vllm/tests/v1/worker/test_gpu_block_table.py:16`、`:110` | 未找到本地 UT | vLLM 有 wrapper 覆盖 |
| `_load_ptr`（bench helper） | 仅随 `_apply_write_kernel` 间接执行 | 未找到本地实现/独立 UT | helper 无独立 UT |
| `_dcp_local_seq_lens_kernel` | 未找到 | 未找到本地同名 UT | 无针对性 UT |
| `_prepare_prefill_inputs_kernel`（清单重复项） | 未找到 | 未找到本地同名 UT | 与前述 input-batch kernel 为同一项 |
| `_prepare_pos_seq_lens_kernel` | 未找到 | 未找到本地同名 UT | 无针对性 UT |
| `_combine_sampled_and_draft_tokens_kernel` | 未找到 | 未找到本地同名 UT | 无针对性 UT |
| `_get_num_sampled_and_rejected_kernel` | 未找到 | 未找到本地同名 UT | 无针对性 UT |
| `_post_update_kernel` | 未找到针对性 UT | 间接：`vllm-ascend-xyz/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_post_update.py:63`，与上游 GPU wrapper 对比 | Ascend 有交叉实现 UT |
| `_post_update_num_computed_tokens_kernel`（wrapper） | 未找到 | Ascend 本地 `post_update` 未使用该同名 kernel | 无针对性 UT |
| `_expand_idx_mapping_kernel` | 未找到 | Ascend 本地 `post_update` 未定义该同名 kernel | 无针对性 UT |
| `_apply_grammar_bitmask_kernel` | 未找到 | 同名适配：`vllm-ascend-xyz/vllm_ascend/worker/v2/structured_outputs.py:35`，monkey patch 替换上游；无精度 UT | Ascend 有实现、无测试 |
| `_zero_kv_blocks_kernel` | 未找到 | 同名适配：`vllm-ascend-xyz/vllm_ascend/worker/utils.py:15`；无精度 UT | Ascend 有实现、无测试 |
| `_topk_topp_kernel` | 间接且针对性强：`vllm/tests/v1/sample/test_topk_topp_sampler.py:298` 起，经 `apply_top_k_top_p_triton` 对比参考结果 | 未找到本地重定义/专属 UT，上游复用 | vLLM 有较完整 UT |
| `_update_min_larger_stats` | 间接：随上述 `apply_top_k_top_p_triton` 测试覆盖 | 未找到本地实现/专属 UT | helper 间接覆盖 |
| `_selective_scan_update_kernel` | 间接且针对性强：`vllm/tests/kernels/mamba/test_mamba_ssm.py:345` 起，经 `selective_state_update` 对比参考实现 | 未找到同名 kernel；Ascend Mamba 测试使用不同实现 | vLLM 有完整 wrapper UT |

## Ascend 改名与替换关系

| vLLM 名称 | vLLM-Ascend 名称/处理 | 位置 |
| --- | --- | --- |
| `gumbel_block_argmax` | `_npu_gumbel_block_argmax`，改用 `tl.rand` | `vllm-ascend-xyz/vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py:34` |
| `_rejection_kernel` | `_probabilistic_rejection_kernel` | `vllm-ascend-xyz/vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py:192` |
| `_prepare_dflash_inputs_kernel` | `_prepare_dflash_inputs_kernel_ascend`，再 monkey patch 回原名 | `vllm-ascend-xyz/vllm_ascend/worker/v2/spec_decode/dflash/speculator.py:153`；patch：`vllm-ascend-xyz/vllm_ascend/patch/worker/patch_v2/patch_triton.py:37` |
| `_compute_global_logsumexp` | 导入后别名 `_compute_global_lse` | `vllm-ascend-xyz/vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py:22` |
| `_compute_local_logits_stats_kernel` | 导入后别名 `_compute_block_stats_kernel` | `vllm-ascend-xyz/vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py:25` |

## 已有测试的限制

- `test_bincount.py:39` 被 skip：`atomic_or operator hangs in current npu_ir version`。
- `test_compute_token_logprobs.py` 的四组测试被 skip：`UB overflow`。
- Ascend `test_rejection_sample.py` 测试 `vllm_ascend.ops.triton.reject_sample` 中的另一套 kernel，不能直接算作 worker v2 `_resample_kernel` / `_probabilistic_rejection_kernel` 的 UT。
- 两仓存在 API 版本错位：Ascend 导入的 `_compute_global_logsumexp`、`_compute_local_logits_stats_kernel` 在当前 vLLM 源码中分别显示为 `_compute_global_lse`、`_compute_block_stats_kernel`。

## 后续维护规则

- 后续算子继续追加到本汇总表，不为每个算子创建独立说明文档。
- 若两个项目均无现有精度 UT，则在本目录新增对应测试文件，并在表中记录位置。
- wrapper/helper 必须能追踪到目标 kernel 的实际调用链，否则仍标记为“未找到”。

## 完全没有现成精度 UT 的算子

以下判定仅针对 `vllm/tests` 与 `vllm-ascend-xyz/tests`。生成在本目录的新测试不参与“原有覆盖”判断。

### 已补充 A3 精度 UT

测试文件统一位于 `accuracy_test/codex/missing_accuracy_tests/`。

| 算子 | 被测实现 | 新增测试文件 |
| --- | --- | --- |
| `_num_nans_kernel` | vLLM 上游实现 | `test_num_nans_kernel.py` |
| `_prepare_rope_positions_kernel` | vLLM 上游实现 | `test_prepare_rope_positions_kernel.py` |
| `_prepare_prefill_inputs_kernel`（input batch） | vLLM 上游实现 | `test_prepare_prefill_inputs_kernel.py` |
| `_prepare_prefill_inputs_kernel`（autoregressive speculator） | vLLM 上游同名实现 | `test_prepare_prefill_inputs_kernel_speculator.py` |
| `_prepare_decode_inputs_kernel` | vLLM 上游实现 | `test_prepare_decode_inputs_kernel.py` |
| `_update_draft_inputs_kernel` | vLLM 上游实现 | `test_update_draft_inputs_kernel.py` |
| `_dcp_local_seq_lens_kernel` | vLLM 上游实现 | `test_dcp_local_seq_lens_kernel.py` |
| `_prepare_pos_seq_lens_kernel` | vLLM 上游实现 | `test_prepare_pos_seq_lens_kernel.py` |
| `_combine_sampled_and_draft_tokens_kernel` | vLLM 上游实现 | `test_combine_sampled_and_draft_tokens_kernel.py` |
| `_get_num_sampled_and_rejected_kernel` | vLLM 上游实现 | `test_get_num_sampled_and_rejected_kernel.py` |
| `_post_update_num_computed_tokens_kernel` | vLLM 上游实现 | `test_post_update_num_computed_tokens_kernel.py` |
| `_expand_idx_mapping_kernel` | vLLM 上游实现 | `test_expand_idx_mapping_kernel.py` |
| `_apply_grammar_bitmask_kernel` | vLLM-Ascend 本地适配 | `test_apply_grammar_bitmask_kernel_patch.py` |
| `_zero_kv_blocks_kernel` | vLLM-Ascend 本地适配 | `test_zero_kv_blocks_kernel_patch.py` |

上述测试均面向 Ascend A3：初始化 Ascend Triton 设备属性，在 NPU 上直接 launch 目标 kernel，并与 CPU/PyTorch reference 或精确预期值比较。

### 当前源码中不存在，暂不能实现

以下名称在当前 `vllm` 和 `vllm-ascend-xyz` 源码中均不存在。它们可能来自其他版本；在取得对应实现前无法编写可运行的精度 UT。

| 名称 | 检索结果 |
| --- | --- |
| `tl_rand32` | 当前 gumbel 路径使用 `tl_rand64` 或 Triton `tl.rand` |
| `_compute_global_residual_mass` | 两仓均无定义 |
| `_compute_global_target_argmax` | 两仓均无定义 |
| `_compute_global_logprobs_and_logsumexp` | 两仓均无定义 |
| `_compute_cumulative_log_p_kernel` | 两仓均无定义 |
| `_compute_local_residual_mass_kernel` | 两仓均无定义 |

### 执行方式

在安装了当前 vLLM、vLLM-Ascend、PyTorch NPU 和 Ascend Triton 后执行：

```bash
pytest -sv accuracy_test/codex/missing_accuracy_tests
```

本地 Windows 环境仅完成 Python AST 语法检查（14 个文件全部通过）；实际 NPU kernel 执行需要在 Ascend A3 环境完成。
