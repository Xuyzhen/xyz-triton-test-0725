# vLLM v0.24 + vLLM-Ascend Triton 算子精度测试全集

**总计 60 个测试文件**，覆盖 vllm 仓库中所有 `@triton.jit` 核函数及其在 vllm-ascend 中的 patch 版本。

## 目录结构

```
all_kernels/
├── test_<kernel_name>.py       # vLLM vanilla 版本 (44个)
├── test_<kernel_name>_patch.py # vLLM-Ascend patch/适配版本 (16个)
├── run_all.sh                  # 一键运行脚本
└── README.md                   # 本文件
```

### 文件命名规则

- **vLLM vanilla**: `test_<函数名>.py`，头部标注 `# vLLM vanilla kernel: <name> from <path>`
- **vLLM-Ascend patch**: `test_<函数名>_patch.py`，头部标注 `# vLLM-Ascend patched kernel: <name> from <path>` + `# PATCH NOTE: ...`

## 完整清单

### vLLM vanilla 核函数 (44个)

| # | 核函数 | 文件 | 源文件(vllm) |
|---|--------|------|-------------|
| 1 | `_num_nans_kernel` | `test_num_nans_kernel.py` | `metrics/logits.py:10` |
| 2 | `_prepare_rope_positions_kernel` | `test_prepare_rope_positions_kernel.py` | `mm/rope.py:150` |
| 3 | `_scatter_num_accepted_kernel` | `test_scatter_num_accepted_kernel.py` | `model_states/mamba_hybrid.py:165` |
| 4 | `_bad_words_kernel` | `test_bad_words_kernel.py` | `sample/bad_words.py:101` |
| 5 | `_temperature_kernel` | `test_temperature_kernel.py` | `sample/gumbel.py:18` |
| 6 | `tl_rand64` | `test_tl_rand64.py` | `sample/gumbel.py:62` |
| 7 | `gumbel_block_argmax` | `test_gumbel_block_argmax.py` | `sample/gumbel.py:77` |
| 8 | `_gumbel_sample_kernel` | `test_gumbel_sample_kernel.py` | `sample/gumbel.py:152` |
| 9 | `_bias_kernel` | `test_bias_kernel.py` | `sample/logit_bias.py:148` |
| 10 | `_topk_log_softmax_kernel` | `test_topk_log_softmax_kernel.py` | `sample/logprob.py:14` |
| 11 | `_ranks_kernel` | `test_ranks_kernel.py` | `sample/logprob.py:56` |
| 12 | `_fill_logprob_token_ids_kernel` | `test_fill_logprob_token_ids_kernel.py` | `sample/logprob.py:174` |
| 13 | `_min_p_kernel` | `test_min_p_kernel.py` | `sample/min_p.py:9` |
| 14 | `_penalties_kernel` | `test_penalties_kernel.py` | `sample/penalties.py:107` |
| 15 | `_bincount_kernel` | `test_bincount_kernel.py` | `sample/penalties.py:219` |
| 16 | `_prompt_logprobs_token_ids_kernel` | `test_prompt_logprobs_token_ids_kernel.py` | `sample/prompt_logprob.py:151` |
| 17 | `_prepare_prefill_inputs_kernel` | `test_prepare_prefill_inputs_kernel.py` | `input_batch.py:175` |
| 18 | `_prepare_decode_inputs_kernel` | `test_prepare_decode_inputs_kernel.py` | `spec_decode/autoregressive/speculator.py:593` |
| 19 | `_update_draft_inputs_kernel` | `test_update_draft_inputs_kernel.py` | `spec_decode/autoregressive/speculator.py:670` |
| 20 | `_prepare_dflash_inputs_kernel` | `test_prepare_dflash_inputs_kernel.py` | `spec_decode/dflash/speculator.py:374` |
| 21 | `_compute_block_max_and_sumexp` | `test_compute_block_max_and_sumexp.py` | `spec_decode/rejection_sampler_utils.py:10` |
| 22 | `_compute_global_lse` | `test_compute_global_logsumexp.py` | `spec_decode/rejection_sampler_utils.py:21` |
| 23 | `_compute_block_stats_kernel` | `test_compute_block_stats_kernel.py` | `spec_decode/rejection_sampler_utils.py:48` |
| 24 | `_rejection_kernel` | `test_rejection_kernel.py` | `spec_decode/rejection_sampler_utils.py:157` |
| 25 | `_resample_kernel` | `test_resample_kernel.py` | `spec_decode/rejection_sampler_utils.py:307` |
| 26 | `_insert_resampled_kernel` | `test_insert_resampled_kernel.py` | `spec_decode/rejection_sampler_utils.py:435` |
| 27 | `_flatten_sampled_kernel` | `test_flatten_sampled_kernel.py` | `spec_decode/rejection_sampler.py:24` |
| 28 | `_gather_block_tables_kernel` | `test_gather_block_tables_kernel.py` | `block_table.py:200` |
| 29 | `_compute_slot_mappings_kernel` | `test_compute_slot_mappings_kernel.py` | `block_table.py:240` |
| 30 | `_apply_write_kernel` | `test_apply_write_kernel.py` | `buffer_utils.py:275` |
| 31 | `_load_ptr` | `test_load_ptr.py` | `buffer_utils.py:313` |
| 32 | `_dcp_local_seq_lens_kernel` | `test_dcp_local_seq_lens_kernel.py` | `cp_utils.py:36` |
| 33 | `_prepare_pos_seq_lens_kernel` | `test_prepare_pos_seq_lens_kernel.py` | `input_batch.py:235` |
| 34 | `_combine_sampled_and_draft_tokens_kernel` | `test_combine_sampled_and_draft_tokens_kernel.py` | `input_batch.py:293` |
| 35 | `_get_num_sampled_and_rejected_kernel` | `test_get_num_sampled_and_rejected_kernel.py` | `input_batch.py:397` |
| 36 | `_post_update_kernel` | `test_post_update_kernel.py` | `input_batch.py:446` |
| 37 | `_post_update_num_computed_tokens_kernel` | `test_post_update_num_computed_tokens_kernel.py` | `input_batch.py:548` |
| 38 | `_expand_idx_mapping_kernel` | `test_expand_idx_mapping_kernel.py` | `input_batch.py:580` |
| 39 | `_apply_grammar_bitmask_kernel` | `test_apply_grammar_bitmask_kernel.py` | `structured_outputs.py:86` |
| 40 | `_zero_kv_blocks_kernel` | `test_zero_kv_blocks_kernel.py` | `worker/utils.py:41` |
| 41 | `_topk_topp_kernel` | `test_topk_topp_kernel.py` | `v1/sample/ops/topk_topp_triton.py:94` |
| 42 | `_update_min_larger_stats` | `test_update_min_larger_stats.py` | `v1/sample/ops/topk_topp_triton.py:71` |
| 43 | `_selective_scan_update_kernel` | `test_selective_scan_update_kernel.py` | `model_executor/layers/mamba/ops/mamba_ssm.py:240` |
| 44 | `_prepare_prefill_inputs_kernel` (speculator) | `test_prepare_prefill_inputs_kernel_speculator.py` | `spec_decode/autoregressive/speculator.py:471` |

### vLLM-Ascend patch 核函数 (16个)

| # | 核函数 | 文件 | 源文件(vllm-ascend) | 备注 |
|---|--------|------|-------------------|------|
| P1 | `_bad_words_kernel` | `test_bad_words_kernel_patch.py` | `worker/v2/sample/bad_words.py` | Ascend 适配 |
| P2 | `_temperature_kernel` | `test_temperature_kernel_patch.py` | `worker/v2/sample/gumbel.py` | Ascend 适配 |
| P3 | `_gumbel_sample_kernel` | `test_gumbel_sample_kernel_patch.py` | `worker/v2/sample/gumbel.py` | Ascend 适配 |
| P4 | `_topk_log_softmax_kernel` | `test_topk_log_softmax_kernel_patch.py` | `worker/v2/sample/logprob.py` | Ascend 适配 |
| P5 | `_ranks_kernel` | `test_ranks_kernel_patch.py` | `worker/v2/sample/logprob.py` | Ascend 适配 |
| P6 | `_min_p_kernel` | `test_min_p_kernel_patch.py` | `worker/v2/sample/min_p.py` | Ascend 适配 |
| P7 | `_penalties_kernel` | `test_penalties_kernel_patch.py` | `worker/v2/sample/penalties.py` | Ascend 适配 |
| P8 | `_bincount_kernel` | `test_bincount_kernel_patch.py` | `worker/v2/sample/penalties.py` | Ascend 适配 |
| P9 | `_resample_kernel` | `test_resample_kernel_patch.py` | `worker/v2/spec_decode/rejection_sampler_utils.py` | Ascend 适配 |
| P10 | `_compute_slot_mappings_kernel` | `test_compute_slot_mappings_kernel_patch.py` | `worker/v2/block_table.py` | Ascend 适配 |
| P11 | `_post_update_kernel` | `test_post_update_kernel_patch.py` | `worker/v2/input_batch.py` | Ascend 适配 |
| P12 | `_apply_grammar_bitmask_kernel` | `test_apply_grammar_bitmask_kernel_patch.py` | `worker/v2/structured_outputs.py` | 通过 monkey patch 替换 |
| P13 | `_zero_kv_blocks_kernel` | `test_zero_kv_blocks_kernel_patch.py` | `worker/utils.py` | Ascend 适配 |
| P14 | `_prepare_dflash_inputs_kernel_ascend` | `test_prepare_dflash_inputs_kernel_ascend_patch.py` | `worker/v2/spec_decode/dflash/speculator.py` | 改名 + 适配，替代原版 |
| P15 | `_npu_gumbel_block_argmax` | `test_npu_gumbel_block_argmax_patch.py` | `worker/v2/spec_decode/rejection_sampler_utils.py` | 替换 `gumbel_block_argmax` |
| P16 | `_probabilistic_rejection_kernel` | `test_probabilistic_rejection_kernel_patch.py` | `worker/v2/spec_decode/rejection_sampler_utils.py` | 替换 `_rejection_kernel` |

## 运行方式

### 前置条件

- Ascend NPU 设备（torch.npu.is_available() == True）
- vllm (v0.24, `VLLM_TARGET_DEVICE=empty` 安装)
- vllm-ascend
- triton (Ascend compatible)

### 一键运行

```bash
bash run_all.sh
```

### 分运行

```bash
# 跑所有 vanilla 原版测试
python -m pytest test_*_kernel.py -v --tb=short
# 注意：排除 _patch 文件
# 或直接：
ls test_*.py | grep -v '_patch' | xargs python -m pytest -v

# 跑所有 patch 测试
python -m pytest test_*_patch.py -v

# 跑单个测试
python -m pytest test_prepare_rope_positions_kernel.py -v -s

# 跑一组相关测试
python -m pytest test_*_kernel.py test_*_patch.py -v --tb=short

# 跳过 mamba 测试（较慢）
python -m pytest . -v --ignore=test_selective_scan_update_kernel.py
```

### 调试失败测试

```bash
python -m pytest test_bad_words_kernel.py -v -s --tb=long 2>&1 | head -80
```

## 测试范式

每个测试文件遵循统一模式：

1. 调用 `init_device_properties_triton()` 初始化
2. 编写 CPU PyTorch 参考实现
3. 创建测试数据并拷贝到 NPU 设备
4. 直接调用 Triton 核函数（grid + 参数）
5. `torch.npu.synchronize()` 等待完成
6. 将结果拷回 CPU，使用 `torch.testing.assert_close` 对比
   - 整数精度：`rtol=0, atol=0`
   - 浮点精度：`rtol=1e-5, atol=1e-5`
   - Mamba SSM：`rtol=1e-4, atol=1e-4`

## 注

- `_insert_resampled_kernel` 在 vllm-ascend 中从 vllm 直接 import，无本地重定义，因此只有 vanilla 测试
- `tl_rand64` 在 vllm-ascend 中未实现（NPU 不支持 float64），改用 `tl.rand`
- `_load_ptr` 是纯工具函数，不做精度测试，仅验证指针加载逻辑
