# vLLM v0.24 Triton 算子精度测试

## 环境要求

- Python 3.11+
- PyTorch (with Ascend NPU support)
- vllm (v0.24, installed with `VLLM_TARGET_DEVICE=empty`)
- vllm-ascend
- triton (Ascend compatible)

## 测试列表 (28个文件)

### metrics
- `test_num_nans.py` — `_num_nans_kernel`: 统计 logits 中的 NaN 数量

### input_batch
- `test_prepare_prefill_inputs.py` — `_prepare_prefill_inputs_kernel`: 复制 prefill 阶段的 token IDs
- `test_prepare_pos_seq_lens.py` — `_prepare_pos_seq_lens_kernel`: 计算 position 和 seq_lens
- `test_get_num_sampled_and_rejected.py` — `_get_num_sampled_and_rejected_kernel`: 计算采样/拒绝的 token 数量
- `test_post_update_num_computed_tokens.py` — `_post_update_num_computed_tokens_kernel`: 更新已计算 token 数
- `test_expand_idx_mapping.py` — `_expand_idx_mapping_kernel`: 扩展 idx_mapping
- `test_combine_sampled_and_draft_tokens.py` — `_combine_sampled_and_draft_tokens_kernel`: 合并采样和 draft tokens
- `test_post_update.py` — `_post_update_kernel`: 采样后状态更新

### rope
- `test_prepare_rope_positions.py` — `_prepare_rope_positions_kernel`: M-RoPE / XD-RoPE 位置计算

### sample
- `test_bias_kernel.py` — `_bias_kernel`: 应用 allowed token IDs 和 logit bias
- `test_fill_logprob_token_ids.py` — `_fill_logprob_token_ids_kernel`: 填充 logprob token IDs
- `test_prompt_logprobs_token_ids.py` — `_prompt_logprobs_token_ids_kernel`: 计算 prompt logprob token IDs
- `test_bad_words.py` — `_bad_words_kernel`: 应用 bad words 过滤
- `test_temperature_kernel.py` — `_temperature_kernel`: 温度缩放
- `test_gumbel_sample_kernel.py` — `_gumbel_sample_kernel`: Gumbel 采样
- `test_min_p_kernel.py` — `_min_p_kernel`: Min-p 采样
- `test_penalties_kernel.py` — `_penalties_kernel`: 重复/存在/频率惩罚
- `test_bincount_kernel.py` — `_bincount_kernel`: Token 出现次数统计
- `test_apply_grammar_bitmask.py` — `_apply_grammar_bitmask_kernel`: 语法约束位掩码

### block_table
- `test_gather_block_tables.py` — `_gather_block_tables_kernel`: 收集 block tables
- `test_compute_slot_mappings.py` — `_compute_slot_mappings_kernel`: 计算 slot mappings
- `test_zero_kv_blocks.py` — `_zero_kv_blocks_kernel`: 清零 KV cache blocks

### buffer_utils
- `test_apply_write.py` — `_apply_write_kernel`: 应用分段写入

### cp_utils
- `test_dcp_local_seq_lens.py` — `_dcp_local_seq_lens_kernel`: DCP 上下文并行下的 seq_lens

### topk_topp
- `test_topk_topp.py` — `_topk_topp_kernel`: Top-k / Top-p 采样

### mamba
- `test_selective_scan_update.py` — `_selective_scan_update_kernel`: Mamba SSM 状态更新

### spec_decode
- `test_speculator_kernels.py` — `_prepare_decode_inputs_kernel` + `_update_draft_inputs_kernel`: 投机解码
- `test_flatten_sampled.py` — `_flatten_sampled_kernel`: 展平采样结果

## 运行方式

```bash
# 方式一：一键运行所有测试
bash run_all_tests.sh

# 方式二：运行单个测试
python -m pytest test_prepare_rope_positions.py -v -s

# 方式三：运行多个测试
python -m pytest test_num_nans.py test_prepare_prefill_inputs.py -v

# 方式四：运行所有测试
python -m pytest . -v

# 只显示错误
python -m pytest . -v --tb=short -q
```
