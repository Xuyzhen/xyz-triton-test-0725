# vLLM Triton 算子精度测试报告（仅 NPU 环境）

> **适用范围声明：本报告仅覆盖 NPU（Ascend）环境下的精度测试。** GPU/CPU 环境的严格 UT 不在本报告范围。
> 数据来源：`accuracy_test/strict_ut/docs/0818log/log-a5-1.txt`（由 `bash run_npu.sh` 触发）
> 规格来源：`accuracy_test/strict_ut/npu_ut_shapes.md`

---

## 1. 测试环境

| 项 | 值 |
|---|---|
| 平台 | Ascend A3（Ascend950PR_9579）|
| CANN | 9.1.0（ccec_compiler / bishengir / hivmc-a5 工具链）|
| Python / pytest | Python 3.11.10 / pytest 8.3.2 / pytest-asyncio 1.3.0（STRICT mode）|
| Triton 后端 | `triton/backends/ascend`（BiSheng IR 编译，`--target=Ascend950PR_9579`）|
| NPU 执行方式 | 每算子一个隔离进程，逐文件 `python -m pytest` |
| 对比基准 | 上游 vLLM vanilla Triton 内核 + 纯 PyTorch CPU reference |

> 固定噪声说明：每个算子段均产生 14 条 `torch.jit.script_method is deprecated` 警告（来自 torch，非本次引入），报告中不再逐条重复。

---

## 2. 每个算子条目需固定记录的信息（统一模板）

报告后续每节（第 4 章）均按以下 **10 个固定维度** 填写：

| # | 固定字段 | 说明 |
|---|---|---|
| 1 | 算子（内核符号） | 精确函数名 + 来源模块（upstream vllm / vllm_ascend 补丁）|
| 2 | 功能域 | 采样 / 拒绝采样 / 预处理 / 状态更新 / 指标统计… |
| 3 | 主输入规格 | 主要输入 tensor、dtype、shape 语义 |
| 4 | 测试矩阵 | 参数化维度及具体取值（逐用例）|
| 5 | 固定参数 | 非参数化的 shape / BLOCK_SIZE / 常量 |
| 6 | 真值基准 | CPU reference / GPU vanilla / golden |
| 7 | 比较准则 | `assert_close` rtol/atol、特殊值断言、诊断字符串 |
| 8 | 关键分支/场景 | 覆盖的控制流（bonus token、early-return、padding…）|
| 9 | 真实模型对齐 | 对应真实模型 config（词表/维度来源）|
| 10 | 运行结果（日志）| collected / passed / failed / skipped / 耗时 |

---

## 3. 总体结论（摘要）

- **被测文件**：41 个（`strict_ut/npu/*.py`）
- **NPU 实际 launch**：39 个
- **通过**：36 个
- **失败**：**3 个**（`fill_logprob_token_ids`、`num_nans`、`topk_topp`）
- **跳过（无 NPU 适配，不 launch）**：2 个（`compute_cumulative_log_p`、`compute_local_residual_mass`）
- **总用例**：约 490+，失败 3 例

| 类别 | 数量 | 算子 |
|---|---|---|
| ✅ 通过 | 36 | 见第 4 章汇总表 |
| ❌ 失败 | 3 | fill_logprob_token_ids / num_nans / topk_topp |
| ⏭️ 跳过 | 2 | compute_cumulative_log_p / compute_local_residual_mass |

---

## 4. 各算子精度测试规格与结果

> 汇总表（结果取自 log-a5-1.txt，顺序即 run_npu.sh 执行顺序）：

| # | 测试文件 | 算子 | 结果 | 用例 | 耗时 |
|---|---|---|---|---|---|
| 1 | apply_grammar_bitmask_kernel | `_apply_grammar_bitmask_kernel` | ✅ | 10 | 12.93s |
| 2 | apply_write_kernel | `_apply_write_kernel` | ✅ | 10 | 5.10s |
| 3 | ar_prepare_prefill_inputs_kernel | `_prepare_prefill_inputs_kernel`(speculator) | ✅ | 11 | 8.14s |
| 4 | bad_words | `_bad_words_kernel` | ✅ | 9 | 6.56s |
| 5 | bias_kernel | `_bias_kernel` | ✅ | 18 | 7.50s |
| 6 | bincount | `_bincount_kernel` | ✅ | 1 | 5.10s |
| 7 | combine_sampled_and_draft_tokens_kernel | `_combine_sampled_and_draft_tokens_kernel` | ✅ | 13 | 5.07s |
| 8 | compute_cumulative_log_p_kernel | `_compute_cumulative_log_p_kernel` | ⏭️ | 1 | 0.01s |
| 9 | compute_local_logits_stats_kernel | `_compute_local_logits_stats_kernel` | ✅ | 73 | 8.22s |
| 10 | compute_local_residual_mass_kernel | `_compute_local_residual_mass_kernel` | ⏭️ | 1 | 0.01s |
| 11 | compute_slot_mappings_kernel | `_compute_slot_mappings_kernel` | ✅ | 4 | 7.59s |
| 12 | dcp_local_seq_lens_kernel | `_dcp_local_seq_lens_kernel` | ✅ | 50 | 5.09s |
| 13 | expand_idx_mapping_kernel | `_expand_idx_mapping_kernel` | ✅ | 11 | 5.14s |
| 14 | fill_logprob_token_ids_kernel | `_fill_logprob_token_ids_kernel` | ❌ | 5/1 | 7.69s |
| 15 | flatten_sampled_kernel | `_flatten_sampled_kernel` | ✅ | 14 | 7.70s |
| 16 | gather_block_tables_kernel | `_gather_block_tables_kernel` | ✅ | 14 | 7.59s |
| 17 | get_num_sampled_and_rejected_kernel | `_get_num_sampled_and_rejected_kernel` | ✅ | 13 | 5.07s |
| 18 | gumbel_sample | `_gumbel_sample_kernel` / `_temperature_kernel` | ✅ | 36 | 5.15s |
| 19 | input_batch_prepare_prefill_inputs_kernel | `_prepare_prefill_inputs_kernel`(input_batch) | ✅ | 11 | 4.85s |
| 20 | insert_resampled_kernel | `_insert_resampled_kernel` | ✅ | 26 | 7.49s |
| 21 | min_p | `_min_p_kernel` | ✅ | 7 | 5.14s |
| 22 | num_nans_kernel | `_num_nans_kernel` | ❌ | 1 | 5.05s |
| 23 | penalties | `_penalties_kernel` | ✅ | 27 | 9.99s |
| 24 | post_update_kernel | `_post_update_kernel` | ✅ | 10 | 5.44s |
| 25 | post_update_num_computed_tokens_kernel | `_post_update_num_computed_tokens_kernel` | ✅ | 11 | 5.08s |
| 26 | prepare_decode_inputs_kernel | `_prepare_decode_inputs_kernel` | ✅ | 9 | 8.14s |
| 27 | prepare_dflash_inputs_kernel | `_prepare_dflash_inputs_kernel_ascend` | ✅ | 8 | 8.34s |
| 28 | prepare_pos_seq_lens_kernel | `_prepare_pos_seq_lens_kernel` | ✅ | 26 | 5.07s |
| 29 | prepare_rope_positions_kernel | `_prepare_rope_positions_kernel` | ✅ | 11 | 7.37s |
| 30 | prompt_logprobs_token_ids_kernel | `_prompt_logprobs_token_ids_kernel` | ✅ | 10 | 7.62s |
| 31 | ranks | `_ranks_kernel` | ✅ | 7 | 7.46s |
| 32 | rejection_kernel | `_probabilistic_rejection_kernel` | ✅ | 21 | 7.50s |
| 33 | resample_kernel | `_resample_kernel` | ✅ | 5 | 7.85s |
| 34 | scatter_num_accepted_kernel | `_scatter_num_accepted_kernel` | ✅ | 10 | 8.46s |
| 35 | selective_scan_update_kernel | `_selective_scan_update_kernel` | ✅ | 13 | 8.04s |
| 36 | temperature | `_temperature_kernel` | ✅ | 8 | 5.30s |
| 37 | topk_log_softmax | `_topk_log_softmax_kernel` | ✅ | 6 | 7.48s |
| 38 | topk_topp_kernel | `_topk_topp_kernel` | ❌ | 1 | 24.35s |
| 39 | update_draft_inputs_kernel | `_update_draft_inputs_kernel` | ✅ | 31 | 8.45s |
| 40 | update_min_larger_stats | `_update_min_larger_stats_kernel` | ✅ | 12 | 5.03s |
| 41 | zero_kv_blocks_kernel | `_zero_kv_blocks_kernel` | ✅ | 9 | 9.96s |

### 4.1 `_apply_grammar_bitmask_kernel`（#1，✅）
- 来源：`vllm_ascend.worker.v2.structured_outputs` 补丁版
- 主输入：`logits[num_logits, vocab_size]` fp32
- 测试矩阵：

| 测试函数 | 参数化 | 固定 |
|---|---|---|
| test_basic_bitmask | vocab ∈ {128, 1024, 8192} | num_bitmasks=2, num_logits=4, padded_vocab_words=ceil(vocab/32) |
| test_all_allowed | vocab ∈ {128, 512, 4096} | num_bitmasks=1, num_logits=2 |
| test_all_blocked | — | vocab=256, num_bitmasks=1 |

- 真值：CPU reference；准则：`assert_close`（标准 RTOL/ATOL）
- 结果：`10 passed, 14 warnings in 12.93s`

### 4.2 `_apply_write_kernel`（#2，✅）
- 来源：`vllm.v1.worker.gpu.buffer_utils`；主输入 `output[num_rows, num_cols]`
- 矩阵：test_prefill `num_rows∈{1,4}×num_cols∈{16,32}`、BLOCK_SIZE=4；test_multi_group `num_groups∈{1,2,4}×num_writes_per_group∈{1,2}`
- 多组能力通过 `kernel.arg_names` 含 `write_group_ids_ptr` / `MULTI_GROUP` 判定
- 结果：`10 passed, 14 warnings in 5.10s`

### 4.3 `_prepare_prefill_inputs_kernel`（speculator 变体）（#3，✅）
- 来源：`vllm.v1.worker.gpu.spec_decode.autoregressive.speculator`
- 主输入：`target_input_ids[max_num_tokens]`、`target_positions[max_num_tokens]` int32
- 矩阵：test_basic_prefill `num_reqs∈{1,2,4}×query_len∈{4,16}`；test_chunked_prefill_path `num_reqs∈{1,2}`；test_rejected_tokens
- 固定：max_num_reqs=8, max_model_len…, seq_len=128
- 结果：`11 passed, 14 warnings in 8.14s`

### 4.4 `_bad_words_kernel`（#4，✅）
- 来源：`vllm_ascend.worker.v2.sample.bad_words`（`apply_bad_words`）
- 矩阵（case → num_tokens / vocab 50257 / reqs / words / word_len）：small `512/16/3/2`、medium `1024/32/5/3`、large `2048/64/8/4`
- 固定 buffer：`bad_word_token_ids[reqs,1024]`、`bad_word_offsets[reqs,129]`、`all_token_ids[reqs,1024]`
- 结果：`9 passed, 14 warnings in 6.56s`

### 4.5 `_bias_kernel`（#5，✅）
- 来源：`vllm.v1.worker.gpu.sample.logit_bias`；主输入 `logits[num_tokens,vocab]` fp32
- 矩阵：num_tokens∈{1,4,8}×vocab∈{128,1024}；test_logit_bias num_tokens∈{1,4}、vocab=64
- 固定：allowed_token_ids/bias_token_ids[reqs,1024]、stop_token_ids[reqs,128]
- 结果：`18 passed, 14 warnings in 7.50s`

### 4.6 `_bincount_kernel`（#6，✅）
- 来源：`vllm_ascend.worker.v2.sample.penalties`
- 固定 shape：`all_token_ids[64,40960]` int32、`prompt_len[64]`、`prefill_len[64]`、`prompt_bin_mask[64,4748]`、`output_bin_counts[64,151936]`、max_prefill_len=10
- 结果：`1 passed, 14 warnings in 5.10s`

### 4.7 `_combine_sampled_and_draft_tokens_kernel`（#7，✅）
- 来源：`vllm.v1.worker.gpu.input_batch`；主输入 `input_ids[num_tokens]`、`draft_tokens[reqs,num_spec_steps]`
- 矩阵：num_reqs∈{1,2,4}×num_spec_steps∈{1,3}×num_new_sampled_tokens∈{0,1}
- 结果：`13 passed, 14 warnings in 5.07s`

### 4.8 `_compute_cumulative_log_p_kernel`（#8，⏭️ 跳过）
- 状态：**无 NPU 适配**，NPU 不 launch，由 GPU strict UT 覆盖
- 日志：`1 skipped in 0.01s`

### 4.9 `_compute_local_logits_stats_kernel`（#9，✅）
- 来源：`vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils`
- 主输入：`target_logits[num_logits,vocab]`、`draft_logits[max_num_reqs,num_spec_steps,vocab]` fp32
- 矩阵：num_logits∈{1,2,4}×vocab∈{128,1024,8192}×num_speculative_steps∈{2,3}；VOCAB_BLOCK_SIZE=8192、max_num_reqs=4
- 结果：**`73 passed, 14 warnings in 8.22s`**（用例最多的通过算子）

### 4.10 `_compute_local_residual_mass_kernel`（#10，⏭️ 跳过）
- 状态：**无 NPU 适配**，NPU 不 launch；`1 skipped in 0.01s`

### 4.11 `_compute_slot_mappings_kernel`（#11，✅）
- 来源：`vllm_ascend.worker.v2.block_table` / upstream
- 矩阵（block_size, positions, cp_size, cp_rank, cp_interleave）：(16,[15,16,17,31,32],1,0,1)/(32,[31,32,33,63,64],1,0,1)/(16,[0..3,16..19],2,0,2)/(16,[…],2,1,2)
- 固定：max_num_tokens=64, max_num_reqs=8, num_reqs=2, block_table[8,64/4096]
- 结果：`4 passed, 14 warnings in 7.59s`

### 4.12 `_dcp_local_seq_lens_kernel`（#12，✅）
- 来源：`vllm.v1.worker.gpu.cp_utils`；主输入 `seq_lens[max_num_reqs]` int32
- 矩阵：num_reqs∈{2,4,8}×max_num_reqs∈{8,16}×dcp_size∈{2,4}×dcp_rank∈{0,1}×cp_interleave∈{1,2}；覆盖最高 rank、全 0 seq
- 结果：`50 passed, 14 warnings in 5.09s`

### 4.13 `_expand_idx_mapping_kernel`（#13，✅）
- 来源：`vllm.v1.worker.gpu.input_batch`；主输入 `idx_mapping[num_reqs]`、`cu_num_logits[num_reqs+1]`
- 矩阵：num_reqs∈{1,2,4}×tokens_per_req∈{1,3,8}；不均等（[2,5,3]）；非连续 idx_mapping
- 结果：`11 passed, 14 warnings in 5.14s`

### 4.14 `_fill_logprob_token_ids_kernel`（#14，❌ 失败）
- 来源：`vllm.v1.worker.gpu.sample.logprob`
- 主输入：`out_token_ids[batch_size,1+PADDED_COLS]`、`topk_indices[batch_size,NUM_TOPK]`；固定 PADDED_COLS=16、MAX_LOGPROB_TOKEN_IDS=128、num_reqs=4
- 矩阵：batch_size∈{1,4,8}×topk∈{0,3,5}
- 日志：`1 failed, 4 passed, 15 warnings in 7.69s`（`stopping after 1 failures`）
- 失败用例：`test_custom_token_ids[3-4]`，详见第 5.1 节（topk 分支漏写 token，非编译错误）

### 4.15 `_flatten_sampled_kernel`（#15，✅）
- 来源：`vllm.v1.worker.gpu.spec_decode.rejection_sampler`；主输入 `sampled[num_reqs,num_spec_steps+1]` int64
- 矩阵：num_reqs∈{1,2,4,8}×num_spec_steps∈{1,3,5}；全 0；单 req 多 logits（num_spec_steps=10）
- 结果：`14 passed, 14 warnings in 7.70s`

### 4.16 `_gather_block_tables_kernel`（#16，✅）
- 来源：`vllm.v1.worker.gpu.block_table`；主输入 `src_block_tables[num_groups,max_num_reqs,max_num_blocks]`
- 矩阵：num_groups∈{1,2,4}×max_num_reqs∈{4,8}×max_num_blocks∈{64,128}；padding 补零
- 结果：`14 passed, 14 warnings in 7.59s`

### 4.17 `_get_num_sampled_and_rejected_kernel`（#17，✅）
- 来源：`vllm.v1.worker.gpu.input_batch`；主输入 `num_sampled[num_reqs]`、`cu_num_logits[num_reqs+1]`
- 矩阵：num_reqs∈{1,2,4}×num_logits_per_req∈{1,3,5}；chunked_prefilling；sampled→rejected 映射 (0,3)/(2,1)/(3,0)
- 结果：`13 passed, 14 warnings in 5.07s`

### 4.18 `_gumbel_sample_kernel` / `_temperature_kernel`（#18，✅）
- 来源：`vllm_ascend.worker.v2.sample.gumbel`；主输入 `logits[num_tokens,vocab]` fp32
- 矩阵（apply_temperature / gumbel_sample）：(1,32000)/(8,32000)/(48,102400)/(64,151936)；(1..16, 1..8, 32000/102400)
- 用例：greedy、deterministic、valid_token_ids、mixed_temp、不同 seed、温度影响分布
- 结果：`36 passed, 14 warnings in 5.15s`

### 4.19 `_prepare_prefill_inputs_kernel`（input_batch 变体）（#19，✅）
- 来源：`vllm.v1.worker.gpu.input_batch`；主输入 `all_token_ids[max_num_reqs,max_model_len]` int32
- 矩阵：num_reqs∈{1,2,4}×query_len∈{1,4,16}；固定 max_model_len=128, max_num_reqs=8, num_lookahead=3, prefill_len=64；prefill done early-return
- 结果：`11 passed, 14 warnings in 4.85s`

### 4.20 `_insert_resampled_kernel`（#20，✅）
- 来源：`vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils`
- 主输入：`sampled[num_reqs,num_spec_steps+1]`、`resampled_local_argmax[num_reqs,num_blocks]`
- 矩阵：num_reqs∈{1,2,4}×num_spec_steps∈{1,3}；vocab=4096, RESAMPLE_BLOCK_SIZE=1024, num_blocks=4
- 结果：`26 passed, 14 warnings in 7.49s`

### 4.21 `_min_p_kernel`（#21，✅）
- 来源：`vllm_ascend.worker.v2.sample.min_p`（`apply_min_p`）
- 矩阵（num_reqs×vocab）：(48,102400)/(96,102400)/(24,151936)/(1,32000)
- 结果：`7 passed, 14 warnings in 5.14s`

### 4.22 `_num_nans_kernel`（#22，❌ 失败）
- 来源：`vllm.v1.worker.gpu.metrics.logits`；主输入 `logits[num_reqs,vocab]` fp32
- 矩阵：num_reqs∈{1,2,4,8}×vocab∈{128,1024,8192,16384}×frac_nan∈{0.0,0.1,0.5,1.0}；BLOCK_SIZE=8192；另含无 nan / 全 nan
- 日志：`1 failed, 14 warnings in 5.05s`（首个用例即停）
- 失败用例：`test_num_nans[0.0-128-1]`，Triton 前端编译错误，详见第 5.2 节

### 4.23 `_penalties_kernel`（#23，✅）
- 来源：`vllm_ascend.worker.v2.sample.penalties`（`apply_penalties`）
- 全组合矩阵：num_tokens{1,4}×vocab{1000}×num_status{1,4}×num_speculative_tokens{0,1,3}×dtype{bfloat16,float16}×seed{42}×device{npu:0} = **24 组**（日志 27 为含附加场景）
- 结果：`27 passed, 14 warnings in 9.99s`

### 4.24 `_post_update_kernel`（#24，✅）
- 来源：upstream `vllm.v1.worker.gpu.input_batch` vs `vllm_ascend.worker.v2.input_batch`（最新 API 对齐）
- 主输入：`output_bin_counts[max_num_reqs,vocab]`、`sampled_tokens[num_reqs,num_spec_steps+1]`、`all_token_ids[max_num_reqs,max_model_len]` int32
- 矩阵：(num_reqs,max_num_reqs,vocab,num_spec_steps)=(36,36,200,2)/(48,48,32000,5)/(128,128,32000,5)；max_model_len=3000
- 结果：`10 passed, 14 warnings in 5.44s`

### 4.25 `_post_update_num_computed_tokens_kernel`（#25，✅）
- 来源：`vllm.v1.worker.gpu.input_batch`；主输入 `num_computed_tokens[max_num_reqs]`、`query_start_loc[num_reqs+1]`
- 矩阵：num_reqs∈{1,2,4}×query_len∈{1,4,8}；非连续 idx；零 query_len
- 结果：`11 passed, 14 warnings in 5.08s`

### 4.26 `_prepare_decode_inputs_kernel`（#26，✅）
- 来源：`vllm.v1.worker.gpu.spec_decode.autoregressive.speculator`
- 主输入：`draft_tokens[num_reqs,1]`、`input_ids[max_num_tokens]`
- 矩阵：num_reqs∈{1,2,4}×advance_pos∈{False,True}；rejected tokens 场景
- 结果：`9 passed, 14 warnings in 8.14s`

### 4.27 `_prepare_dflash_inputs_kernel_ascend`（#27，✅）
- 来源：`vllm_ascend.worker.v2.spec_decode.dflash.speculator`（含 `copy_and_expand_dflash_inputs_kernel_single_grid`）
- 矩阵：num_reqs∈{1,2}×SAMPLE_FROM_ANCHOR∈{False,True}（旧版 vLLM 回退 {False}）；num_speculative_steps=3
- 固定：按内核签名传 30 个位置参数（含 out_temperature/out_seeds/temperature/seeds），CPU reference 同序对齐
- 结果：`8 passed, 14 warnings in 8.34s`

### 4.28 `_prepare_pos_seq_lens_kernel`（#28，✅）
- 来源：`vllm.v1.worker.gpu.input_batch`；主输入 `pos[num_tokens]`、`seq_lens[max_num_reqs]`
- 矩阵：num_reqs∈{1,2,4,8}×max_num_reqs∈{8,16}×tokens_per_req∈{1,4,8}；cudagraph padding；event-driven（num_tokens=0）
- 结果：`26 passed, 14 warnings in 5.07s`

### 4.29 `_prepare_rope_positions_kernel`（#29，✅）
- 来源：`vllm.v1.worker.gpu.mm.rope`
- 主输入：`positions[num_dims,max_num_tokens]`、`prefill_positions[num_reqs*num_dims,max_model_len]`
- 矩阵：num_dims∈{3,4}×num_reqs∈{1,4,8}(prefill)/{1,4}(decode)；max_model_len=512, max_num_tokens=256, prefill_len=20
- 结果：`11 passed, 14 warnings in 7.37s`

### 4.30 `_prompt_logprobs_token_ids_kernel`（#30，✅）
- 来源：`vllm.v1.worker.gpu.sample.prompt_logprob`
- 主输入：`all_token_ids[max_num_reqs,max_model_len]`、`token_ids[num_tokens]`
- 矩阵：num_reqs∈{1,2,4}×query_len∈{1,4,16}；max_model_len=128, max_num_reqs=8；nonzero num_computed
- 结果：`10 passed, 14 warnings in 7.62s`

### 4.31 `_ranks_kernel`（#31，✅）
- 来源：`vllm_ascend.worker.v2.sample.logprob`（`compute_topk_logprobs`）
- 矩阵（batch×vocab×num_logprobs）：(48,1024,5)/(96,1024,0)/(24,1519,1)/(1,320,10)
- 结果：`7 passed, 14 warnings in 7.46s`

### 4.32 `_probabilistic_rejection_kernel`（#32，✅）
- 来源：`vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils` 补丁版（替换 `_rejection_kernel`）
- 主输入：`target_logits[num_logits,vocab]`、`draft_logits[max_num_reqs,num_spec_steps,vocab]`
- 矩阵：greedy num_draft∈{1,3,5}；non-greedy temp∈{0.5,1.0,2.0}；vocab∈{32,64,128}
- 结果：`21 passed, 14 warnings in 7.50s`

### 4.33 `_resample_kernel`（#33，✅）
- 来源：`vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils` 补丁版
- 主输入：`target_logits[num_logits,vocab]`、`resampled_local_argmax[num_reqs,num_blocks]`
- 场景：greedy bonus_token（vocab=512, num_blocks=1）、non_bonus_greedy、非贪婪
- 结果：`5 passed, 14 warnings in 7.85s`

### 4.34 `_scatter_num_accepted_kernel`（#34，✅）
- 来源：`vllm.v1.worker.gpu.model_states.mamba_hybrid`
- 主输入：`idx_mapping[num_reqs]`、`num_accepted[max_num_reqs]`
- 矩阵：num_reqs∈{1,4,8,16}×max_num_reqs∈{16,32}；负 idx 跳过；clamp-to-one
- 结果：`10 passed, 14 warnings in 8.46s`

### 4.35 `_selective_scan_update_kernel`（#35，✅）
- 来源：上游 `vllm/model_executor/layers/mamba/ops/mamba_ssm.py`，NPU 复用
- 主输入：`state[b,h,d,dstate]` fp32 及相关 A/B/C/D/dt/dt_bias/z
- 常量开关：HAS_DT_BIAS/HAS_D/HAS_Z/HAS_STATE_BATCH_INDICES/IS_SPEC_DECODING/IS_VARLEN/TIE_HDIM/DT_SOFTPLUS/BLOCK_SIZE_DSTATE；BLOCK_SIZE_M 由 `try_get_optimal_ssm_config` 选取
- 矩阵：full `bathch{1,4}×nheads32×dim64×dstate128×ngroups8` 及 Jamba/Mamba2（128/8/1、64/16/1）变体；no_z_no_d、变 dstate(8/16/128)、tie_hdim(expand stride=0)
- 数值稳定固定：A=−rand−1.0（[-2,-1)），dt_bias=rand−4.0（[-4,-3)），dt=randn；不同 seed(42/123/99/55)
- 准则：`assert_close(out.cpu(), expected, rtol=1e-4, atol=1e-4)` + state 校验
- 结果：`13 passed, 14 warnings in 8.04s`

### 4.36 `_temperature_kernel`（#36，✅）
- 来源：`vllm_ascend.worker.v2.sample.gumbel`（`apply_temperature`）
- 矩阵：`num_tokens=randint(1,64)` × vocab∈{**32000,50257,65024,128256,151936**}；另 test 跳过/跳过 0 和 1 温度
- 结果：`8 passed, 14 warnings in 5.30s`

### 4.37 `_topk_log_softmax_kernel`（#37，✅）
- 来源：`vllm_ascend.worker.v2.sample.logprob`；主输入 `logits[batch,vocab]` fp32、`token_ids[batch,num_logprobs]`
- 矩阵（batch,vocab,num_logprobs）：(48,102400,50)/(96,102400,1)/(24,151936,8)
- 结果：`6 passed, 14 warnings in 7.48s`

### 4.38 `_topk_topp_kernel`（#38，❌ 失败）
- 来源：`vllm/v1/sample/ops/topk_topp_triton.py`（grid=(1,)，核内按 BATCH_SIZE 循环）
- 主输入：`logits[batch_size,vocab]` fp32；含 topk_only / topp_only / 合并路径
- 日志：`1 failed, 14 warnings in 24.35s`（首个即停，编译耗时高）
- 失败用例：`test_topk_topp_realistic[topk_only-1-32000]`，BiSheng 后端链接错误，详见第 5.3 节

### 4.39 `_update_draft_inputs_kernel`（#39，✅）
- 来源：`vllm.v1.worker.gpu.spec_decode.autoregressive.speculator`
- 主输入：`output_draft_tokens[max_num_reqs,num_spec_steps+1]`、`hidden_states[num_reqs,hidden_size]` fp16
- 矩阵：num_reqs∈{1,2,4}×hidden_size∈{128,512}×advance_pos∈{False,True}；final-step 跳过更新
- 结果：`31 passed, 14 warnings in 8.45s`

### 4.40 `_update_min_larger_stats_kernel`（#40，✅）
- 来源：`vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils`
- 主输入：`target_logits[...]` 相关统计输出
- 结果：`12 passed, 14 warnings in 5.03s`

### 4.41 `_zero_kv_blocks_kernel`（#41，✅）
- 来源：`_resolve_kernel()` 优先 vllm_ascend 补丁（GRID_SIZE），回退 vllm upstream
- 结果：`9 passed, 14 warnings in 9.96s`

---

## 5. 失败用例详细分析（仅 NPU 环境）

> 3 个失败均在本环境（Ascend950PR_9579 / CANN 9.1.0 / triton ascend backend）复现。

### 5.1 `_fill_logprob_token_ids_kernel` — 精度/控制流 bug（非编译错）

- 失败用例：`TestFillLogprobTokenIdsKernel::test_custom_token_ids[3-4]`（batch=4, topk=4）
- 统计：`1 failed, 4 passed`（`stopping after 1 failures`，耗时 7.69s）
- 诊断（`_fill_logprob_token_ids_kernel mismatch diagnostic`）：

```
batch=1 req_state_idx=1 num_custom=0 expected_branch=topk status=MISMATCH
  out_token_ids[b]=[359, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  expected_ids[b]=[359, 859,714,467,0,0,0,0,0,0,0,0,0,0,0,0,0]
  out_valid_mask 与 expected_valid_mask 均=[True,True,True,True,False,...]
  （batch=2、batch=3 同样：仅第 1 个 token 被写出）
```

- 结论：**仅 `num_custom=0` 走 topk 分支的 batch 出错** —— topk 分支只写入了第 1 个 top token，后续 `num_topk`(4) 个未写出；valid_mask 反而正确。属内核 topk 分支写回逻辑缺陷。
- 后续动作：检查 `_fill_logprob_token_ids_kernel` topk 分支的写回循环/掩码。

### 5.2 `_num_nans_kernel` — Triton 前端编译错误

- 失败用例：`test_num_nans[0.0-128-1]`
- 统计：`1 failed`（首个即停，耗时 5.05s）
- 诊断：

```
triton.compiler.errors.CompilationError: at 17:17:
    is_nan = libdevice.isnan(logits).to(tl.int1)
                       ^
AttributeError("'NoneType' object has no attribute 'to'")
```

- 结论：NPU Triton 后端的 `libdevice.isnan` 返回 `None`，**不支持该 libdevice 原语**。需改为 `tl.where(logits != logits, 1, 0)` 等等价写法，或在 UT 侧替换实现。

### 5.3 `_topk_topp_kernel` — BiSheng 后端链接错误

- 失败用例：`test_topk_topp_realistic[topk_only-1-32000]`
- 统计：`1 failed`（24.35s，编译耗时高；`stopping after 1 failures`）
- 诊断：

```
triton.compiler.errors.MLIRCompilationError
  [ConvertLinalgRToBinary] encounters error:
  ld.lld: error: undefined symbol: _mlir_ciface_cumsum_1d_bool_dim0
  >>> referenced by LLVMDialectModule  kernel.o:(cumsum_1d_bool_dim0)
  调用链：linalg_to_bin_enable_npu_compile_910_95 → bishengir-compile(
  --target=Ascend950PR_9579) → hivmc-a5 ... Failed to run BiShengIR pipeline
```

- 结论：内核中 **bool 类型的 `cumsum(..., dim=0)`（`cumsum_1d_bool`）在 bishengir/Ascend950PR_9579 上无链接实现**。需规避 bool 前缀和（改 int32 累加）。Triton 缓存：`/root/.triton/cache/...`。

---

## 6. 编写规范性自审（41 算子对照标准 2.1）

> 依据：《昇腾算子精度标准 2.1》测试方案（`ASCEND_OPERATOR_ACCURACY_2_1_TEST_PLAN.md`）。
> 判定尺度：本套日志属 **PR strict UT 层**，故 L0/L1/L2 的用例量级、重复次数、百万输出点、三指标（MARE/MERE/RMSE）统计归 Nightly/Release 层，本次不强求；按 §6.1 PR 层下限（6–15 case、tile 边界、1 生产 shape、确定性重复 3–10 次、sentinel/guard）+ 第 3 章各算子核心判据判定。

| # | 算子 | 档 | 判据落实情况 / 规范性缺口 |
|---|---|---|---|
| 1 | `_apply_grammar_bitmask_kernel` | ✅ | 允许值 bitwise、禁止值 `-inf`；缺 32-bitmask tile 边界（31/32/33 words）与乱序 idx |
| 2 | `_apply_write_kernel` | ⚠️ | 单/多组，`assert_close rtol=0` 精确；缺 sentinel/guard 区，无法发现越界多写 |
| 3 | `_prepare_prefill_inputs`(spec) | ⚠️ | 离散输出精确；缺 R<Rmax、graph padding、乱序 idx |
| 4 | `_bad_words_kernel` | ⚠️ | 命中 `-inf` 语义正确；形状偏小、缺词汇边界与乱序 idx |
| 5 | `_bias_kernel` | ✅ | 浮点 Ratio + mask `-inf`；覆盖生产 vocab；缺 NaN 注入、tile `B-1/B/B+1` |
| 6 | `_bincount_kernel` | ❌⚠️ | **仅 1 case（低于 6–15 下限）**；atomic 路径未重复运行验证确定性 |
| 7 | `_combine_sampled_and_draft_tokens_kernel` | ✅ | token id/offset 精确；缺 ragged/非连续组合 |
| 8 | `_compute_cumulative_log_p_kernel` | ✅ | **NPU 跳过**（无适配），符合 §7.8，不冒充 rejection 覆盖 |
| 9 | `_compute_local_logits_stats_kernel` | ⚠️ | 73 case；vocab 仅 {128,1024,8192}，缺 8191/8192/8193、动态范围≈200、`-inf`/NaN、平移不变性；未算 L2 三指标 |
| 10 | `_compute_local_residual_mass_kernel` | ✅ | **NPU 跳过**（无适配），符合 §7.8 |
| 11 | `_compute_slot_mappings_kernel` | ✅ | block 16/32、position `B-1/B/B+1`、cp interleave、PAD_ID；缺 block_size=64 |
| 12 | `_dcp_local_seq_lens_kernel` | ✅ | 50 case，最高 rank、全 0 seq；缺非 2 次幂请求数 |
| 13 | `_expand_idx_mapping_kernel` | ✅ | 不均匀 `[2,5,3]`、非连续 `[5,2,8]`；缺 guard 区 |
| 14 | `_fill_logprob_token_ids_kernel` | ❌ | **失败**：topk 分支漏写 token（逻辑 bug，需修内核写回） |
| 15 | `_flatten_sampled_kernel` | ✅ | 全 0、单 req 多 logits；缺「padding token 排除」显式 guard 断言 |
| 16 | `_gather_block_tables_kernel` | ✅ | 多 group、padding 补零；缺非 2 次幂行数 |
| 17 | `_get_num_sampled_and_rejected_kernel` | ✅ | chunked prefill、3 类映射已覆盖 |
| 18 | `_gumbel_sample_kernel` | ⚠️ | 确定性+可复现覆盖完整；**缺 §2.2 强制 KS/卡方检验与 N≥96 通过次数**（用启发式替代） |
| 19 | `_prepare_prefill_inputs`(input_batch) | ✅ | 含 prefill done early-return |
| 20 | `_insert_resampled_kernel` | ✅ | 26 case，局部 max→离散输出判据正确 |
| 21 | `_min_p_kernel` | ✅ | 显式分离 `-inf` mask 与保留值；缺 NaN、tile 边界 |
| 22 | `_num_nans_kernel` | ❌ | **编译失败**（`libdevice.isnan` 返回 None），不符合 |
| 23 | `_penalties_kernel` | ✅ | bf16/fp16 全组合，judged 阈值随 dtype 调整；缺特殊值、tile 边界 |
| 24 | `_post_update_kernel` | ⚠️ | 10 case 真实 shape；缺「防越界多写」的 guard 区显式校验 |
| 25 | `_post_update_num_computed_tokens_kernel` | ✅ | 非连续 idx、零 query_len；缺重复执行验证 |
| 26 | `_prepare_decode_inputs_kernel` | ✅ | advance_pos、rejected 覆盖 |
| 27 | `_prepare_dflash_inputs_kernel_ascend` | ✅ | 30 位置参数对齐、anchor 分支 |
| 28 | `_prepare_pos_seq_lens_kernel` | ✅ | cuda graph padding、event-driven；缺 guard |
| 29 | `_prepare_rope_positions_kernel` | ✅ | decode/prefill 多 dim |
| 30 | `_prompt_logprobs_token_ids_kernel` | ✅ | ragged 边界、nonzero num_computed |
| 31 | `_ranks_kernel` | ⚠️ | 仅正模输入；**缺唯一最大/唯一最小/全 ties/多 ties/`-inf` ties 边界**（§4.2 明列） |
| 32 | `_probabilistic_rejection_kernel` | ⚠️ | greedy 精确；NPU 补丁 `u=0` 恒接受 → 非贪婪路径分布检验被结构性降级（需声明） |
| 33 | `_resample_kernel` | ❌⚠️ | **仅 5 case（低于下限）**，无 KS，无 residual Ratio 统计 |
| 34 | `_scatter_num_accepted_kernel` | ✅ | 负 idx skip、clamp-to-one、`rtol=0`；缺 sentinel/guard 区 |
| 35 | `_selective_scan_update_kernel` | ✅ | A 取负 `[-2,-1)`、dt_bias 强负 `[-4,-3)`、dt=randn 数值稳定固定正确；TIE_HDIM 用 expand 保 stride=0 |
| 36 | `_temperature_kernel` | ⚠️ | 覆盖主流词表、注入 0/1 温度；no-op 用 `allclose` 非 bitwise；缺 NaN/Inf、tile 边界 |
| 37 | `_topk_log_softmax_kernel` | ⚠️ | 仅 `allclose 1e-3`；缺全相等/单极大/动态范围 200/平移/`-inf`/NaN 分布与 L2 三指标 |
| 38 | `_topk_topp_kernel` | ❌ | **编译失败**（BiSheng 缺 bool cumsum），不符合 |
| 39 | `_update_draft_inputs_kernel` | ✅ | 31 case，final_step 早退；缺乱序 idx |
| 40 | `_update_min_larger_stats_kernel` | ⚠️ | 12 case，判据偏功能验证，缺特殊值/分布统计 |
| 41 | `_zero_kv_blocks_kernel` | ✅ | `_resolve_kernel()` 优先 vllm_ascend 补丁、回退 upstream，符合约定 |

### 6.1 自审小结

- **✅ 规范：24 个**（含 2 个合理跳过的 block-verification 算子，符合 §7.8）。
- **⚠️ 基本规范但缺判据：13 个**，集中缺口为：
  1. 浮点算子缺特殊值（NaN/Inf/动态范围≈200/平移不变性）；
  2. 缺 tile 边界 `B-1/B/B+1`；
  3. 缺 sentinel/guard 区（输出预填哨兵 + 尾部 guard）；
  4. 随机类缺 §2.2 强制 KS/卡方分布检验；
  5. L2 算子未算三指标与小值域 ErrorCount（该项归 Nightly 层，可延后）。
- **❌ 不达标：4 个**：`num_nans`（编译）、`topk_topp`（编译）、`fill_logprob`（逻辑错）、`bincount`（用例严重不足且缺 atomic 确定性）。

---

## 7. 覆盖度统计（仅 NPU 环境）

- **NPU 实际 launch 算子**：39（37 个文件中 2 个跳过项不计）
- 常用 Shape 维度汇总：

| 维度 | 覆盖取值 |
|---|---|
| vocab_size（主流词表） | 32000, 50257, 65024, 102400, 128256, 151936 |
| vocab_size（小规模） | 128 ~ 16384 |
| num_reqs / batch_size | 1 ~ 128（bad_words num_tokens 至 2048）|
| num_speculative_steps / num_draft | 1~5 / 1~5 |
| hidden_size | 64, 128, 512 |
| num_groups / max_num_blocks | 1~4 / 32~128 |
| num_logprobs | 0,1,5,8,10,50 |
| dcp_size / cp_interleave | 2,4 / 1,2 |

- 逐算子 vocab 等覆盖明细见 `npu_ut_shapes.md` 附录 A–D。

---

## 8. 结论与建议

1. **39/41 文件正常**，其中 36 个通过、3 个失败、2 个因无 NPU 适配跳过。
2. **Top 待修复**：
   - `_topk_topp_kernel`（bool cumsum 无链接）——优先，影响 topk/topp 采样路径。
   - `_num_nans_kernel`（libdevice.isnan None）——替换实现即可。
   - `_fill_logprob_token_ids_kernel`（topk 分支漏写）——纯逻辑 bug，需要修内核写回。
3. 通过项中已覆盖主流模型词表（151936/128256/102400 等）与投机解码主流配置，可作为回归基线。

---

*本报告由测试产物（`npu_ut_shapes.md` + `log-a5-1.txt`）整理生成，仅适用于 NPU 环境。*