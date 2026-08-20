# strict_ut/npu 算子 UT 测试 Shape 统计

> 统计范围：`xyz-triton-test-0725/accuracy_test/strict_ut/npu/` 下所有 NPU 算子 UT 测试。
> 说明：`parametrize` 列出的为参数化组合；未参数化测试的 shape 以“固定值”列出。

---

## 1. test_apply_grammar_bitmask_kernel.py
- 算子：`_apply_grammar_bitmask_kernel`（vllm_ascend.worker.v2.structured_outputs 补丁版）
- 主输入：`logits[num_logits, vocab_size]` fp32

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_basic_bitmask | `vocab_size` ∈ {128, 1024, 8192} | `num_bitmasks=2, num_logits=4`, `padded_vocab_words=ceil(vocab/32)` |
| test_all_allowed | `vocab_size` ∈ {128, 512, 4096} | `num_bitmasks=1, num_logits=2` |
| test_all_blocked | — | `vocab_size=256, num_bitmasks=1` |

---

## 2. test_apply_write_kernel.py
- 算子：`_apply_write_kernel`（vllm.v1.worker.gpu.buffer_utils）
- 主输入：`output[num_rows, num_cols]`

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_prefill | `num_rows` ∈ {1, 4}, `num_cols` ∈ {16, 32} | `BLOCK_SIZE=4` |
| test_multi_group | `num_groups` ∈ {1, 2, 4}, `num_writes_per_group` ∈ {1, 2} | `num_rows=num_writes_per_group, num_cols=16` |
| 其它（单组/多组写入） | — | 使用相同 num_rows/num_cols 组合 |

---

## 3. test_ar_prepare_prefill_inputs_kernel.py
- 算子：`_prepare_prefill_inputs_kernel`（speculator 变体，vllm.v1.worker.gpu.spec_decode.autoregressive.speculator）
- 主输入：`target_input_ids[max_num_tokens]`, `target_positions[max_num_tokens]` int32

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_basic_prefill | `num_reqs` ∈ {1, 2, 4}, `query_len` ∈ {4, 16} | `max_num_reqs=8, max_num_tokens=num_reqs*(query_len+4)`, `seq_len=128` |
| test_chunked_prefill_path | `num_reqs` ∈ {1, 2} | `max_num_reqs=4, query_len=4, max_num_tokens=num_reqs*(query_len+2)` |
| test_rejected_tokens | `num_reqs` ∈ {1, 2} | — |

---

## 4. test_bad_words.py
- 算子：`_bad_words_kernel` via `apply_bad_words`（vllm_ascend.worker.v2.sample.bad_words）
- 主输入：`logits[num_tokens, vocab_size]` fp32

| 测试函数 | 参数化 shape（id） | 说明 |
|---|---|---|
| test_apply_bad_words | small-case, medium-case, large-case | 见下表 |

| case | num_tokens | vocab_size | num_requests | num_bad_words_per_req | bad_word_length |
|---|---|---|---|---|---|
| small-case | 512 | 50257 | 16 | 3 | 2 |
| medium-case | 1024 | 50257 | 32 | 5 | 3 |
| large-case | 2048 | 50257 | 64 | 8 | 4 |

- 固定 shape：`bad_word_token_ids[reqs, 1024]`, `bad_word_offsets[reqs, 129]`, `all_token_ids[reqs, 1024]`

---

## 5. test_bias_kernel.py
- 算子：`_bias_kernel`（vllm.v1.worker.gpu.sample.logit_bias）
- 主输入：`logits[num_tokens, vocab_size]` fp32

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_allowed_token_ids | `num_tokens` ∈ {1, 4, 8}, `vocab_size` ∈ {128, 1024} | `num_reqs=4` |
| test_logit_bias | `num_tokens` ∈ {1, 4} | `num_reqs=2, vocab_size=64` |
| test_min_tokens | — | — |

- 固定 shape：`allowed_token_ids[reqs, 1024]`, `bias_token_ids[reqs, 1024]`, `stop_token_ids[reqs, 128]`

---

## 6. test_bincount.py
- 算子：`_bincount_kernel`（vllm_ascend.worker.v2.sample.penalties）
- 主输入：`all_token_ids[64, 40960]` int32

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_bincount_kernel | — | `expanded_idx_mapping[1]`, `all_token_ids[64, 40960]`, `prompt_len[64]`, `prefill_len[64]`, `prompt_bin_mask[64, 4748]`, `output_bin_counts[64, 151936]`, `max_prefill_len=10` |

---

## 7. test_combine_sampled_and_draft_tokens_kernel.py
- 算子：`_combine_sampled_and_draft_tokens_kernel`（vllm.v1.worker.gpu.input_batch）
- 主输入：`input_ids[num_tokens]`, `draft_tokens[reqs, num_spec_steps]`

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_combine_basic | `num_reqs` ∈ {1, 2, 4}, `num_spec_steps` ∈ {1, 3}, `num_new_sampled_tokens` ∈ {0, 1} | `vocab_size=100, seq_len=20, prefill_len=5` |

---

## 8. test_compute_cumulative_log_p_kernel.py
- 算子：`_compute_cumulative_log_p_kernel`
- 状态：**无 NPU 适配，测试跳过**（NPU 上不 launch）

---

## 9. test_compute_local_logits_stats_kernel.py
- 算子：`_compute_local_logits_stats_kernel`（vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils）
- 主输入：`target_logits[num_logits, vocab_size]`, `draft_logits[max_num_reqs, num_spec_steps, vocab_size]` fp32

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_compute_block_max_and_sumexp | `num_logits` ∈ {1, 2, 4}, `vocab_size` ∈ {128, 1024, 8192}, `num_speculative_steps` ∈ {2, 3} | `max_num_reqs=4, VOCAB_BLOCK_SIZE=8192` |

---

## 10. test_compute_local_residual_mass_kernel.py
- 算子：`_compute_local_residual_mass_kernel`
- 状态：**无 NPU 适配，测试跳过**（NPU 上不 launch）

---

## 11. test_compute_slot_mappings_kernel.py
- 算子：`_compute_slot_mappings_kernel`（vllm_ascend.worker.v2.block_table / upstream）
- 主输入：`positions[pos_len]`, `block_table[reqs, blocks]`

| 测试函数 | 参数化 shape（block_size, positions, cp_size, cp_rank, cp_interleave） |
|---|---|
| test_compute_slot_mappings | (16, [15,16,17,31,32], 1, 0, 1) / (32, [31,32,33,63,64], 1, 0, 1) / (16, [0,1,2,3,16,17,18,19], 2, 0, 2) / (16, [0,1,2,3,16,17,18,19], 2, 1, 2) |

- 固定 shape：`max_num_tokens=64, max_num_reqs=8, num_reqs=2, block_table[8, 64/4096]`

---

## 12. test_dcp_local_seq_lens_kernel.py
- 算子：`_dcp_local_seq_lens_kernel`（vllm.v1.worker.gpu.cp_utils）
- 主输入：`seq_lens[max_num_reqs]` int32

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_dcp_local_seq_lens | `num_reqs` ∈ {2, 4, 8}, `max_num_reqs` ∈ {8, 16}, `dcp_size` ∈ {2, 4}, `dcp_rank` ∈ {0, 1}, `cp_interleave` ∈ {1, 2} | `seq_lens` 值 ∈ [1, 128) |
| test_dcp_rank_highest | — | `num_reqs=4, max_num_reqs=8, dcp_size=4, dcp_rank=3, cp_interleave=1`, `seq_lens=[10,15,20,25,0,0,0,0]` |
| test_zero_seq_lens | — | `num_reqs=2, max_num_reqs=4, dcp_size=2, dcp_rank=0, cp_interleave=1` |

---

## 13. test_expand_idx_mapping_kernel.py
- 算子：`_expand_idx_mapping_kernel`（vllm.v1.worker.gpu.input_batch）
- 主输入：`idx_mapping[num_reqs]`, `cu_num_logits[num_reqs+1]`

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_basic_expand | `num_reqs` ∈ {1, 2, 4}, `tokens_per_req` ∈ {1, 3, 8} | `total_logits=num_reqs*tokens_per_req` |
| test_uneven_tokens | — | `num_reqs=3, tokens_per_req=[2,5,3], total=10` |
| test_non_contiguous_idx_mapping | — | `num_reqs=3, tokens_per_req=2, idx_mapping=[5,2,8]` |

---

## 14. test_fill_logprob_token_ids_kernel.py
- 算子：`_fill_logprob_token_ids_kernel`（vllm.v1.worker.gpu.sample.logprob）
- 主输入：`out_token_ids[batch_size, 1+PADDED_COLS]`, `topk_indices[batch_size, NUM_TOPK]`

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_custom_token_ids | `batch_size` ∈ {1, 4, 8}, `topk` ∈ {0, 3, 5} | `num_reqs=4, PADDED_COLS=16, MAX_LOGPROB_TOKEN_IDS=128` |

---

## 15. test_flatten_sampled_kernel.py
- 算子：`_flatten_sampled_kernel`（vllm.v1.worker.gpu.spec_decode.rejection_sampler）
- 主输入：`sampled[num_reqs, num_spec_steps+1]` int64

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_flatten_basic | `num_reqs` ∈ {1, 2, 4, 8}, `num_spec_steps` ∈ {1, 3, 5} | `total_num_logits=num_reqs*(num_spec_steps+1)` |
| test_all_zeros_num_sampled | — | `num_reqs=3, num_spec_steps=2` |
| test_single_req_multi_logits | — | `num_reqs=1, num_spec_steps=10, total=11` |

---

## 16. test_gather_block_tables_kernel.py
- 算子：`_gather_block_tables_kernel`（vllm.v1.worker.gpu.block_table）
- 主输入：`src_block_tables[num_groups, max_num_reqs, max_num_blocks]`

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_gather_basic | `num_groups` ∈ {1, 2, 4}, `max_num_reqs` ∈ {4, 8}, `max_num_blocks` ∈ {64, 128} | `num_reqs=max_num_reqs, BLOCK_SIZE=16` |
| test_padding_zeros | `num_groups` ∈ {1, 2} | `max_num_reqs=8, num_reqs=4, max_num_blocks=32` |

---

## 17. test_get_num_sampled_and_rejected_kernel.py
- 算子：`_get_num_sampled_and_rejected_kernel`（vllm.v1.worker.gpu.input_batch）
- 主输入：`num_sampled[num_reqs]`, `cu_num_logits[num_reqs+1]`

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_basic | `num_reqs` ∈ {1, 2, 4}, `num_logits_per_req` ∈ {1, 3, 5} | `max_num_reqs=max(num_reqs,4), seq_len=20, prefill_len=10` |
| test_chunked_prefilling | — | `num_reqs=2, max_num_reqs=2, cu_num_logits=[0,4,10]` |
| test_various_sampled_counts | `(num_sampled_val, expected_rejected)` ∈ {(0,3),(2,1),(3,0)} | `num_reqs=1, num_logits=3` |

---

## 18. test_gumbel_sample.py
- 算子：`_gumbel_sample_kernel` / `_temperature_kernel` via `gumbel_sample` / `apply_temperature`（vllm_ascend.worker.v2.sample.gumbel）
- 主输入：`logits[num_tokens, vocab_size]` fp32

| 测试函数 | 参数化 shape（num_tokens, [num_reqs,] vocab_size） |
|---|---|
| test_apply_temperature | (1, 32000) / (8, 32000) / (48, 102400) / (64, 151936) |
| test_gumbel_sample_greedy | (1, 1, 32000) / (4, 4, 32000) / (8, 4, 32000) / (16, 8, 102400) |
| test_gumbel_sample_deterministic | (4, 4, 32000) / (8, 4, 32000) / (16, 8, 102400) |
| test_gumbel_sample_valid_token_ids | (4, 4, 32000) / (8, 4, 32000) / (16, 8, 102400) |
| test_gumbel_sample_mixed_temperature | (4, 4, 32000) / (8, 4, 32000) |

| 非参数化测试 | 固定 shape |
|---|---|
| test_apply_temperature_skip_zero_and_one | `num_tokens=4, vocab_size=32000` |
| test_gumbel_sample_greedy_apply_temp_flag_irrelevant | `num_tokens=4, num_reqs=4, vocab_size=32000` |
| test_gumbel_sample_different_seeds | `num_tokens=16, num_reqs=16, vocab_size=32000` |
| test_gumbel_sample_temperature_affects_distribution | — |

---

## 19. test_input_batch_prepare_prefill_inputs_kernel.py
- 算子：`_prepare_prefill_inputs_kernel`（vllm.v1.worker.gpu.input_batch）
- 主输入：`all_token_ids[max_num_reqs, max_model_len]` int32

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_prepare_prefill_inputs | `num_reqs` ∈ {1, 2, 4}, `query_len` ∈ {1, 4, 16} | `max_model_len=128, max_num_reqs=8, num_lookahead=3, prefill_len=64` |
| test_early_return_when_prefill_done | — | `num_reqs=2, query_len=4, max_num_tokens=8, max_num_reqs=4, max_model_len=32, num_lookahead=2` |

---

## 20. test_insert_resampled_kernel.py
- 算子：`_insert_resampled_kernel`（vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils）
- 主输入：`sampled[num_reqs, num_spec_steps+1]`, `resampled_local_argmax[num_reqs, num_blocks]`

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_insert_resampled_basic | `num_reqs` ∈ {1, 2, 4}, `num_spec_steps` ∈ {1, 3} | `vocab_size=4096, RESAMPLE_BLOCK_SIZE=1024, num_blocks=4` |

---

## 21. test_min_p.py
- 算子：`_min_p_kernel` via `apply_min_p`（vllm_ascend.worker.v2.sample.min_p）
- 主输入：`logits[num_reqs, vocab_size]` fp32

| 测试函数 | 参数化 shape（num_reqs, vocab_size） |
|---|---|
| test_apply_min_p_kernel | (48, 102400) / (96, 102400) / (24, 151936) / (1, 32000) |

---

## 22. test_num_nans_kernel.py
- 算子：`_num_nans_kernel`（vllm.v1.worker.gpu.metrics.logits）
- 主输入：`logits[num_reqs, vocab_size]` fp32

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_num_nans | `num_reqs` ∈ {1, 2, 4, 8}, `vocab_size` ∈ {128, 1024, 8192, 16384}, `frac_nan` ∈ {0.0, 0.1, 0.5, 1.0} | `BLOCK_SIZE=8192` |
| test_no_nans | — | `num_reqs=4, vocab_size=4096` |
| test_all_nans | — | `num_reqs=3, vocab_size=512` |

---

## 23. test_penalties.py
- 算子：`_penalties_kernel` via `apply_penalties`（vllm_ascend.worker.v2.sample.penalties）
- 主输入：`logits[num_tokens, vocab_size]` fp32

| 参数 | 取值 |
|---|---|
| num_tokens | {1, 4} |
| vocab_size | {1000} |
| num_status | {1, 4} |
| num_speculative_tokens | {0, 1, 3} |
| dtype | {bfloat16, float16} |
| seed | {42} |
| device | {"npu:0"} |

- 组合数：2 × 1 × 2 × 3 × 2 × 1 × 1 = 24 组

---

## 24. test_post_update_kernel.py
- 算子：`_post_update_kernel`（upstream vllm.v1.worker.gpu.input_batch vs vllm_ascend.worker.v2.input_batch）
- 主输入：`output_bin_counts[max_num_reqs, vocab_size]`, `sampled_tokens[num_reqs, num_spec_steps+1]`, `all_token_ids[max_num_reqs, max_model_len]` int32

| 测试函数 | 参数化 shape（num_reqs, max_num_reqs, vocab_size, num_speculative_steps） |
|---|---|
| test_post_update | (36, 36, 200, 2) / (48, 48, 32000, 5) / (128, 128, 32000, 5) |

- 固定：`max_model_len=3000, query_lengths ∈ [1,20), total_len ∈ [50,200)`

---

## 25. test_post_update_num_computed_tokens_kernel.py
- 算子：`_post_update_num_computed_tokens_kernel`（vllm.v1.worker.gpu.input_batch）
- 主输入：`num_computed_tokens[max_num_reqs]`, `query_start_loc[num_reqs+1]`

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_basic_increment | `num_reqs` ∈ {1, 2, 4}, `query_len` ∈ {1, 4, 8} | `max_num_reqs=max(num_reqs,4), num_computed=10` |
| test_non_contiguous_idx_mapping | — | `num_reqs=3, max_num_reqs=6, idx_mapping=[5,0,3]` |
| test_zero_query_len | — | `num_reqs=2, max_num_reqs=2` |

---

## 26. test_prepare_decode_inputs_kernel.py
- 算子：`_prepare_decode_inputs_kernel`（vllm.v1.worker.gpu.spec_decode.autoregressive.speculator）
- 主输入：`draft_tokens[num_reqs, 1]`, `input_ids[max_num_tokens]`

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_prepare_decode_inputs | `num_reqs` ∈ {1, 2, 4}, `advance_pos` ∈ {False, True} | `max_num_tokens=num_reqs, max_num_reqs=8, max_model_len=2048` |
| test_with_rejected_tokens | — | `num_reqs=2, max_num_reqs=4, max_model_len=64, draft_tokens=[42,99]` |

---

## 27. test_prepare_dflash_inputs_kernel.py
- 算子：`_prepare_dflash_inputs_kernel_ascend` / `copy_and_expand_dflash_inputs_kernel_single_grid`（vllm_ascend.worker.v2.spec_decode.dflash.speculator）
- 主输入：`block_table[num_reqs, num_blocks_in_table]`, 多输出 [max_num_tokens] / [max_num_reqs]

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_prepare_dflash_inputs | `num_reqs` ∈ {1, 2}, `SAMPLE_FROM_ANCHOR` ∈ {False, True}（旧版 vLLM 时回退为 {False}） | `num_speculative_steps=3`, `max_num_tokens`, `max_num_reqs*num_spec_steps` 输出 |

---

## 28. test_prepare_pos_seq_lens_kernel.py
- 算子：`_prepare_pos_seq_lens_kernel`（vllm.v1.worker.gpu.input_batch）
- 主输入：`pos[num_tokens]`, `seq_lens[max_num_reqs]`

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_pos_seq_lens | `num_reqs` ∈ {1, 2, 4, 8}, `max_num_reqs` ∈ {8, 16}, `tokens_per_req` ∈ {1, 4, 8} | `num_tokens=num_reqs*tokens_per_req, num_computed ∈ [0,64)` |
| test_cuda_graph_padding | — | `num_reqs=2, max_num_reqs=8, tokens_per_req=4, num_tokens=8` |
| test_event_driven | — | `num_reqs=2, max_num_reqs=8, num_tokens=0` |

---

## 29. test_prepare_rope_positions_kernel.py
- 算子：`_prepare_rope_positions_kernel`（vllm.v1.worker.gpu.mm.rope）
- 主输入：`positions[num_dims, max_num_tokens]`, `prefill_positions[num_reqs*num_dims, max_model_len]`

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_prefill | `num_dims` ∈ {3, 4}, `num_reqs` ∈ {1, 4, 8} | `max_model_len=512, max_num_tokens=256, prefill_len=20` |
| test_decode | `num_dims` ∈ {3, 4}, `num_reqs` ∈ {1, 4} | 同上 |

---

## 30. test_prompt_logprobs_token_ids_kernel.py
- 算子：`_prompt_logprobs_token_ids_kernel`（vllm.v1.worker.gpu.sample.prompt_logprob）
- 主输入：`all_token_ids[max_num_reqs, max_model_len]`, `token_ids[num_tokens]`

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_prompt_logprobs_token_ids | `num_reqs` ∈ {1, 2, 4}, `query_len` ∈ {1, 4, 16} | `max_model_len=128, max_num_reqs=8, num_tokens=num_reqs*query_len, vocab_size=64` |
| test_nonzero_num_computed_tokens | — | `num_reqs=2, query_len=3, num_tokens=6, max_model_len=32, max_num_reqs=4` |

---

## 31. test_ranks.py
- 算子：`_ranks_kernel` via `compute_topk_logprobs`（vllm_ascend.worker.v2.sample.logprob）
- 主输入：`logits[batch_size, vocab_size]` fp32

| 测试函数 | 参数化 shape（batch_size, vocab_size, num_logprobs） |
|---|---|
| test_compute_topk_logprobs | (48, 1024, 5) / (96, 1024, 0) / (24, 1519, 1) / (1, 320, 10) |

---

## 32. test_rejection_kernel.py
- 算子：`_probabilistic_rejection_kernel`（vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils 补丁版，替换 `_rejection_kernel`）
- 主输入：`target_logits[num_logits, vocab_size]`, `draft_logits[max_num_reqs, num_spec_steps, vocab_size]`

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_greedy_varying_lengths | `num_draft_tokens` ∈ {1, 3, 5} | `num_reqs=1, num_logits=num_draft_tokens+1, vocab_size=128` |
| test_non_greedy_varying_temps | `temp` ∈ {0.5, 1.0, 2.0} | `num_reqs=1, num_spec_steps=2, num_logits=3, vocab_size=128` |
| test_varying_vocab_sizes | `vocab_size` ∈ {32, 64, 128} | `num_reqs=1, num_spec_steps=3, num_logits=4` |
| 其它（greedy/non-greedy 基础） | — | `num_reqs=1, num_spec_steps∈{2,3}, num_logits=num_spec_steps+1` |

---

## 33. test_resample_kernel.py
- 算子：`_resample_kernel`（vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils 补丁版）
- 主输入：`target_logits[num_logits, vocab_size]`, `resampled_local_argmax[num_reqs, num_blocks]`

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_greedy_bonus_token | — | `num_reqs=1, vocab_size=512, num_blocks=1, num_logits=2` |
| test_non_bonus_greedy | — | `num_reqs=1, vocab_size=128, num_blocks=1, num_logits=2` |
| 其它（非贪婪采样） | — | `num_reqs=1` |

---

## 34. test_scatter_num_accepted_kernel.py
- 算子：`_scatter_num_accepted_kernel`（vllm.v1.worker.gpu.model_states.mamba_hybrid）
- 主输入：`idx_mapping[num_reqs]`, `num_accepted[max_num_reqs]`

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_scatter_basic | `num_reqs` ∈ {1, 4, 8, 16}, `max_num_reqs` ∈ {16, 32} | `num_sampled ∈ [0,5)` |
| test_skip_negative | — | `num_reqs=6, max_num_reqs=8, idx_mapping=[-1,3,-1,1,5,0]` |
| test_clamp_to_one | — | `num_reqs=4, max_num_reqs=4, num_sampled=[0,-5,-1,0]` |

---

## 35. test_temperature.py
- 算子：`_temperature_kernel` via `apply_temperature`（vllm_ascend.worker.v2.sample.gumbel）
- 主输入：`logits[num_tokens, vocab_size]` fp32

| 测试函数 | 参数化 shape |
|---|---|
| test_temperature_kernel | `num_tokens = random.randint(1,64)`，`vocab_size` ∈ {**32000, 50257, 65024, 128256, 151936**} |

- 备注：`num_tokens` 每次运行随机（1~64），`vocab_size` 固定取主流模型词表大小。

---

## 36. test_topk_log_softmax.py
- 算子：`_topk_log_softmax_kernel`（vllm_ascend.worker.v2.sample.logprob）
- 主输入：`logits[batch_size, vocab_size]` fp32, `token_ids[batch_size, num_logprobs]`

| 测试函数 | 参数化 shape（batch_size, vocab_size, num_logprobs） |
|---|---|
| test_topk_log_softmax_kernel | (48, 102400, 50) / (96, 102400, 1) / (24, 151936, 8) |

---

## 37. test_update_draft_inputs_kernel.py
- 算子：`_update_draft_inputs_kernel`（vllm.v1.worker.gpu.spec_decode.autoregressive.speculator）
- 主输入：`output_draft_tokens[max_num_reqs, num_spec_steps+1]`, `hidden_states[num_reqs, hidden_size]` fp16

| 测试函数 | 参数化 shape | 固定 shape |
|---|---|---|
| test_update_draft_inputs | `num_reqs` ∈ {1, 2, 4}, `hidden_size` ∈ {128, 512}, `advance_pos` ∈ {False, True} | `num_speculative_steps=3, max_num_reqs=8, max_model_len=2048` |
| test_final_step_skips_update | — | `num_reqs=2, hidden_size=64, num_speculative_steps=3, max_num_reqs=4, max_model_len=128` |

---

## 汇总：NPU 上实际 launch 的算子（不含跳过项）

在 NPU 上实际执行/launch 的算子共 **35 个**（37 个文件中，`test_compute_cumulative_log_p_kernel.py` 与 `test_compute_local_residual_mass_kernel.py` 两个为跳过状态）。

常用 Shape 维度统计：
- **vocab_size 覆盖**（词表维度）：32000, 50257, 65024, 102400, 128256, 151936 等主流模型词表；小规模测试 128~16384。
- **num_tokens / num_reqs / batch_size**：1 ~ 2048（bad_words 到 2048）。
- **num_speculative_steps**：1 ~ 5；**periodic steps**：2~3。
- **hidden_size**：128 / 512。
- **block / 分组**：num_groups 1~4，max_num_blocks 32~128。

---

## 附录：按维度汇总（各算子的 Shape 覆盖）

以下表格**去重**列出每个算子实际覆盖的维度取值，便于横向对比覆盖广度。
（`--` 表示该算子不涉及此维度；`n` 前的 `~` 表示近似/随机取值。）

### 附录 A：vocab_size（词表大小）覆盖

| 算子 | 覆盖的 vocab_size |
|---|---|
| _apply_grammar_bitmask_kernel | 128, 256, 512, 1024, 4096, 8192 |
| _bad_words_kernel | 50257 |
| _bias_kernel | 64, 128, 1024 |
| _bincount_kernel | 40960（buffer），151936（bin） |
| _compute_local_logits_stats_kernel | 128, 1024, 8192 |
| _gumbel_sample_kernel / _temperature_kernel | 32000, 50257, 65024, 102400, 128256, 151936 |
| _min_p_kernel | 32000, 102400, 151936 |
| _num_nans_kernel | 128, 512, 1024, 4096, 8192, 16384 |
| _penalties_kernel | 1000 |
| _post_update_kernel | 200, 32000 |
| _ranks_kernel | 320, 1024, 1519 |
| _rejection_kernel | 32, 64, 128 |
| _resample_kernel | 128, 512 |
| _temperature_kernel | 32000, 50257, 65024, 128256, 151936 |
| _topk_log_softmax_kernel | 102400, 151936 |

### 附录 B：请求数 / batch_size / num_reqs 覆盖

| 维度 | 覆盖取值 |
|---|---|
| num_reqs（请求数） | 1, 2, 3, 4, 6, 8, 16, 36, 48, 96, 128 |
| batch_size（topk/ranks） | 1, 4, 8, 24, 48, 96 |
| num_tokens（token 数） | 1~2048（bad_words 到 2048；temperature 随机 1~64） |
| max_num_reqs（缓冲上限） | 2, 4, 6, 8, 16, 32, 36, 48, 128 |

### 附录 C：投机解码相关覆盖

| 维度 | 覆盖取值 | 涉及算子 |
|---|---|---|
| num_speculative_steps | 1, 2, 3, 5 | combine/rejection/update/flatten/computer |
| num_draft_tokens | 1, 3, 5 | rejection |
| num_logits | 1, 2, 3, 4 | compute_local_stats / rejection |
| num_logits_per_req | 1, 3, 5 | get_num_sampled_and_rejected |
| num_new_sampled_tokens | 0, 1 | combine_sampled_and_draft |

### 附录 D：其它维度覆盖

| 维度 | 覆盖取值 | 涉及算子 |
|---|---|---|
| hidden_size | 64, 128, 512 | update_draft_inputs |
| query_len | 1, 4, 8, 16 | 各类 prepare_prefill_inputs / prompt_logprobs |
| max_num_blocks | 32, 64, 128 | gather_block_tables |
| num_groups | 1, 2, 4 | gather_block_tables / apply_write |
| num_dims | 3, 4 | prepare_rope_positions |
| dcp_size | 2, 4 | dcp_local_seq_lens |
| cp_interleave | 1, 2 | dcp_local_seq_lens / compute_slot_mappings |
| num_logprobs | 0, 1, 5, 8, 10, 50 | topk_log_softmax / fill_logprob / ranks |