# easy_ut_026 算子 shape 与位置汇总

> 目录：`accuracy_test/easy_ut_026`
> 说明：以下 9 个算子均为上游 vLLM Triton kernel（无下游 vllm-ascend 覆写），直接通过 `kernel[grid](...)` 调用。分类与判据依据 `ASCEND_OPERATOR_ACCURACY_2_1_TEST_PLAN.md`。

---

## 1. _cache_inputs_kernel

- **算子**：`_cache_inputs_kernel`（Triton kernel，grid = `(num_reqs, cdiv(hidden_size, BLOCK_SIZE))`）
- **位置**：`vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py`
- **分类**：非计算类 + 浮点拷贝融合（将最后 `num_speculative_steps-1` 个 draft 的 id/embeds/hidden 快照进缓存）
- **主输入**：
  - `draft_input_ids[num_tokens]` int32
  - `draft_input_embeds[num_tokens, hidden_size]` fp32（可选，MM 模型）
  - `draft_input_hidden_states[num_tokens, hidden_size]` fp32
  - `idx_mapping[num_reqs]` int64
  - `query_start_loc[max_num_reqs+1]` int32
  - `last_token_indices[num_reqs]` int32
- **输出**（判据：bitwise 一致）：
  - `cached_draft_input_ids[max_num_reqs, num_spec_steps-1]` int64
  - `cached_draft_input_embeds[max_num_reqs, num_spec_steps-1, hidden_size]` fp32（可选）
  - `cached_target_hidden_states[max_num_reqs, num_spec_steps-1, hidden_size]` fp32

| 测试函数 | 参数化 shape（id） | 说明 |
| --- | --- | --- |
| test_cache_inputs | num_reqs, hidden_size, num_speculative_steps, scenario, use_embeds, block_size | 见下表 |

| case | num_reqs | hidden_size | num_speculative_steps | scenario | use_embeds | block_size |
| --- | --- | --- | --- | --- | --- | --- |
| full_window 基础 | 1 | 16 | 2 | full_window | False | 8 |
| full_window 基础 | 2 | 32 | 3 | full_window | False | 8 |
| full_window 含 embeds | 2 | 32 | 3 | full_window | True | 8 |
| short_query 部分窗口 | 1 | 16 | 3 | short_query | False | 8 |
| short_query 含 embeds | 2 | 32 | 4 | short_query | True | 8 |
| exact_window 精确窗口 | 1 | 16 | 3 | exact_window | False | 8 |
| exact_window 含 embeds | 2 | 32 | 4 | exact_window | True | 8 |
| skip_padded 跳过 cudagraph 填充 | 4 | 32 | 3 | skip_padded | False | 8 |
| skip_padded 含 embeds | 4 | 32 | 3 | skip_padded | True | 8 |
| mixed 混合 | 4 | 64 | 3 | mixed | False | 8 |
| mixed 含 embeds | 4 | 64 | 3 | mixed | True | 8 |
| tile 边界 H=BLOCK | 1 | 8 | 2 | full_window | False | 8 |
| tile 边界 H=BLOCK+1 | 1 | 9 | 2 | full_window | False | 8 |
| tile 边界 H=BLOCK-1 | 1 | 7 | 2 | full_window | False | 8 |
| 更大 num_spec_steps | 2 | 32 | 5 | full_window | True | 8 |
| 真实 block_size | 2 | 1024 | 3 | full_window | True | 1024 |

- **固定 shape**：`max_num_reqs = max(8, num_reqs*2)`；`num_tokens = 256`；缓存 buffer 第 2 维固定为 `max(1, num_spec_steps-1)`。

---

## 2. _pad_trailing_draft_slots_kernel

- **算子**：`_pad_trailing_draft_slots_kernel`（Triton kernel，grid = `(num_groups, num_reqs)`）
- **位置**：`vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py`
- **分类**：非计算散写/填充（对每组/每请求的 `[last_token_index+1, query_end)` 写入 `PAD_SLOT_ID=-1`）
- **主输入**：
  - `slot_mappings[num_groups, num_tokens]` int32（输入输出）
  - `query_start_loc[num_reqs+1]` int32
  - `last_token_indices[num_reqs]` int32
  - `pad_id`（标量，PAD_SLOT_ID = -1）
- **输出**（判据：bitwise 一致）：`slot_mappings[num_groups, num_tokens]` int32

| 测试函数 | 参数化 shape（id） | 说明 |
| --- | --- | --- |
| test_pad_trailing_draft_slots | num_groups, num_reqs, num_tokens, scenario, block_size | 见下表 |

| case | num_groups | num_reqs | num_tokens | scenario | block_size |
| --- | --- | --- | --- | --- | --- |
| 无填充 | 1 | 1 | 32 | no_pad | 8 |
| 无填充 | 2 | 4 | 64 | no_pad | 8 |
| 全填充 | 1 | 1 | 32 | full_pad | 8 |
| 全填充 | 2 | 4 | 64 | full_pad | 8 |
| 部分填充 | 1 | 1 | 32 | partial_pad | 8 |
| 部分填充 | 1 | 4 | 64 | partial_pad | 8 |
| 部分填充 | 3 | 4 | 128 | partial_pad | 8 |
| tile 边界 | 1 | 1 | 32 | full_pad | 8 |
| 更大 batch | 4 | 8 | 256 | partial_pad | 8 |
| 较小 block | 2 | 8 | 128 | partial_pad | 4 |
| 真实 block_size | 2 | 4 | 1024 | partial_pad | 256 |

- **固定 shape**：`PAD_SLOT_ID=-1`；`query_lens` 依据场景取 `[3,10)` 或 `[5,12)` 随机。

---

## 3. _shift_input_ids_kernel

- **算子**：`_shift_input_ids_kernel`（Triton kernel，grid = `(num_reqs,)`）
- **位置**：`vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py`
- **分类**：非计算移位（每个请求窗口内 `input_ids` 左移 1 位，末尾插入 draft token）
- **主输入**：
  - `input_ids[num_tokens]` int32（输入输出，就地）
  - `idx_mapping[num_reqs]` int64
  - `query_start_loc[max_num_reqs+1]` int32
  - `last_token_indices[num_reqs]` int64
  - `draft_tokens[num_reqs]` int64
- **输出**（判据：bitwise 一致）：`input_ids[num_tokens]` int32

| 测试函数 | 参数化 shape（id） | 说明 |
| --- | --- | --- |
| test_shift_input_ids | num_reqs, scenario, block_size | 见下表 |

| case | num_reqs | scenario | block_size |
| --- | --- | --- | --- |
| 单 token（仅插入） | 1 | single_token | 8 |
| 单 token | 2 | single_token | 8 |
| 双 token（移位 1） | 1 | two_tokens | 8 |
| 双 token | 2 | two_tokens | 8 |
| query_len==block | 1 | tile_boundary | 8 |
| query_len==block | 2 | tile_boundary | 8 |
| query_len==block | 1 | tile_boundary | 4 |
| query_len==block+1（尾块） | 1 | tile_boundary_plus_one | 8 |
| query_len==block+1 | 2 | tile_boundary_plus_one | 8 |
| 随机长度 | 4 | random | 8 |
| 随机长度 | 8 | random | 8 |
| 跳过 cudagraph 填充 | 4 | skip_padded | 8 |
| 更大 batch | 16 | random | 8 |
| 不同 block | 4 | random | 4 |
| 不同 block | 4 | random | 16 |
| 真实 block_size | 4 | random | 1024 |

- **固定 shape**：`max_num_reqs = max(8, num_reqs*2)`；`num_tokens = 256`。

---

## 4. _shift_input_embeds_kernel

- **算子**：`_shift_input_embeds_kernel`（Triton kernel，grid = `(num_reqs, cdiv(hidden_size, BLOCK_SIZE_H))`）
- **位置**：`vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py`
- **分类**：非计算类 + 浮点拷贝/移位融合（每个请求窗口内嵌入左移 1 位，末尾插入 draft embed）
- **主输入**：
  - `input_embeds[num_tokens, hidden_size]` fp32（输入输出，就地）
  - `draft_embeds[num_reqs, hidden_size]` fp32
  - `idx_mapping[num_reqs]` int64
  - `query_start_loc[max_num_reqs+1]` int32
  - `last_token_indices[num_reqs]` int64
- **输出**（判据：bitwise 一致）：`input_embeds[num_tokens, hidden_size]` fp32

| 测试函数 | 参数化 shape（id） | 说明 |
| --- | --- | --- |
| test_shift_input_embeds | num_reqs, hidden_size, scenario, bq(BLOCK_SIZE_Q), bh(BLOCK_SIZE_H) | 见下表 |

| case | num_reqs | hidden_size | scenario | bq | bh |
| --- | --- | --- | --- | --- | --- |
| 单 token | 1 | 16 | single_token | 4 | 8 |
| 单 token | 2 | 32 | single_token | 4 | 8 |
| 双 token | 1 | 16 | two_tokens | 4 | 8 |
| 双 token | 2 | 32 | two_tokens | 4 | 8 |
| query_len==BQ | 1 | 16 | tile_boundary_q | 4 | 8 |
| query_len==BQ | 2 | 32 | tile_boundary_q | 4 | 8 |
| query_len==BQ+1 | 1 | 16 | tile_boundary_q_plus_one | 4 | 8 |
| query_len==BQ+1 | 2 | 32 | tile_boundary_q_plus_one | 4 | 8 |
| H==BH | 1 | 8 | random | 4 | 8 |
| H==BH+1 | 1 | 9 | random | 4 | 8 |
| H==BH-1 | 1 | 7 | random | 4 | 8 |
| 随机长度 | 4 | 64 | random | 4 | 8 |
| 随机长度 | 8 | 64 | random | 4 | 8 |
| 跳过 cudagraph 填充 | 4 | 32 | skip_padded | 4 | 8 |
| 更大 batch | 16 | 64 | random | 4 | 8 |
| 不同 block | 4 | 64 | random | 8 | 16 |
| 不同 block | 4 | 64 | random | 2 | 4 |
| 真实 block | 4 | 256 | random | 16 | 256 |

- **固定 shape**：`max_num_reqs = max(8, num_reqs*2)`；`num_tokens = 256`。

---

## 5. _prepare_input_buffers_kernel

- **算子**：`_prepare_input_buffers_kernel`（Triton kernel，grid = `(num_reqs,)`）
- **位置**：`vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py`
- **分类**：非计算 + 整数索引操作（input-id/position 移位、缓存重填、填充）
- **主输入**：
  - `idx_mapping[num_reqs]` int64
  - `target_input_ids[max_num_tokens]` int32
  - `target_positions[max_num_tokens]` int64
  - `target_seq_lens[num_reqs]` int32
  - `cached_draft_input_ids[max_num_reqs, num_spec_steps-1]` int64
  - `last_sampled[max_num_reqs]` int64
  - `next_prefill_tokens[num_spec_steps, max_num_reqs]` int32
  - `num_sampled[num_reqs]` int32
  - `num_rejected[num_reqs]` int32
  - `query_start_loc[max_num_reqs+1]` int32
- **输出**（判据：bitwise 一致）：
  - `draft_input_ids[max_num_tokens]` int32
  - `draft_positions[max_num_tokens]` int64
  - `draft_seq_lens[max_num_reqs]` int32
  - `last_token_indices[max_num_reqs]` int64
  - `draft_input_id_overrides[max_num_reqs, num_spec_steps-1]` int64
  - `query_start_loc[max_num_reqs+1]` int32

| 测试函数 | 参数化 shape（id） | 说明 |
| --- | --- | --- |
| test_prepare_input_buffers | num_reqs, max_num_reqs, max_num_tokens, num_speculative_steps, scenario, block_size | 见下表 |

| case | num_reqs | max_num_reqs | max_num_tokens | num_spec_steps | scenario | block_size |
| --- | --- | --- | --- | --- | --- | --- |
| 基础 decode 无拒绝 | 1 | 4 | 64 | 2 | decode_no_reject | 8 |
| 基础 decode 无拒绝 | 1 | 4 | 64 | 3 | decode_no_reject | 8 |
| decode 含拒绝 | 1 | 4 | 64 | 2 | decode_with_reject | 8 |
| decode 含拒绝 | 2 | 8 | 128 | 3 | decode_with_reject | 8 |
| decode 含拒绝 | 4 | 8 | 256 | 4 | decode_with_reject | 8 |
| chunked prefill | 1 | 4 | 64 | 2 | chunked_prefill | 8 |
| chunked prefill | 3 | 8 | 128 | 4 | chunked_prefill | 8 |
| 混合 | 4 | 8 | 256 | 3 | mixed | 8 |
| 混合 | 2 | 8 | 128 | 2 | mixed | 4 |
| tile 边界 | 1 | 4 | 64 | 2 | tile_boundary | 8 |
| tile 边界 | 2 | 8 | 128 | 3 | tile_boundary | 8 |
| 最小输入 | 1 | 4 | 16 | 2 | minimal_input | 8 |
| 最小输入 | 3 | 8 | 64 | 3 | minimal_input | 8 |
| 拒绝 1 | 1 | 4 | 32 | 2 | reject_one | 8 |
| 拒绝 1 | 2 | 8 | 64 | 3 | reject_one | 4 |
| 大 batch 填充校验 | 1 | 16 | 64 | 2 | decode_no_reject | 8 |
| 大 batch 填充校验 | 2 | 16 | 128 | 3 | decode_with_reject | 8 |
| 更大 num_spec_steps | 2 | 8 | 128 | 5 | decode_with_reject | 8 |
| 真实 block_size | 2 | 8 | 4096 | 3 | decode_no_reject | 1024 |

- **固定 shape**：`next_prefill_tokens[num_spec_steps, max_num_reqs]` int32（`num_prefill_lookahead = num_spec_steps`）。

---

## 6. _prepare_input_hidden_states_and_embeddings_kernel

- **算子**：`_prepare_input_hidden_states_and_embeddings_kernel`（Triton kernel，grid = `(num_reqs, cdiv(max_query_len, BLOCK_SIZE_Q), cdiv(hidden_size, BLOCK_SIZE_H))`）
- **位置**：`vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py`
- **分类**：非计算类 + 浮点拷贝/收集融合（target hidden 右移 re-prefill 间隙；缓存 hidden/embeddings 填充间隙）
- **主输入**：
  - `target_hidden_states[num_tokens, hidden_size]` fp32
  - `cached_target_hidden_states[max_num_reqs, num_spec_steps-1, hidden_size]` fp32
  - `input_embeds[num_tokens, hidden_size]` fp32（可选）
  - `cached_draft_input_embeds[max_num_reqs, num_spec_steps-1, hidden_size]` fp32（可选）
  - `idx_mapping[num_reqs]` int64
  - `num_rejected[num_reqs]` int32
  - `query_start_loc[max_num_reqs+1]` int32
- **输出**（判据：bitwise 一致）：
  - `draft_input_hidden_states[num_tokens, hidden_size]` fp32
  - `input_embeds[num_tokens, hidden_size]` fp32（可选）

| 测试函数 | 参数化 shape（id） | 说明 |
| --- | --- | --- |
| test_prepare_input_hidden_states_and_embeddings | num_reqs, hidden_size, num_speculative_steps, scenario, use_embeds, bq, bh | 见下表 |

| case | num_reqs | hidden_size | num_spec_steps | scenario | use_embeds | bq | bh |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 基础无拒绝 | 1 | 16 | 2 | no_reject | False | 4 | 8 |
| 基础无拒绝 | 2 | 32 | 3 | no_reject | False | 4 | 8 |
| 含拒绝 | 1 | 16 | 2 | with_reject | False | 4 | 8 |
| 含拒绝 | 2 | 32 | 3 | with_reject | False | 4 | 8 |
| 含 embeds | 1 | 16 | 2 | no_reject | True | 4 | 8 |
| 含 embeds | 2 | 32 | 3 | with_reject | True | 4 | 8 |
| tile 边界 | 1 | 16 | 2 | tile_boundary | False | 4 | 8 |
| tile 边界 | 2 | 32 | 3 | tile_boundary | True | 4 | 8 |
| 最大 re-prefill | 1 | 16 | 3 | max_reprefill | False | 4 | 8 |
| 最大 re-prefill | 2 | 32 | 4 | max_reprefill | True | 4 | 8 |
| H==BH | 1 | 8 | 2 | with_reject | False | 4 | 8 |
| H==BH+1 | 1 | 9 | 2 | with_reject | False | 4 | 8 |
| H==BH-1 | 1 | 7 | 2 | with_reject | False | 4 | 8 |
| 混合 | 4 | 64 | 3 | mixed | False | 4 | 16 |
| 混合 | 4 | 64 | 3 | mixed | True | 4 | 16 |
| 更大 num_spec_steps | 2 | 32 | 5 | with_reject | True | 4 | 8 |
| 真实 block | 2 | 256 | 3 | with_reject | True | 16 | 256 |

- **固定 shape**：`max_num_reqs = max(8, num_reqs*2)`；`num_tokens = 256`。

---

## 7. preprocess_mamba_align_fused_kernel

- **算子**：`preprocess_mamba_align_fused_kernel`（Triton kernel，grid = `(cdiv(num_reqs, BLOCK_SIZE),)`）
- **位置**：`vllm/v1/worker/mamba_utils.py`
- **分类**：整数/索引计算（每请求产出 src_col/src_off，推进 state_idx，条件重置 num_accepted）
- **主输入**：
  - `idx_mapping[num_reqs]` int32
  - `state_idx[max_state_slots]` int32（输入输出）
  - `num_computed_tokens[max_state_slots]` int32
  - `query_start_loc[num_reqs+1]` int32
  - `num_accepted_tokens[max_state_slots]` int32（输入输出）
  - `src_col[max_state_slots]` int32（输出）
  - `src_off[max_state_slots]` int32（输出）
- **输出**（判据：bitwise 一致）：`src_col`、`src_off`、`state_idx`、`num_accepted_tokens`（全 int32）

| 测试函数 | 参数化 shape（id） | 说明 |
| --- | --- | --- |
| test_preprocess_mamba_align_fused | (num_reqs, block_size, mamba_block_size) × (fresh_state, reset_path, zero_accepted) | SHAPE_PARAMS × BRANCH_PARAMS |

**SHAPE_PARAMS（num_reqs, block_size, mamba_block_size）**：

| case | num_reqs | block_size | mamba_block_size |
| --- | --- | --- | --- |
| 退化 | 1 | 1 | 16 |
| — | 1 | 4 | 32 |
| 非 2 幂 | 3 | 1 | 64 |
| — | 3 | 4 | 128 |
| 非 2 幂 | 17 | 1 | 16 |
| — | 17 | 16 | 32 |
| — | 64 | 1 | 64 |
| — | 64 | 16 | 128 |
| — | 128 | 1 | 16 |
| — | 128 | 16 | 64 |

**BRANCH_PARAMS（fresh_state, reset_path, zero_accepted）**：

| case | fresh_state | reset_path | zero_accepted |
| --- | --- | --- | --- |
| nominal | False | False | False |
| 强制跨块重置 | False | True | False |
| 全 fresh（state_idx=-1） | True | False | False |
| num_accepted=0 边界 | False | False | True |

- **固定 shape**：`max_state_slots = num_reqs`（基础 case 无填充）。

---

## 8. _thinking_budget_kernel

- **算子**：`_thinking_budget_kernel`（Triton kernel，grid = `(num_tokens,)`）
- **位置**：`vllm/v1/worker/gpu/sample/thinking_budget.py`
- **分类**：浮点 + 离散融合（离散输出 force_token_id 需精确；浮点输出为字面量 `1e9` 写入，bitwise 一致；其余 logits 保持 NaN 哨兵）
- **主输入**：
  - `logits[num_tokens, vocab_size]` fp32（输入输出）
  - `expanded_idx_mapping[num_tokens]` int32
  - `thinking_token_budget[max_state_slots]` int32
  - `all_token_ids[max_state_slots, stride]` int32（stride = max_model_len）
  - `total_len[max_state_slots]` int32
  - `input_ids[num_tokens]` int32
  - `expanded_local_pos[num_tokens]` int32
  - `cached_last_start[max_state_slots]` int32
  - `cached_last_end[max_state_slots]` int32
  - `reasoning_start_token_ids[start_len]` int32
  - `natural_reasoning_end_token_ids[natural_end_len]` int32
  - `reasoning_end_token_ids[end_len]` int32
- **输出**（判据：bitwise，字面量 1e9 / NaN 哨兵）：`logits`（超预算时仅覆写某一列）

| 测试函数 | 参数化 shape（id） | 说明 |
| --- | --- | --- |
| test_thinking_budget | (start_len, natural_end_len, end_len) × scenario | SHAPE_PARAMS × SCENARIOS |

**SHAPE_PARAMS（start_len, natural_end_len, end_len）**：

| case | start_len | natural_end_len | end_len |
| --- | --- | --- | --- |
| — | 1 | 1 | 1 |
| — | 2 | 2 | 2 |
| 使能 write_prefix_len_2 | 1 | 2 | 3 |
| — | 3 | 2 | 4 |

**SCENARIOS**：

| id | 说明 |
| --- | --- |
| skip_budget | budget=-1，不写 |
| skip_no_start | 扫描范围内无 start 标记，last_start=-1 |
| skip_last_start_le | last_start<=last_end |
| skip_under_budget | num_reasoning_tokens < budget |
| write_no_prefix | 超预算、无 end-prefix，写 end_marker[0] |
| write_prefix_len_1 | 尾部匹配 end_marker[0]，写 end_marker[1] |
| write_prefix_len_2 | 尾部匹配 end_marker[0:2]，写 end_marker[2]（需 end_len>=3） |

- **固定 shape**：`num_tokens=1`；`vocab_size=256`；`max_state_slots=1`；`all_token_ids` 第 2 维 stride = `max_model_len`（内部固定为 32）。

---

## 9. _update_committed_marker_cache_kernel

- **算子**：`_update_committed_marker_cache_kernel`（Triton kernel，grid = `(num_reqs,)`）
- **位置**：`vllm/v1/worker/gpu/sample/thinking_budget.py`
- **分类**：整数计算（标记扫描，含冷启动/增量路径）。budget<0 路径无写入，哨兵预填捕获越界写
- **主输入**：
  - `req_ids[num_reqs]` int32
  - `thinking_token_budget[max_state_slots]` int32
  - `all_token_ids[max_state_slots, stride]` int32（stride = max_model_len）
  - `total_len[max_state_slots]` int32
  - `cached_last_start[max_state_slots]` int32（输入输出）
  - `cached_last_end[max_state_slots]` int32（输入输出）
  - `cached_scan_pos[max_state_slots]` int32（输入输出）
  - `reasoning_start_token_ids[start_len]` int32
  - `natural_reasoning_end_token_ids[natural_end_len]` int32
- **输出**（判据：bitwise 一致）：`cached_last_start`、`cached_last_end`、`cached_scan_pos`（全 int32）

| 测试函数 | 参数化 shape（id） | 说明 |
| --- | --- | --- |
| test_update_committed_marker_cache | (start_len, natural_end_len, max_len, block) × scenario | SHAPE_PARAMS × SCENARIOS |

**SHAPE_PARAMS（start_len, natural_end_len, max_len, block）**：

| case | start_len | natural_end_len | max_len | block |
| --- | --- | --- | --- | --- |
| — | 1 | 1 | 1 | 4 |
| — | 2 | 2 | 2 | 8 |
| — | 1 | 2 | 2 | 4 |
| 单 chunk 路径 | 3 | 2 | 3 | 1024 |
| — | 2 | 3 | 3 | 8 |

**SCENARIOS**：

| id | 说明 |
| --- | --- |
| cold_none | 冷启动，无任何标记 |
| cold_both_last_chunk | 冷启动，两标记均在最后 BLOCK chunk |
| cold_both_diff_chunks | 冷启动，start 在前 chunk、end 在末 chunk |
| cold_start_only | 冷启动，仅 start 标记在末 chunk |
| cold_end_only | 冷启动，仅 end 标记在末 chunk |
| cold_boundary | start 标记 offs+START_LEN==total_len 边界 |
| incr_none | 增量，scan_pos 后无标记 |
| incr_new_start | 增量，scan_pos 后新 start 标记 |
| incr_new_end | 增量，scan_pos 后新 end 标记 |
| incr_both | 增量，scan_pos 后两者皆有 |

- **固定 shape**：`vocab_size=256`；`max_state_slots=1`；`all_token_ids` 第 2 维 stride = `max(total_len, 32)`。

---

## 附：算子位置速查

| 算子 | 位置 |
| --- | --- |
| _cache_inputs_kernel | vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py |
| _pad_trailing_draft_slots_kernel | vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py |
| _shift_input_ids_kernel | vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py |
| _shift_input_embeds_kernel | vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py |
| _prepare_input_buffers_kernel | vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py |
| _prepare_input_hidden_states_and_embeddings_kernel | vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py |
| preprocess_mamba_align_fused_kernel | vllm/v1/worker/mamba_utils.py |
| _thinking_budget_kernel | vllm/v1/worker/gpu/sample/thinking_budget.py |
| _update_committed_marker_cache_kernel | vllm/v1/worker/gpu/sample/thinking_budget.py |