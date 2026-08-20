# easy_ut_026 — 算子精度 UT（批次 2026-08-18）

本批次针对 9 个 vLLM Triton 内核编写昇腾 NPU 精度 UT，遵循
`ASCEND_OPERATOR_ACCURACY_2_1_TEST_PLAN.md` 标准。所有用例复用 `strict_ut` 的
`runtime_npu.py` 辅助（设备初始化、同步、`STRICT_DEVICE`），不依赖任何
vllm-ascend 补丁。

## 1. 算子分析

| 算子 | 上游 vLLM 路径 | 下游 vllm-ascend | 复用/覆写 | 算子类别 | 精度判据 |
|---|---|---|---|---|---|
| `_update_committed_marker_cache_kernel` | `vllm/v1/worker/gpu/sample/thinking_budget.py` | 无 | 复用上游 | 整数计算（marker 扫描） | bitwise |
| `_thinking_budget_kernel` | `vllm/v1/worker/gpu/sample/thinking_budget.py` | 无 | 复用上游 | 浮点 + 离散融合 | force_token_id bitwise；logits 1e9 字面量 bitwise；未写入槽位 NaN 哨兵 |
| `_prepare_input_buffers_kernel` | `vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py` | 无 | 复用上游 | 非计算 + 整数（index/shift/cache re-prefill/pad） | bitwise |
| `_prepare_input_hidden_states_and_embeddings_kernel` | `vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py` | 无 | 复用上游 | 非计算 + 浮点拷贝融合 | bitwise（纯 copy，无算术） |
| `_pad_trailing_draft_slots_kernel` | `vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py` | 无 | 复用上游 | 非计算 scatter/pad | bitwise |
| `_cache_inputs_kernel` | `vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py` | 无 | 复用上游 | 非计算 + 浮点拷贝融合 | bitwise（纯 snapshot） |
| `_shift_input_ids_kernel` | `vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py` | 无 | 复用上游 | 非计算 + 整数 shift | bitwise |
| `_shift_input_embeds_kernel` | `vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py` | 无 | 复用上游 | 非计算 + 浮点 shift 融合 | bitwise（纯 copy） |
| `preprocess_mamba_align_fused_kernel` | `vllm/v1/worker/mamba_utils.py` | 无 | 复用上游 | 整数/索引 | bitwise |

**结论**：

- 上游 vLLM 主分支中 9 个内核均已实现且为唯一权威实现。
- 下游 vllm-ascend 仓库无任何同名列覆写；测试均直接
  `from vllm.* import ...`。
- 所有算子无需 patch，全部复用上游内核；测试用例独立提供 CPU 参考实现。

## 2. 调用方式

### 2.1 通用约定

- 设备：`runtime_npu.STRICT_DEVICE`（昇腾 NPU）。
- 同步：`runtime_npu.synchronize()` 在内核 launch 后调用。
- Triton 上下文初始化：`runtime_npu.init_device_properties_triton()`。
- 重复确定性：`conftest.py` 注册 `deterministic` marker；用例失败即失败，
  不使用 `xfail`。

### 2.2 内核 launch 调用

| 测试文件 | 内核 grid | BLOCK_SIZE | 主要参数 |
|---|---|---|---|
| `test_update_committed_marker_cache_kernel.py` | `(num_reqs,)` | — | `req_ids`, `thinking_token_budget`, `all_token_ids`, `total_len`, `cached_last_start/end/scan_pos`, `reasoning_start_token_ids`, `natural_reasoning_end_token_ids`, `start_len`, `natural_end_len`, `max_len` |
| `test_thinking_budget_kernel.py` | `(num_tokens,)` | — | `logits`, `expanded_idx_mapping`, `thinking_token_budget`, `all_token_ids`, `total_len`, `input_ids`, `expanded_local_pos`, `cached_last_start/end`, `reasoning_*_token_ids`, `start_len`, `natural_end_len`, `end_len` |
| `test_prepare_input_buffers_kernel.py` | `(num_reqs,)` | `BLOCK_SIZE=8/1024` | `last_token_indices`, `draft_input_ids/positions/seq_lens`, `target_input_ids/positions`, `cached_draft_input_ids`, `draft_input_id_overrides`, `idx_mapping`, `last_sampled`, `next_prefill_tokens`, `num_sampled`, `num_rejected`, `target_seq_lens`, `query_start_loc`, `max_num_reqs`, `num_speculative_steps` |
| `test_prepare_input_hidden_states_and_embeddings_kernel.py` | `(num_reqs, cdiv(max_q_len, BQ), cdiv(H, BH))` | `BQ=4/16`, `BH=8/256` | `draft_input_hidden_states`, `target_hidden_states`, `cached_target_hidden_states`, `input_embeds`, `cached_draft_input_embeds`, `idx_mapping`, `num_rejected`, `query_start_loc`, `num_speculative_steps`, `hidden_size`, `USE_INPUT_EMBEDS` |
| `test_pad_trailing_draft_slots_kernel.py` | `(num_groups, num_reqs)` | `BLOCK_SIZE=8/256` | `slot_mappings`, `query_start_loc`, `last_token_indices`, `PAD_ID` |
| `test_cache_inputs_kernel.py` | `(num_reqs, cdiv(H, BLOCK_SIZE))` | `BLOCK_SIZE=8/1024` | `draft_input_ids`, `draft_input_embeds`, `draft_input_hidden_states`, `cached_draft_input_ids/embeds/target_hidden_states`, `idx_mapping`, `last_token_indices`, `query_start_loc`, `num_speculative_steps`, `hidden_size`, `USE_INPUT_EMBEDS` |
| `test_shift_input_ids_kernel.py` | `(num_reqs,)` | `BLOCK_SIZE=8/1024` | `input_ids`, `idx_mapping`, `query_start_loc`, `last_token_indices`, `draft_tokens` |
| `test_shift_input_embeds_kernel.py` | `(num_reqs, cdiv(H, BH))` | `BQ=4/16`, `BH=8/256` | `input_embeds`, `draft_embeds`, `idx_mapping`, `query_start_loc`, `last_token_indices`, `hidden_size` |
| `test_preprocess_mamba_align_fused_kernel.py` | 见文件 | 见文件 | Mamba 对齐 fused 内核的索引参数 |

### 2.3 测试结构（统一）

1. **导入 kernel**：`try: from vllm.* import _kernel as kernel`，失败时保存
   traceback，测试统一通过 `test_import_error` 上报。
2. **CPU 参考实现** `_ref(...)`：独立于被测内核，纯 Python/PyTorch CPU 实现，
   逐元素模拟内核语义，包括 dtype 截断（如 int64→int32 的位级匹配）。
3. **输入生成** `_gen_inputs(...)`：参数化场景（`decode_no_reject`、
   `with_reject`、`chunked_prefill`、`tile_boundary`、`minimal_input`、
   `skip_padded`、`max_reprefill` 等），预填哨兵值以检测越界写入。
4. **内核执行** `_run_kernel(...)`：在 `STRICT_DEVICE` 上克隆输入缓冲区，按
   上游 `prepare_*`/`update_draft_inputs`/`cache_inputs` 函数同样的参数顺序
   调用内核，随后 `synchronize()`。
5. **断言** `_assert_bitwise(name, expected, actual)`：bitwise 比较；
   失败时打印前 10 个差异位置及值，便于定位。

## 3. 精度判据

依据 `ASCEND_OPERATOR_ACCURACY_2_1_TEST_PLAN.md`：

| 类别 | 判据 | 本批次应用 |
|---|---|---|
| 整数计算 / 索引 / 非计算（整数） | bitwise 一致 | `_update_committed_marker_cache_kernel`、`_prepare_input_buffers_kernel`、`_pad_trailing_draft_slots_kernel`、`_shift_input_ids_kernel`、`preprocess_mamba_align_fused_kernel` |
| 浮点纯拷贝 / gather / scatter | bitwise 一致（无算术） | `_prepare_input_hidden_states_and_embeddings_kernel`、`_cache_inputs_kernel`、`_shift_input_embeds_kernel` |
| 浮点 + 离散融合 | 离散输出 bitwise；浮点字面量 bitwise；未写入槽位 NaN 哨兵保留 | `_thinking_budget_kernel`（`force_token_id` 精确；`1e9` 字面量 bitwise；NaN 哨兵） |

**为何对浮点拷贝类使用 bitwise**：本批次涉及的浮点类内核（hidden states、
embeds、logits 1e9）均为字面量 `tl.load`→`tl.store`，无加法/乘法/归约；
fp32 位级在 NPU/CPU 之间应保持一致，故采用更严格的 bitwise 判据。

## 4. 测试用例覆盖维度

每文件至少覆盖以下维度（满足 2.1 标准的最低用例数要求）：

- **基础场景**：单 req、单 token、单 step（`single_token`/`minimal_input`）。
- **典型场景**：多 req、多 step、有/无 rejection（`decode_no_reject`/
  `with_reject`/`full_window`）。
- **边界场景**：query_len/hidden_size 恰好为 BLOCK_SIZE 的倍数（
  `tile_boundary`/`tile_boundary_q`），或 +1/-1（`*_plus_one`/`BH-1`）。
- **特殊分支**：`chunked_prefill`（`num_sampled=0`）、`max_reprefill`
  （全 gap）、`reject_one`（`num_reprefill=0`）、`skip_padded`
  （`idx_mapping<0` 早返回）。
- **混合场景**：`mixed`（同一 batch 内 req 类型异质）。
- **真实规模**：`BLOCK_SIZE=1024`、`num_reqs=16`，贴近生产配置。
- **哨兵/守卫**：输出缓冲区预填 `_SENTINEL`（int）/`NaN`（float），任何越界
  写入都会触发断言失败。

## 5. 运行方式

```bash
# 单文件
pytest accuracy_test/easy_ut_026/test_prepare_input_buffers_kernel.py -v

# 全部 easy_ut_026
pytest accuracy_test/easy_ut_026/ -v

# 单参数化用例
pytest accuracy_test/easy_ut_026/test_shift_input_ids_kernel.py \
  -v -k "test_shift_input_ids[1-single_token-8]"
```

## 6. 文件清单

```
easy_ut_026/
├── __init__.py
├── conftest.py                                    # deterministic marker
├── runtime_npu.py                                 # 复用 strict_ut 运行时
├── test_preprocess_mamba_align_fused_kernel.py
├── test_update_committed_marker_cache_kernel.py
├── test_thinking_budget_kernel.py
├── test_prepare_input_buffers_kernel.py
├── test_prepare_input_hidden_states_and_embeddings_kernel.py
├── test_pad_trailing_draft_slots_kernel.py
├── test_cache_inputs_kernel.py
├── test_shift_input_ids_kernel.py
├── test_shift_input_embeds_kernel.py
└── README.md
```
