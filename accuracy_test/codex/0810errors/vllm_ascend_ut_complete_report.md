# vLLM / vLLM-Ascend 已有精度 UT 运行结果完整报告（0810）

> **报告性质**：综合完整报告。数据来源为 `0810errors` 同目录下的两份运行日志：
> - `ut_in_vllm.txt`：`run_from_vllm.sh` 运行 `existing_accuracy_tests/from_vllm/` 的 30 个测试文件（7719 行，226 用例），日志时间 08-10 17:16 起。
> - `ut_in_vllm_ascend.txt`：`run_from_vllm_ascend.sh` 运行 `existing_accuracy_tests/from_vllm_ascend/` 的 10 个正式测试文件（325 行，81 用例），日志时间 08-10 19:16 起。
>
> 本报告综合 `precision_test_analysis_report.md`（单测标准/方式/优缺点）与 `existing_tests_report.md`（静态元数据盘点）撰写，覆盖 **vLLM 与 vLLM-Ascend 范围内已搬运的全部已有精度 UT（共 40 个测试文件：from_vllm 30 个含 *_patch 路径 + from_vllm_ascend 10 个）**，并在原「结果列待填」的总结表格基础上**补齐 0810 实际运行结果**。
>
> `diagnose_bincount_atomic_or.py` 为独立诊断脚本（非正式 pytest 用例，不含在 run_from_vllm_ascend.sh 的正规用例里），本次不计入用例数，单独说明。
>
> **判定三态口径**：`PASS`=数值校验通过；`FAIL`=失败（含真实精度缺陷，以及环境/版本/编译/设备层问题）；`SKIP`=未执行（**SKIP≠通过**，精度「未受验」）。XFAIL（编译/后端不兼容）同样不算通过。

---

## 1. 环境与运行口径（复现前提）

| 项 | 值 | 说明 |
| --- | --- | --- |
| 运行命令 | `bash .../existing_accuracy_tests/run_from_vllm.sh`、`bash .../existing_accuracy_tests/run_from_vllm_ascend.sh` | 每个文件启动独立 pytest session |
| 平台 | Linux，Python 3.11.10，pytest 8.3.2（asyncio 1.3.0、xdist 3.6.1、anyio 4.14.2、mock 3.15.1、cov 7.1.0） | `run_from_vllm_ascend.sh` 输出头部已确认 |
| 硬件 | 昇腾 NPU(A3) | 所有 kernel 直接 `torch.device("npu")` launch + `torch.npu.synchronize()` |
| vLLM 平台插件 | ascend 已激活（`vllm_ascend:register`），`Breakable cudagraph` 被强制关闭 | 日志 `[platform.py:62]` |
| 设备初始化 | 各文件内联 `init_device_properties_triton()` + `npu` | 不依赖外部 fixture |
| 依赖 | 已安装 vLLM、vLLM-Ascend（`vllm_ascend`）、PyTorch NPU、Ascend Triton | 测试 import 生产 kernel 与 wrapper |
| 关键环境事实 | 安装的 vLLM 版本**未提供**多个 block-verification kernel 符号（`_compute_local_logits_stats_kernel`、`_compute_cumulative_log_p_kernel`、`_load_ptr` 等），导致对应 UT 被 SKIP（收集期）或 FAIL（import 期） | 是本次 from_vllm 目录大量未受验的主因 |

> 汇总口径沿用 `existing_tests_report.md`：**结果列三态 PASS / FAIL / SKIP**；XFAIL(编译) 记为「未受验」，不计入通过。

---

## 2. 总体结果（40 文件 / 307 用例）

| 集合 | 文件数 | 用例数 | PASSED | FAILED | SKIPPED | 通过率(占收集) |
| --- | --- | --- | --- | --- | --- | --- |
| from_vllm（含 10 个 *_patch） | 30 | 226 | 154 | 44 | 28 | 68.1% |
| from_vllm_ascend（官方搬运） | 10 | 81 | 81 | 0 | 0 | 100% |
| **合计** | **40** | **307** | **235** | **44** | **28** | **76.5%** |

> **要点**：
> 1. from_vllm 的 **44 个失败** 按根因分四类（第 5 章细析）：**数值不符合 6**、**编译/API 不兼容 19**、**import 符号缺失 18**、**NPU 设备异常 1**；其中**真正的数值精度缺陷候选仅 6 例**（global_lse 全 -inf→nan 2 例 + fill_logprob custom 多行 4 例）。
> 2. from_vllm 的 **28 个 SKIP** 多为「安装版本较旧未含对应 kernel」或「收集期整体跳过」，**SKIP≠通过，精度未受验**。
> 3. **from_vllm_ascend 目录 81/81 全绿**，且均为**数值断言通过**（非 XFAIL 遮掩），即被测的 Ascend 生产采样/logprob/penalty 全链路与参考实现一致。

---

## 3. from_vllm/ 逐文件结果（补齐 existing_tests_report.md 第 1 章结果列）

> 「结果」列三态：`PASS`（通过）、`FAIL(根因代号)`（失败）、`SKIP(原因)`（未运行）。根因代号见第 5 章。
> 用例数 = 该 session 收集 items 数；`0/1 skipped` 表示收集阶段整体跳过。

| 文件 | 被测对象 | 对象类型 | 调用类型 | 参考基准 | 容差判据 | 用例数 | 通过 | 失败 | 跳过 | 结果(逐类) | 根因 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test_apply_write_kernel | `_apply_write_kernel` | kernel | A | CPU 参考 | 精确(0/0) | 10 | 4 | 0 | 6 | PASS(single) / SKIP(multi) | 旧版无 fused multi-group |
| test_bias_kernel | `_bias_kernel` | kernel | A | CPU 参考 | 1e-5 | 9 | 9 | 0 | 0 | ALL PASS | — |
| test_compute_block_max_and_sumexp | `_compute_local_logits_stats_kernel` | kernel | A | CPU 参考 | 1e-5 | 1 | 0 | 0 | 1 | SKIP(全部) | 符号不存在 |
| test_compute_block_max_and_sumexp_patch | Ascend `_compute_block_stats_kernel` | kernel | A | CPU 参考 | 1e-5 | 3 | 3 | 0 | 0 | ALL PASS | — |
| test_compute_block_stats_kernel | `_compute_cumulative_log_p_kernel` | kernel | E | CPU 全局 LSE | 1e-4 | 1 | 0 | 0 | 1 | SKIP(全部) | 符号不存在 |
| test_compute_block_stats_kernel_patch | Ascend `_compute_block_stats_kernel` | kernel | A | CPU 参考 | max/sumexp 1e-5 | 2 | 2 | 0 | 0 | ALL PASS | — |
| test_compute_global_logsumexp | `_compute_global_lse`(helper) | helper | C | CPU `_global_logsumexp_ref` | 1e-5 | 8 | 7 | 1 | 0 | FAIL: test_all_neg_inf_blocks | **数值**: 全 -inf→nan |
| test_compute_global_logsumexp_patch | `_compute_global_lse`(Ascend 别名) | helper | C | CPU 参考 | 1e-5 | 8 | 7 | 1 | 0 | FAIL: test_all_neg_inf_blocks | **数值**: 全 -inf→nan |
| test_fill_logprob_token_ids_kernel | `_fill_logprob_token_ids_kernel` | kernel | A | CPU 参考 | 精确+bool | 10 | 6 | 4 | 0 | FAIL: custom_token_ids 3/5×batch4/8 | **数值**: 张量不等 |
| test_fill_logprob_token_ids_kernel_patch | `compute_topk_logprobs`(生产wrapper) | 生产wrapper | D | PyTorch topk/softmax | token 精确; logp 1e-4 | 4 | 4 | 0 | 0 | ALL PASS | — |
| test_flatten_sampled_kernel | `_flatten_sampled_kernel` | kernel | A | CPU 参考 | 精确+行为 | 14 | 14 | 0 | 0 | ALL PASS | — |
| test_gather_block_tables_kernel | `_gather_block_tables_kernel` | kernel | A | CPU 参考 | 精确+padding | 14 | 14 | 0 | 0 | ALL PASS | — |
| test_gumbel_block_argmax | `gumbel_block_argmax`(helper) | helper | C | 行为 | idx 精确 | 3 | 0 | 3 | 0 | ALL FAIL | **编译**: `PER_TOKEN_COL` kwarg 不匹配 |
| test_gumbel_block_argmax_patch | `_npu_gumbel_block_argmax`(Ascend) | helper | C | 行为+精确 | idx 精确; processed 1e-5 | 4 | 4 | 0 | 0 | ALL PASS | — |
| test_insert_resampled_kernel | `_insert_resampled_kernel` | kernel | A | CPU 参考 | 精确+行为 | 8 | 8 | 0 | 0 | ALL PASS | — |
| test_insert_resampled_kernel_patch | Ascend `_insert_resampled_kernel` | kernel | A | CPU 参考 | 精确 | 8 | 8 | 0 | 0 | ALL PASS | — |
| test_load_ptr | `_load_ptr`(helper) | helper | C | 精确值 | int32/fp32 | 1 | 0 | 0 | 1 | SKIP(全部) | **符号不存在** |
| test_prepare_dflash_inputs_kernel | `_prepare_dflash_inputs_kernel`(vanilla) | kernel | A | 手工字段断言 | 精确 | 1 | 0 | 0 | 1 | SKIP(全部) | 无 dflash 模块 |
| test_prepare_dflash_inputs_kernel_patch | Ascend `_prepare_dflash_inputs_kernel_ascend` | kernel | A | CPU 参考 | 精确(10输出) | 1 | 0 | 0 | 1 | SKIP(全部) | 无 vllm_ascend dflash |
| test_prompt_logprobs_token_ids_kernel | `_prompt_logprobs_token_ids_kernel` | kernel | A | CPU ref | 精确 | 10 | 10 | 0 | 0 | ALL PASS | — |
| test_rejection_kernel | `_rejection_kernel`(vanilla) | kernel | E | 行为 | greedy 精确 | 1 | 0 | 0 | 1 | SKIP(全部) | **符号不存在** |
| test_rejection_kernel_patch | Ascend `_probabilistic_rejection_kernel` | kernel | E | CPU 参考 | greedy 精确 | 18 | 0 | 18 | 0 | ALL FAIL | **import**: `_compute_local_logits_stats_kernel` 不存在 |
| test_resample_kernel | `_resample_kernel`(vanilla) | kernel | A | CPU 参考 | argmax 精确; max 1e-5 | 27 | 15 | 0 | 12 | PASS(15) / SKIP(12: block-ver 分支) | 旧版无 block-verification 签名 |
| test_resample_kernel_patch | Ascend `_resample_kernel` | kernel | B | 行为/范围 | bonus 精确; no-op 精确 | 2 | 2 | 0 | 0 | ALL PASS | — |
| test_scatter_num_accepted_kernel | `_scatter_num_accepted_kernel` | kernel | A | CPU 参考 | 精确 | 10 | 10 | 0 | 0 | ALL PASS | — |
| test_selective_scan_update_kernel | `_selective_scan_update_kernel` | kernel | A | PyTorch CPU 参考 | 1e-4 | 7 | 7 | 0 | 0 | ALL PASS | — |
| test_tl_rand64 | `tl_rand64`+`gumbel_sample` | helper+生产wrapper | C+D | 统计+行为 | 范围/均值/主导 | 5 | 4 | 1 | 0 | FAIL: test_fp32_business_path | **运行时**: AI Core vector exception(507035) |
| test_tl_rand64_patch | `tl_rand64`+Ascend `gumbel_sample` | helper+生产wrapper | C+D | 同 tl_rand64 | 同 tl_rand64 | 5 | 5 | 0 | 0 | ALL PASS | — |
| test_topk_topp_kernel | `_topk_topp_kernel` | kernel | A | CPU `_apply_topk_topp_cpu` | 组合 1e-5 | 20 | 0 | 16 | 4 | FAIL(16) / SKIP(4: False-False noop) | **编译**: legalize 失败; noop 分支 SKIP |
| test_update_min_larger_stats | `_update_min_larger_stats`(helper) | helper | C | CPU 参考 | min 1e-5; cnt 精确 | 11 | 11 | 0 | 0 | ALL PASS | — |

> **合计**：30 文件 / 226 用例 = **154 PASS + 44 FAIL + 28 SKIP**。

### 3.1 结果雷达（被测算子维度，仅标注「真的执行过且通过」的）

| 被测算子 | 文件 | 结果 | 说明 |
| --- | --- | --- | --- |
| `_bias_kernel` | bias | PASS | 9/9 |
| `_compute_global_lse`(helper) | global_logsumexp(+patch) | FAIL | 全 -inf 边界 nan（2 例） |
| `_fill_logprob_token_ids_kernel` | fill_logprob | FAIL | custom 多行映射不一致（4 例） |
| `_flatten_sampled_kernel` | flatten | PASS | 14/14 |
| `_gather_block_tables_kernel` | gather | PASS | 14/14 |
| `gumbel_block_argmax`(helper) | gumbel | FAIL | 编译 API 不匹配（3 例） |
| `_npu_gumbel_block_argmax`(Ascend) | gumbel_patch | PASS | 4/4 |
| `_insert_resampled_kernel` | insert(+patch) | PASS | 16/16 |
| `_prompt_logprobs_token_ids_kernel` | prompt_logprobs | PASS | 10/10 |
| `_resample_kernel` | resample | PASS(部分) | 15 通过, 12 未执行 |
| Ascend `_resample_kernel` | resample_patch | PASS | 2/2 |
| `_scatter_num_accepted_kernel` | scatter | PASS | 10/10 |
| `_selective_scan_update_kernel` | selective_scan | PASS | 7/7 |
| `tl_rand64`/`gumbel_sample` | tl_rand64(+patch) | FAIL(未patch)/PASS(patch) | 未patch 触发设备异常 |
| `_topk_topp_kernel` | topk_topp | FAIL | 编译 legalize 失败（16 例） |
| `_update_min_larger_stats`(helper) | update_min_larger | PASS | 11/11 |

> **关键观察**：凡被测对象是 **vLLM 上游 vanilla**（未 patch）的，本次大量失败/SKIP 属于「版本/编译/设备」层问题；而 **Ascend 自有实现（*_patch）普遍更稳**——block_max/-stats、gumbel、insert、fill、resample、tl_rand64 的 *_patch 版**均全 PASS**。

---

## 4. from_vllm_ascend/ 逐文件结果（补齐 existing_tests_report.md 第 2 章结果列）

> 「结果」列三态：`PASS`（通过）、`SKIP`（未运行）。**本日志无 FAIL**。
> 用例数 = 该 session 收集 items 数。全部 81 例均**数值断言通过**（非 XFAIL 遮掩）。

| 文件 | 被测对象 | 对象类型 | 调用类型 | 参考基准 | 容差判据 | 覆盖场景 | 用例数 | 结果 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test_bad_words | `bad_words.apply_bad_words`(`_bad_words_kernel`) | 生产wrapper | C(生产wrapper包helper) | 行为(前后对比是否被修改) | allclose(相等性) | 3 规格 512/1024/2048; 无 bad; max128; 在/超限 1024/1056 | 6 | **ALL PASS** |
| test_bincount | `_bincount_kernel`(penalties.py) | kernel | A | 自写 `torch_bincount` | torch.equal(int32) | 单 token63, 64 req, 单 block, seed42, BLOCK1024 | 1 | **PASS** |
| test_compute_slot_mapping | `_compute_slot_mappings_kernel`(Ascend vs vllm) | kernel | F(Ascend vs 上游头对头) | 另一实现(上游 vllm kernel) | torch.equal(int64) | 固定:1组KV,1req,5tokens,CP1,320 block | 1 | **PASS** |
| test_compute_topk_logprobs | `compute_topk_logprobs`(`_topk_log_softmax_kernel`+`_ranks_kernel`) | kernel(生产wrapper+级联) | D+E | PyTorch topk+log_softmax+计数rank | ID/rank equal; logprobs 1e-4 | 4 规格 batch 48/96/24/1 × vocab 1024/1519/320 × lp 5/0/1/10(含 lp=0) | 4 | **ALL PASS** |
| test_gumbel_sampling | `apply_temperature`+`gumbel_sample`(`_gumbel_sample_kernel`) | kernel+生产wrapper | D+C | 自写PyTorch+统计/行为/确定性 | temp 1e-4/1e-5; greedy equal; processed 1e-4 | 温度/greedy/确定性/种子/分布/EAGLE 共 30 例 | 30 | **ALL PASS** |
| test_log_softmax | `_topk_log_softmax_kernel`(logprob.py) | kernel | A | PyTorch log_softmax+gather | allclose 1e-3/1e-3 | 3 规格 batch 48/96/24×vocab 102400/151936×lp 50/1/8(含 lp=1) | 3 | **ALL PASS** |
| test_min_p | `min_p.apply_min_p`(`_min_p_kernel`) | kernel | D+E | 自写 `torch_min_p_torch` | inf mask equal; 有效值 1e-4 | 4 规格 req 48/96/24/1×vocab 102400/151936/32000 | 4 | **ALL PASS** |
| test_penality | `penalties.apply_penalties`(`_penalties_kernel`) | kernel+生产wrapper | D | 自写 `pytorch_apply_penalties`(packed mask+累积) | allclose 1e-3; bf16 1e-2 | tokens{1,4}×vocab{1000}×status{1,4}×spec{0,1,3}×dtype{bf16,fp16} | 24 | **ALL PASS** |
| test_post_update | `_post_update_kernel`(Ascend vs vllm) | kernel | F+E +独立CPU oracle | 串行 oracle `post_update_ref` + 上游 | assert_close rtol=0/atol=0(int32) | 3 规格 req 36/48/128×vocab 200/32000×steps 2/5 | 3 | **ALL PASS** |
| test_temperature | `gumbel.apply_temperature`(`_temperature_kernel`) | kernel+生产wrapper | D | 自写 `torch_apply_temperature`(纯Python) | allclose 1e-4/1e-5 | 5 主流 vocab 32000..151936 × 随机 token | 5 | **ALL PASS** |

> **合计**：10 文件 / 81 用例 = **81 PASS + 0 FAIL + 0 SKIP**。

### 4.1 被测算子结果雷达（全部通过）

| 被测算子 | 文件 | 结果 | 覆盖规模 |
| --- | --- | --- | --- |
| `_bad_words_kernel` / `apply_bad_words` | bad_words | PASS | 512/1024/2048 tokens × req 16/32/64；无/最大/超限坏词共 6 例 |
| `_bincount_kernel` | bincount | PASS | 固定单例 1 例 |
| `_compute_slot_mappings_kernel` | compute_slot_mapping | PASS | 固定配置 1 例（头对头） |
| `compute_topk_logprobs`(`_topk_log_softmax`+`_ranks`) | compute_topk_logprobs | PASS | batch 48/96/24/1 × vocab 1024/1519/320 × lp 5/0/1/10 共 4 例 |
| `apply_temperature`+`gumbel_sample`(`_gumbel_sample_kernel`) | gumbel_sampling | PASS | temperature/greedy/确定性/种子/分布/EAGLE 等 30 例 |
| `_topk_log_softmax_kernel` | log_softmax | PASS | batch 48/96/24 × vocab 102400/151936 × lp 50/1/8 共 3 例 |
| `_min_p_kernel` / `apply_min_p` | min_p | PASS | req 48/96/24/1 × vocab 102400/151936/32000 共 4 例 |
| `_penalties_kernel` / `apply_penalties` | penality | PASS | 2 dtype(bf16/fp16)×status{0,1,4}×spec{1,3}×tokens{1,4}×vocab1000 共 24 例 |
| `_post_update_kernel` | post_update | PASS | req 36/48/128 × vocab 200/32000 × steps 2/5 共 3 例 |
| `_temperature_kernel` / `apply_temperature` | temperature | PASS | 5 主流 vocab × 随机 token 共 5 例 |

> 与 from_vllm 形成鲜明对比：**本目录（Ascend 生产路径）全部通过**，而上游 vanilla `from_vllm` 目录有 44 失败 / 28 跳过。说明 vllm_ascend 的采样/logprob/penalty 各生产 kernel 在本次运行中数值校验全部与参考一致，且**不依赖上游缺失的 block-verification 符号**。

---

## 5. from_vllm/ 失败逐项详析（44 例，按根因分 4 类）

### 5.1 数值不符合（6 例）—— 需优先关注

**① `_compute_global_lse` 全 -inf 边界 → nan（2 例）**
- 文件：`test_compute_global_logsumexp.py` 与 `_patch.py` 的 `test_all_neg_inf_blocks`（两个文件各 1 例，共 2 例）。
- 断言：`assert output[0].item() == float("-inf")`；实际输出是 `tensor(nan)`，报错 `AssertionError: Global LSE of all -inf should be -inf`。
- 说明：当所有块 max 均为 -inf（示意「无可选 token」）时，Ascend 端 `_compute_global_lse` 的 `tl.max`/`exp` 归约返回 **nan**，与 CPU 参考约定的 -inf 不符。**属真实边界数值语义差异（精度缺陷候选）**。
- 影响：正常采样全 -inf 罕见；但作为「全不可用」哨兵语义，nan 会在 argmax/log-softmax 下游污染。
- 建议：最小复现并对齐 `_compute_global_lse` 对 -inf 的归约实现（返回 -inf 而非 nan）。

**② `_fill_logprob_token_ids_kernel` 自定义 token 多行映射不一致（4 例）**
- 文件：`test_fill_logprob_token_ids_kernel.py` 的 `test_custom_token_ids[3-4 / 3-8 / 5-4 / 5-8]`。
- 触发条件：`batch∈{4,8}` 且 `topk∈{3,5}`，请求 0 携带 custom token。
- 断言：`assert_close(out_token_ids, expected_ids, rtol=0, atol=0)` 精确；实际 Mismatched elements 9/68(13.2%)~30/136(22.1%)，最大绝对差 852~960。
- 根因线索：`expanded_idx_mapping = arange(batch) % num_reqs` 使多个 batch 行映射到同一请求 0（custom 归属者）；Ascend 端在「同一请求对应多行各自写入 custom token」时的填充/覆盖语义与 CPU 参考不一致。`[1-1]`、`[0-*]`、`[3-1]`、`[5-1]` 全 PASS，独 batch>1 的 4 例失败 → **指向多行共享同一 custom 源时按行偏移/repeat 的索引错位**。**真实精度缺陷候选**。
- 上游已修（commit d7af6b34d8 #41761）；本次运行针对**安装版 vLLM**仍复现，刻意保留 FAILED。

### 5.2 Triton 编译 / API 不兼容（19 例）—— 测试与安装版本脱节

**③ `gumbel_block_argmax` API 参数不匹配（3 例）**
- 文件：`test_gumbel_block_argmax.py` 的 3 个测试全部。
- 错误：`CompilationError: gumbel_block_argmax() got an unexpected keyword argument ''PER_TOKEN_COL''`。
- 说明：测试 wrapper 以 `PER_TOKEN_COL=...` 关键字调用，而**安装的 vLLM 该函数签名不含此参数**——UT 与版本签名脱节，**非精度问题**。
- 佐证：Ascend 版 `_npu_gumbel_block_argmax`（`_patch`）4/4 通过，表明 Ascend 路径签名匹配。

**④ `_topk_topp_kernel` Triton legalize 编译失败（16 例）**
- 文件：`test_topk_topp_kernel.py` 的 16 个失败。
- 错误：`error: failed to legalize unresolved materialization from () to ''tensor<1xf32>'' that remained live after conversion`（`topk_topp_triton.py:153/:646`）。
- 说明：**Ascend Triton 后端无法 legalize 该 kernel 的某条标量→tensor 物化**，编译期直接失败，kernel 无法编译运行，谈不上数值。属**后端编译器兼容缺陷**。
- 另 4 例 SKIP：`test_topk_topp_combined[False-False-*]`（topk 与 topp 都关闭的 noop）被跳过。

### 5.3 import / 环境符号缺失 → 运行期错误（18 例）

**⑤ `_probabilistic_rejection_kernel`（rejection_kernel_patch）全部 18 例失败**
- 文件：`test_rejection_kernel_patch.py` 全部 18 个测试。
- 错误：`ImportError: cannot import name ''_compute_local_logits_stats_kernel'' from ''vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils''`。
- 位置：测试 `_run_kernel`（内部 `from vllm... import _compute_local_logits_stats_kernel`）因懒加载在**每个测试体**报错 → 18 例 FAIL。
- 同一符号在 `test_compute_block_max_and_sumexp`、`test_rejection_kernel`、`test_compute_block_stats_kernel` 因收集期 import 而**整体 SKIP**（3 例）。
- 说明：安装的 vLLM 该文件**不含** `_compute_local_logits_stats_kernel`（或已更名）；**非数值缺陷**，属版本/符号对齐问题。Ascend patch 测试依赖的上游符号缺失导致 rejection 系列精度**完全未验证**。

### 5.4 NPU 设备 / AI Core 运行异常（1 例）

**⑥ `tl_rand64` 的 `gumbel_sample` FP32 业务路径触发设备异常（1 例）**
- 文件：`test_tl_rand64.py` 的 `TestTlRand64ViaGumbelSample::test_fp32_business_path`。
- 错误：`RuntimeError: npuSynchronizeDevice ..., error code is 507035`；`[Error]: The vector core execution is abnormal.`；`EZ9999 ... exception of aivec error ... Illegal instruction, which is usually caused by unaligned UUB addresses`；`retCode=0x31 [vector core exception]`。
- 说明：运行上游 gumbel FP32 业务路径时昇腾 AI Core（vector core）执行异常/非法指令，疑似对齐/指令问题，**非数值不符合**。
- 佐证：同文件 4 个统计/范围测试 PASS；`test_tl_rand64_patch.py`（Ascend gumbel 替换路径）**5/5 全 PASS**——Ascend 自家 gumbel 路径正常，问题集中在**上游 vLLM 的 gumbel_sample(FP32) 在昇腾上触发**。

---

## 6. 汇总与结论

### 6.1 关键结论

1. **40 文件 307 用例中，数值真正「通过且被执行」的为 235（76.5%）**；28(SKIP) + 44(FAIL 中非数值 38) = 抵消后 from_vllm 有约三成被测对象在本日志里**未真正完成精度验证**；而 from_vllm_ascend 的 81/81 **全部完成且通过**。
2. 真正的**数值精度缺陷候选仅 6 例**，全部集中在 from_vllm 目录：`_compute_global_lse` 全 -inf→nan(2)、`_fill_logprob_token_ids_kernel` custom 多行(4)。这些须优先排查。
3. 其余 38 例失败（gumbel 3 + topk_topp 16 + rejection 18 + tl_rand64 1）全部是**环境/版本/编译/设备**层面问题，**不是 kernel 数值算错**，但同样导致「精度未验证」。
4. **Ascend 自有实现（*_patch 与 from_vllm_ascend）普遍更稳**：from_vllm 的 10 个 *_patch 路径中除 `rejection_kernel_patch`（18 import FAIL）与 `prepare_dflash_inputs_kernel_patch`（1 SKIP）外均全 PASS；from_vllm_ascend 全绿。印证 vllm_ascend 已替换/实现的生产 kernel 是**可验证且正确**的。
5. **SKIPPED 的「block-verification」系列（resample 12 例 + compute_block_* 3 例 + rejection 系列）恰好是被测对象的「生产新路径」**，安装的 vLLM 版本未含这些 kernel → 这些新功能的精度**等于没测**，是最大的覆盖空洞。

### 6.2 建议动作（按优先级）

| 优先级 | 动作 | 针对 |
| --- | --- | --- |
| P0 | 定位并修复 `_fill_logprob_token_ids_kernel` custom 多行映射错位（最小复现 batch=4,topk=3） | 精度缺陷 ② |
| P0 | 修复 `_compute_global_lse` 全 -inf 返回 nan（对齐 CPU 参考为 -inf） | 精度缺陷 ① |
| P1 | 对齐 vLLM 版本符号：修复 rejection_patch / block_max / block_stats / rejection 的 `_compute_local_logits_stats_kernel` import，或升级/核对版本 | import 18+收集 SKIP |
| P1 | 核对 `gumbel_block_argmax` 签名，更新 wrapper `PER_TOKEN_COL` | 编译 3 |
| P1 | 核对 Ascend 生产路径的 `_ranks_kernel` 用 `>`（非 `>=`）在等值 token 时的语义（from_vllm_ascend 关注点，见 7.2） | 盲区 |
| P1 | 昇腾侧排查 `topk_topp_triton.py:153/646` legalize 失败与 `tl_rand64` vector core 异常(507035) | 编译 16 + 运行时 1 |
| P2 | 安装含 block-verification kernel 的 vLLM 版本，补测 resample/compute_block_*/rejection 新路径 | SKIP 覆盖空洞 |
| P2 | from_vllm 补试 penality / min_p / gumbel 等其余 Ascend 生产路径（本报告 from_vllm 目录未运行，已在 from_vllm_ascend 覆盖） | 覆盖 |

---

## 7. 覆盖盲区与测试本身局限（非本次失败）

> 综合 `existing_tests_report.md`（§5-§6）与 `precision_test_analysis_report.md`（§4.2），列出**即使全 PASS 也没覆盖**的地方。

### 7.1 from_vllm 通用盲区

1. **PRNG 无法精确对比**：rejection/resample/gumbel 的 CPU 参考多数只支持 temp=0 确定性情形；temp>0 的接受/拒绝概率只做行为/属性断言，无统计等价。
2. **等值/并列边界**：`_ranks_kernel` 上游 `>=` vs Ascend `>` 可能差 1；topk_topp/topp 并列值；后需按两端语义校验。
3. **dtype 覆盖弱**：多数仅 fp32；只有 penality 覆盖 bf16/fp16；gumbel/bad_words/min_p/temperature/ranks 未半精度。
4. **多块/跨块归约与 PADDED 脏块**：block_stats、global_lse、cumulative_log_p、insert 普遍未测 vocab 不整除 block、跨块归约、PADDED 含脏值。
5. **生产分支未触发**：min_p=0、num_sampled=0(prefill)、num_rejected>0、USE_BLOCK_VERIFICATION=True、HAS_DRAFT_LOGITS=False 等常未独立触发。
6. **弱测试预警**：bad_words 只查「是否改变」无数值对照；slot_mapping 用 try/except 吞异常(失败不报)；tl_rand64 不验证 FP64 位级等价。
7. **版本条件 skip**：apply_write 的 multi-group 在旧 vLLM 直接 pytest.skip，实际精度未受验。

### 7.2 from_vllm_ascend 覆盖盲区

| 被测对象 | 未覆盖点 | 影响 |
| --- | --- | --- |
| `_bad_words_kernel` | 无数值对照（只查「是否改变」）；仅 fp32；词长<=3 | 无法确认具体 mask 位置正确性 |
| `_bincount_kernel` | 仅 1 个固定例；token_id<10；单 block | 多 block/大数值未验证 |
| `_compute_slot_mappings_kernel` | 仅 1 个固定小场景；CP/多 group/非 interleave 未测；try/except 吞异常隐患 | 复杂拓扑未验证 |
| `compute_topk_logprobs`/`_ranks_kernel` | rank 用 `>`（非 `>=`）；大 vocab；dtype 变体 | 等值并列时语义未明确 |
| `_topk_log_softmax_kernel` | 无 num_logprobs=0；容差较宽(1e-3) | 边界与精度余量 |
| `_min_p_kernel` | min_p=0 分支、min_p>=1、dtype 变体 | 边界未独立触发 |
| `_penalties_kernel` | 无 fp32 广参考；仅 vocab1000 | 大 vocab packed 未测 |
| `_gumbel_sample_kernel` | 采样正确性不定量对 logprobs/分布；random 无 RNG 对齐 | 统计口径宽松 |
| `_post_update_kernel` | 未用请求槽、依赖整数语义 | 边界槽位未验证 |
| `_temperature_kernel` | 无扩展 idx_mapping；dtype 变体 | 多 token 同 req 未测 |

---

## 8. 调用类型分布与会总量的结果汇总（综合 existing_tests_report.md §3-§4）

> 调用类型 A-F 定义沿用 `precision_test_calltype_report.md`：A=直接launch+CPU/PyTorch参考；B=直接launch+精确断言；C=helper用测试wrapper包裹；D=生产wrapper公共函数；E=多kernel级联；F=两实现头对头。

### 8.1 from_vllm（含 *_patch）按调用类型统计

| 调用类型 | 含义 | 涉及文件 | 本次结果 |
| --- | --- | --- | --- |
| A | 直接launch+CPU/PyTorch参考 | apply_write, bias, block_max(+patch), block_stats_patch, fill_logprob(原版), flatten, gather, insert(+patch), prompt_logprobs, resample(vanilla), scatter, selective_scan, topk_topp, prepare_dflash(+patch) | 大量 PASS；topk_topp 编译 FAIL；fill_logprob 数值 FAIL；resample 部分 SKIP |
| B | 直接launch+精确/行为断言 | resample_kernel_patch | PASS |
| C | helper用测试wrapper包裹 | compute_global_logsumexp(+patch), gumbel_block_argmax(+patch), load_ptr, update_min_larger | global_lse 数值 FAIL；gumbel 编译 FAIL；load_ptr SKIP；其余 PASS |
| D | 生产wrapper公共函数 | fill_logprob_patch(compute_topk_logprobs) | PASS |
| E | 多kernel级联 | compute_block_stats(vanilla, SKIP), rejection(vanilla, SKIP), rejection_patch(import FAIL), topk_topp rank | 全未通过或未执行 |
| — | helper+生产wrapper 混合 | tl_rand64(+patch) | FAIL(未patch)/PASS(patch) |

### 8.2 from_vllm_ascend 按调用类型统计

| 调用类型 | 涉及文件 | 本次结果 |
| --- | --- | --- |
| A | bincount, log_softmax | ALL PASS |
| C | bad_words(生产wrapper包helper) | ALL PASS |
| D | topk_logprobs, gumbel_sampling, min_p, penality, temperature | ALL PASS |
| D+E | topk_logprobs(含 rank) | ALL PASS |
| F | compute_slot_mapping, post_update | ALL PASS |
| F+E | post_update(+独立CPU oracle) | ALL PASS |

### 8.3 对象类型汇总的本次结果

| 对象类型 | 涉及算子 | 本次通过度 |
| --- | --- | --- |
| kernel(含生产wrapper包kernel) | bias, insert, gather, resample, rejection, topk_topp, bincount, log_softmax, min_p, penality, temperature, gumbel... | 上游 vanilla 多处 FAIL/SKIP；Ascend 路径(patch + from_vllm_ascend)基本全 PASS |
| helper(测试wrapper包裹) | global_lse, gumbel_block_argmax, load_ptr, update_min_larger_stats, tl_rand64 | global_lse 数值 FAIL；gumbel 编译 FAIL；其余 PASS |

---

## 9. 结论模板（综合上一版「结论模板」并填入本次运行结果）

| 被测对象 | 一致性 | 盲区是否影响生产 | 建议 | 风险等级 |
| --- | --- | --- | --- | --- |
| `_compute_global_lse`(helper) | **不一致(FAIL)** | 全 -inf 返回 nan 会污染下游 | 对齐 -inf 归约语义返回 -inf | **高** |
| `_fill_logprob_token_ids_kernel` | **不一致(FAIL)** | custom 多行映射错位 | 最小复现修复 row 偏移/repeat | **高** |
| `_topk_topp_kernel` | 无法编译(FAIL) | 后端 legalize 不支持 | 昇腾工具链侧修复 topk_topp_triton:153/646 | **中** |
| `gumbel_block_argmax`(helper) | 无法编译(FAIL) | UT 与版本签名脱节 | 对齐 `PER_TOKEN_COL` 签名 | **低**(测试问题) |
| `_probabilistic_rejection_kernel` | 无法验证(import FAIL) | 上游符号缺失 | 对齐 `_compute_local_logits_stats_kernel` 符号 | **中** |
| `tl_rand64`+gumbel FP32 | 设备异常(FAIL) | vector core 执行异常 | 昇腾侧排查; 改用 Ascend gumbel 路径 | **中** |
| `_ranks_kernel` | 通过(> vs >= 未专测) | 等值差1 | 确认 `>` vs `>=` 语义 | **低-中** |
| `_resample_kernel`(vanilla) | PASS(非block-ver部分) | block-ver 分支未测 | 补测生产新路径 | 中 |
| 其余 from_vllm 通过项 | 一致(PASS) | 见 7.1 盲区 | 按盲区补 | 低 |
| from_vllm_ascend 全部 | **一致(ALL PASS)** | 见 7.2 盲区 | 按盲区补（bad_words 数值对照、rank 等值等） | 低 |

---

## 10. 附：本次运行汇总统计速查

| 度量 | from_vllm | from_vllm_ascend | 合计 |
| --- | --- | --- | --- |
| 文件数 | 30 | 10 | 40 |
| 收集用例 | 226 | 81 | 307 |
| PASSED | 154 | 81 | 235 |
| FAILED | 44 | 0 | 44 |
| SKIPPED | 28 | 0 | 28 |
| 数值缺陷候选(FAIL) | 6 | 0 | 6 |
| 运行时间参考 | ~30 session / 约 15min | ~10 session / 约 15min | — |

> **诊断脚本**：`diagnose_bincount_atomic_or.py`（from_vllm_ascend）为独立原子探针，父进程子进程超时隔离排查 `_bincount_kernel` 的 atomic_or 挂起，未作为正式 pytest 用例在 run_from_vllm_ascend.sh 中执行，本次不计入用例数；历史挂起在本次运行中未复现（bincount 正式用例 PASS）。

---

*报告生成：基于 0810errors/ut_in_vllm.txt 与 ut_in_vllm_ascend.txt 两份运行日志，综合 precision_test_analysis_report.md、existing_tests_report.md、precision_test_calltype_report.md 与 README.md 撰写。*

---

## 11. 算子精度 UT 出现位置矩阵（vllm / vllm-ascend 三分类）

> 对全表所有被测算子，归类其在 **vLLM** 与 **vLLM-Ascend** 源码测试树中**是否出现精度 UT**，并标注**出现位置**（仓库 + 文件 + 行号，已对照 `git\vllm` 与 `git\vllm-ascend-xyz` 实际源码核验）。
>
> **三类划分**：
> - **① 存在于 vllm 的 UT**：`git\vllm\tests\...` 中存在对该算子的精度测试（多为间接/经 wrapper 覆盖，若直接 launch 会标注「直接」）。
> - **② 存在于 vllm-ascend 的 UT**（含已 patch / 已适配）：`git\vllm-ascend-xyz\tests\...` 中存在针对 Ascend 实现的精度测试，或已配 adapt / monkey-patch 替换路径。
> - **③ 完全没有精度 UT**：两仓测试树中均无针对该算子的精度测试（仅本 codex 目录补写/搬运的除外，见第 11.3 节）。
>
> 位置列以「仓库:文件路径:行号」表示；「±」表示该测试为间接（wrapper/集成）覆盖，非直接 launch 目标 kernel。

### 11.1 from_vllm/ 被测算子

| 被测算子 | ① vllm 中 UT | ② vllm-ascend 中 UT（含patch/适配） | ③ 完全没有 | 出现位置 |
| --- | --- | --- | --- | --- |
| `_apply_write_kernel` | 有 | — | — | vllm: tests/v1/worker/test_gpu_block_table.py:16,110（±wrapper: apply_staged_writes） |
| `_bias_kernel` | 有 | — | — | vllm: tests/v1/sample/test_sampler.py（±sampler logit-bias 用例） |
| `_compute_local_logits_stats_kernel`（`_compute_max_and_sumexp` helper） | 有 | — | — | vllm: tests/v1/spec_decode/test_rejection_sampler_utils.py:325（±block-verification）；Ascend 以别名 `_compute_block_stats_kernel` 导入/再导出 |
| `_compute_cumulative_log_p_kernel` | 有 | — | — | vllm: tests/v1/spec_decode/test_rejection_sampler_utils.py:325,372（±block-verification 链路） |
| `_compute_block_stats_kernel`（Ascend 别名） | — | 有（已适配别名） | — | vllm-ascend: vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py:25（`_compute_local_logits_stats_kernel` 别名） |
| `_compute_global_lse` / `_compute_global_logsumexp`（helper） | 有 | — | — | vllm: tests/v1/spec_decode/test_rejection_sampler_utils.py:325（±）；Ascend 以旧名导入别名（rejection_sampler_utils.py:22） |
| `_fill_logprob_token_ids_kernel` | 有 | — | — | vllm: tests/v1/sample/test_logprobs.py（±）；Ascend 生产 `compute_topk_logprobs` 已 tensor 拼装替换，不再调用该 kernel |
| `_flatten_sampled_kernel` | 有 | — | — | vllm: tests/v1/spec_decode/test_rejection_sampler_utils.py（±rejection sampler 链路） |
| `_gather_block_tables_kernel` | 有 | — | — | vllm: tests/v1/worker/test_gpu_block_table.py:16,110（±） |
| `gumbel_block_argmax`（helper） | 有 | — | — | vllm: tests/v1/worker/test_gpu_gumbel_sample.py:114（±gumbel_sample）；Ascend 改名 `_npu_gumbel_block_argmax`（rejection_sampler_utils.py:34） |
| `_npu_gumbel_block_argmax`（Ascend helper） | — | 有（已适配/补测） | — | vllm-ascend: vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py:34（实现）；本 codex patch 测试 test_gumbel_block_argmax_patch.py |
| `_insert_resampled_kernel` | 有 | — | — | vllm: tests/v1/spec_decode/test_rejection_sampler_utils.py（±）；Ascend rejection sampler 直接再导出（rejection_sampler_utils.py:28） |
| `_load_ptr`（helper） | 有 | — | — | vllm: tests/v1/worker/test_gpu_block_table.py（仅随 `_apply_write_kernel`/gather 间接执行，无独立 UT） |
| `_prepare_dflash_inputs_kernel`（vanilla） | 有 | — | — | vllm: tests/v1/spec_decode/test_dflash_lookahead.py（±集成覆盖，非 kernel UT） |
| `_prepare_dflash_inputs_kernel_ascend`（Ascend） | — | 有（已适配 patch，monkey-patch 回原名） | — | vllm-ascend: vllm_ascend/worker/v2/spec_decode/dflash/speculator.py:153；patch: patch/worker/patch_v2/patch_triton.py:37 |
| `_prompt_logprobs_token_ids_kernel` | 有 | — | — | vllm: tests/v1/sample/test_logprobs.py:1192（±）；Ascend 生产用 `compute_topk_logprobs` 替代 |
| `_rejection_kernel`（vanilla） | 有 | — | — | vllm: tests/v1/spec_decode/test_rejection_sampler_utils.py:141,183（±）；Ascend 重写 `_probabilistic_rejection_kernel`（rejection_sampler_utils.py:192） |
| `_probabilistic_rejection_kernel`（Ascend） | — | 有（已适配 patch，Ascend 无官方专属 UT） | — | vllm-ascend: vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py:192（实现）；本 codex patch 测试 test_rejection_kernel_patch.py |
| `_resample_kernel`（vanilla + Ascend 同名适配） | 有 | 有（已适配同名实现，用 `_npu_gumbel_block_argmax`） | — | vllm: tests/v1/spec_decode/test_rejection_sampler_utils.py（±）；vllm-ascend: vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py:82 |
| `_scatter_num_accepted_kernel` | 有 | — | — | vllm: tests/kernels/mamba/test_mamba_ssm.py:861（±经 selective_state_update） |
| `_selective_scan_update_kernel` | 有 | — | — | vllm: tests/kernels/mamba/test_mamba_ssm.py:356 等（±经 selective_state_update） |
| `tl_rand64` | 有 | —（Ascend 改用 tl.rand FP32） | — | vllm: tests/v1/worker/test_gpu_gumbel_sample.py:114（±）；Ascend 生产 Gumbel 用 tl.rand（vllm_ascend gumbel.py:154） |
| `_topk_topp_kernel` | 有 | — | — | vllm: tests/v1/sample/test_topk_topp_sampler.py:298（±经 apply_top_k_top_p_triton） |
| `_update_min_larger_stats`（helper） | 有 | — | — | vllm: tests/v1/sample/test_topk_topp_sampler.py（随 apply_top_k_top_p 间接覆盖） |

---

### 11.2 from_vllm_ascend/ 被测算子

| 被测算子 | ① vllm 中 UT | ② vllm-ascend 中 UT（含patch/适配） | ③ 完全没有 | 出现位置 |
| --- | --- | --- | --- | --- |
| `_bad_words_kernel` / `apply_bad_words` | 有（间接） | 有（官方 wrapper UT） | — | vllm: tests/v1/sample/test_sampler.py（±）；vllm-ascend: tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_bad_words.py:93 |
| `_bincount_kernel` | — | 有（官方直接 UT） | — | vllm-ascend: tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_bincount.py:40（直接 launch，A3 已确认通过） |
| `_compute_slot_mappings_kernel` | 有（间接） | 有（官方直接 UT，与上游头对头） | — | vllm: tests/v1/worker/test_gpu_block_table.py（±）；vllm-ascend: tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_slot_mapping.py:7 |
| `_topk_log_softmax_kernel` | 有（间接） | 有（官方直接 UT + wrapper） | — | vllm: tests/v1/sample/test_logprobs.py（±）；vllm-ascend: tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_log_softmax.py:16（直接） |
| `_ranks_kernel` | 有（间接） | 有（经 compute_topk_logprobs wrapper） | — | vllm: tests/v1/sample/test_logprobs.py（±）；vllm-ascend: tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_topk_logprobs.py:17（±） |
| `compute_topk_logprobs` | — | 有（官方 UT，含 num_logprobs=0） | — | vllm-ascend: tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_topk_logprobs.py:17 |
| `_gumbel_sample_kernel` / `gumbel_sample` / `apply_temperature` | 有（间接） | 有（官方 a2 wrapper UT） | — | vllm: tests/v1/worker/test_gpu_gumbel_sample.py:114（±）；vllm-ascend: tests/ut/sample/a2/test_gumbel_sampling.py:44（±wrapper） |
| `_min_p_kernel` / `apply_min_p` | 有（间接） | 有（官方 wrapper UT） | — | vllm: tests/v1/sample/test_sampler.py（±）；vllm-ascend: tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_min_p.py:41 |
| `_penalties_kernel` / `apply_penalties` | 有（间接） | 有（官方 wrapper UT） | — | vllm: tests/v1/sample/test_sampler.py、tests/v1/sample/test_logprobs.py:1075（±）；vllm-ascend: tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_penality.py:191 |
| `_post_update_kernel` | — | 有（官方交叉实现 UT，GPU vs Ascend + oracle） | — | vllm-ascend: tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_post_update.py:63 |
| `_temperature_kernel` / `apply_temperature` | 有（间接） | 有（官方 wrapper UT） | — | vllm: tests/v1/worker/test_gpu_gumbel_sample.py:114（±）；vllm-ascend: tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_temperature.py:43 |

### 11.3 完全没有精度 UT 的算子（③类，两仓源代码测试树均无）

> 以下算子**在 vllm 与 vllm-ascend 两仓测试树中都没有**针对其本身的精度 UT，仅在**本 codex 目录**（missing_accuracy_tests / existing_accuracy_tests）补写或搬运了测试。逐项列出「无 UT」的被测算子与补写位置。

| 被测算子 | 两仓情形 | 补写/搬运测试（codex 目录） |
| --- | --- | --- |
| `_num_nans_kernel` | vLLM 无直接 kernel UT；Ascend 无本地实现/测试 | missing: test_num_nans_kernel.py |
| `_prepare_rope_positions_kernel` | vLLM wrapper 已迁移、无同名 UT；Ascend 无适配（`test_mrope.py` 是另一套 `triton_mrope`） | missing: test_prepare_rope_positions_kernel.py |
| `_prepare_prefill_inputs_kernel`（input_batch） | vLLM 无；Ascend 无 | missing: test_prepare_prefill_inputs_kernel.py |
| `_prepare_prefill_inputs_kernel`（AR speculator） | 同前，不同签名 | missing: test_prepare_prefill_inputs_kernel_speculator.py |
| `_prepare_decode_inputs_kernel` | vLLM 无；Ascend 无 | missing: test_prepare_decode_inputs_kernel.py |
| `_update_draft_inputs_kernel` | vLLM 无；Ascend 无 | missing: test_update_draft_inputs_kernel.py |
| `_dcp_local_seq_lens_kernel` | 两仓均无 | missing: test_dcp_local_seq_lens_kernel.py |
| `_prepare_pos_seq_lens_kernel` | 两仓均无 | missing: test_prepare_pos_seq_lens_kernel.py |
| `_combine_sampled_and_draft_tokens_kernel` | 两仓均无（Ascend 生产复用上游） | missing: test_combine_sampled_and_draft_tokens_kernel.py |
| `_get_num_sampled_and_rejected_kernel` | 两仓均无 | missing: test_get_num_sampled_and_rejected_kernel.py |
| `_post_update_num_computed_tokens_kernel` | vLLM 无；Ascend 生产 post_update 未用同名 kernel | missing: test_post_update_num_computed_tokens_kernel.py |
| `_expand_idx_mapping_kernel` | 两仓均无 | missing: test_expand_idx_mapping_kernel.py |
| `_apply_grammar_bitmask_kernel` | Ascend 有实现、无精度 UT | missing: test_apply_grammar_bitmask_kernel_patch.py |
| `_zero_kv_blocks_kernel` | Ascend 有实现、无精度 UT | missing: test_zero_kv_blocks_kernel_patch.py |


> **综合判定**（本次 40 个已有精度 UT 文件，交叉参考 §11.1–§11.3）：
> - **同时有 vllm 与 vllm-ascend 官方 UT 覆盖**的算子，集中在 from_vllm_ascend（sampling/logprob/penalty/block_table 生产路径），且两仓测试都存在、位置明确。
> - **仅有 vllm（上游）UT、Ascend 侧只作 改名/替换/再导出（patch）而没有 vllm-ascend 专属官方测试**的算子，集中在 from_vllm 的 rejection/resample/global_lse/gumbel/topk_topp 等（如 `_probabilistic_rejection_kernel`、`_npu_gumbel_block_argmax`、`_compute_block_stats_kernel`、`_prepare_dflash_inputs_kernel_ascend`）。
> - **两仓均无精度 UT、完全依赖本 codex 目录补写**的算子见 §11.3（input_batch 各 kernel、num_nans、zero_kv、grammar_bitmask、rope/prefill 等）。
>
> 注：本 codex 目录自身的 `existing_accuracy_tests/from_vllm/*_patch.py` 与 `missing_accuracy_tests/*_patch.py` 是**针对 vllm-ascend 实现路径**补充的独立精度测试，不属于上述两仓官方测试树；其出现位置以本 codex 目录为准（详见本报告 §3–§4 各文件结果表）。


---

## 12. 完整算子清单精度 UT 出现位置（vllm / vllm-ascend 三分类）

> 针对用户给定**完整算子清单**（含 wrapper / launch kernel / bench helper 标注），逐一给出其在 **vLLM** 与 **vLLM-Ascend** 中**是否出现精度 UT** 及**出现位置**。已对照 `git\vllm`、`git\vllm-ascend-xyz` 实际源码与测试树核验。
>
> **三类**：① **存在于 vllm 的 UT**；② **存在于 vllm-ascend 的 UT**（含已打 patch 与已适配的 Ascend 实现路径）；③ **完全没有精度 UT**（两仓测试树均无，仅本 codex 目录补写或也没有）。
>
> 位置格式「仓库:文件路径:行号」；「±」表示**间接**覆盖（经 wrapper / 集成调用，非直接 launch 目标 kernel）。「codex」表示本 work区 `accuracy_test/codex` 目录的补写/搬运测试（非两仓官方测试树）。

| # | 算子（用户列表） | ① vllm 中 UT | ② vllm-ascend 中 UT（含patch/适配） | ③ 完全没有 | 出现位置 |
| --- | --- | --- | --- | --- | --- |
| 1 | `_num_nans_kernel`（wrapper） | — | — | **无** | 两仓均无 kernel 精度 UT；codex: missing/test_num_nans_kernel.py |
| 2 | `_prepare_rope_positions_kernel`（wrapper） | — | — | **无** | vllm 源码: vllm/v1/worker/gpu/mm/rope.py；两仓无 UT；codex: missing/test_prepare_rope_positions_kernel.py |
| 3 | `_scatter_num_accepted_kernel`（launch） | 有（±） | — | — | vllm: tests/kernels/mamba/test_mamba_ssm.py:861（经 selective_state_update） |
| 4 | `_bad_words_kernel` | 有（±） | 有（wrapper） | — | vllm: tests/v1/sample/test_sampler.py；vllm-ascend: tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_bad_words.py:93 |
| 5 | `_temperature_kernel` | 有（±） | 有（wrapper） | — | vllm: tests/v1/worker/test_gpu_gumbel_sample.py:114；vllm-ascend: tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_temperature.py:43 |
| 6 | `gumbel_block_argmax`（bench helper） | 有（±） | 有（已适配 `_npu_gumbel_block_argmax`） | — | vllm: tests/v1/worker/test_gpu_gumbel_sample.py:114；vllm-ascend 实现: worker/v2/spec_decode/rejection_sampler_utils.py:34；codex patch: from_vllm/test_gumbel_block_argmax_patch.py |
| 7 | `_gumbel_sample_kernel` | 有（±） | 有（wrapper） | — | vllm: tests/v1/worker/test_gpu_gumbel_sample.py:114；vllm-ascend: tests/ut/sample/a2/test_gumbel_sampling.py:44 |
| 8 | `_bias_kernel` | 有（±） | — | — | vllm: tests/v1/sample/test_sampler.py（logit-bias 用例） |
| 9 | `_topk_log_softmax_kernel` | 有（±） | 有（直接 + wrapper） | — | vllm: tests/v1/sample/test_logprobs.py；vllm-ascend: tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_log_softmax.py:16（直接） |
| 10 | `_ranks_kernel` | 有（±） | 有（经 compute_topk_logprobs） | — | vllm: tests/v1/sample/test_logprobs.py；vllm-ascend: tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_topk_logprobs.py:17 |
| 11 | `_fill_logprob_token_ids_kernel`（launch） | 有（±） | — | — | vllm: tests/v1/sample/test_logprobs.py（±）；Ascend 生产 `compute_topk_logprobs` 已 tensor 拼装替换，不再调用该 kernel |
| 12 | `_min_p_kernel` | 有（±） | 有（wrapper） | — | vllm: tests/v1/sample/test_sampler.py；vllm-ascend: tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_min_p.py:41 |
| 13 | `_penalties_kernel` | 有（±） | 有（wrapper） | — | vllm: tests/v1/sample/test_sampler.py、tests/v1/sample/test_logprobs.py:1075；vllm-ascend: tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_penality.py:191 |
| 14 | `_bincount_kernel` | — | 有（直接） | — | vllm-ascend: tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_bincount.py:40（直接 launch，A3 已确认通过） |
| 15 | `_prompt_logprobs_token_ids_kernel` | 有（±） | — | — | vllm: tests/v1/sample/test_logprobs.py:1192（±）；Ascend 生产用 `compute_topk_logprobs` 替代 |
| 16 | `_prepare_prefill_inputs_kernel`（ar） | — | — | **无** | 两仓无 UT；codex: missing/test_prepare_prefill_inputs_kernel_speculator.py |
| 17 | `_prepare_decode_inputs_kernel`（wrapper） | — | — | **无** | 两仓无 UT；codex: missing/test_prepare_decode_inputs_kernel.py |
| 18 | `_update_draft_inputs_kernel`（wrapper） | — | — | **无** | 两仓无 UT；codex: missing/test_update_draft_inputs_kernel.py |
| 19 | `_prepare_dflash_inputs_kernel`（wrapper） | 有（±集成） | 有（已适配 `_prepare_dflash_inputs_kernel_ascend`） | — | vllm: tests/v1/spec_decode/test_dflash_lookahead.py；vllm-ascend: worker/v2/spec_decode/dflash/speculator.py:153（适配）+ patch/worker/patch_v2/patch_triton.py:37；codex patch: from_vllm/test_prepare_dflash_inputs_kernel_patch.py |
| 20 | `_compute_block_stats_kernel`（= `_compute_local_logits_stats_kernel`，launch） | 有（±） | 有（已适配别名） | — | vllm: tests/v1/spec_decode/test_rejection_sampler_utils.py:325（±）；vllm-ascend 别名: worker/v2/spec_decode/rejection_sampler_utils.py:25；codex patch: from_vllm/test_compute_block_stats_kernel_patch.py |
| 21 | `_compute_cumulative_log_p_kernel`（launch） | 有（±） | — | — | vllm: tests/v1/spec_decode/test_rejection_sampler_utils.py:325,372（± block-verification 链路） |
| 22 | `_compute_local_residual_mass_kernel`（launch） | — | — | **无** | vllm 源码: vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py；两仓无直接精度 UT（仅经 rejection 链路间接）；codex 亦未补写 |
| 23 | `_rejection_kernel`（launch） | 有（±） | 有（已适配 `_probabilistic_rejection_kernel`） | — | vllm: tests/v1/spec_decode/test_rejection_sampler_utils.py:141,183（±）；vllm-ascend: worker/v2/spec_decode/rejection_sampler_utils.py:192；codex patch: from_vllm/test_rejection_kernel_patch.py |
| 24 | `_resample_kernel`（launch） | 有（±） | 有（已适配同名实现） | — | vllm: tests/v1/spec_decode/test_rejection_sampler_utils.py（±）；vllm-ascend: worker/v2/spec_decode/rejection_sampler_utils.py:82（用 `_npu_gumbel_block_argmax`）；codex patch: from_vllm/test_resample_kernel_patch.py |
| 25 | `_insert_resampled_kernel`（launch） | 有（±） | —（Ascend 直接再导出） | — | vllm: tests/v1/spec_decode/test_rejection_sampler_utils.py（±）；vllm-ascend 再导出: worker/v2/spec_decode/rejection_sampler_utils.py:28；codex patch: from_vllm/test_insert_resampled_kernel_patch.py |
| 26 | `_flatten_sampled_kernel` | 有（±） | — | — | vllm: tests/v1/spec_decode/test_rejection_sampler_utils.py（±rejection sampler 链路） |
| 27 | `_gather_block_tables_kernel` | 有（±） | — | — | vllm: tests/v1/worker/test_gpu_block_table.py:16,110（±） |
| 28 | `_compute_slot_mappings_kernel` | 有（±） | 有（直接，与上游头对头） | — | vllm: tests/v1/worker/test_gpu_block_table.py（±）；vllm-ascend: tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_slot_mapping.py:7 |
| 29 | `_apply_write_kernel` | 有（±） | — | — | vllm: tests/v1/worker/test_gpu_block_table.py:16,110（±wrapper: apply_staged_writes） |
| 30 | `_dcp_local_seq_lens_kernel` | —（仅 Python 包装函数 `get_dcp_local_seq_lens` 间接） | — | **无（kernel 本身）** | vllm 源码: vllm/v1/worker/gpu/cp_utils.py；Python wrapper 间接测于 tests/v1/worker/test_cp_utils.py 与 tests/v1/attention/test_indexer_dcp_localize.py:258；**Triton kernel 本身无直接精度 UT**；codex: missing/test_dcp_local_seq_lens_kernel.py |
| 31 | `_prepare_prefill_inputs_kernel`（input_batch） | — | — | **无** | 两仓无 UT；codex: missing/test_prepare_prefill_inputs_kernel.py |
| 32 | `_prepare_pos_seq_lens_kernel` | —（wrapper 间接？未发现直接） | — | **无（直接 UT）** | vllm 源码: vllm/v1/worker/gpu/input_batch.py；两仓无直接精度 UT；codex: missing/test_prepare_pos_seq_lens_kernel.py |
| 33 | `_combine_sampled_and_draft_tokens_kernel` | — | — | **无** | 两仓无 UT（Ascend 生产复用上游）；codex: missing/test_combine_sampled_and_draft_tokens_kernel.py |
| 34 | `_get_num_sampled_and_rejected_kernel` | — | — | **无** | 两仓无 UT；codex: missing/test_get_num_sampled_and_rejected_kernel.py |
| 35 | `_post_update_kernel` | — | 有（交叉实现 UT） | — | vllm-ascend: tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_post_update.py:63（GPU vs Ascend + oracle）；vllm 无专属 UT |
| 36 | `_post_update_num_computed_tokens_kernel`（wrapper） | — | — | **无** | vllm 无；Ascend 生产 post_update 未用同名 kernel；codex: missing/test_post_update_num_computed_tokens_kernel.py |
| 37 | `_expand_idx_mapping_kernel` | — | — | **无** | 两仓无 UT；codex: missing/test_expand_idx_mapping_kernel.py |
| 38 | `_apply_grammar_bitmask_kernel` | — | 有实现、无官方 UT（已适配 monkey-patch） | **无（官方 UT）** | vllm-ascend: worker/v2/structured_outputs.py:35（monkey patch 替换上游）；无官方测试；codex: missing/test_apply_grammar_bitmask_kernel_patch.py |


### 12.1 三分类统计与小结

| 分类 | 涉及算子上限（38 项） | 代表 |
| --- | --- | --- |
| **① 仅 vllm 有 UT**（Ascend 侧无专属官方精度测试，多为改名/替换/再导出） | ~13 | `_bias_kernel`、`_fill_logprob_token_ids_kernel`、`_prompt_logprobs_token_ids_kernel`、`_gather_block_tables_kernel`、`_apply_write_kernel`、`_flatten_sampled_kernel`、`_insert_resampled_kernel`、`_compute_cumulative_log_p_kernel`、`_scatter_num_accepted_kernel` 等 |
| **② 仅 vllm-ascend 有 UT**（含已 patch / 已适配实现路径，多与 vllm 并存） | ~8 | `_bincount_kernel`、`_post_update_kernel`、`_compute_slot_mappings_kernel`、`compute_topk_logprobs` 等 |
| **①② 两仓都有 UT** | ~7 | `_bad_words_kernel`、`_temperature_kernel`、`_gumbel_sample_kernel`、`_topk_log_softmax_kernel`、`_ranks_kernel`、`_min_p_kernel`、`_penalties_kernel`、`_wrap` 等 |
| **③ 完全没有精度 UT**（两仓测试树均无） | ~11 | `_num_nans_kernel`、`_prepare_rope_positions_kernel`、`_prepare_prefill_inputs_kernel`(ar/input_batch)、`_prepare_decode_inputs_kernel`、`_update_draft_inputs_kernel`、`_compute_local_residual_mass_kernel`、`_dcp_local_seq_lens_kernel`(kernel本身)、`_prepare_pos_seq_lens_kernel`、`_combine_sampled_and_draft_tokens_kernel`、`_get_num_sampled_and_rejected_kernel`、`_post_update_num_computed_tokens_kernel`、`_expand_idx_mapping_kernel`、`_apply_grammar_bitmask_kernel` |

> **要点**：
> - **ASCend 生产采样/logprob/penalty/block_table 路径（from_vllm_ascend 及 *_patch）** 在两仓中均有官方 或 适配型精度 UT，位置已在上表标注。
> - **③ 类（完全没有精度 UT）** 绝大多数为 **input_batch / AR speculator / 公共辅助 kernel**（`_prepare_*`、`_get_num_*`、`_combine_*`、`_expand_idx_mapping`、`_num_nans`、`_dcp_*`、`_zero_kv_blocks`、`_grammar_bitmask` 等），仅能依赖本 work区 `missing_accuracy_tests/` 补写测试（已在列中标注 codex 位置）。
> - **`_compute_local_residual_mass_kernel`** 在 vllm 源码存在（rejection_sampler_utils.py），但两仓均无对其的直接精度 UT，且本 codex 目录也未补写——属「完全无 UT」且尚无覆盖。
> - 标注「±」的 vllm 侧测试多为**间接**（经 wrapper/集成链路）覆盖，非直接 launch 目标 kernel；真正直接 launch 的精度 UT 主要存在于 vllm-ascend（如 `_bincount_kernel`、`_topk_log_softmax_kernel`、`_compute_slot_mappings_kernel`）。
