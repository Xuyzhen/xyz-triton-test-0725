# vLLM-Ascend 内部已有 UT 运行结果报告（0810 from_vllm_ascend）

> 数据来源：`ut_in_vllm_ascend.txt`（`run_from_vllm_ascend.sh` 运行 from_vllm_ascend 测试文件的标准输出，325 行）。
> 本报告**仅覆盖 vLLM-Ascend 内部已实现的 UT**，即 `existing_accuracy_tests/from_vllm_ascend/` 目录中的正式测试文件（含被 `run_from_vllm_ascend.sh` 执行的全部 10 个文件；`diagnose_bincount_atomic_or.py` 为诊断脚本，未作为正式用例运行）。
> 综合 `precision_test_analysis_report.md`（单测标准/方式/优缺点）与 `existing_tests_report.md`（静态元数据盘点）撰写，并在后者表格基础上**补齐 0810 实际运行结果**。

---

## 1. 环境与运行口径

| 项 | 值 |
| --- | --- |
| 运行命令 | `bash .../accuracy_test/codex/existing_accuracy_tests/run_from_vllm_ascend.sh` |
| 平台 | Linux, Python 3.11.10, pytest 8.3.2 |
| 硬件 | 昇腾 NPU(A3)；`torch.device("npu")` 直接 launch |
| 插件 | pytest-asyncio 1.3.0, xdist 3.6.1 |
| vLLM 平台插件 | ascend 已激活（`vllm_ascend:register`） |
| 执行方式 | 每次运行启动独立 pytest session（共 10 个 session，逐个文件） |
| 被测来源 | 均调用 vLLM-Ascend / vLLM 的**生产 wrapper 公共函数与 kernel**（`apply_bad_words`、`apply_penalties`、`apply_min_p`、`gumbel_sample`、`compute_topk_logprobs` 等） |

## 2. 总体结果（10 文件 / 81 用例）

| 指标 | 数量 | 占比 |
| --- | --- | --- |
| 收集用例总数 | 81 | 100% |
| **PASSED（通过）** | **81** | 100% |
| **FAILED（失败）** | **0** | 0% |
| **SKIPPED（未运行）** | **0** | 0% |

> **结论：本次 vllm_ascend 目录全部 81 用例通过，无任何失败或跳过。**
> 且所有断言均为**数值校验通过**（非 XFAIL/skip 遮掩），即被测的 Ascend 生产路径与各自的 CPU/PyTorch 参考实现一致。
---

## 3. 逐文件结果（补齐 `existing_tests_report.md` from_vllm_ascend 结果列）

> 「结果」列三态：`PASS`（通过）、`SKIP`（未运行）。本日志无 FAIL。
> 用例数 = 该 session 收集 items 数。

### 3.1 from_vllm_ascend/ 10 文件结果表

| 文件 | 被测对象 | 对象类型 | 调用类型 | 参考基准 | 容差判据 | 用例数 | 结果 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test_bad_words | `bad_words.apply_bad_words`(`_bad_words_kernel`) | 生产wrapper | C(生产wrapper包helper) | 行为(前后对比是否被修改) | allclose(相等性) | 6 | **ALL PASS** |
| test_bincount | `_bincount_kernel`(penalties.py) | kernel | A | 自写 `torch_bincount` | torch.equal(int32) | 1 | **PASS** |
| test_compute_slot_mapping | `_compute_slot_mappings_kernel`(Ascend vs vllm) | kernel | F(Ascend vs 上游头对头) | 另一实现(上游 vllm kernel) | torch.equal(int64) | 1 | **PASS** |
| test_compute_topk_logprobs | `compute_topk_logprobs`(`_topk_log_softmax_kernel`+`_ranks_kernel`) | kernel(生产wrapper+级联) | D+E | PyTorch topk+log_softmax+计数rank | ID/rank equal; logprobs 1e-4 | 4 | **ALL PASS** |
| test_gumbel_sampling | `apply_temperature`+`gumbel_sample`(`_gumbel_sample_kernel`) | kernel+生产wrapper | D+C | 自写PyTorch+统计/行为/确定性 | temp 1e-4/1e-5; greedy equal; processed 1e-4 | 30 | **ALL PASS** |
| test_log_softmax | `_topk_log_softmax_kernel`(logprob.py) | kernel | A | PyTorch log_softmax+gather | allclose 1e-3/1e-3 | 3 | **ALL PASS** |
| test_min_p | `min_p.apply_min_p`(`_min_p_kernel`) | kernel | D+E | 自写 `torch_min_p_torch` | inf mask equal; 有效值 1e-4 | 4 | **ALL PASS** |
| test_penality | `penalties.apply_penalties`(`_penalties_kernel`) | kernel+生产wrapper | D | 自写 `pytorch_apply_penalties`(packed mask+累积) | allclose 1e-3; bf16 1e-2 | 24 | **ALL PASS** |
| test_post_update | `_post_update_kernel`(Ascend vs vllm) | kernel | F+E +独立CPU oracle | 串行 oracle `post_update_ref` + 上游 | assert_close rtol=0/atol=0(int32) | 3 | **ALL PASS** |
| test_temperature | `gumbel.apply_temperature`(`_temperature_kernel`) | kernel+生产wrapper | D | 自写 `torch_apply_temperature`(纯Python) | allclose 1e-4/1e-5 | 5 | **ALL PASS** |

### 3.2 被测算子结果雷达（全部通过）

| 被测算子 | 文件 | 结果 | 覆盖规模 |
| --- | --- | --- | --- |
| `_bad_words_kernel` / `apply_bad_words` | bad_words | PASS | 512/1024/2048 tokens × requests 16/32/64；无/最大/超限坏词共 6 例 |
| `_bincount_kernel` | bincount | PASS | 固定单例 1 例 |
| `_compute_slot_mappings_kernel` | compute_slot_mapping | PASS | 固定配置 1 例 |
| `compute_topk_logprobs`(`_topk_log_softmax`+`_ranks`) | compute_topk_logprobs | PASS | batch 48/96/24/1 × vocab 1024/1519/320 × lp 5/0/1/10 共 4 例 |
| `apply_temperature`+`gumbel_sample`(`_gumbel_sample_kernel`) | gumbel_sampling | PASS | temperature/greedy/确定性/种子/分布/EAGLE 等 30 例 |
| `_topk_log_softmax_kernel` | log_softmax | PASS | batch 48/96/24 × vocab 102400/151936 × lp 50/1/8 共 3 例 |
| `_min_p_kernel` / `apply_min_p` | min_p | PASS | req 48/96/24/1 × vocab 102400/151936/32000 共 4 例 |
| `_penalties_kernel` / `apply_penalties` | penality | PASS | 2 dtype(bf16/fp16)×status{0,1,4}×spec{1,3}×tokens{1,4}×vocab1000 共 24 例 |
| `_post_update_kernel` | post_update | PASS | req 36/48/128 × vocab 200/32000 × steps 2/5 共 3 例 |
| `_temperature_kernel` / `apply_temperature` | temperature | PASS | 5 主流 vocab × 随机 token 共 5 例 |

> 与上一份 `from_vllm` 报告对比：**本目录（Ascend 生产路径）全部通过**，而上游 vanilla `from_vllm` 目录有 44 失败/28 跳过。这说明 vllm_ascend 的采样/logprob/penalty 各生产 kernel 在本次运行中数值校验全部与参考一致，**且不依赖上游缺失的 block-verification 符号**。
---

## 4. 各文件实际运行用例明细

### 4.1 test_bad_words（6 例，全 PASS）
- `test_apply_bad_words_different_shapes`：small-case(tokens=512,requests=16)、medium-case(1024,32)、large-case(2048,64)
- `test_apply_bad_words_no_bad_words`：无坏词
- `test_apply_bad_words_edge_cases`：最大坏词数量
- `test_apply_bad_words_token_limit`：token 数在限内 / 超限两种
> 行为级校验：仅断言 logits「是否被修改」，未比对具体数值位置。

### 4.2 test_bincount（1 例，PASS）
- `test_bincount_kernel`：固定单例（63 单 token、64 req、单 block、seed42）
> 仅 1 个固定用例，数值范围窄。

### 4.3 test_compute_slot_mapping（1 例，PASS）
- `test_compute_slot_mapping_npu_kernel`：固定配置（1 组 KV、1 req、5 tokens）
> Ascend vs 上游头对头，int64 slot 精确相等。

### 4.4 test_compute_topk_logprobs（4 例，全 PASS）
- `[48-1024-5]`、`[96-1024-0]`、`[24-1519-1]`、`[1-320-10]`（含 num_logprobs=0 边界）
> 覆盖 `_ranks_kernel` 的 rank 输出与 logprobs（1e-4）。注意：rank 用 `>`（非 `>=`）。

### 4.5 test_gumbel_sampling（30 例，全 PASS）
- 温度：`[1-32000]`、`[8-32000]`、`[48-102400]`、`[64-151936]`；skip_zero_and_one
- greedy：`[1-1-32000]`、`[4-4-32000]`、`[8-4-32000]`、`[16-8-102400]`；apply_temp_flag_irrelevant
- 确定性：`[4-4-32000]`、`[8-4-32000]`、`[16-8-102400]`
- 其他：different_seeds、valid_token_ids×3、temperature_affects_distribution、mixed_temperature×2、expanded_idx_mapping、shared_seed_same_request、apply_temperature_true/false_nonzero、processed_logits_req_state_idx / _col / _mixed_temp、single_token、large_vocab、extreme_temperatures
> 覆盖温度缩放、greedy vs 采样、seed 确定性、分布倾向、EAGLE processed_logits 等。

### 4.6 test_log_softmax（3 例，全 PASS）
- `[48-102400-50]`、`[96-102400-1]`、`[24-151936-8]`（含 num_logprobs=1）

### 4.7 test_min_p（4 例，全 PASS）
- `[48-102400]`、`[96-102400]`、`[24-151936]`、`[1-32000]`

### 4.8 test_penality（24 例，全 PASS）
- 参数组合 `[dtype0/1 × num_status{0,1,4} × num_spec{1,3} × vocab1000 × tokens{1,4}]`（dtype0=bf16, dtype1=fp16）
> 覆盖 bf16/fp16、packed prompt mask、累积 draft counts。

### 4.9 test_post_update（3 例，全 PASS）
- `[36-36-200-2]`、`[48-48-32000-5]`、`[128-128-32000-5]`

### 4.10 test_temperature（5 例，全 PASS）
- `[60-32000]`、`[44-50257]`、`[10-65024]`、`[9-128256]`、`[14-151936]`

---

## 5. 汇总与结论

### 5.1 关键结论

1. **本目录（vllm_ascend）81/81 全部通过**，0 失败、0 跳过，且均为**数值断言通过**（非 XFAIL 遮掩）。
2. 覆盖对象是 **Ascend 生产采样/logprob/penalty 全链路**：`_bad_words_kernel`、`_bincount_kernel`、`_compute_slot_mappings_kernel`、`_gumbel_sample_kernel`、`_topk_log_softmax_kernel`、`_min_p_kernel`、`_penalties_kernel`、`_post_update_kernel`、`_temperature_kernel`，以及 `_ranks_kernel`（经 compute_topk_logprobs）。
3. **与 from_vllm 报告形成鲜明对比**：上游 vanilla 目录 44 失败/28 跳过，而本 Ascend 生产路径全绿——印证 vllm_ascend 已替换/实现的生产 kernel 是**可验证且正确**的。
4. **未在本次运行中出现的**：`diagnose_bincount_atomic_or.py`（诊断脚本，非正式用例）；上游 block-verification 类 kernel（本目录本就不测）。

### 5.2 仍需注意的覆盖盲区（非本次失败，而是测试本身未覆盖处）

| 被测对象 | 未覆盖点 | 影响 |
| --- | --- | --- |
| `_bad_words_kernel` | 无数值对照（只查"是否改变"）；仅 fp32；词长<=3 | 无法确认具体 mask 位置正确性 |
| `_bincount_kernel` | 仅 1 个固定例；token_id<10；单 block | 多 block/大数值未验证 |
| `_compute_slot_mappings_kernel` | 仅 1 个固定小场景；CP/多 group/非 interleave 未测 | 复杂拓扑未验证 |
| `compute_topk_logprobs`/`_ranks_kernel` | rank 用 `>`（非 `>=`）；大 vocab；dtype 变体 | 等值并列时语义未明确 |
| `_topk_log_softmax_kernel` | 无 num_logprobs=0；容差较宽(1e-3) | 边界与精度余量 |
| `_min_p_kernel` | min_p=0 分支、min_p>=1、dtype 变体 | 边界未独立触发 |
| `_penalties_kernel` | 无 fp32 广参考；仅 vocab1000 | 大 vocab packed 未测 |
| `_gumbel_sample_kernel` | 采样正确性不定量对 logprobs/分布；random 无 RNG 对齐 | 统计口径宽松 |
| `_post_update_kernel` | 未用请求槽、依赖整数语义 | 边界槽位未验证 |
| `_temperature_kernel` | 无扩展 idx_mapping；dtype 变体 | 多 token 同 req 未测 |

### 5.3 建议

| 优先级 | 动作 | 针对 |
| --- | --- | --- |
| P1 | 为 `_bad_words_kernel` 补具体 mask 位置的数值参考（目前只查"是否改变"） | 盲区 |
| P1 | 核对 `_ranks_kernel` 的 `>` 与上游 `>=` 在等值 token 时的语义（linked to from_vllm 关注点） | 盲区 |
| P2 | 扩大 `_bincount_kernel`/`_compute_slot_mapping` 的尺寸与拓扑覆盖 | 盲区 |
| P2 | 对半精度与极端边界（min_p=0、num_logprobs=0、大 vocab packed）补用例 | 盲区 |
| P3 | 明确区分 `diagnose_bincount_atomic_or.py`（诊断）与正式精度测试，避免混淆 | 范围界定 |

> 总体评价：**vllm_ascend 内部 UT 本次运行结果健康（100% 通过）**，无需修复动作；建议后续重点补齐上表盲区以增强覆盖置信度，而不需要处理任何失败。