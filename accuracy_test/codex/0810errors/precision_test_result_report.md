# vLLM 内部已有 UT 运行结果误差报告（0810 from_vllm）

> 数据来源：`ut_in_vllm.txt`（`run_from_vllm.sh` 运行 30 个 from_vllm 测试文件的标准输出，7719 行）。
> 本报告**仅覆盖 vLLM 内部已实现的 UT（含 *_patch 版本）**，即 `existing_accuracy_tests/from_vllm/` 目录 30 个文件；不含 `from_vllm_ascend/`。
> 综合 `precision_test_analysis_report.md`（单测标准/方式/优缺点）与 `existing_tests_report.md`（静态元数据盘点）撰写，并在后者表格基础上**补齐 0810 实际运行结果**。

---

## 1. 环境与运行口径

| 项 | 值 |
| --- | --- |
| 运行命令 | `bash .../accuracy_test/codex/existing_accuracy_tests/run_from_vllm.sh` |
| 平台 | Linux, Python 3.11.10, pytest 8.3.2 |
| 硬件 | 昇腾 NPU(A3)；`torch.device("npu")` 直接 launch |
| 插件 | pytest-asyncio 1.3.0, xdist 3.6.1 |
| vLLM 平台插件 | ascend 已激活（`vllm_ascend:register`） |
| 执行方式 | 每次运行启动独立 pytest session（共 30 个 session，逐个文件） |
| 关键环境事实 | 安装的 vLLM 版本**未提供**多个 block-verification kernel 符号（`_compute_local_logits_stats_kernel`、`_compute_cumulative_log_p_kernel`、`_load_ptr` 等），导致对应 UT 在收集/运行期被 SKIP 或 ImportError |

## 2. 总体结果（30 文件 / 226 用例）

| 指标 | 数量 | 占比 |
| --- | --- | --- |
| 收集用例总数 | 226 | 100% |
| **PASSED（通过，精度判据满足）** | **154** | 68.1% |
| **FAILED（失败）** | **44** | 19.5% |
| **SKIPPED（未运行）** | **28** | 12.4% |

> 44 个失败按根因分四类（第 4 章细析）：
> - **数值不符合（6 例）**：global_logsumexp 全 -inf→nan(2)、fill_logprob custom_token_ids(4)。
> - **Triton 编译/API 不兼容（19 例）**：gumbel_block_argmax `PER_TOKEN_COL` 参数不匹配(3)、topk_topp `legalize unresolved materialization` 编译失败(16)。
> - **import/环境缺失→运行期错误（18 例）**：rejection_kernel_patch 全部 18 例 ImportError(`_compute_local_logits_stats_kernel` 不存在)。
> - **NPU 设备/AI Core 运行异常（1 例）**：tl_rand64 的 gumbel FP32 业务路径 `vector core exception`(507035)。

> 另 28 例 SKIPPED：多为「安装的 vLLM 版本较旧，未提供对应 kernel」。**SKIP≠通过**，其精度「未受验」。
---

## 3. 逐文件结果（补齐 `existing_tests_report.md` 结果列）

> 「结果」列三态：`PASS`（通过）、`FAIL`（失败+根因代号）、`SKIP`（未运行+原因）。根因代号见 §3.3。
> 用例数=该 session 收集 items 数（`collected N items`；`0/1 skipped` 表示收集阶段即整体跳过）。

### 3.1 from_vllm/ 30 文件全新结果表

| 文件 | 被测对象 | 调用类型 | items | 通过 | 失败 | 跳过 | 结果(逐类) | 根因 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test_apply_write_kernel | `_apply_write_kernel` | A | 10 | 4 | 0 | 6 | PASS(single) / SKIP(multi) | 旧版vLLM无 fused multi-group |
| test_bias_kernel | `_bias_kernel` | A | 9 | 9 | 0 | 0 | ALL PASS | — |
| test_compute_block_max_and_sumexp | `_compute_local_logits_stats_kernel` | A | 1 | 0 | 0 | 1 | SKIP(全部) | 符号不存在 |
| test_compute_block_max_and_sumexp_patch | Ascend `_compute_block_stats_kernel` | A | 3 | 3 | 0 | 0 | ALL PASS | — |
| test_compute_block_stats_kernel | `_compute_cumulative_log_p_kernel` | E | 1 | 0 | 0 | 1 | SKIP(全部) | 符号不存在 |
| test_compute_block_stats_kernel_patch | Ascend `_compute_block_stats_kernel` | A | 2 | 2 | 0 | 0 | ALL PASS | — |
| test_compute_global_logsumexp | `_compute_global_lse`(helper) | C | 8 | 7 | 1 | 0 | FAIL: test_all_neg_inf_blocks | **数值**: 全 -inf→nan |
| test_compute_global_logsumexp_patch | `_compute_global_lse`(Ascend) | C | 8 | 7 | 1 | 0 | FAIL: test_all_neg_inf_blocks | **数值**: 全 -inf→nan |
| test_fill_logprob_token_ids_kernel | `_fill_logprob_token_ids_kernel` | A | 10 | 6 | 4 | 0 | FAIL: custom_token_ids 3/5×batch4/8 | **数值**: 张量不等 |
| test_fill_logprob_token_ids_kernel_patch | `compute_topk_logprobs`(生产wrapper) | D | 4 | 4 | 0 | 0 | ALL PASS | — |
| test_flatten_sampled_kernel | `_flatten_sampled_kernel` | A | 14 | 14 | 0 | 0 | ALL PASS | — |
| test_gather_block_tables_kernel | `_gather_block_tables_kernel` | A | 14 | 14 | 0 | 0 | ALL PASS | — |
| test_gumbel_block_argmax | `gumbel_block_argmax`(helper) | C | 3 | 0 | 3 | 0 | ALL FAIL | **编译**: `PER_TOKEN_COL` kwarg 不匹配 |
| test_gumbel_block_argmax_patch | `_npu_gumbel_block_argmax`(Ascend) | C | 4 | 4 | 0 | 0 | ALL PASS | — |
| test_insert_resampled_kernel | `_insert_resampled_kernel` | A | 8 | 8 | 0 | 0 | ALL PASS | — |
| test_insert_resampled_kernel_patch | Ascend `_insert_resampled_kernel` | A | 8 | 8 | 0 | 0 | ALL PASS | — |
| test_load_ptr | `_load_ptr`(helper) | C | 1 | 0 | 0 | 1 | SKIP(全部) | 符号不存在 |
| test_prepare_dflash_inputs_kernel | `_prepare_dflash_inputs_kernel` | A | 1 | 0 | 0 | 1 | SKIP(全部) | 无 dflash 模块 |
| test_prepare_dflash_inputs_kernel_patch | Ascend `_prepare_dflash_inputs_kernel_ascend` | A | 1 | 0 | 0 | 1 | SKIP(全部) | 无 vllm_ascend dflash |
| test_prompt_logprobs_token_ids_kernel | `_prompt_logprobs_token_ids_kernel` | A | 10 | 10 | 0 | 0 | ALL PASS | — |
| test_rejection_kernel | `_rejection_kernel`(vanilla) | E | 1 | 0 | 0 | 1 | SKIP(全部) | 符号不存在 |
| test_rejection_kernel_patch | Ascend `_probabilistic_rejection_kernel` | E | 18 | 0 | 18 | 0 | ALL FAIL | **import**: `_compute_local_logits_stats_kernel` 不存在 |
| test_resample_kernel | `_resample_kernel`(vanilla) | A | 27 | 15 | 0 | 12 | PASS(15) / SKIP(12: block-ver 分支) | 旧版无 block-verification _resample_kernel |
| test_resample_kernel_patch | Ascend `_resample_kernel` | B | 2 | 2 | 0 | 0 | ALL PASS | — |
| test_scatter_num_accepted_kernel | `_scatter_num_accepted_kernel` | A | 10 | 10 | 0 | 0 | ALL PASS | — |
| test_selective_scan_update_kernel | `_selective_scan_update_kernel` | A | 7 | 7 | 0 | 0 | ALL PASS | — |
| test_tl_rand64 | `tl_rand64`(helper)+`gumbel_sample` | C+D | 5 | 4 | 1 | 0 | FAIL: test_fp32_business_path | **运行时**: AI Core vector exception(507035) |
| test_tl_rand64_patch | `tl_rand64`+Ascend `gumbel_sample` | C+D | 5 | 5 | 0 | 0 | ALL PASS | — |
| test_topk_topp_kernel | `_topk_topp_kernel` | A | 20 | 0 | 16 | 4 | FAIL(16) / SKIP(4: False-False noop) | **编译**: legalize 失败; noop 分支 SKIP |
| test_update_min_larger_stats | `_update_min_larger_stats`(helper) | C | 11 | 11 | 0 | 0 | ALL PASS | — |

### 3.2 结果雷达（被测算子维度，仅标注「真的执行过且通过」的）

| 被测算子 | 文件 | 结果 | 说明 |
| --- | --- | --- | --- |
| `_bias_kernel` | bias | PASS | 9/9 |
| `_compute_global_lse`(helper) | global_logsumexp(+patch) | FAIL | 全 -inf 边界 nan |
| `_fill_logprob_token_ids_kernel` | fill_logprob | FAIL | custom 多行映射不一致 |
| `_flatten_sampled_kernel` | flatten | PASS | 14/14 |
| `_gather_block_tables_kernel` | gather | PASS | 14/14 |
| `gumbel_block_argmax`(helper) | gumbel | FAIL | 编译 API 不匹配 |
| `_npu_gumbel_block_argmax`(Ascend) | gumbel_patch | PASS | 4/4 |
| `_insert_resampled_kernel` | insert(+patch) | PASS | 16/16 |
| `_prompt_logprobs_token_ids_kernel` | prompt_logprobs | PASS | 10/10 |
| `_resample_kernel` | resample | PASS(部分) | 15 通过, 12 未执行 |
| Ascend `_resample_kernel` | resample_patch | PASS | 2/2 |
| `_scatter_num_accepted_kernel` | scatter | PASS | 10/10 |
| `_selective_scan_update_kernel` | selective_scan | PASS | 7/7 |
| `tl_rand64`/`gumbel_sample` | tl_rand64(+patch) | FAIL(未patch) / PASS(patch) | 未patch 触发设备异常 |
| `_topk_topp_kernel` | topk_topp | FAIL | 编译 legalize 失败 |
| `_update_min_larger_stats`(helper) | update_min_larger | PASS | 11/11 |
---

## 4. 失败逐项详析（44 例，按根因分 4 类）

### 4.1 数值不符合（6 例）—— 需重点关注

**① `_compute_global_lse` 全 -inf 边界 → nan（2 例）**
- 文件：test_compute_global_logsumexp.py / _patch.py 的 `test_all_neg_inf_blocks`
- 断言：`assert output[0].item() == float("-inf")`
- 实际：`assert nan == -inf`（输出是 `tensor(nan)`）
- 说明：当所有块 max 均为 -inf（示意"无可选 token"）时，得到的 bix 应为 -inf，Ascend 实现返回 **nan**。这与 CPU 参考约定不符，属于**边界数值语义差异**。
- 影响评估：正常采样时全部块 -inf 罕见；但作为"全不可用"哨兵语义，nan 会在下游污染(如 argmax/log-softmax)。**判定为真实精度缺陷候选**，建议单独核查 `_compute_global_lse` 的 `tl.max`/`exp` 对 -inf 的归约实现。

**② `_fill_logprob_token_ids_kernel` 自定义 token 多行映射不一致（4 例）**
- 文件：test_fill_logprob_token_ids_kernel.py 的 `test_custom_token_ids[3-4 / 3-8 / 5-4 / 5-8]`
- 触发条件：`batch_size∈{4,8}`（> num_reqs=4）且 `topk∈{3,5}`，请求 0 携带 custom token
- 断言：`assert_close(out_token_ids, expected_ids, rtol=0, atol=0)` 精确
- 实际：Mismatched elements 9/68(13.2%)~30/136(22.1%)，最大绝对差 852~960（相对差 1.0）
- 根因线索：`expanded_idx_mapping = arange(batch_size) % num_reqs` 使多个 batch 行映射到**同一请求 0**（custom 归属者）；Ascend 端在"同一请求对应多行各自写入其 custom token"时的填充/覆盖语义与 CPU 参考 `_fill_logprob_token_ids_ref` 不一致：`[1-1]`(batch=1)、`[0-*]`、`[3-1]`、`[5-1]` 全 PASS，独独 batch>1 的 4 个失败 → **指向多行共享同一 custom 源时按行偏移/repeat 的索引错位**。**真实精度缺陷候选。**
- 建议：构造最小复现（batch=4, topk=3, req0 custom=[100,200,300]），逐行对照定位是"未重复写入"还是"填充到错列"。

### 4.2 Triton 编译 / API 不兼容（19 例）—— 测试与安装版本脱节

**③ `gumbel_block_argmax` API 参数不匹配（3 例）**
- 文件：test_gumbel_block_argmax.py 的 3 个测试全部
- 错误：`CompilationError: gumbel_block_argmax() got an unexpected keyword argument 'PER_TOKEN_COL'`
- 说明：测试的 wrapper 以 `PER_TOKEN_COL=...` 关键字调用，而**安装的 vLLM 该函数签名不含此参数**。是 UT 代码与所测 vLLM 版本签名的脱节，非精度问题。
- 处理：核对目标 vLLM 版本 `gumbel_block_argmax` 签名，更新 wrapper 参数或对齐版本。Ascend 版 `_npu_gumbel_block_argmax`（_patch）通过(4/4)，表明 Ascend 路径签名匹配。

**④ `_topk_topp_kernel` Triton legalize 编译失败（16 例）**
- 文件：test_topk_topp_kernel.py，16 个失败
- 错误：`loc(".../vllm/v1/sample/ops/topk_topp_triton.py":153:54) / (:646:34): error: failed to legalize unresolved materialization from () to 'tensor<1xf32>' that remained live after conversion`
- 说明：**Triton 后端（Ascend）无法 legalize 该 kernel 中的某条标量→tensor 物化**，编译期直接失败。属**后端编译器兼容缺陷**，kernel 根本没法编译运行，谈不上数值。
- 另 4 例 SKIP：`test_topk_topp_combined[False-False-*]`（topk 与 topp 都关闭的 noop）被跳过，未执行。
- 处理：属昇腾 Triton 编译器对 `topk_topp_triton.py:153/:646` 的兼容问题，需工具链侧修复或改写该处；期间该 kernel 精度**无法验证**。

### 4.3 import / 环境缺失 → 运行期错误（18 例）

**⑤ `_probabilistic_rejection_kernel`（rejection_kernel_patch）全部 18 例失败**
- 文件：test_rejection_kernel_patch.py 全部 18 个测试
- 错误：`ImportError: cannot import name '_compute_local_logits_stats_kernel' from 'vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils'`
- 位置：测试的 `_run_kernel`（内部 `from vllm... import _compute_local_logits_stats_kernel`），因懒加载故在**每个测试体**报错 → 18 例 FAIL。
- 同一符号在 test_compute_block_max_and_sumexp、test_rejection_kernel、test_compute_block_stats_kernel 里因在收集期 import 而**整体 SKIP**（3 例）。
- 根因：安装的 vLLM 该文件**不含** `_compute_local_logits_stats_kernel`（或已更名），Ascend patch 测试依赖的上游符号缺失。**非数值缺陷**，属版本/符号对齐问题。
- 处理：确认目标 vLLM 该 kernel 的正确符号名，修复 import；否则 rejection 系列精度**完全未验证**。

### 4.4 NPU 设备 / AI Core 运行异常（1 例）

**⑥ `tl_rand64` 的 `gumbel_sample` FP32 业务路径触发设备异常**
- 文件：test_tl_rand64.py 的 `TestTlRand64ViaGumbelSample::test_fp32_business_path`
- 错误：`RuntimeError: npuSynchronizeDevice ..., error code is 507035`；`[Error]: The vector core execution is abnormal.`；`EZ9999 ... exception of aivec error ... errorStr: Illegal instruction, which is usually caused by unaligned UUB addresses`；`retCode=0x31 [vector core exception]`
- 说明：运行 gumbel FP32 业务路径时昇腾 **AI Core（vector core）执行异常/非法指令**，属设备侧运行时故障（疑似对齐/指令问题），非数值不符合。
- 补注：同文件的 4 个统计/范围测试 PASS；`test_tl_rand64_patch.py`（Ascend gumbel 替换路径）**5 例全 PASS**——说明 Ascend 自家 gumbel 路径正常，问题集中在**上游 vLLM 的 gumbel_sample(FP32) 在昇腾上触发**。
- 处理：需昇腾侧排查（对照 ascend log），或改用 Ascend 自家 gumbel 路径（_patch 已通过）。

---

## 5. 汇总与结论

### 5.1 关键结论

1. **30 文件 226 用例中，性能/数值真正"通过且被执行"的为 154(68.1%)**；28(SKIP)+37(编译/import/运行时)=65 例**未完成数值校验**，即**约三成被测对象在这份日志里没有真正完成精度验证**。
2. 真正的**数值精度缺陷候选仅 6 例**：`_compute_global_lse` 全 -inf→nan(2)、`_fill_logprob_token_ids_kernel` custom 多行(4)。这些须优先排查。
3. 其余 38 例失败（gumbel 3 + topk_topp 16 + rejection 18 + tl_rand64 1）全部是**环境/版本/编译/设备**层面的问题，**不是 kernel 数值算错**，但同样导致"精度未验证"。
4. **Ascend 自有实现（_patch）普遍更稳**：block_max/-stats、gumbel、insert、fill、resample、tl_rand64 的 _patch 版均全 PASS，而上游 vanilla 版多处 SKIP/FAIL——印证现有精度测试套件更依赖 Ascend 替换路径。
5. **SKIPPED 的「block-verification」系列（resample 12 例 + compute_block_* 3 例 + rejection 系列）恰好是被测对象的"生产新路径"**，安装的 vLLM 版本未含这些 kernel → 这些新功能的精度**等于没测**，是最大的覆盖空洞。

### 5.2 建议动作（按优先级）

| 优先级 | 动作 | 针对 |
| --- | --- | --- |
| P0 | 定位并修复 `_fill_logprob_token_ids_kernel` custom 多行映射错位（最小复现 batch=4,topk=3） | 精度缺陷 ② |
| P0 | 修复 `_compute_global_lse` 全 -inf 返回 nan（对齐 CPU 参考为 -inf） | 精度缺陷 ① |
| P1 | 对齐 vLLM 版本符号：修复 rejection_patch / block_max / block_stats / rejection 的 `_compute_local_logits_stats_kernel` import，或升级/核对版 | import 18+3 |
| P1 | 核对 `gumbel_block_argmax` 签名，更新 wrapper `PER_TOKEN_COL` | 编译 3 |
| P1 | 昇腾侧排查 `topk_topp_triton.py:153/646` legalize 失败与 `tl_rand64` vector core 异常(507035) | 编译 16 + 运行时 1 |
| P2 | 安装含 block-verification kernel 的 vLLM 版本，补测 resample/compute_block_*/rejection 新路径 | SKIP 覆盖空洞 |