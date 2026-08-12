# codex 精度测试：算子调用类型（UT 对应关系）详解

> 目的：解释“helper”的含义，并列出全部被测算子/辅助函数的**调用类型**，以及每个 UT 文件对应的调用类型。
> 本文与 `precision_test_analysis_report.md` 配套使用，后者给出每个测试的标准/方式/流程/优缺点，本文聚焦“被测对象究竟是怎么被调用的”。

---

## 1. 首先讲清楚：什么是 helper？

在 Ascend Triton（`@triton.jit`）语境下，一个 `@triton.jit` 修饰的函数按**能否独立以网格（grid）launch** 分为两类：

| 类别 | 判定特征 | 能否独立 launch | 例子 |
| --- | --- | --- | --- |
| **kernel（内核）** | 顶层函数，内部用 `tl.program_id(...)` 取网格索引，通过 `kernel[grid](...)` 启动 | 能 | `_bias_kernel`、`_num_nans_kernel`、`_resample_kernel` |
| **helper（辅助函数）** | 非顶层、通常在 kernel 内部被内联调用；**没有 `tl.program_id`**，以返回值形式把结果交给调用方 | 不能（无独立网格） | `tl_rand64`、`gumbel_block_argmax`、`_compute_global_lse`、`_update_min_larger_stats`、`_load_ptr`、`_npu_gumbel_block_argmax`、`_compute_max_and_sumexp` |

> **关键结论：“helper” = 不能被网格直接 launch 的 Triton JIT 函数。**
> 它必须由某个 kernel 在内部调用（内联展开），或由**测试代码自己写一个本地 wrapper kernel** 包裹起来才能跑在 NPU 上。
>
> 因此凡是被测对象是 helper 的测试文件，都会在文件里 `@triton.jit` 定义一个 `*_wrapper` / `_wrapper_kernel` 之类的小 kernel，把 helper 调进去、把结果写进张量，再从 CPU/统计读回来验证。

> **文档里“wrapper”可能指两种不同含义，务必区分：**
> 1. **测试 wrapper kernel**：测试文件内自建、包裹 helper 的 `@triton.jit` 小 kernel（作用是让 helper 能独立跑在 NPU 上）。
> 2. **生产 wrapper 公共函数**：vLLM / vLLM-Ascend 里调用真实 kernel 的 Python 函数（如 `apply_temperature`、`gumbel_sample`、`compute_topk_logprobs`、`apply_min_p`、`apply_penalties`、`apply_bad_words`）。测试直接调用它，间接覆盖内部 kernel。

---

## 2. 调用类型总分类系

结合“被测对象本身是什么”和“测试怎么调它”，把全部 UT 的调用类型归纳为 **6 类**：

| 类型代号 | 类型名称 | 被测对象 | 测试如何调用 | 参考来源 |
| --- | --- | --- | --- | --- |
| **A** | 直接 launch kernel + CPU/PyTorch 参考 | 独立 kernel | `kernel[grid](...)` 直接启动 | 纯 Python/NumPy/PyTorch 串行参考 |
| **B** | 直接 launch kernel + 精确值断言 | 独立 kernel（整数/位/布尔） | `kernel[grid](...)` 直接启动 | `assert x.item()==...` / `torch.equal`（无浮点容差） |
| **C** | helper 包裹测试（测试 wrapper kernel） | helper（不可独立 launch 的 JIT 函数） | 测试文件内建 `_wrapper` kernel 包裹后 launch | CPU 参考 / 精确断言 / 统计校验 |
| **D** | 生产 wrapper 公共函数测试 | 会 launch kernel 的 Python wrapper | 调用 `apply_temperature` / `gumbel_sample` 等 | PyTorch 参考 / 行为 / 统计 |
| **E** | 多 kernel 级联测试 | 多个 kernel（先算前置 stats，再跑主 kernel） | 直接 launch 多个 kernel，串联数据 | CPU 参考 / 行为断言 |
| **F** | 两实现头对头比较 | Ascend 版 vs 上游版同一 kernel | 分别 launch 两个 kernel，输出互比 | `torch.equal` 另一实现输出 |

> 说明：
> - **E** 是 **A** 的扩展——主 kernel 依赖前置 kernel 生成的中间数据（如 rejection 需要先 `_compute_block_stats`）。
> - 一个测试文件可能**同时属于多类**（例如既直接 launch 主 kernel（B），又对 helper 用 wrapper（C），又调生产 wrapper（D）），下表会逐个标注“类型主/次”。

---

## 3. 全部算子 UT 的调用类型对照表

> 下表按目录分组，列满 6 类代号。`类型` 列第一个为主类型，后续为次类型。

### 3.1 missing_accuracy_tests/（补写测试）

| UT 文件 | 被测对象 | 对象本身 | 类型 | 说明 |
| --- | --- | --- | --- | --- |
| test_apply_grammar_bitmask_kernel_patch.py | `_apply_grammar_bitmask_kernel`（Ascend） | kernel | B | 直接 launch，位级精确断言 |
| test_combine_sampled_and_draft_tokens_kernel.py | `_combine_sampled_and_draft_tokens_kernel` | kernel | A | 直接 launch + CPU 参考（constexpr 用本地副本） |
| test_dcp_local_seq_lens_kernel.py | `_dcp_local_seq_lens_kernel` | kernel | A | 直接 launch + CPU 参考 |
| test_expand_idx_mapping_kernel.py | `_expand_idx_mapping_kernel` | kernel | A | 直接 launch + CPU 参考 |
| test_get_num_sampled_and_rejected_kernel.py | `_get_num_sampled_and_rejected_kernel` | kernel | A | 直接 launch + CPU 参考 |
| test_npu_gumbel_block_argmax_patch.py | `_npu_gumbel_block_argmax`（Ascend） | helper | C | 测试 wrapper kernel 包裹 |
| test_num_nans_kernel.py | `_num_nans_kernel` | kernel | A | 直接 launch + CPU 参考 |
| test_post_update_num_computed_tokens_kernel.py | `_post_update_num_computed_tokens_kernel` | kernel | A | 直接 launch + CPU 参考 |
| test_prepare_decode_inputs_kernel.py | `_prepare_decode_inputs_kernel` | kernel | B | 直接 launch + 原位精确断言 |
| test_prepare_dflash_inputs_kernel_ascend_patch.py | `_prepare_dflash_inputs_kernel_ascend`（Ascend） | kernel | A | 直接 launch + CPU 参考（10 输出精确） |
| test_prepare_pos_seq_lens_kernel.py | `_prepare_pos_seq_lens_kernel` | kernel | A | 直接 launch + CPU 参考 |
| test_prepare_prefill_inputs_kernel.py | `_prepare_prefill_inputs_kernel`（input_batch） | kernel | A | 直接 launch + CPU 参考 |
| test_prepare_prefill_inputs_kernel_speculator.py | `_prepare_prefill_inputs_kernel`（speculator） | kernel | A | 直接 launch + CPU 参考 |
| test_prepare_rope_positions_kernel.py | `_prepare_rope_positions_kernel` | kernel | A | 直接 launch + CPU 参考 |
| test_probabilistic_rejection_kernel_patch.py | `_probabilistic_rejection_kernel`（Ascend） | kernel | E | 先 `_compute_block_stats_kernel` 再主 kernel，行为/精确断言 |
| test_resample_kernel_patch.py | `_resample_kernel`（Ascend） | kernel | B | 直接 launch + 精确断言 |
| test_update_draft_inputs_kernel.py | `_update_draft_inputs_kernel` | kernel | A | 直接 launch + CPU 参考 |
| test_zero_kv_blocks_kernel_patch.py | `_zero_kv_blocks_kernel`（Ascend） | kernel | B | 直接 launch + 精确断言（data_ptr 绝对地址） |

### 3.2 existing_accuracy_tests/from_vllm/（从 vLLM 搬运/适配文件）

| UT 文件 | 被测对象 | 对象本身 | 类型 | 说明 |
| --- | --- | --- | --- | --- |
| test_apply_write_kernel.py | `_apply_write_kernel` | kernel | A | 直接 launch + CPU 参考 |
| test_bias_kernel.py | `_bias_kernel` | kernel | A | 直接 launch + CPU 参考 |
| test_compute_block_max_and_sumexp.py | `_compute_max_and_sumexp`（helper）经 `_compute_local_logits_stats_kernel` | helper（间接） | E | 直接 launch parent kernel，间接覆盖 helper |
| test_compute_block_stats_kernel.py | `_compute_cumulative_log_p_kernel` | kernel | E | 先 stats 再 cumulative log-p，CPU 参考 |
| test_compute_global_logsumexp.py | `_compute_global_lse` | helper | C | 测试 wrapper kernel 包裹 |
| test_fill_logprob_token_ids_kernel.py | `_fill_logprob_token_ids_kernel` | kernel | A | 直接 launch + CPU 参考 |
| test_flatten_sampled_kernel.py | `_flatten_sampled_kernel` | kernel | A | 直接 launch + CPU 参考 |
| test_gather_block_tables_kernel.py | `_gather_block_tables_kernel` | kernel | A | 直接 launch + CPU 参考（含 `_load_ptr` 间接寻址） |
| test_gumbel_block_argmax.py | `gumbel_block_argmax` | helper | C | 测试 wrapper kernel 包裹 |
| test_insert_resampled_kernel.py | `_insert_resampled_kernel` | kernel | A | 直接 launch + CPU 参考 |
| test_load_ptr.py | `_load_ptr` | helper | C | 测试 wrapper kernel 包裹 |
| test_prepare_dflash_inputs_kernel.py | `_prepare_dflash_inputs_kernel` | kernel | B | 直接 launch + 原位精确断言 |
| test_prompt_logprobs_token_ids_kernel.py | `_prompt_logprobs_token_ids_kernel` | kernel | A | 直接 launch + CPU 参考 |
| test_rejection_kernel.py | `_rejection_kernel` | kernel | E | 先 stats 再 rejection，行为断言 |
| test_resample_kernel.py | `_resample_kernel` | kernel | A | 直接 launch + CPU 参考（argmax 精确/max 容差） |
| test_scatter_num_accepted_kernel.py | `_scatter_num_accepted_kernel` | kernel | A | 直接 launch + CPU 参考 |
| test_selective_scan_update_kernel.py | `_selective_scan_update_kernel` | kernel | A | 直接 launch + PyTorch CPU 参考（`@heuristics`） |
| test_tl_rand64.py | `tl_rand64`（经 FP32 替代）及生产 `gumbel_sample` | helper + 生产 wrapper | C + D | 本地 wrapper 统计校验 + 生产 Gumbel 路径 |
| test_topk_topp_kernel.py | `_topk_topp_kernel` | kernel | A | 直接 launch + CPU 参考 + 属性断言 |
| test_update_min_larger_stats.py | `_update_min_larger_stats` | helper | C | 测试 wrapper kernel 包裹 |

### 3.3 from_vllm/ 下的 `_patch.py`（Ascend 改名/替换/再导出路径）

| UT 文件 | 被测对象 | 对象本身 | 类型 | 说明 |
| --- | --- | --- | --- | --- |
| test_compute_block_max_and_sumexp_patch.py | `_compute_block_stats_kernel`（Ascend 别名，间接覆盖 max/sumexp helper） | kernel（别名路径） | E | 直接 launch Ascend 别名，CPU 参考 |
| test_compute_block_stats_kernel_patch.py | `_compute_block_stats_kernel`（Ascend 别名） | kernel（别名路径） | E | 直接 launch Ascend 别名，CPU 参考 |
| test_compute_global_logsumexp_patch.py | `_compute_global_lse`（Ascend 导出） | helper | C | 测试 wrapper kernel 包裹 |
| test_fill_logprob_token_ids_kernel_patch.py | Ascend `compute_topk_logprobs`（tensor 替换原 Triton kernel） | 生产 wrapper（替换路径） | D | 调用 Python 函数，PyTorch 参考 |
| test_gumbel_block_argmax_patch.py | `_npu_gumbel_block_argmax`（Ascend） | helper | C | 测试 wrapper kernel 包裹 |
| test_insert_resampled_kernel_patch.py | `_insert_resampled_kernel`（Ascend 再导出） | kernel | A | 直接 launch + CPU 参考 |
| test_prepare_dflash_inputs_kernel_patch.py | `_prepare_dflash_inputs_kernel_ascend`（Ascend） | kernel | A | 直接 launch + CPU 参考 |
| test_rejection_kernel_patch.py | `_probabilistic_rejection_kernel`（Ascend） | kernel | E | 先 stats 再主 kernel，行为/精确断言 |
| test_resample_kernel_patch.py | `_resample_kernel`（Ascend） | kernel | B | 直接 launch + 精确断言 |
| test_tl_rand64_patch.py | `tl_rand64` FP32 替代 + 生产 `gumbel_sample` | helper + 生产 wrapper | C + D | 本地 wrapper 统计 + 生产路径 |

### 3.4 existing_accuracy_tests/from_vllm_ascend/（从 vLLM-Ascend 搬运文件）

| UT 文件 | 被测对象 | 对象本身 | 类型 | 说明 |
| --- | --- | --- | --- | --- |
| test_bad_words.py | `_bad_words_kernel` 经 `apply_bad_words` | 生产 wrapper | D | 行为/一致性校验（无数值 oracle） |
| test_bincount.py | `_bincount_kernel` | kernel | A | 直接 launch + 位级参考（`torch.equal`） |
| test_compute_slot_mapping.py | Ascend `_compute_slot_mappings_kernel` vs 上游 | kernel | F | 两实现头对头 equal |
| test_compute_topk_logprobs.py | `_topk_log_softmax_kernel`+`_ranks_kernel` 经 `compute_topk_logprobs` | 生产 wrapper | D | PyTorch 参考 |
| test_gumbel_sampling.py | `_gumbel_sample_kernel` 经 `gumbel_sample`/`apply_temperature` | 生产 wrapper | D | 数值/行为/统计混合 |
| test_log_softmax.py | `_topk_log_softmax_kernel` | kernel | A | 直接 launch + PyTorch 参考 |
| test_min_p.py | `_min_p_kernel` 经 `apply_min_p` | 生产 wrapper | D | PyTorch 参考（mask+值） |
| test_penality.py | `_penalties_kernel` 经 `apply_penalties` | 生产 wrapper | D | PyTorch 参考 |
| test_post_update.py | 上游 + Ascend `_post_update_kernel` | kernel（两实现） | F（+CPU oracle） | 头对头 + 独立 CPU 参考 |
| test_temperature.py | `_temperature_kernel` 经 `apply_temperature` | 生产 wrapper | D | PyTorch 参考 |
| diagnose_bincount_atomic_or.py | `_bincount_kernel` 原子原语探针 | kernel（诊断，非 pytest） | A/B（子进程隔离） | 精确断言 + 超时隔离 |

---

## 4. 各类别统计（48 个文件 + 1 诊断脚本）

| 类型 | 约计 | 典型代表 |
| --- | --- | --- |
| A（直接 launch + CPU/PyTorch 参考） | ~20 | input_batch 各 kernel、selective_scan、topk_topp、bincount |
| B（直接 launch + 精确断言） | ~7 | apply_grammar_bitmask、prepare_decode、zero_kv、resample_patch |
| C（helper 测试 wrapper 包裹） | ~7 | tl_rand64、gumbel_block_argmax、compute_global_lse、load_ptr、update_min_larger_stats |
| D（生产 wrapper 公共函数） | ~9 | apply_temperature、gumbel_sample、apply_min_p、apply_penalties、apply_bad_words、compute_topk_logprobs |
| E（多 kernel 级联） | ~7 | rejection/resample 系列（先 stats 再主 kernel） |
| F（两实现头对头） | 2 | compute_slot_mapping、test_post_update |

---

## 5. 关键判断口诀（如何一眼看调用类型）

1. 被测对象名以 `_xxx_kernel` 结尾且测试里出现 `_xxx_kernel[grid](...)` → **kernel**，先看参考：有 CPU 函数对比→**A**；只有 `==`/`torch.equal`/`torch.all(x==0)`→**B**。
2. 被测对象是单词（无 `_kernel`，如 `gumbel_block_argmax`、`_compute_global_lse`、`tl_rand64`、`_load_ptr`、`_update_min_larger_stats`）且测试文件里定义了一个 `_wrapper`/`_wrapper_kernel` → **helper，C 类**。
3. 测试直接 `from vllm_ascend... import apply_xxx` 或 `gumbel_sample`/`compute_topk_logprobs` 并调用它们（不带 `[grid]`）→ **生产 wrapper，D 类**。
4. 测试先 launch 一个 `_compute_block_stats_kernel`/`_compute_local_logits_stats_kernel` 再用其结果 launch 主 kernel → **多 kernel 级联，E 类**。
5. 测试同时 import Ascend 版与上游版同名 kernel 并各自 launch 互比 → **头对头，F 类**。

---

## 6. helper 清单汇总（本目录出现的所有 helper 及其作用）

| helper 名 | 所在源码 | 作用 | 被哪个 kernel 调用 | 测试文件 |
| --- | --- | --- | --- | --- |
| `gumbel_block_argmax` | vllm/gumbel.py | 块级 Gumbel-argmax（温度缩放+噪声） | `_gumbel_sample_kernel` | test_gumbel_block_argmax.py |
| `_npu_gumbel_block_argmax`（Ascend 替换） | vllm_ascend/rejection_sampler_utils.py | 同上，FP32 `tl.rand` | Ascend `_resample_kernel`/Gumbel | test_npu_gumbel_block_argmax_patch.py、test_gumbel_block_argmax_patch.py |
| `tl_rand64` | vllm/gumbel.py | FP64 随机均匀（A3 用 FP32 `tl.rand` 替代） | `gumbel_block_argmax` | test_tl_rand64.py、test_tl_rand64_patch.py |
| `_compute_global_lse`（旧名 `_compute_global_logsumexp`） | rejection_sampler_utils.py | 全局 logsumexp 归约 | `_rejection_kernel`、`_compute_cumulative_log_p_kernel` | test_compute_global_logsumexp.py、_patch.py |
| `_compute_max_and_sumexp` | rejection_sampler_utils.py | 块内 max+sumexp | `_compute_local_logits_stats_kernel` | test_compute_block_max_and_sumexp.py |
| `_update_min_larger_stats` | topk_topp_triton.py | top-k/top-p 的 pivot 之上最小合并 | `_topk_topp_kernel` | test_update_min_larger_stats.py |
| `_load_ptr` | buffer_utils.py | 间接寻址（ptr-to-ptr）读 | `_gather_block_tables_kernel`、`_apply_write_kernel` | test_load_ptr.py |

> 注：`_compute_block_stats_kernel` / `_compute_local_logits_stats_kernel` 本身是 **kernel**（可 launch），但内联使用上述 helper；`_compute_max_and_sumexp` 是它的 helper。


---

## 7. 深入解释：grid / wrapper（wrap）/ helper 到底是什么意思

> 本节用**实际源码地址 + 具体算子**把三个概念讲透。涉及两套源码根目录：
> - vLLM 上游：`C:\Users\x30084275\Desktop\git\vllm\vllm\...`
> - vLLM-Ascend：`C:\Users\x30084275\Desktop\git\vllm-ascend-xyz\vllm_ascend\...`

---

### 7.1 grid —— 网格启动参数，决定“开多少线程块、每个块做哪一块数据”

在 `kernel[grid](...)` 语法里，`grid` 是一个 tuple，声明本次 launch 的 **program（线程块）拓扑**。kernel 内部用 `tl.program_id(axis)` 取当前块的索引，据此决定它处理数据中的哪一段。

**关键点：**
- `grid` 的每个维度对应一个 `tl.program_id(axis)`（axis=0,1,2）。
- 一个块内再用 `tl.arange(0, BLOCK_SIZE)` 展开成若干 lane，做向量化。
- 同一 kernel 可以以不同 `grid` 启动（如行数多少、是否分块处理 vocab）。

**具体例子（含地址）：**

1. `_num_nans_kernel` 每个 program 负责一个请求（一行），`grid = (num_reqs,)`：
   - 源：`vllm\vllm\v1\worker\gpu\metrics\logits.py:10`（定义），`:35`（launch）
   - launch 语句：`_num_nans_kernel[(num_reqs,)](logits, ...)`
   - 即 `grid=(num_reqs,)`，一个块处理一行的全部 vocab（块内 `tl.arange(0, BLOCK_SIZE)` 循环）。

2. `_bias_kernel` 每个 program 负责一个 token，`grid = (num_tokens,)`：
   - 源：`vllm\vllm\v1\worker\gpu\sample\logit_bias.py:148`（定义），`:264`（launch）
   - launch：`_bias_kernel[(num_tokens,)](...)`

3. 二维 grid：`_gumbel_sample_kernel` 用 `grid=(num_tokens, num_blocks)`，即第一个维度是 token、第二个维度是 vocab 分块；每个 program 内部用 `tl.program_id(0)`（token）与 `tl.program_id(1)`（block）定位：
   - 源：`vllm\vllm\v1\worker\gpu\sample\gumbel.py:162`（定义），`:241`（launch `_gumbel_sample_kernel[(num_tokens, num_blocks)]`）

4. rejection/resample 系列常用网格 `(num_reqs, num_blocks)` 或 `(num_logits, vocab_num_blocks)`，前者处理请求，后者处理 logit×块统计。

---

### 7.2 wrapper —— “包装层”，分两层含义

#### 7.2.1 生产环境里的 wrapper（Python 公共函数包装 kernel）

Ascend 侧很多 Triton kernel 不是被上层直接 launch，而是被一个 **Python 公共函数**包装，这个函数负责：预分配/准备张量 → 按 `[grid]` launch kernel → 返回结果。测试里这类函数常叫 `apply_xxx` 或 `xxx_sample`。

**具体例子（含地址）：**

1. `apply_temperature` 包装 `_temperature_kernel`：
   - Ascend 源：`vllm-ascend-xyz\vllm_ascend\worker\v2\sample\gumbel.py:50`（定义 `apply_temperature`），`:65`（内部 `_temperature_kernel[(num_tokens, num_blocks)](...)`）
   - 调用链：用户代码 → `apply_temperature(logits, idx_mapping, temperature)` → launch `_temperature_kernel`。

2. `gumbel_sample` 包装 `_gumbel_sample_kernel`：
   - Ascend 源：`vllm-ascend-xyz\vllm_ascend\worker\v2\sample\gumbel.py:156`（定义 `gumbel_sample`），`:184`（内部 `_gumbel_sample_kernel[(num_tokens, num_blocks)](...)`）
   - 调用链：上层采样 → `gumbel_sample(...)` → launch `_gumbel_sample_kernel`；kernel 内再调用 helper（见 7.3）。

**测试意义：** 实测文件里写 `from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample` 然后直接 `gumbel_sample(...)`，属于 **D 类（生产 wrapper 测试）**，间接覆盖内部 kernel。

#### 7.2.2 测试环境里的 wrapper kernel（本地自建、包裹 helper）

当被测对象是 **helper**（不能独立 launch）时，测试文件里会 `@triton.jit` 定义一个极小的“wrapper kernel”：给它一个 grid、把 helper 调进去、把返回值 `tl.store` 到输出张量，从而让 helper 能在 NPU 上跑。

**具体例子（含地址）：**
- `test_tl_rand64.py` 内定义 `_tl_rand64_wrapper`（`@triton.jit`），`grid=(NUM_SAMPLES,)`，每个 program 调用 `_tl_rand64_a3(seed, offset, ...)`（helper）并 `tl.store`。
- `test_compute_global_logsumexp.py` 内定义 `_global_logsumexp_wrapper`，`grid=(1,)`，调用 helper `_compute_global_lse(...)` 并 store。
- `test_update_min_larger_stats.py` 内定义 `_update_min_larger_wrapper`，调用 helper `_update_min_larger_stats(...)` 并 store。
- `test_load_ptr.py` 内定义 `_load_ptr_wrapper_kernel` / `_load_float_ptr_kernel` / `_load_and_store_multi_kernel`。

**一句话总结 wrapper：** 生产 wrapper 是“真实 kernel 的 Python 门面”；测试 wrapper kernel 是“让 helper 能跑起来的小壳”。

---

### 7.3 helper —— 不能独立 launch、被 kernel 内联调用的 Triton JIT 纯函数

helper 是 `@triton.jit` 函数，但**没有 `tl.program_id`、不能以 `[grid]` 启动**；它以返回值把结果交给调用它的 kernel。它要么被某个真实 kernel 内联展开，要么被测试 wrapper kernel 包裹。

**具体例子（含地址及“由谁调用”）：**

1. `gumbel_block_argmax`（helper）→ 被 `_gumbel_sample_kernel`（kernel）调用：
   - 定义：`vllm\vllm\v1\worker\gpu\sample\gumbel.py:85`（`def gumbel_block_argmax(...)`）
   - 调用点：同文件 `:193`（`value, idx = gumbel_block_argmax(...)`，在 `_gumbel_sample_kernel` 内）
   - Ascend 等价替换：`_npu_gumbel_block_argmax`，`vllm-ascend-xyz\...\rejection_sampler_utils.py:34`

2. `tl_rand64`（helper）→ 被 `gumbel_block_argmax`（helper）调用 → 最终被 `_gumbel_sample_kernel`（kernel）调用：
   - 定义：`vllm\vllm\v1\worker\gpu\sample\gumbel.py:62`（`def tl_rand64(...)`）
   - 调用点：同文件 `:142`（`u = tl_rand64(gumbel_seed, block, includes_zero=False)`，在 `gumbel_block_argmax` 内）
   - 链：`_gumbel_sample_kernel(kernel) → gumbel_block_argmax(helper) → tl_rand64(helper)`
   - A3 用 FP32 `tl.rand` 替代：`vllm-ascend-xyz\...\gumbel.py:143`

3. `_compute_global_lse`（helper，旧名 `_compute_global_logsumexp`）→ 被 `_probabilistic_rejection_kernel`（Ascend kernel）调用：
   - 导入/别名：`vllm-ascend-xyz\...\rejection_sampler_utils.py:23`（`_compute_global_logsumexp as _compute_global_lse`）
   - 调用点：同文件 `:277`、`:296`（`target_lse = _compute_global_lse(...)`、`draft_lse = _compute_global_lse(...)`，在 `_probabilistic_rejection_kernel` 内）

4. `_compute_max_and_sumexp`（helper）→ 被 `_compute_local_logits_stats_kernel`（kernel）调用：
   - 定义与调用均在 `vllm\vllm\v1\worker\gpu\spec_decode\rejection_sampler_utils.py`（块内 max+sumexp；Ascend 以 `_compute_block_stats_kernel` 别名导入，`rejection_sampler_utils.py:26`）

5. `_update_min_larger_stats`（helper）→ 被 `_topk_topp_kernel`（kernel）调用：
   - 定义：`vllm\vllm\v1\sample\ops\topk_topp_triton.py`（`_update_min_larger_stats`）
   - 测试：`test_update_min_larger_stats.py` 用本地 wrapper kernel 包裹

6. `_load_ptr`（helper）→ 被 `_gather_block_tables_kernel` / `_apply_write_kernel`（kernel）调用：
   - 定义：`vllm\vllm\v1\worker\gpu\buffer_utils.py`（`_load_ptr`）
   - 测试：`test_load_ptr.py` 用本地 wrapper kernel 包裹

---

### 7.4 三段式调用关系的直观示例（Gumbel 采样链路）

以“温度采样 + Gumbel 采样”为例，把三层关系串起来：

```
[上层业务代码]
   │  调用公共 wrapper
   ▼
apply_temperature(logits, idx_mapping, temperature)      # 生产 wrapper（Python 函数）
   │  vllm-ascend-xyz\...\gumbel.py:50
   ▼
_temperature_kernel[(num_tokens, num_blocks)](...)        # kernel（grid=(num_tokens,num_blocks)）
   │  gumbel.py:65
   ...
[再次进入]
gumbel_sample(logits, idx_mapping, temperature, seed, pos)  # 生产 wrapper
   │  gumbel.py:156
   ▼
_gumbel_sample_kernel[(num_tokens, num_blocks)](...)       # kernel
   │  gumbel.py:184
   ▼（kernel 内部）
gumbel_block_argmax(...)                                    # helper（内联调用）
   │  gumbel.py:193（helper 定义在 :85）
   ▼（helper 内部）
tl_rand64(seed, block, includes_zero)                      # helper（内联调用）
   │  gumbel.py:142（helper 定义在 :62）
```

**测试对应的三档粒度：**
- 测 `tl_rand64` / `gumbel_block_argmax`：文件内建测试 wrapper kernel 包裹（== C 类）。
- 测 `_gumbel_sample_kernel` / `_temperature_kernel`：直接 `[grid]` launch（== A 类），或经 `gumbel_sample`/`apply_temperature` 走生产 wrapper（== D 类）。

---

### 7.5 一眼判断：某个名字是 kernel 还是 helper

| 现象 | 结论 |
| --- | --- |
| 源码里出现 `某函数[grid](...)` 或定义体内有 `tl.program_id` | **kernel** |
| 定义体内只有 `return 值`、调用方写 `x = 函数(...)`，且无 `[grid]` | **helper** |
| 测试文件里 `from vllm_ascend... import apply_xxx` / `gumbel_sample` 并直接调用 | 被测的是**生产 wrapper**（内部 launch kernel） |
| 测试文件里 `@triton.jit def *_wrapper(...)` 包裹某个名字再 launch | 既有**测试 wrapper kernel**，被包裹的通常是 **helper** |
