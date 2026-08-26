# strict_ut 精度 UT 失败原因分析（026 批次，修复后归档于 028）

> 本文记录 strict_ut_026 快照中 5 个精度 UT 失败实例（4 个用例）的完整根因分析，
> 明确每一处是**上游（vLLM 主仓）**还是**下游（vllm-ascend 适配仓）**的变更引起，
> 以及对应的修复方式。所有修复已落入本目录（strict_ut_028）。
>
> 所有 PR 编号与合入日期均取自本地 vllm / vllm-ascend 仓 git log（2026-08-26 查证）。

---

## 0.5 相关 PR 时间线（按合入日期排序）

| 合入日期 | 仓 | PR | 标题 | 对本批 UT 的影响 |
|---|---|---|---|---|
| 2026-06-27 | 上游 vllm | [#46878](https://github.com/vllm-project/vllm/pull/46878) | [Model Runner V2][Spec Decode] Use fp32 uniform threshold for acceptance | 引入 fp32 均匀阈值概率采样（`tl_rand32`），"恒接受"简化实现的消除自此开始 |
| 2026-06-30 | 上游 vllm | [#46781](https://github.com/vllm-project/vllm/pull/46781) | [Model Runner V2][Spec Decode] Implement block verification for rejection sampling | 引入 `_compute_local_logits_stats_kernel`（失败 #1 的被测对象） |
| 2026-07-13 | 下游 vllm-ascend | [#11765](https://github.com/vllm-ascend/vllm-ascend/pull/11765) | [Feature] Add qwen/glm dspark for mrv1 | DSpark kernel 引入，`ops/triton/spec_decode/utils` modern 路径的前身 |
| 2026-07-24 | 下游 vllm-ascend | [#12612](https://github.com/vllm-ascend/vllm-ascend/pull/12612) | [Bugfix] Fix multi dp in dspark | DSpark 演进 |
| 2026-07-28 | 下游 vllm-ascend | [#12791](https://github.com/vllm-ascend/vllm-ascend/pull/12791) | [CI] Add full ut protection for dspark | DSpark kernel 定型为 `copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid` |
| 2026-07-30 | 上游 vllm | [#48892](https://github.com/vllm-project/vllm/pull/48892) | [Model Runner V2][Spec Decode] Add multi-layer MTP speculator | `_prepare_prefill_inputs_kernel` 新增 lookahead 参数 → **失败 #2** |
| 2026-08-05 | 上游 vllm | [#50910](https://github.com/vllm-project/vllm/pull/50910) | [Model Runner V2] Cache draft logits in model's LM head dtype | draft logits 改存除温前值，kernel 加载后 `/ temp` → **失败 #1** |
| 2026-08-07 | 上游 vllm | [#46727](https://github.com/vllm-project/vllm/pull/46727) | thinking_budget 相关 | 环境约束（UT 运行环境需含此 PR，非本批直接根因） |
| 2026-08-14 | 下游 vllm-ascend | [#13191](https://github.com/vllm-ascend/vllm-ascend/pull/13191) | [Performance] Optimized Kernel copy_and_expand_dflash_and_dspark_inputs_kernel | dflash/dspark kernel 模块重组与优化 → **失败 #3a** |
| 2026-08-17 | 上游 vllm | [#52188](https://github.com/vllm-project/vllm/pull/52188) | [Spec decode] Support Kimi-K3 DCP with DSpark | kernel 新增 DCP 参数 `cp_rank`/`CP_SIZE`/`CP_INTERLEAVE` → **失败 #3b** |
| 2026-08-18 | 下游 vllm-ascend | [#13470](https://github.com/vllm-ascend/vllm-ascend/pull/13470) | [Bugfix][MRV2] Support probabilistic rejection sampling for spec decode | NPU `_probabilistic_rejection_kernel` 从 u=0 恒接受改为真实概率采样 → **失败 #4** |
| 2026-08-24 | 下游 vllm-ascend | [#14746](https://github.com/vllm-ascend/vllm-ascend/pull/14746) | [Ci] main2main vllm 0821 | main2main 同步（dflash speculator 跟随上游 #52188 的 DCP 签名等） |

---

## 0. 阅读指南（术语速查，面向 0 基础读者）

| 术语 | 含义 |
|---|---|
| **上游（upstream）** | vLLM 主项目（`vllm/` 仓），运行在 GPU/CUDA 上的参考实现，是事实标准 |
| **下游（downstream）** | vllm-ascend 适配项目（`vllm-ascend/` 仓），把 vLLM 移植到昇腾 NPU，API/语义需跟随上游 |
| **UT（单元测试）** | 本目录 `gpu/`、`npu/` 下的 pytest 用例：构造输入 → 跑 kernel → 与 **CPU 参考实现**（用纯 PyTorch 重写的期望结果）逐元素比对 |
| **CPU 参考实现** | 测试文件里手写的 `_xxx_ref` 函数，模拟 kernel "应该" 算出什么。kernel 改了而 ref 没改 → 误报失败 |
| **philox / tl_rand32** | GPU/NPU 上的确定性随机数发生器。kernel 内用它画均匀随机数 u ∈ (0,1) |
| **温度缩放（/ temp）** | 采样前把 logits 除以温度。temp=0 表示贪婪采样（取 argmax） |
| **MTP / lookahead** | Multi-Token Prediction 推测解码：一次 prefill 顺带写入后续 lookahead token |
| **DCP** | Decode Context Parallel（解码上下文并行），把解码期序列切到多卡 |

**本批失败的共同本质**：项目（上游 vLLM / 下游 vllm-ascend）更新后，kernel 的
**签名**或**行为**变了，而旧 UT 仍按旧签名传参、或 CPU 参考仍按旧行为计算。
属于"UT 与 kernel 版本不匹配"，**不是 kernel 精度 bug**——唯一的例外见 3.4：
下游 kernel 漏除温是一处真实缺陷，已在 kernel 侧修复。

---

## 1. 结论速览

| # | 失败用例 | 报错现象 | 根因 | 变更来源（PR / 日期） | 修复位置 |
|---|---|---|---|---|---|
| 1 | `TestComputeBlockMaxAndSumexp::test_compute_block_max_and_sumexp[True-1-128-4]` | `AssertionError: Scalars are not close!` | 上游 kernel 对 draft logits 加了 `/ temp` 温度缩放，CPU 参考未同步 | **上游 vLLM** PR #50910（2026-08-05 合入） | NPU/GPU UT 的 CPU 参考实现 |
| 2 | `TestPreparePrefillInputsKernel::test_prepare_prefill_inputs[1-1]` | `TypeError: dynamic_func() missing 3 required positional arguments: 'prefill_lens_ptr', 'num_computed_tokens_ptr', 'LOOKAHEAD_BLOCK'` | 上游 kernel 为 MTP lookahead 扩展了签名，UT 按旧签名传参错位 | **上游 vLLM** PR #48892（2026-07-30 合入） | NPU/GPU UT 的 kernel 调用与 CPU 参考 |
| 3a | `TestPrepareDFlashInputsKernelAscendPatch::test_prepare_dflash_inputs[False-3-1]` | 环境兼容性失败（探测不到 kernel） | 下游模块重组 + kernel 符号重命名，UT 探测列表过旧；旧 vLLM 环境缺新模块 | **下游 vllm-ascend** PR #13191（2026-08-14 合入）等 DSpark 系列 | UT 的 kernel 符号探测列表 |
| 3b | `TestPrepareDFlashInputsKernelAscendPatch::test_prepare_dflash_inputs[False-3-1]` | `TypeError: dynamic_func() missing 3 required positional arguments: 'cp_rank', 'CP_SIZE', 'CP_INTERLEAVE'` | 上游 PR #52188 给 kernel 新增 DCP 参数，下游为 main2main 兼容跟进签名 | **上游 vLLM** PR #52188（2026-08-17 合入），下游 #14746（2026-08-24 main2main 跟进） | UT 参数探测 + 动态传参 |
| 4 | `TestProbabilisticRejectionKernelPatch::test_non_greedy_always_accept` | `AssertionError: Expected 2 accepted, got 1` | 下游 kernel 从"u=0 恒接受"简化实现改为真实概率采样（对齐上游 `tl_rand32` 语义）；另发现下游漏除温缺陷 | **下游 vllm-ascend** PR #13470（2026-08-18 合入，对齐上游 #46878/#46781 的 2026-06-27/06-30 语义）+ 下游缺陷 | UT 的 CPU 参考 + 测试数据；**kernel 侧补 / temp** |

---

## 2. 变更传播链路：为什么"上游一改，下游 UT 跟着炸"

```
┌─────────────────────────────────────────────────────────────────────┐
│  上游 vLLM 主仓（vllm/）                                             │
│  ─ PR：draft logits 温度缩放 / lookahead 签名 / PR #52188 DCP 参数  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  下游跟随（main2main 兼容：签名必须一致，
                           │  否则 vllm-ascend 无法对接上游调用方）
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  下游 vllm-ascend 仓（vllm_ascend/）                                 │
│  ─ 自有 kernel（_probabilistic_rejection_kernel、                    │
│    _prepare_dflash_inputs_kernel_ascend）按上游语义/签名更新          │
│  ─ 模块重组：dflash kernel 移入 ops/triton/spec_decode/utils 并重命名 │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  strict_ut 的 NPU 用例里，
                           │  一部分直接 import 上游 kernel，
                           │  一部分 import 下游 kernel
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  strict_ut 精度 UT（本目录）                                         │
│  ─ 旧 UT：旧签名传参 → TypeError（#2、#3b）                          │
│  ─ 旧 UT：CPU 参考按旧行为 → 数值对不上（#1、#4）                     │
│  ─ 旧 UT：探测不到重命名后的 kernel → 环境兼容性失败（#3a）            │
└─────────────────────────────────────────────────────────────────────┘
```

关键点：**UT 导入哪个仓的 kernel，决定了变更源头是谁**。

| UT 文件（npu/） | 导入的 kernel | 来源仓 |
|---|---|---|
| `test_compute_local_logits_stats_kernel.py` | `_compute_local_logits_stats_kernel` ← `vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils` | 上游 |
| `test_input_batch_prepare_prefill_inputs_kernel.py` | `_prepare_prefill_inputs_kernel` ← `vllm.v1.worker.gpu.input_batch` | 上游 |
| `test_prepare_dflash_inputs_kernel.py` | `_prepare_dflash_inputs_kernel_ascend` ← `vllm_ascend.worker.v2.spec_decode.dflash.speculator` | 下游（签名跟随上游 PR #52188） |
| `test_rejection_kernel.py` | `_probabilistic_rejection_kernel` ← `vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils` | 下游（语义对齐上游 `_rejection_kernel`） |

---

## 3. 逐项详细分析

### 3.1 `test_compute_block_max_and_sumexp[True-1-128-4]` — 上游：draft logits 温度缩放

**变更来源：上游 vLLM PR [#50910](https://github.com/vllm-project/vllm/pull/50910)**
（[Model Runner V2] Cache draft logits in model's LM head dtype，
**2026-08-05 合入**，commit `81bc196913`）

**变更内容**：该 PR 为让 draft logits 以模型 LM head 的原始 dtype 缓存，把 draft logits
的存储约定改为"**除温前的原始值**"，kernel 加载后必须先 `/ temp` 再算统计量
（本次 PR diff 中可见 `_compute_local_logits_stats_kernel`、`_rejection_kernel`、
`_resample_kernel` 三处同步加 `/ temp`）。上游源码（带官方注释）：

```python
# vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
# _compute_local_logits_stats_kernel 内部：
if HAS_DRAFT_LOGITS:
    # draft_logits is stored pre-temperature, so apply scale first.
    draft_logits = (
        tl.load(
            draft_logits_ptr + ... + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        / temp          # ← 上游新增：先除温，再算 max / sumexp
    )
    draft_max, draft_sumexp = _compute_max_and_sumexp(draft_logits)
```

**失败机理**（`has_draft_logits=True` 的参数组合才触发）：

```
kernel：  draft_block / temp  →  max, sumexp     （新行为）
CPU ref： draft_block         →  max, sumexp     （旧行为，未除温）
                        ↓
        temp ≠ 1 时两边的 max / sumexp 数值不同
                        ↓
        AssertionError: Scalars are not close!
```

**修复**（[npu/test_compute_local_logits_stats_kernel.py](npu/test_compute_local_logits_stats_kernel.py)，GPU 侧同步修改）：

```python
# CPU 参考实现：draft logits 与 kernel 一致，先除温再计算
if has_draft_logits:
    drf_block = drf_cpu[rs, draft_step, start:end] / temp   # ← 修复点
    drf_max, drf_sumexp = _compute_max_and_sumexp_ref(drf_block)
```

---

### 3.2 `test_prepare_prefill_inputs[1-1]` — 上游：lookahead 签名扩展

**变更来源：上游 vLLM PR [#48892](https://github.com/vllm-project/vllm/pull/48892)**
（[Model Runner V2][Spec Decode] Add multi-layer MTP speculator，
**2026-07-30 合入**，commit `dec13a33b7`）

**变更内容**：上游为 MTP/lookahead 支持扩展了 `_prepare_prefill_inputs_kernel`
签名，新增（部分为位置参数，插在中间导致旧调用**参数错位**而非简单缺参）：

| 新参数 | 作用 |
|---|---|
| `next_prefill_tokens` + `next_prefill_tokens.stride(0)` | 后续 lookahead token 及其行跨度 |
| `num_lookahead` | lookahead token 数量 |
| `prefill_lens_ptr`、`num_computed_tokens_ptr` | prefill 长度 / 已算 token 数（决定实际写入量） |
| `LOOKAHEAD_BLOCK: tl.constexpr` | lookahead 的 2 的幂 padded 块大小 |

上游调用方式：

```python
# vllm/v1/worker/gpu/input_batch.py
_prepare_prefill_inputs_kernel[(num_reqs,)](
    input_ids, next_prefill_tokens, next_prefill_tokens.stride(0), num_lookahead,
    idx_mapping, query_start_loc, all_token_ids, all_token_ids.stride(0),
    prefill_lens, num_computed_tokens,
    BLOCK_SIZE=1024,
    LOOKAHEAD_BLOCK=triton.next_power_of_2(num_lookahead),   # ← 新 constexpr
)
```

**失败现象**：

```
TypeError: dynamic_func() missing 3 required positional arguments:
'prefill_lens_ptr', 'num_computed_tokens_ptr', and 'LOOKAHEAD_BLOCK'
```

**修复**（[npu/test_input_batch_prepare_prefill_inputs_kernel.py](npu/test_input_batch_prepare_prefill_inputs_kernel.py)，GPU 侧同步）：
UT 的三处 kernel 调用全部改为上游新签名（含 `LOOKAHEAD_BLOCK=triton.next_power_of_2(num_lookahead)`），
同时 CPU 参考实现扩展为支持多 lookahead token 的写入语义
（`num_lookahead > 0` 时逐个写入 `next_prefill_tokens`，并尊重
`prefill_lens - num_computed_tokens` 的实际写入上限）。

---

### 3.3 `test_prepare_dflash_inputs[False-3-1]` — 两个错误实例，两种来源

#### 3.3a 环境兼容性失败 — 下游模块重组 + 符号重命名

**变更来源：下游 vllm-ascend**（DSpark 系列演进：PR
[#11765](https://github.com/vllm-ascend/vllm-ascend/pull/11765) 2026-07-13 引入
DSpark → [#12612](https://github.com/vllm-ascend/vllm-ascend/pull/12612) 2026-07-24
修复多 DP → [#12791](https://github.com/vllm-ascend/vllm-ascend/pull/12791)
2026-07-28 kernel 定型 →
[#13191](https://github.com/vllm-ascend/vllm-ascend/pull/13191) 2026-08-14
优化重组，commit `acbd2bb28`）。

DFlash 输入准备 kernel 在下游经历了模块重组与重命名：

```
旧：vllm_ascend.worker.v2.spec_decode.dflash.speculator
        ._prepare_dflash_inputs_kernel_ascend
新：vllm_ascend.ops.triton.spec_decode.utils
        .copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid   （DFlash/DSpark 合并版）
        .copy_and_expand_dflash_inputs_kernel_single_grid              （旧名单体版）
```

旧环境（旧 vLLM）缺少新依赖模块、且 UT 的符号探测列表没跟上重命名，
导致探测失败、用例直接报环境兼容性错误。

**修复**（[npu/test_prepare_dflash_inputs_kernel.py](npu/test_prepare_dflash_inputs_kernel.py)）：
kernel 符号探测改为**多路径降级探测**——先试 legacy worker-v2 的
`_prepare_dflash_inputs_kernel_ascend`，失败再试 modern
`ops.triton.spec_decode.utils` 的合并版/单体版 kernel，并记录
`_dflash_import_errors` 便于诊断；DSpark 合并版才支持
`sample_from_anchor=True`。

#### 3.3b 缺 `cp_rank / CP_SIZE / CP_INTERLEAVE` — 上游 PR #52188（下游跟随签名）

**变更来源：上游 vLLM PR [#52188](https://github.com/vllm-project/vllm/pull/52188)**
（[Spec decode] Support Kimi-K3 DCP with DSpark，**2026-08-17 合入**，commit
`d1e3eee6fb`；下游 vllm-ascend 经 #14746 2026-08-24 main2main 跟进签名）。
下游源码注释写得很清楚：

```python
# vllm_ascend/worker/v2/spec_decode/dflash/speculator.py
# main2main compat: upstream ``_prepare_dflash_inputs_kernel`` added
# ``cp_rank``/``CP_SIZE``/``CP_INTERLEAVE`` for DCP support (see
# vllm-project/vllm#52188). The extra parameters are unused: Ascend does not
# run dflash with DCP (CP_SIZE is always 1, where upstream ``cp_local_slot``
# yields the same slots as this kernel).
```

即：上游为 DCP 加了 3 个参数；昇腾实际不跑 DCP（CP_SIZE 恒 1，结果与上游一致），
但**签名必须保持一致**（main2main 兼容），于是下游 kernel 也多了这 3 个参数。
旧 UT 按旧签名调用 → `TypeError`。

**失败现象**：

```
TypeError: dynamic_func() missing 3 required positional arguments:
'cp_rank', 'CP_SIZE', and 'CP_INTERLEAVE'
```

**修复**（[npu/test_prepare_dflash_inputs_kernel.py](npu/test_prepare_dflash_inputs_kernel.py)）：
按项目约束"UT 不改 kernel、通过参数探测适配版本差异"，做**签名探测 + 动态传参**：

```python
# vllm-project/vllm#52188 added context-parallel (DCP) args to the kernel:
# cp_rank (positional) plus CP_SIZE / CP_INTERLEAVE (constexpr). vllm-ascend
# 0.27.1 still ships the old signature, so probe before launching.
_DFLASH_KERNEL_HAS_DCP_ARGS = (
    _prepare_dflash_inputs_kernel_ascend is not None
    and "cp_rank" in getattr(_prepare_dflash_inputs_kernel_ascend, "arg_names", ())
)

# 发射处：
if _DFLASH_KERNEL_HAS_DCP_ARGS:
    launch_kwargs.update(cp_rank=0, CP_SIZE=1, CP_INTERLEAVE=False)
```

`cp_rank=0, CP_SIZE=1, CP_INTERLEAVE=False` 与"不跑 DCP"等价，新旧签名下
数值结果完全一致，探测式写法同时兼容带/不带 DCP 参数的 vllm-ascend 版本。

---

### 3.4 `test_non_greedy_always_accept` — 下游 kernel 语义对齐上游 + 下游漏除温缺陷

**变更来源：下游 vllm-ascend PR [#13470](https://github.com/vllm-ascend/vllm-ascend/pull/13470)**
（[Bugfix][MRV2] Support probabilistic rejection sampling for spec decode，
**2026-08-18 合入**，commit `27a94764b`）。该 PR 使 NPU
`_probabilistic_rejection_kernel` 对齐**上游 vLLM** `_rejection_kernel` 的真实
概率采样语义（上游语义由 PR
[#46878](https://github.com/vllm-project/vllm/pull/46878) 2026-06-27"Use fp32
uniform threshold for acceptance"与 PR
[#46781](https://github.com/vllm-project/vllm/pull/46781) 2026-06-30"Implement
block verification for rejection sampling"先后确立）。

**变更内容**（两处叠加）：

**(1) 采样行为**。旧版下游 kernel 用 `u = 0` 的简化实现，代入
`接受 ⇔ target_log_prob > log(u) + draft_log_prob` 后 log(0)=-inf，
**非贪婪场景恒接受**。新版改为真实 philox 采样，对齐上游：

```python
# 上游 vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py（_rejection_kernel）
u = tl_rand32(seed, pos, includes_zero=False)   # 真实均匀随机数

# 新版下游 vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py
# NPU Triton 缺 float64 tl_rand64 和标量 tl.rand，从 1 元素 block 生成 u，
# clamp 到 [2^-31, 1) 保证 log(u) 有限 —— 与上游语义等价
```

旧 UT 的"非贪婪恒接受"假设随之失效 → `AssertionError: Expected 2 accepted, got 1`
（构造的 logits 落入"随机区间"，某次 u 画得偏大导致该步被拒）。

**(2) 下游漏除温（真实缺陷，唯一一处 kernel 侧修复）**。时间线：上游 #50910
（2026-08-05）先给 `_rejection_kernel` / `_resample_kernel` 的 draft logit 加了
`/ temp`；下游 #13470（2026-08-18）重写 NPU kernel 对齐采样语义时，两处 draft
logit 加载都漏了 `/ temp`——即下游在追赶上游 06-27/06-30 语义时，漏同步了上游
08-05 的除温约定。**已直接修复 kernel**（本批 5 个失败中唯一的非 UT 修复）：

```python
# vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py（修复后）
# _resample_kernel 内：
draft_logits = (
    tl.load(draft_logits_ptr + ..., mask=mask, other=float("-inf"))
    .to(tl.float32)
    / temp        # ← 修复：与上游一致，draft logits 存的是除温前的值
)

# _probabilistic_rejection_kernel 内：
draft_logit = (
    tl.load(draft_logits_ptr + ... + draft_sampled)
    .to(tl.float32)
    / temp        # ← 修复
)
```

**UT 修复**（[npu/test_rejection_kernel.py](npu/test_rejection_kernel.py)）：

CPU 无法复现 philox 随机流，所以参考实现改为评估 **u 取值范围的两个确定性极端**，
测试数据相应构造为落入确定区间的 near-one-hot 场景：

```python
def _non_greedy_accept_cpu_ref(...):
    """kernel 画 u ~ U(0,1)（philox，clamp 到 [2^-31, 1)），当且仅当
    target_log_prob > log(u) + draft_log_prob 时接受。CPU 复现不了 philox
    随机流，因此本参考实现只评估 u 范围的两个确定性极端：
      - target_log_prob - draft_log_prob >= 0        -> 对所有 u < 1 恒接受
      - target_log_prob - draft_log_prob < log(2^-31) -> 对所有 u 恒拒绝
    返回 (accepted_count, "always" | "never" | "stochastic")。
    """
```

```python
def test_non_greedy_always_accept(self):
    """非贪婪 + draft logits：target 在 draft token 上 near-one-hot。

    target_log_prob ~= 0 >= log(u) + draft_log_prob 对每次 philox 采样
    u < 1 都成立，因此接受是确定性的（全部步接受）。
    """
    ...
    target_logits[i, draft_sampled[i + 1]] = 100.0   # ← 近 one-hot → 恒接受
```

`test_mixed_greedy_and_non_greedy` 中的非贪婪请求同样改为上述确定性构造，
消除随机翻转导致的偶发失败。

---

## 4. GPU 参照用例统一情况（合规性检查结论）

用户要求"检查对应 GPU 精度参照用例是否合规统一"，逐项核对结果：

| UT（gpu/ 侧） | 统一动作 | 结论 |
|---|---|---|
| `test_compute_local_logits_stats_kernel.py` | CPU 参考与 NPU 侧同步加 `/ temp` | ✅ 两侧一致 |
| `test_input_batch_prepare_prefill_inputs_kernel.py` | 三处调用同步改为新签名（含 `LOOKAHEAD_BLOCK`） | ✅ 两侧一致 |
| `test_rejection_kernel.py` | 新增 `_run_non_greedy_kernel` 辅助函数统一非贪婪调用路径；新增 `test_non_greedy_always_accept` / `test_non_greedy_always_reject`，与 NPU 侧 near-one-hot 场景**镜像** | ✅ 两侧一致 |
| `test_prepare_dflash_inputs_kernel.py` | dflash 为 NPU 专属 kernel（下游仓），GPU 侧无对应参照用例 | ➖ 不适用 |

---

## 5. 修复文件清单

**UT 侧（strict_ut_028，GPU/NPU 成对修改）**：

| 文件 | 修改内容 |
|---|---|
| `npu/test_compute_local_logits_stats_kernel.py`、`gpu/test_compute_local_logits_stats_kernel.py` | CPU 参考对 draft logits 先 `/ temp` |
| `npu/test_input_batch_prepare_prefill_inputs_kernel.py`、`gpu/test_input_batch_prepare_prefill_inputs_kernel.py` | kernel 调用改新签名（+`LOOKAHEAD_BLOCK`）；CPU 参考支持多 lookahead |
| `npu/test_prepare_dflash_inputs_kernel.py` | kernel 符号多路径降级探测；DCP 参数探测 + 动态传 `cp_rank=0, CP_SIZE=1, CP_INTERLEAVE=False` |
| `npu/test_rejection_kernel.py` | 重写 `_non_greedy_accept_cpu_ref`（确定性极端评估）；`test_non_greedy_always_accept` 等改 near-one-hot 确定性构造 |
| `gpu/test_rejection_kernel.py` | 新增 `_run_non_greedy_kernel` + always_accept / always_reject 镜像用例 |

**kernel 侧（vllm-ascend 仓，本批唯一的 kernel 修复）**：

| 文件 | 修改内容 |
|---|---|
| `vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py` | `_probabilistic_rejection_kernel` 与 `_resample_kernel` 的 draft logit 加载后补 `/ temp`，对齐上游 |

---

## 6. 环境约束与经验教训（历史踩坑沉淀）

- **环境版本硬约束**：vLLM ≥ 0.26.0 且含 PR #48892（multi_module_mtp，2026-07-30
  合入）、PR #46727（thinking_budget，2026-08-07 合入）；vllm-ascend ≥ 0.16.0rc1；
  Triton ≥ 3.3。版本过低会出现模块缺失类 ImportError（如 3.3a 的环境兼容性失败
  即属此类）。
- **签名变更优先用参数探测**：UT 不得修改 kernel；对"上游加参、下游版本参差"的
  场景（如 DCP 参数），用 `arg_names` 探测 + 兜底传参，一套 UT 兼容多版本。
- **随机类 kernel 的 UT 用确定性场景**：CPU 复现不了 philox 随机流时，构造
  near-one-hot 等使判定落在 u 取值范围极端的数据，把"随机"变成"恒接受/恒拒绝"
  两种确定结果来断言。
- **两侧文件必须同步**：GPU/NPU 用例成对修改；本地与 a5 环境的测试文件版本
  必须一致，历史多次 UT "失败"实为跑了旧文件。
