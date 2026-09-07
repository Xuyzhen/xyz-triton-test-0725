# vllm-ascend 接口变更影响报告（2026-09-07 pull）

> 范围：`accuracy_test/acc_ut_260907` 全部 NPU 算子精度 UT
> 触发事件：vllm-ascend 仓库于 2026-09-07 16:09 执行 `pull --ff`，
> HEAD 由 `80c833fb8`（08-26 基线）快进至 `78cd10dec`（09-07），
> 一次性带入 08-26 ~ 09-07 约 12 天的上游提交。
> acc_ut_260907 套件的构建与核验基线为 08-26 布局，因此部分 UT 出现
> 模块路径失效 / kernel 签名漂移。

---

## 1. 事件时间线

| 时间 | 事件 |
|---|---|
| 2026-07-30 15:47 | vllm-ascend 仓库 clone |
| 2026-08-26 15:26 | 最后一次 pull，HEAD=`80c833fb8`（**acc_ut_260907 核验基线**） |
| 2026-08-31 20:24 | 上游合入 `f8c81e379`（grammar_bitmask 迁移） |
| 2026-09-02 ~ 09-03 | 上游合入 rejection 采样相关变更（`4e6fb74b2`、`637417dbd`） |
| 2026-09-07 15:52 | 上游合入 `fd815467c`（main2main vllm 0828，dflash 同步） |
| **2026-09-07 16:09:58** | **本地 pull --ff，`80c833fb8` → `78cd10dec`，变更进入本环境** |
| 2026-09-07 16:09 之后 | 节点运行 `run_npu.sh`，grammar 测试 10 用例 FAIL（`ModuleNotFoundError`） |

节点报错能定位到 `vllm_ascend.worker.v2.structured_outputs` 这一层，说明节点
安装的 vllm-ascend 已是 09-07 HEAD（与共享目录代码同步）。

---

## 2. 变更一（核心）：grammar_bitmask 算子迁移

**commit**：`f8c81e379`（2026-08-31，"[MRV2][Feature] Optimize triton ops
grammar_bitmask on A2/A3 (#14525)"，作者 AuroraEmiya）

**文件级变化**：

| 文件 | 变化 |
|---|---|
| `vllm_ascend/worker/v2/structured_outputs.py` | **删除**（-68 行） |
| `vllm_ascend/ops/triton/v2/apply_grammar_bitmask.py` | **新增**（+110 行） |
| `vllm_ascend/ops/triton/v2/docs/apply_grammar_bitmask.md` | 新增算子文档 |
| `vllm_ascend/patch/worker/patch_v2/patch_triton.py` | patch 导入源由旧模块改为新模块 |
| `tests/.../test_apply_grammar_bitmask_triton.py` | 新增官方 e2e 测试（+95 行） |

### 2.1 变更前（08-26 基线）

- **算子位置**：`vllm_ascend/worker/v2/structured_outputs.py:35`
- **符号**：`@triton.jit def _apply_grammar_bitmask_kernel(...)` —— 纯 Triton kernel
- **签名**（7 参数）：

```python
_apply_grammar_bitmask_kernel(
    logits_ptr,          # [num_logits, vocab_size] fp32
    logits_stride,       # stride(0)
    logits_indices_ptr,  # [num_bitmasks]
    bitmask_ptr,         # [num_bitmasks, padded_vocab//32] packed
    bitmask_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,   # 8192
)
```

- **实现要点**：`BLOCK_SIZE_SUB=1024` 子块 tiling 循环
  （`tl.range(0, BLOCK_SIZE, BLOCK_SIZE_SUB)`），规避 Ascend UB 溢出；
  2D grid `(num_bitmasks, num_blocks)` 直接启动。
- **patch 方式**：`patch_triton.py` 从 `vllm_ascend.worker.v2.structured_outputs`
  导入，替换 `vllm.v1.worker.gpu.structured_outputs._apply_grammar_bitmask_kernel`。

### 2.2 变更后（09-07 HEAD）

- **算子位置**：`vllm_ascend/ops/triton/v2/apply_grammar_bitmask.py`
- **符号分两层**：
  1. `_apply_grammar_bitmask_kernel_impl`（内部 Triton kernel，1D grid 重写）
  2. `_ApplyGrammarBitmaskKernelLauncher`（**对外兼容层**），
     模块级单例 `_apply_grammar_bitmask_kernel = _ApplyGrammarBitmaskKernelLauncher()`
- **新实现要点**：
  - launcher 把上游逻辑 2D grid `(num_masks, num_vocab_blocks)` 映射为
    Ascend VectorCore 的 1D grid：`total_tasks = num_masks * num_vocab_blocks`，
    `num_programs = min(get_vectorcore_num(), total_tasks)`；
  - kernel 内部按 `NUM_PROGRAMS / NUM_VOCAB_BLOCKS` constexpr 自行划分任务；
  - `multibuffer=False` 以更充分利用 UB（A2/A3 性能优化，A5 可恢复）。
- **对外调用签名不变**（launcher 兼容层，7 参数，与旧版逐参一致）：

```python
_apply_grammar_bitmask_kernel[(num_masks, num_vocab_blocks)](
    logits, logits.stride(0), logits_indices,
    bitmask, bitmask.stride(0), vocab_size,
    BLOCK_SIZE=8192,
)
```

- **patch 方式**：`patch_triton.py:10` 改为
  `from vllm_ascend.ops.triton.v2.apply_grammar_bitmask import _apply_grammar_bitmask_kernel`，
  `patch_triton.py:46` 仍替换 `structured_outputs._apply_grammar_bitmask_kernel`。

### 2.3 对 UT 的影响（本次 10 个 FAIL 的直接原因）

| 测试文件 | 失效点 | 现象 |
|---|---|---|
| `npu/test_apply_grammar_bitmask_kernel.py:73` | `from vllm_ascend.worker.v2.structured_outputs import _apply_grammar_bitmask_kernel` | 模块已删除 → `ModuleNotFoundError`，10 用例 FAIL |
| `npu/test_apply_grammar_bitmask_kernel_upstream.py:73` | 同上（复制时两文件保留了同一导入） | 同样会 FAIL（尚跑到） |

两个文件均在 `_run_kernel()` 方法内**懒加载**导入，且没有模块级
`try/except + pytest.skip` 保护，因此 ImportError 直接以 FAIL 暴露
（套件内其他文件均有 skip 保护，仅这两个 grammar 文件没有）。

### 2.4 修复方式

导入改为多路径 fallback（新位置优先，旧位置兼容老版本），调用代码
**零改动**（launcher 兼容层签名与旧版一致）。

---

## 3. 变更二：`_probabilistic_rejection_kernel` 签名扩展

**commit**：`637417dbd`（2026-09-03，"[Feature][MRV2][Spec Decode] support
synthetic rejection sampling on MRV2 (#15556)"）
配套：`4e6fb74b2`（2026-09-02，Disable AutoBlockify workaround）

**文件**：`vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py`（±53 行）

**算子位置不变**（同文件 192 行），仅签名与内部逻辑扩展：

### 3.1 变更前（08-26 基线）

```python
_probabilistic_rejection_kernel(
    ...,
    seed_ptr,
    pos_ptr,
    vocab_num_blocks,                    # ← pos 之后直接是块数
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
)
```

非 greedy 路径：`u = tl.full([], 0.0)`（更早版本）→ philox 随机数
（08-26 时点版本）；无 synthetic 采样支持，`rejection_sample()` 遇到
`synthetic_conditional_rates is not None` 直接 `raise NotImplementedError`。

### 3.2 变更后（09-07 HEAD）

```python
_probabilistic_rejection_kernel(
    ...,
    seed_ptr,
    pos_ptr,
    synthetic_conditional_rates_ptr,      # ← 新增位置参数（[num_spec_steps] 或 None）
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    SYNTHETIC_MODE: tl.constexpr,         # ← 新增 constexpr
)
```

- greedy / 非 greedy 分支内部按 `SYNTHETIC_MODE` 分流：
  `SYNTHETIC_MODE=False` 时行为与旧版**完全一致**（diff 确认）；
- `rejection_sample()` 相应传入 `synthetic_conditional_rates` 与
  `SYNTHETIC_MODE=synthetic_conditional_rates is not None`，并删除了旧的
  `raise NotImplementedError`；
- `_resample_kernel` 签名**未变**（已核对）；
- `rejection_sample()` 内部对 `_compute_block_stats_kernel` / `_resample_kernel`
  的调用新增 launcher 选项 `has_auto_blockify_blacklist_op=True`
  （AutoBlockify bug workaround，属可选 kwarg，不影响直接调用 kernel 的 UT）。

### 3.3 对 UT 的影响

6 处直接调用缺新参数，运行时会因 Triton 参数校验失败（TypeError /
missing argument）：

| 测试文件 | 调用处 |
|---|---|
| `npu/test_rejection_kernel_downstream.py` | 301、500、787 行附近（3 处） |
| `npu/test_rejection_kernel_upstream.py` | 256、455、652 行附近（3 处） |

**语义不受影响**：全部测试场景均为标准（非 synthetic）路径，
`SYNTHETIC_MODE=False` 下新旧行为一致，期望值无需调整。

### 3.4 修复方式

在 `pos,` 之后插入 `None,`（占位 `synthetic_conditional_rates`），
在 `HAS_DRAFT_LOGITS=...` 之后补 `SYNTHETIC_MODE=False,`。

---

## 4. 变更三：dflash speculator 重构（UT 已兼容，无需修改）

**commit**：`fd815467c`（2026-09-07，"[CI] main2main vllm 0828 (#14872)"）

**文件**：`vllm_ascend/worker/v2/spec_decode/dflash/speculator.py`（-303/+173）

- `_prepare_dflash_inputs_kernel_ascend` 新增 DCP 参数（同步 vllm#52188）：
  位置参数 `cp_rank` + constexpr `CP_SIZE` / `CP_INTERLEAVE`；
  CP 关闭时 `cp_rank=0, CP_SIZE=1, CP_INTERLEAVE=False`，行为不变。
- acc_ut_260907 的两个 dflash UT
  （`test_prepare_dflash_inputs_kernel_downstream.py` / `..._upstream.py`）
  在生成时即内置了 `"cp_rank" in arg_names` 探测与
  `launch_kwargs.update(cp_rank=0, ...)` 分支；
- 已逐参数核对当前签名（11 输出 → 9 target 输入 → 2 采样参数 →
  block_table + stride → 7 标量 → cp_rank → 4 constexpr）与 UT 调用
  完全匹配。**状态：无需修改**。

---

## 5. 其他相关变更（核验通过，不影响 UT）

| commit | 内容 | 与 UT 的关系 |
|---|---|---|
| `843dd09a9` (08-29) | 新增 `worker/v2/sample/apply_top_k_top_p.py`，re-export `_apply_top_k_top_p_pytorch` | UT 导入的 `vllm_ascend.sample.sampler._apply_top_k_top_p_pytorch` 符号与签名未变 |
| `83f9ef38f` / `d4ebe8a0e` 等 | 设备能力判断迁移至 `hardware_profile` | `sampler.py` 模块级 `apply_top_k_top_p = (...)` 选择逻辑变化，但被测符号本体未变 |

**vllm 侧注意**：本地 `git/024/vllm` 仓库的
`vllm/v1/worker/gpu/structured_outputs.py:126` kernel 为 8 参签名
（额外 `cu_num_logits_ptr` 位置参数 + `MASK_STRIDE` constexpr），系 vllm main
后续演进；而 vllm-ascend 09-07 HEAD 的 launcher 为 7 参。二者存在版本错配
风险（vllm-ascend `fd815467c` 配套声明为 vllm 0828）。节点实际运行以
site-packages 安装版本为准；UT 通过 launcher（7 参）调用，不受该错配影响。

---

## 6. 影响范围总表（NPU 侧全部 UT 核验结果）

| 测试文件 | 依赖符号 | 状态 |
|---|---|---|
| test_apply_grammar_bitmask_kernel.py | `worker.v2.structured_outputs._apply_grammar_bitmask_kernel` | **FAIL（模块删除）→ 本次修复** |
| test_apply_grammar_bitmask_kernel_upstream.py | 同上 | **将 FAIL → 本次修复** |
| test_rejection_kernel_downstream.py | `_probabilistic_rejection_kernel` | **将 FAIL（签名）→ 本次修复** |
| test_rejection_kernel_upstream.py | 同上 | **将 FAIL（签名）→ 本次修复** |
| test_resample_kernel_downstream/upstream.py | `_resample_kernel` | 签名未变，通过 |
| test_compute_block_stats_kernel_downstream.py | `_compute_block_stats_kernel`（re-export vllm `_compute_local_logits_stats_kernel`） | 通过 |
| test_compute_block_max_and_sumexp_downstream.py | 同上 | 通过 |
| test_compute_global_logsumexp_downstream.py | `_compute_global_lse`（re-export vllm `_compute_global_logsumexp`） | 通过 |
| test_gumbel_block_argmax_downstream.py | `_npu_gumbel_block_argmax` | 通过（34 行，未动） |
| test_prepare_dflash_inputs_kernel_downstream/upstream.py | `_prepare_dflash_inputs_kernel_ascend` | 通过（探测机制已覆盖 cp_rank） |
| test_topk_topp_kernel_downstream.py | `sample.sampler._apply_top_k_top_p_pytorch` | 通过（签名未变） |
| test_bad_words / test_bincount / test_penalties / test_min_p / test_temperature / test_gumbel_sample / test_ranks / test_topk_log_softmax / test_num_nans_* / test_fill_logprob_token_ids_* | 对应 `worker.v2.sample.*`、`ops.triton.v2.*` 符号 | 模块与签名均未变，通过 |

（核验方法：`git diff 80c833fb8..78cd10dec --stat` 圈定触碰文件 →
对照 UT 全部 `vllm_ascend` 导入清单 → 逐符号核对当前签名与调用。）

---

## 7. 本次修复清单

| # | 文件 | 修改内容 |
|---|---|---|
| 1 | `npu/test_apply_grammar_bitmask_kernel.py` | 导入改为新位置优先 + 旧位置 fallback；更新头部位置注释与 docstring 的实现描述 |
| 2 | `npu/test_apply_grammar_bitmask_kernel_upstream.py` | 同上；走 vllm 模块入口 + 手动应用 patch_triton 同款 patch（模拟运行时行为），失败 fallback 直连 |
| 3 | `npu/test_rejection_kernel_downstream.py` | 3 处调用补 `None`（synthetic_conditional_rates）+ `SYNTHETIC_MODE=False`；docstring 签名同步 |
| 4 | `npu/test_rejection_kernel_upstream.py` | 同上（3 处） |

调用侧的其他代码（grid、数据构造、期望值）**均不需要改动**：
- grammar：launcher 兼容层保持 7 参调用约定；
- rejection：`SYNTHETIC_MODE=False` 分支与旧版逐行为一致。

## 8. 后续建议

1. 依赖核验报告具有时效性——vllm-ascend 仓库每次 pull 后建议重跑
   `git diff <旧HEAD>..<新HEAD> --stat -- vllm_ascend/` 与 UT 导入清单对照；
2. grammar 两个 UT 本次顺带补上模块级 skip 保护（与其余 UT 对齐），
   未来再发生路径迁移时将以 SKIP 而非 FAIL 暴露；
3. 节点 vllm / vllm-ascend 安装版本建议固定（如 pip freeze 快照），
   避免共享目录代码与 site-packages 悄然错配。

---

## 9. 第二轮观测：test_all_neg_inf_blocks 失败（2026-09-07 复测发现）

### 9.1 现象与定性：非精度回归

- 现象：`npu/test_compute_global_logsumexp_downstream.py::test_all_neg_inf_blocks`
  全 `-inf` block max 输入，kernel 返回 `nan`，UT 断言期望 `-inf`。
- **同文件 7 个有限值域用例（parametrize 6 + single_block）在 rtol/atol=1e-5
  下全部通过**——失败仅出现在退化输入域用例，故不是数值精度漂移。
- 数学根因：kernel 裸公式
  `global_max + log(sum(sumexp*exp(maxes-global_max)))` 在全 `-inf` 时
  `maxes - global_max = -inf - (-inf) = NaN`（IEEE754），**CPU 上同样为 nan，
  与硬件无关**。UT 的 CPU 参考实现了 `global_max > -inf` 特判，kernel 没有
  → 期望与实现不匹配。

### 9.2 版本错配排查（本地 git 证据）

| 检查项 | 结果 |
|---|---|
| `git log -L 21,44`（vllm 该函数完整谱系） | 仅两次变更：04-07 `5daf62271d` 引入即裸公式；06-30 `db808b3961` 仅改名 `_compute_global_lse→_compute_global_logsumexp`。**任何版本均无 -inf 特判** |
| `git log -S "tl.where(global_max"` / `-S "isinf"`（vllm 该文件） | 零命中 |
| vllm-ascend 侧 `_compute_global_lse` 来源 | 07-06 `cd76505e8` 起为对 vllm 的**纯 re-export**，无本地实现 |
| 节点可运行性反推 | UT 未 skip 且已执行 ⇒ 节点安装版 vllm ≥ 06-30（重命名后）⇒ 该函数与本地 checkout 逐字相同 |
| 结论 | **版本错配无法解释该 nan**：不存在任何"全 -inf 返回 -inf"的可安装版本。节点 site-packages 无法本地直接检查，留探针实锤（见 9.3） |

注：仓库中确实存在一处已知版本错配——本地 vllm checkout（HEAD `2cf0a6915c`，
08-23）的 `structured_outputs.py` kernel 为 8 参，而 vllm-ascend 09-07 的
launcher 为 7 参——但那是 grammar 模块，与本失败（rejection_sampler_utils）
无关，且 grammar UT 经 launcher 调用不受影响。

### 9.3 探针

新增 [probe/probe_global_lse.py](../probe/probe_global_lse.py)
（standalone，仅依赖 vllm/vllm-ascend/torch/triton）：

```bash
cd accuracy_test/acc_ut_260907
python probe/probe_global_lse.py               # NPU 节点全量探针
python probe/probe_global_lse.py --device cpu  # 仅环境 + CPU 数学对照
```

输出分四节，判读：

| 节 | 回答 | 判读要点 |
|---|---|---|
| `[ENV]` | Q2 静态部分 | vllm / vllm-ascend 版本与 `__file__`（site-packages 还是源码目录） |
| `[IDENTITY]` | Q2 | `vllm_ascend._compute_global_lse is vllm._compute_global_logsumexp` 为 True = 纯 re-export |
| `[SOURCE]` | Q2 | dump 节点实际参与 JIT 编译的 helper 源码（看有无 -inf 特判） |
| `[CPU]` | Q1 | 公式逐项分解：全 -inf 时 CPU 同样 nan，与硬件无关 |
| `[KERNEL]` | Q1+Q3 | 安装版 helper vs 裸公式参考 kernel 四场景对比（A 全 -inf / B 混合 / C 全有限 / D 单块）；两者一致（A 均 nan）→ 固有行为非错配；helper 在 A 返回 -inf → 节点安装版与共享目录代码不一致，错配实锤 |

### 9.4 处置（待定，2026-09-07 已撤回 xfail 标记）

节点实测确认：`test_compute_global_logsumexp_downstream.py` 与
`_upstream.py` 的 `test_all_neg_inf_blocks` 均失败（两个文件测的是同一
kernel 对象：upstream 直连 vllm 实现，downstream 走 vllm-ascend
re-export），其余 14 个有限值域用例全部 1e-5 通过。

曾按方案 A 给两个用例标注 `@pytest.mark.xfail(strict=True)`，
经确认后已撤回，测试文件保持原断言不变。后续处置待定：

| 方案 | 做法 | 说明 |
|---|---|---|
| A | 用例标 `xfail` 并注明原因 | 保留边界覆盖痕迹："上游 vllm 裸公式固有行为（04-07 引入起无 -inf 特判），GPU/NPU/CPU 一致为 nan" |
| B | 断言放宽为 `nan or -inf` | 不区分"数学正确"与"实现行为" |
| C | 删除该用例 | 信息丢失最多，不推荐 |
| D | 改 kernel 加特判 | **违反"不改原文件"约束，排除** |

探针 `probe/probe_global_lse.py` 保留在位，随时可在节点实锤
（判读表见 9.3）。

---

## 10. 变更四：AutoBlockify 规避选项（UT 侧缺失导致 248320 档失败）

### 10.1 现象

2026-09-07 全量跑 [16/69] `npu/test_compute_local_logits_stats_kernel.py`
出现 4 个 FAIL，全部为 `vocab_size=248320` 档：

```
[False-2-248320-4] / [False-2-248320-8] / [False-3-248320-4] / [False-3-248320-8]
（has_draft_logits=False, num_spec_steps=2/3, num_reqs=4/8）
Expected 3.77 ~ 4.02 but got 0.0   ← greedy 分支 target_local_max
```

### 10.2 根因

**commit `4e6fb74b2`（2026-09-02，在本次 pull 区间内）**：
*"[Bugfix][Spec Decode] Disable AutoBlockify for rejection sampling (#15118)"*。
commit 说明原文：*"AutoBlockify can corrupt the max-with-index reductions
used by `_compute_block_stats_kernel` and `_resample_kernel`, which can
produce an incorrect argmax/token ID even when the input logits are finite."*

- **位置不变、签名不变**，仅生产调用点新增 launcher 选项
  `has_auto_blockify_blacklist_op=True`（vllm-ascend
  `rejection_sampler_utils.py:437` stats kernel、`:512` resample kernel 两处），
  并附 TODO：Triton Ascend 修复 max-with-index 归约的 AutoBlockify bug 后移除。
- 被破坏的正是 stats kernel greedy 分支的 `tl.max(..., return_indices=True)`
  （max-with-index 归约）——`got 0.0` 是归约结果被 AutoBlockify 改写，不是算错。
- **UT 直接 launch kernel 对象**（绕过 vllm-ascend `rejection_sample()` 包装），
  生产侧的规避选项没被带上 → 大 grid 档被 AutoBlockify 破坏。

失败模式与 AutoBlockify 启发式触发条件完全吻合（按 grid 规模/内核复杂度决定
是否启用）：

| 观察 | 解释 |
|---|---|
| 仅 248320（31 blocks）失败，129280（16 blocks）通过 | grid 第二维超过启发式阈值 |
| 仅 has_draft_logits=False 失败 | True 编译变体代码量大，AutoBlockify 不启用 |
| 仅 num_reqs≥4 且 steps≥2（grid ≥ 12×31=372 programs）失败 | 总 program 数超过阈值 |
| got 0.0（非近似值） | 归约被破坏，非数值精度漂移 |

### 10.3 修复（2026-09-07 已执行：全量对齐生产）

按生产配置给 UT 侧这两个 kernel 的**全部 18 处 launch**（9 个文件）补
`has_auto_blockify_blacklist_op=True`，附 TODO 注释引用 `4e6fb74b2`：

| 文件 | 处数 |
|---|---|
| test_compute_local_logits_stats_kernel.py | 2 |
| test_compute_block_stats_kernel_upstream.py | 1 |
| test_compute_block_stats_kernel_downstream.py | 2 |
| test_compute_block_max_and_sumexp_upstream.py | 2 |
| test_compute_block_max_and_sumexp_downstream.py | 1 |
| test_rejection_kernel_downstream.py | 3 |
| test_rejection_kernel_upstream.py | 3 |
| test_resample_kernel_downstream.py | 2 |
| test_resample_kernel_upstream.py | 2 |

其中 rejection（vocab 参数含 248320）与 resample（同）的 10 处为**预防性修复**
——生产 commit 明确两个 kernel 都受影响，本轮跑到必然同样失败；
其余小 grid 文件（vocab ≤ 8192，单 block）为一致性对齐（生产无条件规避，
AutoBlockify 对小 grid 不启用，该选项无副作用）。

**不在范围内**：`test_insert_resampled_kernel.py`（`_insert_resampled_kernel`
不在生产规避清单）；`_probabilistic_rejection_kernel`（生产未规避，无
max-with-index 归约）。

**验证方式**：节点重跑
`pytest npu/test_compute_local_logits_stats_kernel.py -k 248320`
（预期 4 FAIL 全部转 PASS），以及 rejection/resample 文件的 248320 档。
