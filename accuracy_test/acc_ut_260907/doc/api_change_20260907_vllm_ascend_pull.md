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
| 2026-09-07 23:01 | 深夜补跑暴露变更七：prefill inputs 签名漂移（vllm #48892，07-30 已在 vllm 侧合入），09-08 UT 侧修复（§13） |

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

### 9.4 处置（已定：保持现状，持续 FAIL 作为已知问题跟踪）

节点实测确认：`test_compute_global_logsumexp_downstream.py` 与
`_upstream.py` 的 `test_all_neg_inf_blocks` 均失败（两个文件测的是同一
kernel 对象：upstream 直连 vllm 实现，downstream 走 vllm-ascend
re-export），其余 14 个有限值域用例全部 1e-5 通过。

曾按方案 A 给两个用例标注 `@pytest.mark.xfail(strict=True)`，经确认后
已撤回。**最终决策（2026-09-07）：保持现状**——测试文件保持原始严格
断言不动，该用例（两个文件共 2 个）在全量跑中持续 FAIL，作为已知问题
由本节跟踪。判读基线：

- 全量跑中这两个 FAIL = 本已知问题，无需重新排查；
- 若其余 14 个有限值域用例出现任何失败 = 新问题，需排查；
- 根因定性不变：非精度回归（kernel 裸公式 IEEE754 固有行为，
  CPU/GPU/NPU 一致为 nan，任何 vllm 版本均无 -inf 特判）；
- 探针 `probe/probe_global_lse.py` 保留在位，随时可在节点实锤
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

---

## 11. 变更五：`_compute_slot_mappings_kernel` 新增 `BLOCK_TABLE_PAD_SIZE`（Ascend 适配版）

### 11.1 现象

2026-09-07 全量跑 `npu/test_compute_slot_mappings_kernel.py` 4 个用例全部
FAIL，报错一致：

```
TypeError: dynamic_func() missing 1 required positional argument: 'BLOCK_TABLE_PAD_SIZE'
```

（绑定阶段即失败，非数值错误。）

### 11.2 根因（代码证据）

**算子位置不变**：`vllm_ascend/ops/triton/v2/block_table/compute_slot_mappings.py`
（经 `vllm_ascend/worker/v2/block_table.py` re-export，UT 走该入口）。

签名变化（仅 vllm-ascend 适配版；vllm 上游 kernel 无此参数，fallback 路径
不受影响）：

```python
# 旧版（UT 生成时基线）：TOTAL_BLOCK_SIZE（UT 已有探测分支）
# 新版（09-07 HEAD）：
def _compute_slot_mappings_kernel(
    ..., cp_rank,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    BLOCK_TABLE_PAD_SIZE: tl.constexpr,   # ← 新增必填 constexpr
)
```

- 语义：`tl.arange(0, BLOCK_TABLE_PAD_SIZE)` 需要编译期 2 的幂长度，
  作为各 KV cache group block table 行加载的上界；运行时用
  `mask=offsets < block_table_stride` 收敛到实际行宽。
- 生产取值（`AscendBlockTables.__init__`，worker/v2/block_table.py:63）：
  `next_power_of_2(max(block_table.stride(0)))`，launch 时以
  `BLOCK_TABLE_PAD_SIZE=self._block_table_pad_size` 传入（L102）。
- UT 直接 launch kernel，原有探测分支只覆盖 `TOTAL_BLOCK_SIZE`
  （旧参数名），新参数未传 → 绑定失败。
- 位置参数（max_num_tokens … cp_rank）新旧完全一致，无需调整。

具体引入 commit 可在节点用
`git log -S BLOCK_TABLE_PAD_SIZE -- vllm_ascend/ops/triton/v2/block_table/`
确认（属 pull 区间 80c833fb8..78cd10dec 内变更）。

### 11.3 修复（2026-09-07 已执行）

扩展 UT 既有探测分支（保留 `TOTAL_BLOCK_SIZE` 分支兼容旧版）：

```python
if "BLOCK_TABLE_PAD_SIZE" in tuple(KERNEL.arg_names):
    kwargs["BLOCK_TABLE_PAD_SIZE"] = triton.next_power_of_2(block_table.stride(0))
```

- 取值镜像生产 `AscendBlockTables.__init__`（单 group 场景
  `max(stride(0))` = `stride(0)`）；
- 节点 ascend_adapted 路径下 `block_table.stride(0)=4096`（2 的幂）→
  `PAD_SIZE=4096`，与运行时 stride mask（4096）一致，整行可载；
- upstream fallback 路径：vllm kernel 无此参数，分支不触发，行为不变。

**验证方式**：节点重跑
`pytest npu/test_compute_slot_mappings_kernel.py -v`
（预期 4 FAIL 全部转 PASS；期望值无需调整——slot 计算逻辑未变）。

---

## 12. 变更六（环境级）：vllm / vllm-ascend 版本错配实锤（dflash 无法 import）

### 12.1 现象

2026-09-07 全量跑 dflash 两个 UT 文件
（`test_prepare_dflash_inputs_kernel_downstream.py` / `_upstream.py`），
`TestPrepareDFlashInputsKernelAscendPatch` 全部参数化用例 FAIL，报错为
测试自带的 revision=2 环境诊断：

```
DFlash environment compatibility failure; this is not a precision failure
and no kernel was tested.
diagnostic revision=2
vllm=0.28.0+empty
vllm-ascend=0.19.1rc2.dev1959+g78cd10dec
import errors=legacy worker-v2: No module named 'vllm.v1.attention.ops.pcp';
  modern spec-decode utils: neither the combined DFlash/DSpark kernel
  nor the legacy DFlash kernel is exported
modern related exports=copy_and_expand_dflash_and_dspark_inputs_kernel,
  dflash2_greedy_selector_walk_kernel
modern source contains expected symbol=False
dflash proposer import failed=No module named 'vllm.v1.attention.ops.pcp'
```

诊断显示**两条探测路径同时失败，且失败原因不同**（详见 12.2 两层根因）。

### 12.2 根因（§5 预警的版本错配首次实际咬人）

**环境组合错配**：节点 vllm 为 08-23 checkout（2cf0a6915c，0.28.0），
vllm-ascend 为 09-07 HEAD（78cd10dec，main2main 配套 vllm 0828）。
完整断链证据：

```
import vllm_ascend.worker.v2.spec_decode.dflash.speculator   # kernel 本体在 L161
  └→ L19  vllm_ascend.worker.v2.attn_utils
       └→ L46  vllm_ascend.attention.attention_v1
            └→ L40  from vllm.v1.attention.ops.pcp import _gather_prefill_cache_inputs
                  ← ModuleNotFoundError：pcp.py 仅存在于 0828+ 的 vllm main
```

- vllm-ascend 侧三处硬依赖（attention_v1.py:40 / mla_v1.py:22 /
  sfa_cp.py:12），均带 `# type: ignore[import-not-found]`——开发者预期
  只在配套 vllm 版本下运行；
- 本地 vllm checkout `vllm/v1/attention/ops/` 下确认无 `pcp.py`；
- **kernel 本体未丢失**：`_prepare_dflash_inputs_kernel_ascend` 仍在
  speculator.py:161，纯粹是模块 import 链断裂导致测试拿不到符号；
- 测试的 revision=2 诊断逻辑（多路径探测 + pytest.fail 带完整报告）
  按设计如实报告了环境问题，非测试 bug，非 kernel 精度问题。

**第二层根因（modern fallback 路径失效，独立于 pcp）**：
legacy 失效后，测试按设计回落到 modern 路径
（`vllm_ascend.ops.triton.spec_decode.utils`）。该模块本身 import 干净
（不经过 pcp 依赖链），但测试探测的符号名已过时：

| 符号 | 引入 / 移除 | 说明 |
|---|---|---|
| `copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid` | `41ff81e1a`（07-13，#11765）引入；`acbd2bb28`（08-14，#13191）改名 | 测试的 modern 首选探测目标 |
| `copy_and_expand_dflash_inputs_kernel_single_grid` | 早期版本 | 测试的 modern 次选探测目标 |
| `copy_and_expand_dflash_and_dspark_inputs_kernel` | `acbd2bb28` 改名后的现名 | **当前环境实际导出**（utils.py:69） |

`acbd2bb28`（[Performance] Optimized Kernel，2026-08-14）不仅去掉了
`_single_grid` 后缀，还将 kernel 从 per-request 串行循环重写为 grid-stride
平铺（新增 `TILE_SIZE: tl.constexpr = 256` 默认参数；multimodal 输入下
query slot 改由 effective_seq_len 推导，文本输入下行为不变）。

**定性要点**：`acbd2bb28` 早在 08-26 基线（`80c833fb8`）内
（`git merge-base --is-ancestor` 确认），即 **UT 的 modern 探测符号从
构建时起就落后于环境，并非 09-07 pull 引入**。此前从未暴露，是因为
08-26 基线下 legacy 路径可用、modern fallback 从未触发；09-07 pull 的
pcp 断链使 legacy 失效、modern 首次触发，才暴露这一滞后。

**其余 UT 通过属侥幸**：它们 import 的模块链恰好未触碰 pcp 依赖。
当前 08-23 vllm + 09-07 vllm-ascend 组合本身就是错配状态。

### 12.3 处置（已更新：modern 符号适配修复）

**初判决策（2026-09-07）：保持现状**——环境不修，测试保持大声 FAIL，
作为环境错配的可见跟踪（与 §9.4 logsumexp 处置逻辑一致）。

**修订（2026-09-08）**：深入解读 revision=2 完整诊断后发现第二层根因
（modern 探测符号过时，§12.2 下半节）可独立于环境修复——modern 模块
import 不经过 pcp 依赖链，仅符号名需适配。据此执行**路径 C：测试侧
modern 符号适配**，使 dflash UT 在当前错配环境下即恢复可测性：

- `npu/test_prepare_dflash_inputs_kernel_downstream.py` 与
  `..._upstream.py` 两文件同步修改：
  - modern 探测顺序改为：无后缀 combined（现名，acbd2bb28 后）→
    `_single_grid` combined（旧名 fallback）→ legacy `_single_grid`；
  - 探测到任一 combined 变体时 `_modern_dflash_supports_sample_from_anchor
    = True`（两者均带 SAMPLE_FROM_ANCHOR 参数），语义与原逻辑一致；
  - 诊断函数 `expected_symbol` 同步改为现名。

**兼容性验证（适配不改动调用逻辑的依据）**：

| 检查项 | 结论 |
|---|---|
| modern 模块 import | 干净，不经过 pcp 链（诊断输出 `modern module=...` 行已证明） |
| 参数签名 | 20 个位置参数与测试调用逐一对齐（utils.py:69-98 vs 测试 `_test_modern_dflash_inputs`）；`TILE_SIZE` 带默认值 256，无需显式传 |
| launch grid | 测试用 `grid=(1,)`，grid-stride 重写后单 program 覆盖全部 tile，与原串行循环语义等价 |
| 参考实现语义 | `_modern_dflash_inputs_ref` 的 `cache_pos = effective_seq_len + query_offset` 及 anchor/非 anchor 的 sample_indices 写法与新 kernel 逐分支一致（参考实现本就按新 kernel 编写，仅探测符号名是旧的） |

**效果**：修复后 legacy 路径仍失效（pcp 未修，环境错配如实保留），
但 modern fallback 恢复可用——dflash 从"环境 FAIL、0 kernel 被测"转为
"实际测试 acbd2bb28 grid-stride 重写版 kernel"。节点的 pcp 错配对
dflash UT 不再构成阻断，仅影响 legacy 探测分支。

若未来仍要修环境（可选，不再必需）：

| 路径 | 操作 | 影响 |
|---|---|---|
| A | 节点 vllm checkout 切到配套版本 `e6bfe03ad`（main2main 0828 范围终点，见 fd815467c commit message）；源码模式（+empty）checkout 即生效，无需重装 | legacy 路径恢复（speculator 版 kernel）；需重跑全套 UT 确认其他 kernel 签名未漂移（grammar/rejection/stats 等 kernel 本体在 vllm 侧，0828 可能有变） |
| B | 测试侧把环境不兼容的 pytest.fail 改为 pytest.skip（诊断降级为 skip reason） | 已被路径 C 取代：modern 恢复后无需 skip |

### 12.4 全量跑已知问题基线（2026-09-08 更新）

判读全量跑日志时，以下 FAIL 为已知问题、无需重新排查：

| # | 已知问题 | 涉及 | 根因定性 |
|---|---|---|---|
| 1 | `test_all_neg_inf_blocks` | logsumexp downstream + upstream（2 用例） | vllm kernel 裸公式 IEEE754 固有行为（§9，非精度回归） |
| ~~2~~ | ~~dflash 环境诊断 FAIL~~ | `test_prepare_dflash_inputs_kernel_*` | **已于 09-08 修复**（§12.3 路径 C：modern 符号适配，不再阻断） |

**除 #1 外，任何 FAIL 均为新问题，需排查。** 已修复待同步验证项：
grammar 导入（§2.4）、rejection 参数（§3.4）、AutoBlockify（§10.3）、
slot mappings（§11.3）、dflash modern 符号（§12.3）、
prefill inputs lookahead（§13.3）、hidden states buffer 尺寸（§14）。
验证命令：`pytest npu/test_prepare_dflash_inputs_kernel_downstream.py
npu/test_prepare_dflash_inputs_kernel_upstream.py -v`
（预期全部参数化用例经 modern 路径 PASS）。

---

## 13. 变更七：_prepare_prefill_inputs_kernel 新增 lookahead 参数（vllm #48892）

### 13.1 现象

2026-09-07 深夜补跑（stderr 时间戳 W907 23:01，晚于 §12.4 基线成文）
`npu/test_prepare_prefill_inputs_kernel.py`，
`TestPreparePrefillInputsKernel` 全部用例 FAIL（9 个参数化组合 +
early_return / boundary），报错一致：

```
TypeError: dynamic_func() missing 3 required positional arguments:
'prefill_lens_ptr', 'num_computed_tokens_ptr', and 'LOOKAHEAD_BLOCK'
```

### 13.2 根因：vllm 侧 #48892 签名扩展，UT 按旧快照调用

**变更来源**：vllm `dec13a33b7`（2026-07-30 合入 main，
#48892 "[Model Runner V2][Spec Decode] Add multi-layer MTP speculator"，
即 §12 已知的 multi_module_mtp 同源 PR）。节点 vllm 0.28.0
（08-23 checkout）已包含该 commit，kernel 为新签名；strict_ut 版 UT
按 #48892 之前的签名调用。**此变更在 vllm 侧，与 09-07 的
vllm-ascend pull 无关**。

**接口具体变化**（算子位置：`vllm/v1/worker/gpu/input_batch.py:254`）：

| 项 | 变更前（< #48892） | 变更后（≥ #48892） |
|---|---|---|
| 签名 | `(input_ids, next_prefill_tokens, idx_mapping, query_start_loc, all_token_ids, all_token_ids_stride, prefill_lens, num_computed_tokens, BLOCK_SIZE)` | 第 3/4 位**插入** `next_prefill_tokens_stride`、`num_lookahead`；`BLOCK_SIZE` 后**新增** `LOOKAHEAD_BLOCK: tl.constexpr` |
| `next_prefill_tokens` 布局 | `[max_num_reqs]`（单 token） | `[num_lookahead, max_num_reqs]`（多 lookahead，按 stride 寻址） |
| 越界 lookahead 语义 | 不写槽位（保留原值） | **写 0**：load mask 为 `in_lookahead & (pos < prefill_len)` 且 `other=0`，store mask 仅查 `in_lookahead` |

**报错形态解释**：UT 按旧签名传 8 个位置参数，在新签名下整体左移
错位绑定（`idx_mapping→next_prefill_tokens_stride`、
`query_start_loc→num_lookahead`、`prefill_lens→all_token_ids_ptr` 等），
末尾恰好剩 `prefill_lens_ptr` / `num_computed_tokens_ptr` /
`LOOKAHEAD_BLOCK` 三个参数无实参——与报错逐字吻合，确认签名错配。

**生产调用方**（同文件 `prepare_prefill_inputs()` :303）：

```python
num_lookahead = next_prefill_tokens.shape[0]
LOOKAHEAD_BLOCK = triton.next_power_of_2(num_lookahead)
```

**波及范围核查**（套件内 4 个引用同名 kernel 的文件）：

| 文件 | kernel 来源 | 是否受影响 |
|---|---|---|
| `npu/test_prepare_prefill_inputs_kernel.py` | vllm input_batch（本变更） | **是**（旧签名调用，本次修复） |
| `npu/test_input_batch_prepare_prefill_inputs_kernel.py` | vllm input_batch（同 kernel） | 否（codex 版，已按新签名 + 0 填充语义编写，期望值正确） |
| `npu/test_ar_prepare_prefill_inputs_kernel.py` | vllm AR speculator（另一同名 kernel，`spec_decode/autoregressive/speculator.py:653`） | 否（17 参数签名未变，#48892 未触碰） |
| `npu/test_prepare_prefill_inputs_kernel_speculator.py` | 同上 | 否 |

### 13.3 修复（UT 侧双签名适配，与 §11 slot mappings 同模式）

`npu/test_prepare_prefill_inputs_kernel.py` 修改点：

1. **签名探测**：模块级 `_HAS_LOOKAHEAD_ARGS`
   （`"num_lookahead" in arg_names`），新旧两代 kernel 均可运行，
   老 vllm 环境不破坏；
2. **launch 收敛**：`_launch_kernel()` 统一探测分支，新签名分支镜像
   生产调用方传参（`stride(0)` / `shape[0]` / `next_power_of_2`），
   旧签名分支保持原调用；`_make_next_prefill_tokens()` 按代际生成
   `[num_lookahead, max_num_reqs]` 或扁平 buffer；
3. **参考实现双语义**：新代所有 in-lookahead 槽位必写（越界为 0），
   旧代仅在 `next_pos < prefill_len` 时写单槽位；
4. **boundary 期望修正**：`num_computed + query_len == prefill_len` 时
   新代 kernel 写 0（非保留 -1）；inactive req 槽位不被写、保留
   sentinel（与 codex 版 `expected_next[:, 0] = 0` 写法对齐）；
5. **新增 `test_multi_lookahead_tokens`**（num_lookahead=3/4，旧签名
   环境 skipif）：覆盖多 lookahead 拷贝 + 越界 0 填充 + 不活跃请求
   保留 sentinel——#48892 新功能首次被精度 UT 覆盖。

**验证方式**：节点重跑
`pytest npu/test_prepare_prefill_inputs_kernel.py -v`
（预期 11 个原用例 PASS + 2 个新 multi-lookahead 用例 PASS，共 13）。

---

## 14. 变更八定性：hidden states UT 自身 buffer 尺寸 bug（非接口变更）

### 14.1 现象

2026-09-08 全量跑
`npu/test_prepare_input_hidden_states_and_embeddings_kernel.py`，
仅 1 个用例 FAIL（其余 27 个 PASS）：

```
test_prepare_input_hidden_states_and_embeddings[16-2048-3-tile_boundary-True-32-256]
IndexError: index 304 is out of bounds for dimension 0 with size 304
  common/prepare_input_hidden_states_and_embeddings_impl.py:88 in _ref
```

### 14.2 根因：UT 尺寸公式漏算 tile_boundary 的 bq 项（UT 自身 bug）

失败点在 **CPU 参考实现**（`_ref` 先于 kernel 执行，自己先越界）：

- `tile_boundary` 场景每请求 `query_len = bq + 2 = 34`（bq=32），
  16 请求共需 **544** token；
- buffer 尺寸公式为 `max(256, num_reqs*(nss+16)) = max(256, 16*19) = 304`，
  注释假设"query lens ≤ ~12 或 nss+3"，**未覆盖 bq+2 项**；
- 越界点复核：req 8 `query_start = 8*34 = 272`，
  `dst_max = 272 + num_reprefill(1) + num_input_hs(32) - 1 = 304`
  ——与报错 `index 304 ... size 304` 逐字吻合；
- 该用例为 strict_ut_027 高规格新增（bq=32），首次打破尺寸假设；
  此前通过的低 bq 用例（bq=4/16）恰未越界，属侥幸未爆。

**定性**：非接口变更、非精度问题、非 09-07 pull 引入——UT 构建时即
埋下的尺寸 bug，被新增高规格用例触发。若参考不崩，kernel 亦会向
304+ 越界写（设备侧 UB），故必须修 UT 尺寸而非绕过。

### 14.3 修复（common 实现单点改动）

`common/prepare_input_hidden_states_and_embeddings_impl.py` 尺寸公式
追加 `num_reqs * (bq + 2)` 项：

```python
num_tokens = max(
    256,
    num_reqs * (num_speculative_steps + 16),
    num_reqs * (bq + 2),
)
```

- 失败用例：`max(256, 304, 544) = 544`，恰等于总 token 数，
  `dst_max = 543 ≤ 543` 边界闭合；
- 其余 27 个通过用例：max 追加项只增不减，行为不变。

**验证方式**：节点重跑
`pytest npu/test_prepare_input_hidden_states_and_embeddings_kernel.py -v`
（预期 28 个用例全 PASS）。
