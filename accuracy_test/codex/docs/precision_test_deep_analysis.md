# 算子深入分析：rejection_sampler 系列 helper/kernel + ranks + tl_rand32

> 本文专门分析以下 6 个被测算子，供补充测试与报告使用：
> 1. `_compute_global_residual_mass`（bench helper）
> 2. `_compute_global_target_argmax`（bench helper）
> 3. `_compute_cumulative_log_p_kernel`（launch kernel）
> 4. `_compute_local_residual_mass_kernel`（launch kernel）
> 5. `tl_rand32`（bench helper）
> 6. `_ranks_kernel`（launch kernel）
>
> 逐一说明：**位置、功能、参数签名、算法特性、调用链（由谁调用/调用了谁）、数据类型与边界、测试现状、建议补充的测试点**。

---

## 0. 顶层背景：block-verification（Sun et al., 2024）算法链

这 6 个算子大多属于 vLLM speculative decoding 的 **block verification** 拒绝采样路径（https://arxiv.org/abs/2403.10444）。整条数据流为：

```
_compute_local_logits_stats_kernel       (预计算 各 block 的 target/draft max+sumexp+argmax)
   ↓
_compute_cumulative_log_p_kernel         (预计算 每个 draft 位置的累积 joint log-p)
   ↓
_compute_local_residual_mass_kernel      (预计算 每个位置×block 的残差质量 M 局部项)
   ↓
_rejection_kernel                        (逐 token 判定 accept/reject)
   ├─ _compute_global_residual_mass      (helper，跨 block 归约残差质量)
   ├─ _compute_global_target_argmax      (helper，求 target 全局 argmax)
   └─ _compute_global_logsumexp          (helper，求全局 logsumexp)
   └─ tl_rand32                          (helper，取随机数 u)
```

其中前 3 个 + `_rejection_kernel` 都是 **kernel**（可 `[grid]` launch），而 `_compute_global_*` 与 `tl_rand32` 是 **helper**（无 program_id，被 kernel 内联调用）。

---

## 1. `_compute_global_target_argmax`（bench helper）

### 1.1 位置
- **定义：** `C:\Users\x30084275\Desktop\git\vllm\vllm\v1\worker\gpu\spec_decode\rejection_sampler_utils.py:95`（`@triton.jit def _compute_global_target_argmax`）
- **类型：** helper（`@triton.jit` 纯函数，`return` 返回值，无 `tl.program_id`，不能独立 launch）

### 1.2 功能
给定某 logit 位置的全局 target 分布，返回 **target 分布的全局 argmax token id**。做法：各 block 的 local max 里找最大块，读出该块记录的 local argmax（即全局最大 token）。

### 1.3 参数签名
```
_compute_global_target_argmax(
    target_local_max_ptr,      # [num_logits, num_blocks] fp32 每块最大 logit
    target_local_max_stride,
    target_local_argmax_ptr,   # [num_logits, num_blocks] int64 每块 argmax token
    target_local_argmax_stride,
    logit_idx,                 # 当前 logit 位置
    vocab_num_blocks,          # 实际块数
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
)
→ (int64) 全局 argmax token id
```

### 1.4 算法特性 / 关键实现点
- `tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)` 加载各块 max，越界块 `other=-inf`。
- `max_block_idx = tl.argmax(local_max, axis=0)` 找最大块。
- 通过 `target_local_argmax_ptr[... + max_block_idx]` 读出该块的 argmax token。
- **数值依赖：** 依赖 `_compute_local_logits_stats_kernel` 写的 `target_local_max` / `target_local_argmax`。若某块全 `-inf`（掩码导致），`argmax` 块索引可能指向 -inf 块——需注意。
- **仅 greedy 路径使用**（见调用链）。

### 1.5 调用链（谁调用它）
- **被 `_rejection_kernel` 调用**（`rejection_sampler_utils.py:568`），在 `USE_BLOCK_VERIFICATION=False` 且 `is_greedy`（temp==0）分支：
  ```
  _rejection_kernel (kernel) @ :460
      └─ elif accepted: if is_greedy:
            target_argmax = _compute_global_target_argmax(...) @ :568
  ```

### 1.6 测试现状与建议
- 现状：本套件**无独立 UT**。README 曾把 `_compute_global_target_argmax` 列为"源码中不存在"，与当前 checkout 不符（实际在 `:95` 存在）。
- 建议补充测试：**C 类（测试 wrapper kernel 包裹 helper）**。
  - 构造多块 `target_local_max` / `target_local_argmax`（含块数非 2 幂、某块全 -inf、多个等大 block max）。
  - 断言返回的全局 argmax token 与 CPU 参考（`argmax` over all blocks）一致。
  - 边界：某块全 -inf 时行为；`PADDED_VOCAB_NUM_BLOCKS > vocab_num_blocks` 时 padding 以 -inf 填充不干扰。

---

## 2. `_compute_global_residual_mass`（bench helper）

### 2.1 位置
- **定义：** `rejection_sampler_utils.py:48`（`@triton.jit def _compute_global_residual_mass`）
- **类型：** helper

### 2.2 功能
计算 block-verification 中某个 draft 位置的**残差质量（residual mass）**，即 `h` 阈值分子的跨 block 归约值。两种模式：
- `HAS_DRAFT_LOGITS=True`：直接累加各 block 的 `local_residual_mass`（`tl.sum(partials)`）。
- `HAS_DRAFT_LOGITS=False`（one-hot draft）：闭式 `prefix_joint_ratio * (1 - M_b(draft_token))`，其中 `M_b` 是 target 在 draft token 处的概率质量。

### 2.3 参数签名
```
_compute_global_residual_mass(
    local_residual_mass_ptr,    # [num_logits, num_blocks] fp32
    local_residual_mass_stride,
    prefix_joint_ratio,         # fp32 该位置前置 joint 概率 p
    target_logits_ptr,          # [num_logits, V]
    target_logits_stride,
    target_local_max_ptr,
    target_local_max_stride,
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    draft_sampled_ptr,          # [num_logits] int64 draft 采样 token
    logit_idx,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
)
→ (fp32) 残差质量
```

### 2.4 算法特性
- `+1` 前缀逻辑：调用方传 `logit_idx + 1`（见 `_rejection_kernel:552`）。
- 依赖 `_compute_local_residual_mass_kernel` 产出的 `local_residual_mass`（HAS_DRAFT_LOGITS=True）。
- one-hot 分支内又调用 helper `_compute_global_logsumexp`（`:78`）。
- **数值敏感：** `1.0 - m_b`、`prefix_joint_ratio * (...)`，当 `m_b` 接近 1（目标几乎确定在 draft token）时残差趋 0。

### 2.5 调用链
- **被 `_rejection_kernel` 调用**（`:541`），在 `USE_BLOCK_VERIFICATION and not is_greedy` 分支：
  ```
  _rejection_kernel (kernel) @ :460
      └─ if USE_BLOCK_VERIFICATION and not is_greedy: @ :535
            if i < num_draft_tokens - 1:
                residual_mass = _compute_global_residual_mass(...) @ :541
                denom = residual_mass + 1.0 - prefix_joint_ratio
                h = tl.where(denom > 0.0, residual_mass / denom, 1.0)
  ```

### 2.6 测试现状与建议
- 现状：**无独立 UT**。
- 建议补充测试：**C 类（helper wrapper 测试）**。
  - HAS_DRAFT_LOGITS=True：构造多块 local_residual_mass + 随机 logit，断言返回等于各块之和（CPU 参考 `sum`）。
  - HAS_DRAFT_LOGITS=False：构造单点 draft，用 `_compute_global_logsumexp` 一致的计算做闭式参考，断言 `prefix_joint_ratio*(1 - exp(target_logit - target_lse))`。
  - 边界：`prefix_joint_ratio` 为 0 / 1；`m_b` 接近 1（残差≈0）；vocab 块数非 2 幂。

## 3. `_compute_cumulative_log_p_kernel`（launch kernel）

### 3.1 位置
- 定义：rejection_sampler_utils.py:296（@triton.jit def _compute_cumulative_log_p_kernel）
- 类型：kernel（tl.program_id(0)，可 [grid] launch；通常 grid=(num_reqs,)，num_warps 常设 1）

### 3.2 功能
为每个请求，从头到尾累积 joint log-probability 前缀 log_p（block verification 的 p_i 前序累计），写入 cumulative_log_p_ptr[logit_idx]：
    log_p = min(log_p + (target_logprob - draft_logprob), 0.0)
- 每个 program 负责一个请求（req_idx = tl.program_id(0)）。
- 只对非 greedy（temp != 0）的请求计算；greedy 直接 return（temp==0 时 :336）。

### 3.3 参数签名
    _compute_cumulative_log_p_kernel(
        cumulative_log_p_ptr,       # [num_logits] fp32 输出
        target_logits_ptr,          # [num_logits, V]
        target_logits_stride,
        target_local_max_ptr,
        target_local_max_stride,
        target_local_sumexp_ptr,
        target_local_sumexp_stride,
        draft_sampled_ptr,          # [num_logits] int64
        draft_logits_ptr,           # [max_num_reqs, num_spec_steps, V]
        draft_logits_stride_0,
        draft_logits_stride_1,
        draft_local_max_ptr,
        draft_local_max_stride,
        draft_local_sumexp_ptr,
        draft_local_sumexp_stride,
        cu_num_logits_ptr,          # [num_reqs + 1] int64
        idx_mapping_ptr,            # [num_reqs] int32
        temp_ptr,                   # [max_num_reqs] fp32
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
        HAS_DRAFT_LOGITS: tl.constexpr,
    )

### 3.4 算法特性 / 关键实现点
- 串行 for 累加：for step in range(num_draft_tokens)，逐位置读 draft token、算 target/draft 条件概率差并 tl.minimum(..., 0.0) 截断（保证 log_p <= 0）。
- 内部调用 helper _compute_global_logprobs_and_logsumexp（:343），该 helper 再调 _compute_global_logsumexp。
- 依赖前置 _compute_local_logits_stats_kernel 的 local max/sumexp。
- 边界/易错点：
  - num_draft_tokens = end_idx - start_idx - 1（减 1 表示扣除 bonus token）。
  - 读取 draft_sampled_ptr + logit_idx + 1（跳过 bonus，draft token 从下一位置读）。
  - greedy 请求整段早退，cumulative_log_p 保持未写。

### 3.5 调用链
- 被上游 wrapper rejection_tokens / rejection sampler 调用（launch kernel）。
- 输出 cumulative_log_p 被 _compute_local_residual_mass_kernel（读 :449 cumulative_log_p_ptr + logit_idx - 1）与 _rejection_kernel（读 :538）使用。
- 现有测试：test_compute_block_stats_kernel.py 已覆盖它（与 _compute_local_logits_stats_kernel 级联，CPU 参考校验，E 类）。

### 3.6 测试现状与建议
- 现状：已有 test_compute_block_stats_kernel.py::test_cumulative_log_p 覆盖（num_reqs∈{1,2} x num_draft∈{1,2,3} x vocab∈{128,1024}，rtol=1e-4）。
- 建议补充：greedy（temp=0）早退路径显式校验（输出保持未写/参考跳过）；log_p 截断到 <=0 的边界；HAS_DRAFT_LOGITS=False 的 one-hot 分支；更大 batch 与块 padding 非 2 幂。

---

## 4. `_compute_local_residual_mass_kernel`（launch kernel）

### 4.1 位置
- 定义：rejection_sampler_utils.py:371（@triton.jit def _compute_local_residual_mass_kernel）
- 类型：kernel（二维 grid：tl.program_id(0)=logit_idx，tl.program_id(1)=block_idx）

### 4.2 功能
为每个 (logit 位置, vocab 块) 计算一个局部的残差质量（block verification 中 max(p_i * M_b - M_s, 0) 的块内求和项），写入 local_residual_mass_ptr。后续被 _compute_global_residual_mass 按块累加。

### 4.3 参数签名
    _compute_local_residual_mass_kernel(
        local_residual_mass_ptr,        # [num_logits, num_blocks] fp32 输出
        local_residual_mass_stride,
        cumulative_log_p_ptr,           # [num_logits] fp32
        target_logits_ptr,              # [num_logits, V]
        target_logits_stride,
        target_local_max_ptr,
        target_local_max_stride,
        target_local_sumexp_ptr,
        target_local_sumexp_stride,
        draft_logits_ptr,               # [max_num_reqs, num_spec_steps, V]
        draft_logits_stride_0,
        draft_logits_stride_1,
        draft_local_max_ptr,
        draft_local_max_stride,
        draft_local_sumexp_ptr,
        draft_local_sumexp_stride,
        expanded_idx_mapping_ptr,       # [num_logits]
        expanded_local_pos_ptr,         # [num_logits]
        temp_ptr,                       # [max_num_reqs]
        vocab_size,
        num_speculative_steps,
        vocab_num_blocks,
        BLOCK_SIZE: tl.constexpr,
        PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    )

### 4.4 算法特性 / 关键实现点
- 早退规则：draft_step_idx == 0 或 draft_step_idx >= num_speculative_steps（bonus）时直接 return（:410，注释说明 h 只看前一个位置）；temp == 0.0（greedy）也 return（:418）。
- 核心公式（:448-452）：
    p = exp(cumulative_log_p[logit_idx - 1])
    m_b = exp(target_log_probs)
    m_s = exp(draft_log_probs)
    partial = sum( max(p * m_b - m_s, 0.0) , over block)
- 内部调用 helper _compute_global_logprobs_and_logsumexp（:424，HAS_DRAFT_LOGITS=True 硬编码）。
- 读 cumulative_log_p_ptr + logit_idx - 1（依赖前一个位置的累积概率）。

### 4.5 调用链
- 被上游 wrapper 调用（launch kernel）；输出被 _rejection_kernel 经 _compute_global_residual_mass 消费。
- 现有测试：无独立 UT（README 曾列为"源码中不存在"，实际在 :371 存在）。

### 4.6 测试现状与建议
- 现状：无独立 UT。
- 建议补充：A/E 类（直接 launch，先跑 _compute_cumulative_log_p_kernel 与 _compute_local_logits_stats_kernel 产出前置数据）。
  - 校验每个 (logit, block) 的局部残差质量 vs CPU 参考 max(p*exp(target_lp) - exp(draft_lp), 0) 逐块和。
  - 早退：draft_step==0、bonus、greedy 三路应保持输出未写。
  - 边界：p=0 或 p=1；m_b<=m_s（残差=0）；块非 2 幂 padding。
---

## 5. `tl_rand32`（bench helper）

### 5.1 位置
- 定义：vLLM 上游 gumbel.py:77（@triton.jit def tl_rand32）
- 类型：helper

### 5.2 功能
生成一个 FP32 均匀随机数（tl.rand），用于 Gumbel 噪声 / rejection 阈值。可指定是否排除 0（includes_zero=False 时 tl.maximum(u, _TL_RAND_MIN) 下夹）。是上游 tl_rand64 的 FP32 版本；Ascend A3 无法编译 FP64 的 tl_rand64，故生产路径走 tl_rand32 / tl.rand。

### 5.3 参数签名
    tl_rand32(seed, offset, includes_zero: tl.constexpr) -> fp32 均匀随机数
- 底层 tl.rand(seed, offset)（Philox 派生）。

### 5.4 算法特性
- 与 tl_rand64 对比（gumbel.py:62，用 tl.randint4x 拼 64bit * 2^-64）：tl_rand32 直接用 tl.rand，精度 FP32。
- _TL_RAND_MIN 下夹保证 u > 0，避免 -log(u) 除零/负无穷。
- 在 gumbel_block_argmax 中按 USE_FP64 二选一：USE_FP64=True 用 tl_rand64，否则用 tl_rand32（gumbel.py:141-145）。

### 5.5 调用链
- 被 gumbel_block_argmax 调用（gumbel.py:145，u = tl_rand32(gumbel_seed, block, includes_zero=False)），最终被 _gumbel_sample_kernel（kernel）调用。
- 被 _rejection_kernel 调用（rejection_sampler_utils.py:534，u = tl_rand32(seed, pos, includes_zero=False)）。import 自 gumbel（:6）。

### 5.6 测试现状与建议
- 现状：无独立 UT（test_tl_rand64.py / test_tl_rand64_patch.py 测的是 tl_rand64 的 FP32 替代契约，不是直接 tl_rand32）。README 曾把 tl_rand32 列为"源码中不存在"，与当前 checkout 不符——实际存在（gumbel.py:77）。
- 建议补充：C 类（helper wrapper 测试），仿照 test_tl_rand64.py：
  - 范围：includes_zero=False -> 0 < u <= 1；True -> 0 <= u <= 1。
  - 统计均匀性：大样本均值约 0.5。
  - seed/offset 派生：同 seed+不同 offset 应不同；不同 seed 不同。
  - 与 _rejection_kernel 组合的行为（u 参与 accept 判定）。

---

## 6. `_ranks_kernel`（launch kernel）

### 6.1 位置（上游与 Ascend 两处实现）
- vLLM 上游：logprob.py:61（@triton.jit def _ranks_kernel）
- vLLM-Ascend：vllm_ascend\worker\v2\sample\logprob.py:88（@triton.jit(do_not_specialize=[...]) def _ranks_kernel）
- 类型：kernel（两端都能 [grid] launch，但拓扑不同，见下）

### 6.2 功能
为每个请求计算 sampled token 在 logits 中的 rank（即 logits >=（或 >）sampled token 值的个数），写入 selected_token_ranks。即"该 token 在概率排序中排第几"（值越大越靠后）。用于 logprobs 输出的 rank 统计。

### 6.3 参数签名（两端差异）
- 上游（vLLM）logprob.py:61：
    _ranks_kernel(output_ptr, logits_ptr, logits_stride, token_ids_ptr, vocab_size,
                  BLOCK_SIZE: tl.constexpr)
    grid = (batch_size,)            # 每个 program 一个请求
    n = sum((logits >= x) for each vocab block)   # 用 >=
- Ascend logprob.py:88：
    _ranks_kernel(output_ptr, logits_ptr, logits_stride, token_ids_ptr, vocab_size,
                  batch_size, rows_per_core, BLOCK_SIZE: tl.constexpr)
    grid = (NUM_CORES,)             # 每个 program 处理多行（按 vectorcore 数分片）
    core_id = tl.program_id(0)
    start_row = core_id * rows_per_core
    for req_idx in range(start_row, end_row):
        n_vec += (logits > x).to(int32)   # 用 > 且向量累加后 sum

### 6.4 算法特性 / 关键实现点
- 等值处理差异：上游用 logits >= x，Ascend 用 logits > x。对存在重复 logit 值等于 sampled 的情形，两端 rank 会差 1——这是关键不一致点，补测时应明确。
- 上游：BLOCK_SIZE 迭代 vocab，n += tl.sum((logits >= x).to(tl.int32))。
- Ascend：每向量核处理多行（rows_per_core = cdiv(batch_size, NUM_CORES)），用 tl.zeros([BLOCK_SIZE]) 向量累加后再 tl.sum，BLOCK_SIZE=8192。
- do_not_specialize 让 batch_size / rows_per_core 不特化（减少编译变体）。
- mask 处理：越界 block 用 other=-inf，由于 -inf >= x / -inf > x 为 false，不会计入 rank。

### 6.5 调用链
- 被 Ascend compute_topk_logprobs 调用（logprob.py:152，_ranks_kernel[grid](...)，在其中计算 selected_token_ranks）。
- 上游被 compute_token_logprobs / sampler 相关流程使用。

### 6.6 测试现状与建议
- 现状：已有 test_compute_topk_logprobs.py（Ascend wrapper 级）间接覆盖 _ranks_kernel（rank 用 torch.equal，PyTorch 参考 (logits > sampled).sum）。
- 建议补充直接测试：A/B 类（直接 launch _ranks_kernel）：
  - 直接 launch Ascend _ranks_kernel，校验每行 rank == CPU (logits > token).sum()。
  - 重点测等值边界：构造多个 logit 恰等于 sampled token 的重复值，明确 > 与 >= 语义。