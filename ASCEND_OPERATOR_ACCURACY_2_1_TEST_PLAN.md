# vLLM Triton 算子类型与昇腾精度标准 2.1 测试方案

> 依据：附件《昇腾算子精度标准 2.1》以及当前工作区 vLLM/vLLM-Ascend 源码。
> 日期：2026-08-11。
> 范围：用户给出的 kernel 清单。忽略真正的 Triton device helper，但不忽略 vLLM `main` 中的 `_compute_cumulative_log_p_kernel` 与 `_compute_local_residual_mass_kernel`。

## 1. 范围澄清

### 1.1 本文忽略的 helper

`gumbel_block_argmax` 是 `@triton.jit` device helper：它被 `_gumbel_sample_kernel` 和 `_resample_kernel` 内联调用，不是独立生产 launch 入口。因此不把它列为独立验收算子，其分支由这两个父 kernel 覆盖。

### 1.2 不应被忽略的两个主线 kernel

以下两个 kernel 存在于 vLLM `main` 的：

`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`

它们由提交 `db808b396`（2026-06-30，block verification）新增：

- `_compute_cumulative_log_p_kernel[(num_reqs,)]`
- `_compute_local_residual_mass_kernel[(num_logits, vocab_num_blocks)]`

当前工作树检出的是 `releases/v0.24.0`，该分支没有包含上述提交，所以不能以当前文件的 `rg` 结果判定它们不存在。

同一提交中的真实改名关系是：

`_compute_block_stats_kernel` -> `_compute_local_logits_stats_kernel`

因此这两个名称在本文合并为同一个实现/测试对象，避免重复计数。

vLLM-Ascend 当前接受 `use_block_verification` 参数，但源码明确说明 NPU 尚未实现该路径。故这两个 kernel 的测试目前是“vLLM 主线 kernel 的 NPU 移植/支持验收方案”，不能用 Ascend 标准 rejection 路径间接冒充覆盖。

## 2. 标准 2.1 的可执行化解释

### 2.1 精度等级

| 等级 | 定位 | 最低用例数 | 每用例重复次数 | 浮点 Ratio 阈值：MARE / MERE / RMSE |
|---|---|---:|---:|---|
| L0 | 常规算子 | 5,000 | 50 | 10 / 2.0 / 2.0 |
| L1 | 重要算子 | 10,000 | 50 | 5 / 1.5 / 1.5 |
| L2 | LLM/MOE 关键算子 | 30,000 | 1,000 | 2 / 1.2 / 1.2 |

所有用例合计输出采样点不少于 1,000,000。L2 泛化用例与真实模型用例比例为 2:1。

这里的“每用例重复次数”属于验收/压测套件，不建议全部放入每次 PR 的快速 UT。工程上应拆为：

- PR strict UT：确定性语义、shape/tile 边界、特殊值、少量生产 shape。
- Nightly accuracy：满足 L0/L1/L2 的用例规模、数据分布与重复次数。
- Release acceptance：双标杆、全平台、百万输出点及统计复检。

### 2.2 四类基础判据

| 类别 | 本文中的典型算子 | 必要判据 |
|---|---|---|
| 非计算类 | copy/gather/scatter/flatten/mask/index preparation | 与 Golden bitwise 一致 |
| 整数计算类 | 长度、计数、slot/index 算术 | bitwise 一致；bitwise 不一致但 AE=0 也通过 |
| 浮点计算类 | softmax、temperature、penalty、residual mass | 双标杆；计算 MARE、MERE、RMSE Ratio |
| 随机数生成类 | Gumbel sampling 的随机部分 | KS test，`p>0.01`；N=100 时按标准公式要求约不少于 96 次通过 |

本清单包含融合算子。融合算子不能只选一个宽松判据：

- 浮点输出按浮点类验收。
- token id、rank、count、mask、length 等离散输出必须精确一致。
- 含 RNG 的算子同时做随机分布检验和确定性不变量检查。
- 只写部分输出的 kernel 必须检查未写区域保持初始化 sentinel。

### 2.3 浮点指标

以更高精度 CPU Golden 为真值，同时计算 GPU 标杆和 NPU 输出相对 Golden 的误差：

- `MARE = max(abs(actual-golden)/(abs(golden)+1e-7))`
- `MERE = mean(abs(actual-golden)/(abs(golden)+1e-7))`
- `RMSE = sqrt(mean((actual-golden)^2))`
- `Ratio = NPU误差 / max(GPU误差, err)`

L2 浮点算子要求三个 Ratio 分别不超过 `2 / 1.2 / 1.2`。

小值域必须额外统计 `ErrorCount`。阈值/误差为：

| dtype | Small Value Threshold | error |
|---|---:|---:|
| FLOAT16 | 2^-11 | 2^-16 |
| BFLOAT16 | 2^-8 | 2^-16 |
| FLOAT32 | 2^-14 | 2^-30 |

通过条件为 `ErrorCount_npu / max(ErrorCount_gpu,1) <= 2`。

### 2.4 通用输入生成规则

每个支持 dtype、shape、attr 必须正交组合，不能让某个 dtype 永远只对应某一种 shape。

- 均匀分布 `[-5,5]`。
- 小值均匀分布 `[-0.001,0.001]`。
- 正态分布，`mu in [-100,100]`、`sigma in [1,25]`。
- 离群点分布：随机约 0.1% 的参与计算浮点值放大 1000 倍，仍需满足合法值域。
- L2 增加真实推理输入：真实 vocab、batch、ragged query、spec steps、block tables。
- 特殊值：NaN、Inf、-Inf、`[-Inf,Inf]` 混合，并按公式和标杆判断。
- 空 tensor、标量、上下边界和非法 attr 仅在算子契约允许或 wrapper 应负责拦截时测试，不能向 kernel 传入生产永远不会出现的非法指针。

执行前浮点输出 buffer 初始化为 NaN，以发现漏写和越界。整数/布尔 buffer 使用不可能成为合法结果的 sentinel，并在尾部增加 guard 区。

## 3. 算子分类总表

“主类别”按实际数学行为划分；“补充判据”用于处理融合输出。

| 算子 | 主类别 | 建议等级 | 核心输出判据 | 说明 |
|---|---|---:|---|---|
| `_num_nans_kernel` | 整数计算（浮点谓词+计数） | L1 | count 精确一致 | NaN 识别语义必须单测 |
| `_prepare_rope_positions_kernel` | 整数计算 | L1 | positions 精确一致 | 含 position/delta 加法，不是纯 copy |
| `_scatter_num_accepted_kernel` | 非计算类（scatter） | L0 | bitwise | 未命中 request state 不变 |
| `_bad_words_kernel` | 非计算类（条件 mask） | L1 | bitwise | 命中 token 写 `-inf`，其他值原样保留 |
| `_temperature_kernel` | 浮点计算 | L1 | 双标杆 Ratio | 除法，temp 0/1 为 bitwise no-op |
| `_gumbel_sample_kernel` | 随机+浮点融合 | L2 | KS + 离散不变量 + processed logits 浮点 Ratio | helper 覆盖在父 kernel 内 |
| `_bias_kernel` | 浮点+mask 融合 | L1 | 浮点 Ratio；mask bitwise | bias 加法、allowed/min-token mask |
| `_topk_log_softmax_kernel` | 浮点规约 | L2 | 双标杆 Ratio | max/sumexp/log，数值敏感 |
| `_ranks_kernel` | 浮点比较规约、整数输出 | L1 | rank 精确一致 | ties 使用 `>=` |
| `_fill_logprob_token_ids_kernel` | 非计算类（select/gather） | L0 | token ids/mask bitwise | custom ids 覆盖 top-k |
| `_min_p_kernel` | 浮点规约+mask | L1 | 保留值 Ratio；mask bitwise | 阈值来自最大概率 |
| `_penalties_kernel` | 浮点计算 | L1 | 双标杆 Ratio | repetition/frequency/presence 融合 |
| `_bincount_kernel` | 整数计算 | L1 | bit mask/count 精确 | 含 atomic 写路径 |
| `_prompt_logprobs_token_ids_kernel` | 非计算类（索引准备） | L0 | token ids/mask bitwise | ragged prompt 边界关键 |
| `_prepare_prefill_inputs_kernel`（AR） | 非计算+整数索引 | L1 | 所有离散输出精确 | 含 shift、拒绝长度和 graph padding |
| `_prepare_decode_inputs_kernel` | 非计算+整数计算 | L1 | ids/positions/lengths 精确 | 含 clamp 和 padding program |
| `_update_draft_inputs_kernel` | 非计算+整数计算 | L1 | copy bitwise；位置精确 | hidden states 是复制，不用浮点容差 |
| `_prepare_dflash_inputs_kernel` | 非计算+整数计算融合 | L1 | 所有离散输出精确 | slot mapping、padding、anchor 分支 |
| `_compute_block_stats_kernel` / `_compute_local_logits_stats_kernel` | 浮点规约 | L2 | max/sumexp Ratio；argmax 精确 | 同一实现的版本前后名称 |
| `_compute_cumulative_log_p_kernel` | 浮点规约 | L2 | 双标杆 Ratio + 单调/上界不变量 | vLLM main block verification |
| `_compute_local_residual_mass_kernel` | 浮点规约 | L2 | block partial 与总和 Ratio | vLLM main block verification |
| `_rejection_kernel` | 随机+浮点+离散融合 | L2 | 接受序列不变量、分布检验、离散输出精确 | 标准/greedy/block verification 分支 |
| `_resample_kernel` | 随机+浮点融合 | L2 | KS/分布 + residual 数值 Ratio | 输出局部候选和值 |
| `_insert_resampled_kernel` | 浮点比较+scatter | L1 | token/count 精确 | local max 只用于选择，最终为离散输出 |
| `_flatten_sampled_kernel` | 非计算类（compact/copy） | L0 | bitwise | padding token 不得进入输出 |
| `_gather_block_tables_kernel` | 非计算类（gather/copy） | L0 | bitwise | 多 group、padded row 清零 |
| `_compute_slot_mappings_kernel` | 整数计算 | L1 | slot ids 精确 | div/mod、CP ownership、PAD_ID |
| `_apply_write_kernel` | 非计算类（scatter/copy） | L0 | bitwise | 单组/多组 pointer table |
| `_dcp_local_seq_lens_kernel` | 整数计算 | L1 | local lengths 精确 | CP rank/interleave 分配 |
| `_prepare_prefill_inputs_kernel`（input batch） | 非计算+整数索引 | L1 | ids/next token 精确 | 与 AR 同名但不是同一 kernel |
| `_prepare_pos_seq_lens_kernel` | 整数计算 | L1 | positions/seq lens 精确 | ragged qsl、event-driven、padding |
| `_combine_sampled_and_draft_tokens_kernel` | 非计算类（拼接/重排） | L1 | token ids/offsets 精确 | spec token ragged 组合 |
| `_get_num_sampled_and_rejected_kernel` | 整数计算 | L1 | counts 精确 | 接受、拒绝、chunked prefill |
| `_post_update_kernel` | 非计算+整数计算融合 | L1 | token/state/length 全部精确 | 原位更新，重点防越界多写 |
| `_post_update_num_computed_tokens_kernel` | 整数计算 | L0 | count 精确 | query length 差分累加 |
| `_expand_idx_mapping_kernel` | 非计算+整数生成 | L0 | mapping/local pos 精确 | ragged expand |
| `_apply_grammar_bitmask_kernel` | 非计算类（bitmask） | L1 | 允许值 bitwise；禁止值为 `-inf` | structured decoding 关键约束 |

## 4. 分组测试方案

### 4.1 非计算类与搬移/索引类

适用算子：

- `_scatter_num_accepted_kernel`
- `_bad_words_kernel`
- `_fill_logprob_token_ids_kernel`
- `_prompt_logprobs_token_ids_kernel`
- 两个 `_prepare_prefill_inputs_kernel`
- `_prepare_decode_inputs_kernel`
- `_update_draft_inputs_kernel`
- `_prepare_dflash_inputs_kernel`
- `_flatten_sampled_kernel`
- `_gather_block_tables_kernel`
- `_apply_write_kernel`
- `_combine_sampled_and_draft_tokens_kernel`
- `_post_update_kernel`
- `_expand_idx_mapping_kernel`
- `_apply_grammar_bitmask_kernel`

验收核心：

1. 所有离散输出、被复制的浮点 bit pattern、mask 与 sentinel 均 bitwise 一致。
2. 输出预填 sentinel；只允许目标区域改变。
3. 输入中使用可追踪值，例如 `request_id*100000 + position`，快速发现跨 request、跨 group、跨 block 串写。
4. 必测非连续、乱序 `idx_mapping`，以及 `R<Rmax`、`T<Tpadded`。
5. 若 wrapper 禁止空输入/重复目标，测试 wrapper 的异常拦截，不直接向 kernel 制造未定义数据竞争。

公共 shape：

- `R={1,3,16,64,128}`，其中 3/13/17 等非 2 次幂用于发现 padding 假设。
- `Rmax={R,R+3,32,128,256}`。
- ragged `Q=[0,1,7,31,64,127,256,511]` 的子集。
- spec steps `S={1,2,4,5,8}`。
- tile 边界按实际 `BLOCK_SIZE` 取 `B-1/B/B+1`。

### 4.2 整数计算类

适用算子：

- `_num_nans_kernel`
- `_prepare_rope_positions_kernel`
- `_ranks_kernel` 的离散结果
- `_bincount_kernel`
- `_compute_slot_mappings_kernel`
- `_dcp_local_seq_lens_kernel`
- `_prepare_pos_seq_lens_kernel`
- `_get_num_sampled_and_rejected_kernel`
- `_post_update_num_computed_tokens_kernel`

验收核心：

1. 使用独立 CPU reference，输出逐元素精确相等。
2. 对 div/mod/prefix-sum/count 测试边界的 `-1/0/+1`。
3. 对 atomic count 重复运行并验证确定性。
4. 整数输出不使用 `rtol/atol` 放宽；即使上游输入包含浮点，最终离散语义仍必须精确。

重点 case：

- NaN 位于 vocab tile 首尾和 8191/8192/8193。
- rank 的唯一最大、唯一最小、全 ties、多个 ties、`-inf` ties。
- slot mapping 的 block size `{16,32,64}`，position 为 `B-1/B/B+1`。
- CP size `{1,2,4,8}`、所有 rank、interleave `{1,2,4}`。
- sequence length `{0,1,63,64,65,2047,2048,8191,8192,8193}`。

### 4.3 普通浮点变换

适用算子：

- `_temperature_kernel`
- `_bias_kernel`
- `_min_p_kernel`
- `_penalties_kernel`

建议 L1，真实模型用例至少覆盖：

- `T={1,16,64,128}`。
- `V={32000,128256,151936}`，并加入 tile 的 `B-1/B/B+1`。
- dtype 覆盖实际支持的 fp16/bf16/fp32。
- 同 batch 混合参数，而不是一次 launch 所有请求使用相同 temperature/penalty。

每种算子都要拆开验证：

- 保留元素的浮点误差 Ratio。
- 被 mask 元素的 `-inf` 精确语义。
- no-op 参数的 bitwise 不变。
- 小值域 ErrorCount。
- Inf/-Inf/NaN 按公式和标杆单独判断，不能送入普通相对误差统计。

### 4.4 Softmax、logsumexp 与 block verification

适用算子：

- `_topk_log_softmax_kernel`
- `_compute_block_stats_kernel` / `_compute_local_logits_stats_kernel`
- `_compute_cumulative_log_p_kernel`
- `_compute_local_residual_mass_kernel`

全部建议 L2。Golden 使用 CPU fp64 稳定实现，GPU 使用同精度生产实现作为第二标杆。

共同 shape：

- `V={8191,8192,8193,32000,128256,151936}`。
- `R={1,8,16,64}`。
- `S={1,3,5,8}`。
- `num_logits=sum(valid_draft_i+1)`，必须包含 ragged 请求。

共同数值分布：

- 全相等 logits。
- 单个极大值。
- 动态范围约 200。
- 所有值整体加常数，验证 softmax/logprob 平移不变性。
- 全 `-inf`、部分 `-inf`、合法范围内离群点。
- target=draft、target/draft 明显不同、概率极接近。

#### `_compute_cumulative_log_p_kernel` 专项

- grid 为 `[R]`，每个 program 顺序处理本请求的 draft steps。
- 输出 `[N]` 只写每请求前 S 个 draft 位置，bonus 位置不写。
- target=draft 时输出全 0。
- 每步概率比为 1/2 时应形成单调递减的负 log 累积。
- 某一步概率比大于 1 时仍受 `log_p<=0` 上界约束。
- temperature=0 请求早退，sentinel 必须保持。
- 覆盖提交 `a47f38f82` 修复的 int64 offset/大 storage offset 回归。

#### `_compute_local_residual_mass_kernel` 专项

- grid 为 `[N,ceil(V/8192)]`，输出同 shape。
- 只在 block verification、完整 draft logits 路径 launch。
- local position 0、bonus position、temperature=0 均早退，输出 sentinel 不变。
- 有效位置计算 `sum(max(p*M_target-M_draft,0))` 的 block partial。
- 先逐 block 对比 fp64 reference，再比较跨 block 总和。
- target=draft 且 `p=1` 时 residual mass 为 0。
- 构造仅一个 token/一个 block 有正 residual，验证 block 定位和尾块 mask。

### 4.5 随机与 rejection sampling

适用算子：

- `_gumbel_sample_kernel`
- `_rejection_kernel`
- `_resample_kernel`
- `_insert_resampled_kernel`（离散收尾阶段）

`gumbel_block_argmax` 不独立验收，由上述父 kernel 覆盖。

测试必须分成三层：

1. **确定性路径**：temperature=0、固定大 margin、固定 residual，仅验证精确 token、count、早退和 mask。
2. **随机可复现性**：固定 seed/position 重复执行结果一致；改变 seed 应产生变化，不能所有 seed 输出同一序列。
3. **分布正确性**：大量样本与 target categorical/Gumbel Golden 做 KS 或更适合离散分布的卡方/频率区间检验。标准明确要求 RNG 使用 KS，故验收报告至少保留 KS `p>0.01` 结果。

生产 shape：

- `R={1,16,64}`、`S={1,3,5,8}`。
- `V={32000,128256}`，统计分布测试另用较小 `V={4,16,128}` 获得足够的每类样本。
- greedy、标准 rejection、block verification。
- 有/无完整 draft logits。
- 全接受、首步拒绝、中间拒绝、bonus。

随机测试不能只断言“token 在合法范围”；必须同时验证采样频率和 target 分布一致。对于稀有 token，合并低期望频数桶或增加样本量，避免统计检验失效。

### 4.6 KV cache 与图模式专项

适用算子：

- `_gather_block_tables_kernel`
- `_compute_slot_mappings_kernel`
- `_apply_write_kernel`
- `_dcp_local_seq_lens_kernel`
- 所有带 `Rmax/Tpadded` 的 input preparation kernel

必测：

- block size `{16,32,64}`。
- position `0/B-1/B/B+1/2B-1`。
- KV groups `{1,2,4}`，各 group 使用不同 stride/block id。
- `CP_SIZE={1,2,4}`、所有 rank、interleave `{1,2,4}`。
- graph padding buffer 先写入合法但陈旧的数据，launch 后必须被 0/PAD_ID 覆盖。
- `R=13,Rmax=32`、`T<Tpadded` 等非整齐形状。

## 5. 特殊值和异常规则按算子裁剪

标准要求覆盖 NaN/Inf/空 tensor，但不能机械地给所有 kernel 注入所有特殊值：

- logits 浮点计算：覆盖 NaN、Inf、-Inf，并根据数学公式判断。
- token ids、offsets、block ids：不注入 NaN；测试合法边界和 wrapper 对负数/越界的拦截。
- RNG seed/position：测试 0、最大合法值、重复 seed、不同 position；不使用浮点特殊值规则。
- 空 batch/空 token：若 wrapper 明确不 launch，则测试 wrapper no-op；不要启动 0 grid。
- vocab size 0 通常不属于模型合法配置，应测试配置层拒绝，而非要求 kernel 定义结果。
- 重复 scatter 目标若会造成数据竞争，应由上层禁止；UT 验证禁止条件，而不是接受非确定输出。

## 6. 推荐执行矩阵

### 6.1 PR strict UT

- 每算子 6–15 个定向 case。
- tiny semantic shape + tile 边界 + 1 个真实生产 shape。
- 确定性算子重复 3–10 次验证一致。
- 全部检查 sentinel/guard。

### 6.2 Nightly accuracy

- L0/L1/L2 按标准达到 5K/10K/30K 用例规模。
- L0/L1 每用例 50 次，L2 每用例 1000 次。
- L2 泛化:模型用例为 2:1。
- 输出样本点合计不少于 100 万。
- 数据分布按均匀、小值、正态、离群点配比生成。

### 6.3 Release acceptance

- CPU fp64 Golden + GPU 标杆 + NPU 被测实现。
- 覆盖目标推理平台。
- 输出 MARE/MERE/RMSE、Ratio、小值域 ErrorCount、Inf/NaN 分类结果。
- 随机算子输出 KS p-value、通过次数和最低要求。
- 单用例失败时更换 seed 复检；N 不得少于 200，推荐 1000，并计算 bootstrap 95% CI。

## 7. 最终通过条件

一个算子只有同时满足下列条件才算通过：

1. 分类对应的数值/bitwise/随机分布指标达标。
2. 所有离散元数据输出精确一致。
3. 特殊值行为符合 Golden、标杆和计算公式。
4. 未写区域、padding、guard 区无变化或被正确清理。
5. 确定性算子重复执行完全一致。
6. shape 覆盖真实生产规模、ragged mapping 和 kernel tile 边界。
7. wrapper 的非法输入有明确资料约束与代码拦截。
8. `_compute_cumulative_log_p_kernel` 与 `_compute_local_residual_mass_kernel` 使用 vLLM `main` 的真实实现直接 launch；在 Ascend 尚未接入 block verification 时不得标记为已由 NPU rejection wrapper 覆盖。
