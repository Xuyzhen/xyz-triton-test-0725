# 已有精度测试报告（existing_accuracy_tests 全量盘点）

> 面向：从 vLLM / vLLM-Ascend 搬运到 NPU(A3) 的**已有精度测试**逐一盘点。
> 每题给出：被测对象、对象类型、调用类型、参考基准、容差判据、覆盖场景、覆盖盲区、用例数。
> **结果/状态列（通过/数值）留空待运行后填写。** 本文档与 `precision_test_calltype_report.md`（A–F 调用类型）配套，调用类型沿用其定义。

## 0. 环境与口径（复现前提）

| 项 | 值 | 说明 |
| --- | --- | --- |
| 硬件 | 昇腾 NPU（A3） | 所有 kernel 直接 `torch.device("npu")` launch |
| 依赖 | vllm / vllm_ascend 已安装 | 测试 import 生产 kernel 与 wrapper |
| 设备初始化 | 各文件内联 `init_device_properties_triton()` + `npu` | 不依赖外部 fixture |
| XFAIL 策略 | conftest.py 的 `pytest_runtest_makereport` 钩子 | 把「后端/NPU 编译兼容失败」(compilation failed / unsupported dtype 等) 标为 XFAIL(skip)；**数值断言失败仍算精度失败** |
| 结果列口径 | PASS / FAIL / XFAIL(skip) + 关键数值 | XFAIL≠通过，标记为「未受验」 |
| 测试用例数口径 | 参数化展开后全部组合 | 括号内为展开 case 数 |

> 说明：结果列三态中 **XFAIL(skip)** 特指"编译/后端不兼容"导致未跑，不代表精度通过，应在结论中与 PASS 区分。

---
# 一、from_vllm/（30 文件）

## 1.1 逐 UT 条目表（结果列待填）

| 文件 | 被测对象 | 对象类型 | 调用类型 | 参考基准 | 容差判据 | 覆盖场景 | 覆盖盲区 | 用例数 | 结果 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test_apply_write_kernel | `_apply_write_kernel` | kernel | A | CPU 参考 | 精确(rtol=0/atol=0) | 单组 rows 1/4 × cols 16/32 | 非0列偏移、组>4、旧版multi-group skip | 2 |  |
| test_bias_kernel | `_bias_kernel` | kernel | A | CPU 参考 | 1e-5 | allowed: tokens 1/4/8×vocab 128/1024; bias: 1/4×64 | allowed+bias 叠加、min_len 临界、MAX 上限、vocab>8192 | 3 |  |
| test_compute_block_max_and_sumexp | `_compute_local_logits_stats_kernel`(含`_compute_max_and_sumexp`) | kernel | A | CPU+Pytorch max/exp | 1e-5; 全-inv 精确 | nlg 1/2/4×vocab 128/1024/8192×steps 2/3; greedy/non-greedy | 块不可分、跨多块、HAS_DRAFT_LOGITS=False 分支 | 2 |  |
| test_compute_block_max_and_sumexp_patch | Ascend `_compute_block_stats_kernel` | kernel | A | CPU 参考 | 1e-5 | vocab 15/31/65(非对齐/跨块), block 16, HAS_DRAFT=False | draft/argmax 未校验、greedy 未测、单 num_logits | 1 |  |
| test_compute_block_stats_kernel | `_compute_cumulative_log_p_kernel`(先跑 stats 前置) | kernel | E | CPU 全局 LSE | 1e-4 | reqs 1/2×draft 1/2/3×vocab 128/1024; temp=1.0 | greedy(temp=0) 输出、PADDED 块污染、跨多块、HAS_DRAFT=False | 1 |  |
| test_compute_block_stats_kernel_patch | Ascend `_compute_block_stats_kernel`(=原 local_logits_stats) | kernel | A | CPU `_expected_block_stats` | max/sumexp 1e-5; greedy argmax 精确 | non-greedy 2×2 全量 target+draft; greedy HAS_DRAFT=False; vocab 37 | 跨多块 argmax、draft 全-inv、temp 混合、idx_mapping 非全0 | 2 |  |
| test_compute_global_logsumexp | `_compute_global_lse`(helper) | helper | C | CPU `_global_logsumexp_ref` | 1e-5; 全-inv 精确 | nlg 1/4×blocks 1/2/4; 全-inv; 单块 | 极大动态范围、NaN、PADDED 脏块 | 3 |  |
| test_compute_global_logsumexp_patch | `_compute_global_lse`(Ascend 别名) | helper | C | CPU `_global_logsumexp_ref` | 1e-5 | 同非patch | 同非patch; 未与上游头对头 | 3 |  |
| test_fill_logprob_token_ids_kernel | `_fill_logprob_token_ids_kernel` | kernel | A | CPU `_fill_logprob_token_ids_ref` | 精确+bool | batch 1/4/8×topk 0/3/5; custom 混合 | 截断、fill 满 PADDED_COLS | 2 |  |
| test_fill_logprob_token_ids_kernel_patch | `compute_topk_logprobs`(Ascend 生产) | 生产wrapper | D | Pytorch topk/log_softmax/gather | token_ids/ranks 精确; logprobs 1e-4 | 4 组合 (batch×vocab×logprobs): 48×1024×5, 96×1024×0, 24×1519×1, 1×320×10 | num_lp>vocab、并列topk、-inf、fp16、无 custom | 1(4组) |  |
| test_flatten_sampled_kernel | `_flatten_sampled_kernel` | kernel | A | CPU `_flatten_sampled_ref` | 精确+行为(-1 保留) | reqs 1/2/4/8×steps 1/3/5; 全0; 单req10步 | 越界写、非均匀起始、大 reqs、非 int64 | 3 |  |
| test_gather_block_tables_kernel | `_gather_block_tables_kernel`(含`_load_ptr`) | kernel | A | CPU `_gather_block_tables_ref` | 精确+padding 行为 | groups 1/2/4×reqs 4/8×blocks 64/128 | 非全满、idx 重排、BLOCK 不可分 | 2 |  |
| test_gumbel_block_argmax | `gumbel_block_argmax`(helper) | helper | C | 行为(期望 idx) | idx 精确(==1/==0) | temp=0; APPLY_TEMP+temp=0; 大 margin 不翻转 | 小 margin 真实分布、USE_FP64、PER_TOKEN_COL、block_start>0 | 3 |  |
| test_gumbel_block_argmax_patch | `_npu_gumbel_block_argmax`(Ascend) | helper | C | 行为+精确(processed_logits) | idx/val 精确; processed 1e-5 | temp=0 全量; temp; 大 margin; processed_logits=logits/temp vocab256 | 小 margin、seed 分布、与上游未头对头 | 4 |  |
| test_insert_resampled_kernel | `_insert_resampled_kernel` | kernel | A | CPU `_insert_resampled_ref` | 精确+行为 | reqs 1/2/4×steps 1/3, vocab 4096; greedy 非 bonus 跳过 | temp 混合、resampled 全-inv、tie、中间位置、PADDED 脏块 | 3 |  |
| 文件 | 被测对象 | 对象类型 | 调用类型 | 参考基准 | 容差判据 | 覆盖场景 | 覆盖盲区 | 用例数 | 结果 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test_insert_resampled_kernel_patch | Ascend `_insert_resampled_kernel` | kernel | A | CPU `_insert_resampled_ref` | 精确(sampled/num_sampled) | reqs 1/2/4×steps 1/3; vocab 4096/1024/512; temp=0 skip; bonus | temp>0 写回、复杂 index 映射、idx 非均匀、padded 越界 | 3(~8) |  |
| test_load_ptr | `_load_ptr`(helper) | helper | C | 精确值 | int32/strided 精确; fp32 1e-5 | int32 单值; fp32; 连续 10 个 int32 步进 | 多元素越界、非16对齐、int64/bool dtype、ptr_to_ptr | 3 |  |
| test_prepare_dflash_inputs_kernel | `_prepare_dflash_inputs_kernel`(vanilla) | kernel | A | 手工逐字段断言 | 精确单值== | reqs 1/2×ctx 2/4×query 2/3; graph padding; SAMPLE_FROM_ANCHOR | 与多req组合、num_sampled=0、超长 clamp | 3(~14) |  |
| test_prepare_dflash_inputs_kernel_patch | Ascend `_prepare_dflash_inputs_kernel_ascend` | kernel | A | CPU `_prepare_dflash_inputs_ref` | 精确(10输出 full 对比) | reqs 1/2×SAMPLE_FROM[F,T]; query=3; block 64 | num_rejected>0、num_sampled=0、clamp、multi-block grid、PAD 非-1 | 1(4组) |  |
| test_prompt_logprobs_token_ids_kernel | `_prompt_logprobs_token_ids_kernel`(vanilla) | kernel | A | CPU ref | 精确 | reqs 1/2/4×query_len 1/4/16; offset; idx 非恒等 | BLOCK 分块、越界、max_num_reqs padding | 2(~10) |  |
| test_rejection_kernel | `_rejection_kernel`(vanilla) | kernel | E(先造 stats) | 行为; greedy CPU argmax; non-greedy 属性 | greedy 精确; non-greedy 属性/范围 | reqs 1/2×draft 1/2×vocab 128/1024; non-greedy vocab128; mismatch; rate 0/1 | 无 PRNG 精确对比、USE_BLOCK_VERIFICATION=True、HAS_DRAFT=False、LSE 精度 | 5(~12) |  |
| test_rejection_kernel_patch | Ascend `_probabilistic_rejection_kernel` | kernel | E(先造 stats) | CPU `_greedy/_non_greedy_accept_cpu_ref` | greedy 精确; non-greedy 行为(u=0 恒 accept) | greedy all-acc/all-rej/varying/sampled/multi_req; non-greedy temps 0.5/1/2; bonus; vocab 32/64/128 | 接受概率数学正确性(u=0 简化)、LSE 数值、reqs>2 | 11(~17) |  |
| test_resample_kernel | `_resample_kernel`(vanilla) | kernel | A | CPU `_resample_ref` + argmax | argmax 精确; max 1e-5; **仅 temp=0** | reqs 1/2/4×steps 1/3×has_draft[F,T]×use_bv[F,T]; bonus | temp>0 被显式排除、USE_FP64、padded | 3(~26) |  |
| test_resample_kernel_patch | Ascend `_resample_kernel` | kernel | B(+范围/行为) | 行为/范围 | greedy bonus: block 范围 [start,end); no-op 精确 | greedy bonus(vocab512,2blk); non-bonus no-op | 无数值 oracle(不验证是否最大)、非贪心、padded | 2 |  |
| test_scatter_num_accepted_kernel | `_scatter_num_accepted_kernel`(vanilla) | kernel | A | CPU `_scatter_num_accepted_ref` | 精确 | reqs 1/4/8/16×max 16/32; randperm; skip-neg; clamp1 | 越界、idx 重复、极大溢出、int64 | 3(~10) |  |
| test_selective_scan_update_kernel | `_selective_scan_update_kernel`(vanilla,@heuristics) | kernel | A | PyTorch CPU `_selective_scan_update_cpu` | 1e-4(out/state) | D/z/dt_bias/DT_SOFTPLUS; no-D-no-z; dstate 2/8/16; TIE_HDIM; batch=nheads=1 | batch>1、ngroups>1、IS_SPEC/VARLEN/HAS_STATE、半精度、大 dim 多块 | 5(~7) |  |
| test_tl_rand64 | `tl_rand64`(A3 FP32 替代)+`gumbel_sample` | helper+生产wrapper | C+D | 统计(范围/均值)+行为(种子/主导 token) | 范围 >0<=1; 均值 0.45-0.55; 异种子 not allclose; 主导 token equal | includes_zero[T,F]×10000; 均匀; 不同种子; gumbel 主导 token(恒7) | FP64 位级等价不验证、非主导分布、0/1 精确值、KS 检验 | 4 |  |
| test_tl_rand64_patch | `tl_rand64`(A3)+`gumbel_sample`(vllm_ascend) | helper+生产wrapper | C+D | 同 tl_rand64 | 同 tl_rand64 | 同上(Ascend gumbel 路径) | 同上; 与上游内容近乎一致仅 import 源不同 | 4 |  |
| test_topk_topp_kernel | `_topk_topp_kernel`(vanilla) | kernel | A | CPU `_apply_topk_topp_cpu` + 属性 | 组合 1e-5; topk 存活数精确; topp 累积>=p-0.05; noop 精确 | batch 1/2×vocab 64/128×topk[F,T]×topp[F,T]; topk exact; topp 0.7/0.9; noop | 多 program 并行、fp16、topp 重复值、k=1/k=vocab/p=0/1 | 5(~20) |  |
| test_update_min_larger_stats | `_update_min_larger_stats`(helper) | helper | C(本地 wrapper) | CPU `_update_min_larger_ref` | min 1e-5; cnt 精确 | new-min/same-min(5.0)/larger-min/no-above/sentinel(inf); BLOCK 16/32/64 | multi-tile 归约、above 部分0、NaN、cnt 溢出 | 5(~13) |  |
---

# 二、from_vllm_ascend/（10 个正式测试 + 1 诊断脚本）

## 2.1 逐 UT 条目表（结果列待填）

| 文件 | 被测对象 | 对象类型 | 调用类型 | 参考基准 | 容差判据 | 覆盖场景 | 覆盖盲区 | 用例数 | 结果 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test_bad_words | `bad_words.apply_bad_words`(`_bad_words_kernel`) | 生产wrapper | C(生产wrapper包helper) | 行为(仅对比是否改变) | allclose(相等性) | 3 规格 512/1024/2048; 无 bad; max128; 在/超限 1024/1056 | 无数值对照(只查是否改变)、仅 fp32、词长<=3、位置校验 | 6 |  |
| test_bincount | `_bincount_kernel`(penalties.py) | kernel | A | 自写 `torch_bincount` | torch.equal(int32) | 单 token63, 64 req, 单 block, seed42, BLOCK1024 | 仅1固定例、token_id<10、不多样化、多 block/超 prefetch 边界 | 1 |  |
| test_compute_slot_mapping | `_compute_slot_mappings_kernel`(Ascend vs vllm) | kernel | F(Ascend vs 上游头对头) | 另一实现(上游 vllm kernel) | torch.equal(int64 slot) | 固定:1组KV,1req,5tokens,CP1,320 block | 异常被 try/except 吞(失败不报)、单小场景、CP/多group/非interleave | 1 |  |
| test_compute_topk_logprobs | `compute_topk_logprobs`(`_topk_log_softmax_kernel`+`_ranks_kernel`) | kernel(生产wrapper+级联) | D+E | PyTorch topk+log_softmax+计数rank | ID/rank equal; logprobs 1e-4 | 4 规格 batch 48/96/24/1 × vocab 1024/1519/320 × lp 5/0/1/10(含 lp=0) | 重复 token 断连(rank 用 >)、大 vocab、dtype 变体 | 4 |  |
| test_gumbel_sampling | `gumbel.apply_temperature`+`gumbel.gumbel_sample`(`_gumbel_sample_kernel`) | kernel+生产wrapper | D+C | 自写 PyTorch 参考+统计/行为/确定性 | temp 1e-4/1e-5; greedy equal; processed 1e-4/1e-4 | 多种 vocab 32000/102400/151936; temp0/1 skip; greedy vs 采样; seed; 统计; EAGLE processed | 采样正确性不定量、分布宽松统计、多token竞态、random 无 RNG 对齐 | 30 |  |
| test_log_softmax | `_topk_log_softmax_kernel`(logprob.py) | kernel | A | PyTorch log_softmax+gather | allclose 1e-3/1e-3 | 3 规格 batch 48/96/24×vocab 102400/151936×lp 50/1/8(含 lp=1) | 无 lp=0、容差宽、仅 fp32、PADDED_TOPK=2 分支 | 3 |  |
| test_min_p | `min_p.apply_min_p`(`_min_p_kernel`) | kernel | D+E | 自写 `torch_min_p_torch` | inf mask equal; 有效值 1e-4/1e-4 | 4 规格 req 48/96/24/1×vocab 102400/151936/32000; 反向 idx; min_p 0.01-0.5 | min_p=0 分支、min_p>=1/负值、dtype、扩展映射 | 4 |  |
| test_penality | `penalties.apply_penalties`(`_penalties_kernel`) | kernel+生产wrapper | D | 自写 `pytorch_apply_penalties`(packed mask+累积) | allclose 1e-3; bf16 1e-2 | tokens{1,4}×vocab{1000}×status{1,4}×spec{0,1,3}×dtype{bf16,fp16}; 随机 rep/freq/pres | 无 fp32 广参考、仅 vocab1000、rep=1/freq=0/pres=0 组合、大vocab packed | 24 |  |
| test_post_update | `_post_update_kernel`(Ascend vs vllm) | kernel | F+E + 独立 CPU oracle | 串行 oracle `post_update_ref` + 上游 vllm | assert_close rtol=0/atol=0(int32) | 3 规格 req 36/48/128×vocab 200/32000×steps 2/5; Async 区分 | grid 取 min(num_rows,vectorcore)、未用槽、累积依赖整数语义 | 3 |  |
| test_temperature | `gumbel.apply_temperature`(`_temperature_kernel`) | kernel+生产wrapper | D | 自写 `torch_apply_temperature`(纯Python) | allclose 1e-4/1e-5 | 5 主流 vocab 32000..151936; random tokens 1-64; temp 0.2-2.0 注入 0/1 | 无扩展 idx、temp 负/超2、批量多token同req、dtype | 5 |  |
| diagnose_bincount_atomic_or(诊断,非正式) | `_bincount_kernel` atomic_or 挂起排查 | — | — | 8 探针子进程限时 | assert_close rtol=0 | 挂起问题排查 | 属编译/兼容排查,非数值覆盖 | (诊断) |  |

---

## 2.2 用例/规模汇总

| 目录 | 文件数 | 展开用例合计 |
| --- | --- | --- |
| from_vllm | 30 | 各条见表(差异大,见批注) |
| from_vllm_ascend | 10 | 6+1+1+4+30+3+4+24+3+5 = **81** |
| 诊断脚本 | 1 | 非正式测试 |
---

# 三、调用类型汇总统计

| 调用类型 | 含义 | 涉及文件(from_vllm) | 涉及文件(from_vllm_ascend) |
| --- | --- | --- | --- |
| A | 直接launch+CU/PyTorch参考 | apply_write, bias, block_max_and_sumexp(+patch), block_stats_patch, fill_logprob原版, flatten, gather, insert_resampled(+patch), load_ptr, prepare_dflash(+patch), prompt_logprobs, scatter, selective_scan, topk_topp | bincount, log_softmax |
| B | 直接launch+精确/行为断言 | resample_kernel_patch | — |
| C | helper用测试wrapper包裹 | compute_global_logsumexp(+patch), gumbel_block_argmax(+patch), load_ptr, update_min_larger_stats, tl_rand64(+patch) | bad_words, gumbel_sampling(Ascend helper) |
| D | 生产wrapper公共函数 | fill_logprob_patch(compute_topk_logprobs) | topk_logprobs, gumbel_sampling, min_p, penality, temperature |
| E | 多kernel级联(先造前置数据) | compute_block_stats, rejection_kernel(+patch) | topk_logprobs(rank), post_update |
| F | 两实现头对头(Ascend vs 上游) | (本目录未真正出现) | compute_slot_mapping, post_update |

> 要点：from_vllm 目录**几乎无 F 型**（Ascend 版与上游版没有直接逐元素头对头，仅各自对照 CPU/参考）。
> 真正 F 型只在 ascend 目录（slot_mapping、post_update）。gumbel/tl_rand64 的 Ascend 版也未与上游输出互比。

# 四、对象类型汇总

| 对象类型 | 说明 | 涉及算子 | 测试方式 |
| --- | --- | --- | --- |
| kernel(含生产wrapper包kernel) | 可 [grid] launch | bias, insert, gather, resample, rejection, topk_topp, etc. | 直接launch(A/B)或经生产wrapper(D) |
| helper | 需测试内 wrapper kernel 包裹 | global_lse, gumbel_block_argmax, load_ptr, update_min_larger_stats, tl_rand64 | C 类 |

# 五、共性与突出盲区（跨全套件）

1. **PRNG 无法精确对比**：rejection/resample/gumbel 的 CPU 参考多数只支持 temp=0 确定性情形；temp>0 的接受/拒绝概率只做行为/属性断言，未做统计等价。
2. **等值/并列边界**：`_ranks_kernel` 上游 `>=` vs Ascend `>` 可能差 1；topk_topp/topp 并列值；后需按两端语义校验。
3. **dtype 覆盖弱**：多数仅 fp32；只有 penality 覆盖 bf16/fp16；gumbel/bad_words/min_p/temperature/ranks 未半精度。
4. **多块/跨块归约与 PADDED 脏块**：block_stats、global_lse、cumulative_log_p、insert 普遍未测 vocab 不整除 block、跨块归约、PADDED 含脏值。
5. **生产分支未触发**：min_p=0、num_sampled=0(prefill)、num_rejected>0、USE_BLOCK_VERIFICATION=True、HAS_DRAFT_LOGITS=False 等常未独立触发。
6. **弱测试预警**：bad_words 只查"是否改变"无数值对照；slot_mapping 用 try/except 吞异常(失败不报)；tl_rand64 不验证 FP64 位级等价。
7. **版本条件 skip**：apply_write 的 multi-group 在旧 vLLM 直接 pytest.skip，实际精度未受验。

# 六、结论模板（每算子一行，结果运行后填）

| 被测对象 | 一致性 | 盲区是否影响生产 | 建议 | 风险等级 |
| --- | --- | --- | --- | --- |
| `_bias_kernel` | (待测) | 叠加分支未测 | 补 allowed+bias 叠加 | 低 |
| `_ranks_kernel` | (待测) | 等值差1 | 确认 >= vs > | 中 |
| `_rejection_kernel` | (待测) | temp>0 概率未统计验证 | 补统计 | 中 |
| `tl_rand32/64` | (待测) | FP64 位级不验证 | 补 FP64 | 中 |
| ... | ... | ... | ... | ... |

# 七、结果列填写指引

1. 运行 `python -m pytest <目录> -v` 收集每个文件 **PASS / FAIL / XFAIL** 与关键数值。
2. FAIL 时抄录断言语与最大误差(如 `max_err=3.1e-5` vs 容差 1e-4)。
3. XFAIL(skip) 记 `XFAIL(编译)`，**不计入通过**。
4. 在"结论模板"按算子汇总一致性，仅对实际运行覆盖的分支下结论；未覆盖分支标「未受验(盲区)」。