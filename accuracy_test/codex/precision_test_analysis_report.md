# codex 目录精度测试（Accuracy Test）UT 全量分析报告

> 分析对象：`accuracy_test/codex` 目录下全部精度测试单元测试。
> 目录共 3 类测试集合：
> - `missing_accuracy_tests/`（18 个文件）：上下游均无现成精度 UT，后补写的测试
> - `existing_accuracy_tests/from_vllm/`（28 个文件）：从 vLLM 已有 UT 搬运/适配到 Ascend A3 的独立版本（含 `_patch.py`）
> - `existing_accuracy_tests/from_vllm_ascend/`（11 个文件）：从 vLLM-Ascend 官方 A3/NPU 测试搬运的独立版本（含诊断脚本）
>
> 本文档逐一给出：**测试标准（判定准则）、测试方式、测试流程、优缺点分析**，供撰写报告使用。

---

## 0. 总体框架与通用约定

### 0.1 执行环境与前置条件
- 需安装并配置：vLLM、vLLM-Ascend（`vllm_ascend`）、PyTorch NPU（`torch_npu`，device=`"npu"`）、Ascend Triton。
- 每个测试文件在使用 Triton kernel 前都调用 `init_device_properties_triton()` 初始化 Ascend Triton 设备属性（向量核数量等）。
- 所有 kernel 都在 **NPU 上直接 launch**，输入输出张量 device=`torch.device("npu")`，launch 后调用 `torch.npu.synchronize()` 保证异步执行就绪再读取结果。
- 本地 Windows 环境只能做 **Python AST 语法检查**（README 记载 48 个文件全部通过）；真正的 kernel 执行需在 Ascend A3 环境完成。

### 0.2 通用测试标准（判定准则）
来自 `conftest.py`（`from_vllm/`、`from_vllm_ascend/`、`missing_accuracy_tests/` 各有相同的策略文件），把**后端兼容性失败**与**精度失败**区分开：

| pytest 结果 | 含义 |
| --- | --- |
| `PASSED` | kernel 输出与参考结果在测试容差内一致 |
| `FAILED`（来自 `assert` / `torch.testing.assert_close`） | 可能存在精度或功能正确性问题，须检查实际差值 |
| `XFAIL`（由 conftest 标记为 `skipped` 带 `wasxfail`） | 已知 NPU/Triton binder、编译器或设备能力限制，未得到可比输出，精度未知 |
| `SKIPPED` | 参数组合无效，或可能挂起等不能安全运行的已知场景 |
| Python `NameError/TypeError/AttributeError/IndexError` 等 | 测试代码问题，记为 `FAILED`，**不会被兼容性策略隐藏** |

`conftest.py` 的判定逻辑（`pytest_runtest_makereport` hook）：
- 遍历异常链（`__cause__`/`__context__`）拼成一个字符串并小写。
- 若命中后端兼容性模式（如 `"backend compiler failed"`、`"cannot find compiler"`、`"failed to run bishengir pipeline"`、`"not implemented for npu"`、`"out of resources"`、`"ub overflow"`、`"unsupported dtype"` 等约 14 个模式），则把报告结果改为 `skipped` 并标记 `wasxfail="Backend compatibility failure; precision is unknown"`。
- `_AUTHORING_ERRORS`（断言、属性、类型、名字等）一律不被隐藏，保持真实失败。

**优点：** 把"编译器/设备绑定限制导致的无法运行"与"真正数值差异"分离，避免因编译失败误报精度失败，也避免用 skip 掩盖真实的 `assert` 失败。
**缺点：** 依赖字符串模式匹配，若后端报错文案不在模式列表内会误报为精度失败；且 XFAIL 意味着"精度未知"，并未真正验证精度。

### 0.3 通用测试方式分类
按"是否直接 launch 目标 kernel"与"参考来源"可分为 5 种方式：

1. **直接 launch + CPU/PyTorch 参考对比**（最主流）：直接以网格 launch 目标 Triton kernel，用纯 Python/NumPy/PyTorch 串行参考实现计算期望值，再 `torch.testing.assert_close`。
2. **直接 launch + 精确值断言**：针对整数/布尔/位操作 kernel，用 `assert x.item()==...` 或 `torch.equal` 做位级精确比较。
3. **wrapper/helper 测试**：对 Triton JIT helper（非独立可 launch 的函数）写一个本地 wrapper kernel 包裹调用，再对比 CPU 参考。
4. **wrapper 公共函数测试**：不直接 launch kernel，而是调用会 launch kernel 的 Ascend wrapper（如 `apply_temperature`、`gumbel_sample`、`compute_topk_logprobs`、`apply_min_p`、`apply_penalties`、`apply_bad_words`），做数值/行为/统计校验。
5. **两 kernel 头对头比较**：Ascend 版本 kernel 与上游（CUDA/GPU）版本 kernel 各自 launch，输出直接 `torch.equal` 对比。

### 0.4 通用测试流程（典型 pytest 流程）
1. 模块导入：加载被测 kernel（很多文件在 `try/except ImportError` 中做模块级 `pytest.skip(allow_module_level=True)`，避免缺包时报错）。
2. `setup` fixture（`autouse=True`）：`init_device_properties_triton()`，设置 `self.device = torch.device("npu")` 及常量（BLOCK_SIZE 等）。
3. 构造输入：随机张量 + 随机种子（多为 `torch.manual_seed(42)`），构造索引/映射/边界条件。
4. Launch kernel（网格 `(num_reqs,)` 或 `(num_reqs, num_blocks)` 等），`torch.npu.synchronize()`。
5. 计算参考：CPU 参考函数 / PyTorch 参考 / 精确值断言。
6. 判定：`torch.testing.assert_close(...)`（整数精确/浮点容差）或属性断言。
7. 多组参数通过 `@pytest.mark.parametrize` 覆盖。

---

## 1. missing_accuracy_tests/（18 个补写测试）

> 特点：全部针对 **vLLM 上游 vanilla kernel 或 vLLM-Ascend 本地 patch kernel**，绝大多数是"直接 launch + CPU 参考 + 精确/小容差对比"。覆盖 input_batch、spec_decode、rope、metrics 等无原有 UT 的算子。

### 1.1 test_apply_grammar_bitmask_kernel_patch.py
- **被测 kernel/来源：** `vllm_ascend/worker/v2/structured_outputs.py:35` 的 `_apply_grammar_bitmask_kernel`（Ascend 适配版）。
- **测试标准：** `torch.testing.assert_close(rtol=0, atol=0)`（位级精确，因输出是 -inf 或原值）+ 布尔断言。
- **测试方式：** 直接 launch kernel，与 CPU 参考 `_apply_grammar_bitmask_ref`（逐 bit 解包 bitmask）对比。
- **测试流程：** ①setup 初始化设备属性；②构造 logits（[num_logits, vocab_size]）、logits_indices、packed bitmask（int32，每字 32 bit）；③launch `_apply_grammar_bitmask_kernel[(num_bitmasks, num_blocks)]`，`BLOCK_SIZE=8192`；④与 CPU 参考逐位对比。
- **用例/参数化：** `test_basic_bitmask`（vocab_size∈{128,1024,8192}，屏蔽前一半词表）、`test_all_allowed`（vocab∈{128,512,4096}，全部 1 不改）、`test_all_blocked`（全部 0 → 全 -inf）。
- **优缺点：** 优点：位打包语义（word-level 32bit）与 NPU sub-block（`BLOCK_SIZE_SUB=1024` 规避 UB overflow）均有覆盖，"全 1/全 0/半屏蔽"三个分支齐全。缺点：只测了 -inf 置位与否，未校验核内 `tl.range` 多 sub-block 迭代的正确性边界；断言为精确比较，对 float 中间路径无容差概念（此 kernel 无非 -inf 运算，故可接受）。

### 1.2 test_combine_sampled_and_draft_tokens_kernel.py
- **被测 kernel/来源：** `vllm/v1/worker/gpu/input_batch.py` 的 `_combine_sampled_and_draft_tokens_kernel`。
- **测试标准：** `torch.testing.assert_close(rtol=0, atol=0)`（int 精确）。
- **测试方式：** 直接 launch，与 CPU 参考 `_combine_sampled_and_draft_tokens_ref` 对比 `input_ids` 与 `logits_indices`。
- **测试流程：** 构造 input_ids/idx_mapping/last_sampled/query_start_loc/seq_lens/prefill_len/draft_tokens/cu_num_logits；launch；同步；对比两个输出。
- **特殊点：** 上游 kernel 的 `NUM_NEW_SAMPLED_TOKENS` 有默认 constexpr `=1`，Ascend Triton runtime binder 无法覆盖 `=0`；因此测试定义了一个**本地同名副本** `_combine_sampled_and_draft_tokens_required_constexpr`（无默认值），通过 `_launch_combine_kernel` 动态选择。`NUM_NEW_SAMPLED_TOKENS=0` 场景以此副本运行。
- **用例：** `test_combine_basic`（num_reqs∈{1,2,4}×num_spec_steps∈{1,3}×num_new_sampled∈{0,1}）、`test_prefill_only`（seq_len≤prefill_len 时只写 logits_indices）。
- **优缺点：** 优点：覆盖了"constexpr 绑定 / 上游 kernel 不可传参"这一 NPU 特有坑，且仍直接运行项目原始 kernel（默认值=1 时）。缺点：`NUM_NEW_SAMPLED_TOKENS=0` 实际测的是本地副本而非上游实现（README 明确记为可能 `XFAIL`）；测试数据做了大量简化（`num_tokens = total_logits`）。

### 1.3 test_dcp_local_seq_lens_kernel.py
- **被测 kernel/来源：** `vllm/v1/worker/gpu/cp_utils.py` 的 `_dcp_local_seq_lens_kernel`（context parallelism 本地 seq_len 计算）。
- **测试标准：** `assert_close(rtol=0, atol=0)` + 布尔断言。
- **测试方式：** 直接 launch，与 CPU 参考 `_dcp_local_seq_lens_ref`（round-robin 分配公式）对比。
- **测试流程：** 构造随机 seq_lens、dcp_size/rank/cp_interleave；launch；同步；对比前 num_reqs 项，padding 项应为 0。
- **用例：** `test_dcp_local_seq_lens`（num_reqs∈{2,4,8}×max∈{8,16}×dcp_size∈{2,4}×rank∈{0,1}×cp_interleave∈{1,2}，rank≥size 时 skip）、`test_dcp_rank_highest`（rank=size-1）、`test_zero_seq_lens`。
- **优缺点：** 优点：参数化覆盖多种 dcp 拓扑与最高秩边界，padding 语义也有校验。缺点：`dcp_rank >= dcp_size` 直接用 `pytest.skip` 跳过（参数组合无效，符合 README 的 SKIP 准则）；未覆盖更大的 dcp_size（如 8、16）与跨 block 边界场景。

### 1.4 test_expand_idx_mapping_kernel.py
- **被测 kernel/来源：** `vllm/v1/worker/gpu/input_batch.py` 的 `_expand_idx_mapping_kernel`。
- **测试标准：** `assert_close(rtol=0, atol=0)`（int64/int32 精确）。
- **测试方式：** 直接 launch，与 CPU 参考 `_expand_idx_mapping_ref` 对比 `expanded_idx_mapping` 与 `expanded_local_pos`。
- **测试流程：** 构造 idx_mapping、cu_num_logits；launch `[(num_reqs,)]`；同步；对比两组输出。
- **用例：** `test_basic_expand`（num_reqs∈{1,2,4}×tokens_per_req∈{1,3,8}）、`test_uneven_tokens`（各请求 token 数不同）、`test_non_contiguous_idx_mapping`（[5,2,8] 非连续映射）。
- **优缺点：** 优点：覆盖均匀、不均匀、非连续三种布局，local_pos 逐 token 定位。缺点：`BLOCK_SIZE` 取 `next_power_of_2(tokens_per_req)`，对不均匀场景取 max，未测超长 token（>16）与 block 边界；参考实现本身逻辑与 kernel 高度一致，独立性强有限。

### 1.5 test_get_num_sampled_and_rejected_kernel.py
- **被测 kernel/来源：** `vllm/v1/worker/gpu/input_batch.py` 的 `_get_num_sampled_and_rejected_kernel`。
- **测试标准：** `assert_close(rtol=0, atol=0)` + 布尔断言。
- **测试方式：** 直接 launch，与 CPU 参考 `_get_num_sampled_and_rejected_ref` 对比 `num_sampled`（原地更新）与 `num_rejected`。
- **测试流程：** 构造 num_sampled/seq_lens/cu_num_logits/idx_mapping/prefill_len；launch `[(num_reqs,)]`；同步；对比两组输出。
- **用例：** `test_basic`（num_reqs∈{1,2,4}×num_logits_per_req∈{1,3,5}）、`test_chunked_prefilling`（seq_len<prefill_len 时两者为 0）、`test_various_sampled_counts`（参数化 (0,3)/(2,1)/(3,0) 的拒绝数）。
- **优缺点：** 优点：对 chunked-prefill 特殊分支有专门用例，拒绝数=logits-采样 的关系有多种取值验证。缺点：测试数据偏小（num_reqs≤4），未测大 batch 与 idx_mapping 非连续场景。

### 1.6 test_npu_gumbel_block_argmax_patch.py
- **被测 kernel/来源：** `vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py:34` 的 `_npu_gumbel_block_argmax`（Ascend 替换 `gumbel_block_argmax`）。
- **测试标准：** 精确值断言（`==`）+ 数值 `assert_close(rtol=1e-5, atol=1e-5)`（processed_logits）。
- **测试方式：** 因为该 helper 非独立可 launch，测试定义了本地 wrapper kernel（`_npu_gumbel_block_argmax_wrapper`）将返回值写入输出，再校验；`processed_logits` 输出路径用逐块 inline wrapper。
- **测试流程：** setup（vocab=128, BLOCK_SIZE=64）；通过 wrapper 单 block 调用；校验 (value, idx)；另有 `test_processed_logits_output` 在 vocab=256 逐 block 调用验证 `processed_logits == logits / temperature`。
- **用例：** `test_no_temperature_no_gumbel`（temp=0 无噪声取 max）、`test_temperature_applied`、`test_gumbel_noise_dominates`（100 分差确保不被噪声翻转）、`test_processed_logits_output`（temp=2.0，验证除温度后存储）。
- **优缺点：** 优点：覆盖零噪声确定语义、温度应用、噪声主导、可选 processed_logits 存储路径四类行为；对 Gumbel 随机采样契约（用 margin 保证确定性）处理得当。缺点：`tl.rand` 的随机唯一性/分布未直接验证（只靠 margin 判 argmax）；`processed_logits` 测试用 inline wrapper 逐块启动，与真实调用链存在差异。

### 1.7 test_num_nans_kernel.py
- **被测 kernel/来源：** `vllm/v1/worker/gpu/metrics/logits.py` 的 `_num_nans_kernel`。
- **测试标准：** `assert_close(rtol=0, atol=0)`（int 精确）。
- **测试方式：** 直接 launch，与 CPU 参考 `_num_nans_ref`（逐行查 NaN）对比每请求 NaN 计数。
- **测试流程：** 构造 logits（num_reqs×vocab_size），按 frac_nan 注入前 num_nan 列为 NaN；launch `[(num_reqs,)]`，`BLOCK_SIZE=8192`；同步；对比。
- **用例：** `test_num_nans`（num_reqs∈{1,2,4,8}×vocab∈{128,1024,8192,16384}×frac_nan∈{0,0.1,0.5,1.0}）、`test_no_nans`、`test_all_nans`。
- **优缺点：** 优点：参数化覆盖 NaN 比例 0/10%/50%/100%，且对全 0、全 NaN 边界专测；`libdevice.isnan` 语义在多种行/列规模下验证。缺点：NaN 始终注入在每行头部连续区间，未测 NaN 散布在任意位置的情况。

### 1.8 test_post_update_num_computed_tokens_kernel.py
- **被测 kernel/来源：** `vllm/v1/worker/gpu/input_batch.py` 的 `_post_update_num_computed_tokens_kernel`。
- **测试标准：** `assert_close(rtol=0, atol=0)`。
- **测试方式：** 直接 launch，与 CPU 参考 `_post_update_num_computed_tokens_ref`（按 query_len 累加）对比。
- **测试流程：** 构造 idx_mapping/num_computed_tokens/query_start_loc；launch `[(num_reqs,)]`；同步；对比。
- **用例：** `test_basic_increment`（num_reqs∈{1,2,4}×query_len∈{1,4,8}）、`test_non_contiguous_idx_mapping`（[5,0,3]）、`test_zero_query_len`（不变化）。
- **优缺点：** 优点：覆盖非连续映射累加与零 query_len 的 no-op。缺点：运算极简单（加法），未覆盖 idx_mapping 越界/重复引用等异常，参考与实现逻辑相似、独立性弱。

### 1.9 test_prepare_decode_inputs_kernel.py
- **被测 kernel/来源：** `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py` 的 `_prepare_decode_inputs_kernel`。
- **测试标准：** `assert_close(rtol=0, atol=0)`（原位手动断言较多）。
- **测试方式：** 直接 launch（`[(num_reqs+1,)]`，最后一个 block 处理 CUDA graph padding），与手写 CPU 参考逐字段对比（input_ids/seq_lens/query_start_loc/positions）。
- **测试流程：** 构造 draft_tokens/target_seq_lens/num_rejected/max_model_len 等；launch；同步；按 CPU 语义（advance 位置、seq_len=tsl-num_rejected+1、clamp）对比。
- **用例：** `test_prepare_decode_inputs`（num_reqs∈{1,2,4}×advance_pos∈{False,True}）、`test_with_rejected_tokens`（num_rejected>0）、`test_model_len_clamp`（clamp 到 max_model_len）。
- **优缺点：** 优点：覆盖 advance 位置推进、拒绝 token 缩减 query_len、max_model_len 截断、CUDA graph padding 四项关键语义。缺点：大量字段用 `.item()` 原位断言而非整体 assert_close，覆盖度高但可读性/原子性稍弱；padding 期望为手写推导。

### 1.10 test_prepare_dflash_inputs_kernel_ascend_patch.py
- **被测 kernel/来源：** `vllm_ascend/worker/v2/spec_decode/dflash/speculator.py:140` 的 `_prepare_dflash_inputs_kernel_ascend`，以及对安装版本的 legacy/modern 双内核自适应（`_modern_dflash_inputs_kernel`，来自 `vllm_ascend.ops.triton.spec_decode.utils`）。
- **测试标准：** `assert_close(rtol=0, atol=0)`（10 个输出全精确）。
- **测试方式：** 直接 launch，系统根据安装的 Ascend 包自动选择 legacy `_prepare_dflash_inputs_kernel_ascend` 或 modern `copy_and_expand_dflash_inputs_kernel_single_grid`，分别与对应 CPU 参考（`_prepare_dflash_inputs_ref` / `_modern_dflash_inputs_ref`）对比 10 个输出。
- **测试流程：** ①构造两种参考输入；②动态解析安装的 DFlash 实现（`_dflash_environment_diagnostics` 提供诊断信息）；③launch 并同步；④逐输出 `assert_close`。
- **用例：** `test_prepare_dflash_inputs`（num_reqs∈{1,2}×SAMPLE_FROM_ANCHOR∈样本值集）。kernel 尺寸固定：max_num_reqs=4、num_query_per_req=3、num_speculative_steps=3、block_size=64、max_model_len=1024。
- **优缺点：** 优点：极强的版本兼容性——同时支持 legacy 与 modern 两代 Ascend DFlash 内核，且 modern 不支持 anchor 采样时自动从参数集剔除；独立 CPU 参考覆盖完整 padding（PAD_SLOT_ID=-1 等）语义。缺点：测试体量大、逻辑复杂（依赖运行时探测安装包结构），若两代内核语义存在细微差异容易被各自参考掩盖；modern 路径下 `SAMPLE_FROM_ANCHOR` 支持与否依赖运行时探测，存在误判风险。

### 1.11 test_prepare_pos_seq_lens_kernel.py
- **被测 kernel/来源：** `vllm/v1/worker/gpu/input_batch.py` 的 `_prepare_pos_seq_lens_kernel`。
- **测试标准：** `assert_close(rtol=0, atol=0)` + 布尔断言。
- **测试方式：** 直接 launch（`[(num_reqs+1,)]`，末 block 处理 padding），与 CPU 参考 `_prepare_pos_seq_lens_ref` 对比 pos 与 seq_lens。
- **测试流程：** 构造 idx_mapping/query_start_loc/num_computed_tokens；launch，`BLOCK_SIZE=4`；同步；对比。
- **用例：** `test_pos_seq_lens`（num_reqs∈{1,2,4,8}×max∈{8,16}×tokens_per_req∈{1,4,8}，num_reqs>max 时 skip）、`test_cuda_graph_padding`（padding 行置 0）、`test_event_driven`（零 query 长度保留 active seq_len）。
- **优缺点：** 优点：覆盖 CUDA graph padding 与 event-driven（无 token 时仍保留已算 seq_len）两种关键语义。缺点：`BLOCK_SIZE=4` 是极小固定值，可能未覆盖大 tile 迭代分支；pos 的 per-token 写入依赖小规模参数。

### 1.12 test_prepare_prefill_inputs_kernel.py（input_batch 版）
- **被测 kernel/来源：** `vllm/v1/worker/gpu/input_batch.py` 的 `_prepare_prefill_inputs_kernel`。
- **测试标准：** `assert_close(rtol=0, atol=0)`。
- **测试方式：** 直接 launch，与 CPU 参考 `_prepare_prefill_inputs_ref` 对比 input_ids 与 next_prefill_tokens。
- **测试流程：** 构造 all_token_ids/idx_mapping/query_start_loc/num_computed_tokens/prefill_lens；launch `[(num_reqs,)]`，`BLOCK_SIZE=1024`；同步；对比。
- **用例：** `test_prepare_prefill_inputs`（num_reqs∈{1,2,4}×query_len∈{1,4,16}）、`test_early_return_when_prefill_done`（num_computed≥prefill_len 时 no-op）、`test_exact_prefill_boundary`（正好到达边界不写 next）。
- **优缺点：** 优点：覆盖 prefill 早退、边界（next_pos==prefill_len 不写 next naive）等易错分支。缺点：`next_prefill_tokens` 的写入条件与 prefill_len/num_computed 组合覆盖较全，但 query_len 偏小（≤16）。

### 1.13 test_prepare_prefill_inputs_kernel_speculator.py
- **被测 kernel/来源：** `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py` 的 `_prepare_prefill_inputs_kernel`（与 1.12 同名不同签名，属 speculator 流程）。
- **测试标准：** `assert_close(rtol=0, atol=0)` + 原位断言。
- **测试方式：** 直接 launch，与 CPU 参考 `_prepare_prefill_inputs_speculator_ref` 对比 last_token_indices/draft_input_ids/draft_positions/draft_query_start_loc/draft_seq_lens，并校验 draft_current_step 复位为 0。
- **测试流程：** 构造大量 speculator 输入；launch `[(num_reqs,)]`，`BLOCK_SIZE=1024`；同步；对比 6 组输出。
- **用例：** `test_basic_prefill`（num_reqs∈{1,2,4}×query_len∈{4,16}）、`test_chunked_prefill_path`（num_sampled=0 时用 next_prefill_tokens）、`test_rejected_tokens_adjustment`（num_rejected 缩减 query_len）、`test_padding_for_cuda_graphs`。
- **优缺点：** 优点：是覆盖 speculator 特有 shift-输入、拒绝收缩、next-token 选择逻辑的最完整测试之一；多种分支（采样/预填充、拒绝/不拒绝）均有参数化。缺点：draft_current_step 始终复位 0 的断言较单一；输入规模较小。

### 1.14 test_prepare_rope_positions_kernel.py
- **被测 kernel/来源：** `vllm/v1/worker/gpu/mm/rope.py` 的 `_prepare_rope_positions_kernel`（M-RoPE / XD-RoPE 多维位置计算）。
- **测试标准：** `assert_close(rtol=0, atol=0)`。
- **测试方式：** 直接 launch，与 CPU 参考 `_prepare_rope_positions_ref` 对比 positions（多维）。参考实现注意用 flat view 模拟 Triton 指针算术（维内 strides）。
- **测试流程：** 构造 positions/prefill_positions 表/prefill_delta/idx_mapping/query_start_loc/prefill_lens/num_computed；launch `[(num_reqs,)]`，`NUM_DIMS` constexpr；同步；对比。
- **用例：** `test_prefill`（num_dims∈{3,4}×num_reqs∈{1,4,8}，prefill 读表）、`test_decode`（num_dims∈{3,4}×num_reqs∈{1,4}，decode 用 orig+delta）、`test_mixed_prefill_decode`（同一 batch 混合 prefill/decode）。
- **优缺点：** 优点：覆盖 prefill 读表、decode 增量、混合 batch 三类核心路径，并跨 num_dims 3/4（M-RoPE/XD-RoPE）。缺点：参考实现对 flat storage 与维 strides 的处理需精确对应 kernel 指针算术，属较复杂的旁路参考；未测最大 max_model_len 边界（max_num_tokens=256）。

### 1.15 test_probabilistic_rejection_kernel_patch.py
- **被测 kernel/来源：** `vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py:192` 的 `_probabilistic_rejection_kernel`（Ascend 重写 `_rejection_kernel`），依赖 `_compute_block_stats_kernel` 预计算块统计。
- **测试标准：** `assert_close(rtol=0, atol=0)`（int 精确）+ 属性/行为断言。
- **测试方式：** 直接 launch；helper `_run_kernel` 先调用 `_compute_block_stats_kernel` 生成 local stats，再调用 `_probabilistic_rejection_kernel`。对 greedy 路径与 CPU 参考 `_greedy_accept_cpu_ref` 对比拒绝步数；非 greedy 路径因 NPU 使用 `u=0`（`tl.log(u)=-inf`）恒接受而做行为断言。
- **测试流程：** ①加载 block-stats 与 rejection kernel（`_load_rejection_kernels` 处理两仓 helper 改名：`_compute_local_logits_stats_kernel`/`_compute_block_stats_kernel` 与 `_compute_global_lse`/`_compute_global_logsumexp` 的兼容别名）；②`_run_kernel` 生成统计并 launch；③校验。
- **用例：** greedy（all accepted/all rejected/varying lengths × CPU ref/sampled output/multi-req）、non-greedy（always accept、无 draft logits）、mixed（一个 greedy 一个 non-greedy）、边界（bonus-only、vocab∈{32,64,128}）。
- **优缺点：** 优点：全面覆盖 greedy 精确接受语义（与 CPU ref 对比）与 NPU `u=0` 恒接受的已适配行为；处理了 rejection 路径两仓 API 改名错位的兼容加载。缺点：non-greedy 路径在 NPU 上"恒接受"本身是一种**功能性降级**（上游是概率性拒绝），本测试只验证了降级后的行为而非概率分布匹配；padded blocks 与 PADDED_VOCAB_NUM_BLOCKS 的处理依赖 next_power_of_2。

### 1.16 test_resample_kernel_patch.py
- **被测 kernel/来源：** `vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py:82` 的 `_resample_kernel`（Ascend 适配版）。
- **测试标准：** 精确值断言（`==`）+ `assert_close(rtol=0, atol=0)`（输出与变化前对比）。
- **测试方式：** 直接 launch。
- **测试流程：** 构造 target_logits/rejected 等；launch `[(num_reqs, num_blocks)]`，`BLOCK_SIZE=1024`，`HAS_DRAFT_LOGITS=False`；同步。
- **用例：** `test_greedy_bonus_token`（bonus token temp=0 带 Gumbel 噪声重采样，校验 argmax 落在对应块范围）、`test_non_bonus_greedy`（非 bonus greedy 早退 no-op，输出与变化前一致）。
- **优缺点：** 优点：覆盖 bonus（重采样生效）与非 bonus greedy（早退 no-op）两个关键分支。缺点：与 1.1/1.6 类似，bonus 用例只用"块内范围"粗校验 argmax，未与"加噪声后的 argmax"参考对比（因随机契约），判定强度有限；未测 HAS_DRAFT_LOGITS=True 路径。

### 1.17 test_update_draft_inputs_kernel.py
- **被测 kernel/来源：** `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py` 的 `_update_draft_inputs_kernel`。
- **测试标准：** `assert_close`（int 精确 / hidden rtol=1e-3,atol=1e-3）。
- **测试方式：** 直接 launch，与 CPU 语义参考对比 5 组输出（draft tokens/hidden/input_ids/positions/seq_lens）。
- **测试流程：** 构造 draft_tokens/current_draft_step/hidden_states/max_model_len/num_speculative_steps；launch `[(num_reqs,)]`，`BLOCK_SIZE=1024`，`ADVANCE_DRAFT_POSITIONS`；同步；对比。
- **用例：** `test_update_draft_inputs`（num_reqs∈{1,2,4}×hidden_size∈{128,512}×advance_pos∈{False,True}）、`test_final_step_skips_update`（末步只写 draft token 早退）。
- **优缺点：** 优点：覆盖 advance 推进、末步早退（只写 draft token）两个关键分支，hidden 用 fp16 并设 1e-3 容差。缺点：hidden 复制本质是 memcpy，1e-3 容差对随机 fp16 可能过严但也可接受；draft token 写入用 `.item()` 校验。

### 1.18 test_zero_kv_blocks_kernel_patch.py
- **被测 kernel/来源：** `vllm_ascend/worker/utils.py:15` 的 `_zero_kv_blocks_kernel`（Ascend 适配版，支持多分段、向量核负载均衡）。
- **测试标准：** 精确断言 `assert torch.all(x==0)` + `assert_close(rtol=0,atol=0)`（no-op）。
- **测试方式：** 直接 launch；通过 `data_ptr()` 取绝对地址构造 `seg_addrs`；用 `get_vectorcore_num()` 与工作量求 `grid`。
- **测试流程：** 分配随机 KV 页/缓冲，构造 seg_addrs（int64 绝对地址）、block_ids；launch `[(grid,)]`（N_SEGS/PAGE_SIZE_EL/BLOCK_SIZE/GRID_SIZE constexpr）；同步；校验清零与未越界。
- **用例：** `test_zero_single_block_single_seg`、`test_zero_multiple_blocks`、`test_zero_multiple_segments`（K/V 两分段）、`test_no_blocks`（n_blocks=0 不执行）。
- **优缺点：** 优点：直接覆盖指针级清零（绝对地址写入），验证多 block/多 segment 与 no-op；用 clone 校验"仅目标块清零、其余不变"。缺点：`test_no_blocks` 中 n_blocks=0 时 `total_work=0` 不 launch（仅比对不变），未真正执行空网格；以 `data_ptr()` 构造地址依赖 NPU 张量连续性。

---

## 2. existing_accuracy_tests/from_vllm/（28 个从 vLLM 搬运/适配文件）

> 特点：每个文件头部注释标明 `Accuracy UT source`（上游 vLLM 测试）与 `Kernel source`。上游 CUDA 硬编码与仓库级 fixture 被改为在 A3 单文件直接 launch NPU kernel 的独立版本。文件含 `_patch.py` 后缀表示测试的是 **vLLM-Ascend 的改名/替换/再导出路径**，而非原始 vLLM 入口。判定策略同 0.2 节。

### 2.1 test_apply_write_kernel.py
- **被测 kernel/来源：** `vllm/v1/worker/gpu/buffer_utils.py` 的 `_apply_write_kernel`（含 MULTI_GROUP 多组融合写）。
- **测试标准：** `assert_close(rtol=0,atol=0)`（int 精确）。
- **测试方式：** 直接 launch；对 single-group 用 `_apply_write_ref` 参考对比；multi-group 直接断言每组输出期望值。`_HAS_MULTI_GROUP` 依据上游 kernel 参数名（`write_group_ids_ptr`/`MULTI_GROUP`）探测，缺失时 single 用适配、multi 用 skip。
- **测试流程：** ①探测安装版本是否支持 MULTI_GROUP；②构造单组写入并对照 ref；③构造多组（ptr-to-ptrs）写入并断言。
- **用例：** `test_single_group`（num_rows∈{1,4}×num_cols∈{16,32}）、`test_multi_group`（num_groups∈{1,2,4}×writes_per_group∈{1,2}，不持 MULTI_GROUP 时 skip）。
- **优缺点：** 优点：自适应两种签名，多组融合写用 data_ptr 构造 ptr-to-ptrs 覆盖。缺点：multi-group 只断言覆盖式最终值（第二组覆盖第一组），未系统对照 ref；`_HAS_MULTI_GROUP` 用参数名探测强依赖版本。

### 2.2 test_bias_kernel.py
- **被测 kernel/来源：** `vllm/v1/worker/gpu/sample/logit_bias.py` 的 `_bias_kernel`（allowed tokens + logit bias + min-token 抑制）。
- **测试标准：** `assert_close(rtol=1e-5,atol=1e-5)`（float logits）。
- **测试方式：** 直接 launch，与 CPU 参考 `_bias_ref` 对比。
- **测试流程：** 构造 logits/expanded_idx_mapping/allowed_token_ids/bias/pos/min_lens/stop_token_ids；launch `[(num_tokens,)]`；同步；对比。
- **用例：** `test_allowed_token_ids`（num_tokens∈{1,4,8}×vocab∈{128,1024}）、`test_logit_bias`、`test_combined`（三种机制叠加）。
- **优缺点：** 优点：三条功能（allowed/bias/min-token）各自与组合都有用例，float 用 1e-5 容差合理。缺点：`BLOCK_SIZE` 取三个表列宽的 next_power_of_2，未覆盖超 1024 的大 allowed 集；`-inf` 置位与 `1e-5` 容差并存的判定对 `-inf-(-inf)` 场景需谨慎。

### 2.3 test_compute_block_max_and_sumexp.py
- **被测 helper/来源：** `_compute_max_and_sumexp`（内部 helper），经 `_compute_local_logits_stats_kernel`（`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`）间接调用。
- **测试标准：** `assert_close`（max/sumexp rtol=1e-5,atol=1e-5；argmax 精确）。
- **测试方式：** 直接 launch `_compute_local_logits_stats_kernel`，分 greedy（temp=0，只校验 max/argmax）与非 greedy（同时校验 target/draft 的 max+sumexp）与 CPU 参考对比。
- **测试流程：** 构造 target/draft logits、expanded_idx_mapping/local_pos、temperature；launch `[(num_logits, vocab_num_blocks)]`，`BLOCK_SIZE=8192`；同步；逐 logit×block 校验。
- **用例：** `test_compute_block_max_and_sumexp`（num_logits∈{1,2,4}×vocab∈{128,1024,8192}×num_spec_steps∈{2,3}）、`test_all_neg_inf`（全 -inf → max=-inf,sumexp=0）。
- **优缺点：** 优点：通过 parent kernel 间接覆盖 helper，并分别覆盖 greedy 与非 greedy 两条路径；全 -inf 边界专测。缺点：reference 对块求和与 kernel tile 语义需精确对应，复杂；只测单 spec step 的 draft（`drf_cpu[rs,0,start:end]`）。

### 2.4 test_compute_block_stats_kernel.py
- **被测 kernel/来源：** `_compute_local_logits_stats_kernel` + `_compute_cumulative_log_p_kernel`（`vllm/v1/.../rejection_sampler_utils.py`）。
- **测试标准：** `assert_close`（cumulative log-p rtol=1e-4,atol=1e-4）。
- **测试方式：** 直接 launch 两个 kernel，先算 block stats 再算 cumulative log p，与 CPU 暴力数值参考 `_global_logsumexp_cpu` 对比。
- **测试流程：** 构造输入；先 launch stats 生成 local max/sumexp，再 launch `_compute_cumulative_log_p_kernel`；同步；逐 req×step 校验 `log_p = min(log_p + (target_lp - draft_lp), 0)`。
- **用例：** `test_cumulative_log_p`（num_reqs∈{1,2}×num_draft∈{1,2,3}×vocab∈{128,1024}）。
- **优缺点：** 优点：覆盖了非默认路径 block-verification 的 cumulative log-p 统计，CPU 参考独立于 kernel 推导。缺点：greedy（temp=0）路径被 `continue` 跳过（数值无意义）；依赖 `num_warps=1` 启动，运行方式与生产有差异。

### 2.5 test_compute_global_logsumexp.py
- **被测 helper/来源：** `_compute_global_lse`（或旧名 `_compute_global_logsumexp`，`vllm/v1/.../rejection_sampler_utils.py`）。
- **测试标准：** `assert_close(rtol=1e-5,atol=1e-5)` + `==`（-inf 边界）。
- **测试方式：** 因 helper 非独立 launch，定义 wrapper kernel `_global_logsumexp_wrapper` 调用它并存储结果，与 CPU 参考 `_global_logsumexp_ref` 对比。
- **测试流程：** 构造 local_max/local_sumexp；逐 logit 用 wrapper launch；同步；对比。
- **用例：** `test_global_logsumexp`（num_logits∈{1,4}×num_blocks∈{1,2,4}）、`test_all_neg_inf_blocks`（→-inf）、`test_single_block`。
- **优缺点：** 优点：覆盖多块 reduce、全 -inf、单块三种场景；兼容新旧 helper 名（try/except 导入）。缺点：README 已确认 `test_all_neg_inf_blocks` 在 2026-08-04 复现为 **NaN（数值正确性问题）**，被刻意保留为 FAILED 而非 skip/XFAIL——这是本测试的重要发现，说明其参考与 kernel 判定足够敏感。缺点方面，wrapper 每 logit 单次 launch，效率低。

### 2.6 test_fill_logprob_token_ids_kernel.py
- **被测 kernel/来源：** `vllm/v1/worker/gpu/sample/logprob.py` 的 `_fill_logprob_token_ids_kernel`。
- **测试标准：** `assert_close(rtol=0,atol=0)`。
- **测试方式：** 直接 launch，与 CPU 参考 `_fill_logprob_token_ids_ref` 对比 token_ids 与 valid_mask。
- **测试流程：** 构造 sampled_token_ids/topk_indices/num_per_req_token_ids/per_req_token_ids；launch `[(batch_size,)]`；同步；对比。
- **用例：** `test_custom_token_ids`（batch∈{1,4,8}×topk∈{0,3,5}）、`test_no_custom_no_topk`。
- **优缺点：** 优点：覆盖"custom token 覆盖 top-k"与"全空只有 sampled"两条逻辑。缺点：README 确认 `test_custom_token_ids` 在安装版 vLLM 上复现**非 custom 行 top-k 列未写入**的正确性问题（上游 commit d7af6b34d8 #41761 已修），也刻意保留为 FAILED——再次体现该测试的判定价值。

### 2.7 test_flatten_sampled_kernel.py
- **被测 kernel/来源：** `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` 的 `_flatten_sampled_kernel`。
- **测试标准：** `assert_close(rtol=0,atol=0)` + `assert all(==-1)`。
- **测试方式：** 直接 launch，与 CPU 参考 `_flatten_sampled_ref` 对比。
- **测试流程：** 构造 sampled/num_sampled/cu_num_logits；launch `[(num_reqs,)]`；同步；对比。
- **用例：** `test_flatten_basic`（num_reqs∈{1,2,4,8}×num_spec_steps∈{1,3,5}）、`test_all_zeros_num_sampled`、`test_single_req_multi_logits`。
- **优缺点：** 优点：覆盖 num_sampled=0 的保留原值（-1）语义与单请求多 logits。缺点：num_sampled 随机夹在 [0,num_spec_steps+1]，未测 num_sampled 越界被截断的真实行为。

### 2.8 test_gather_block_tables_kernel.py
- **被测 kernel/来源：** `vllm/v1/worker/gpu/block_table.py` 的 `_gather_block_tables_kernel`（借助 `_load_ptr` 间接寻址）。
- **测试标准：** `assert_close(rtol=0,atol=0)` + 行断言。
- **测试方式：** 直接 launch，与 CPU 参考 `_gather_block_tables_ref` 对比。
- **测试流程：** 构造 batch_idx_to_req_idx/多基准组 block tables（ptr-to-ptrs 用 data_ptr 构造）；launch `[(num_groups, num_reqs)]`；同步；对比。
- **用例：** `test_gather_basic`（num_groups∈{1,2,4}×max_num_reqs∈{4,8}×max_num_blocks∈{64,128}）、`test_padding_zeros`（num_reqs 之后行清零）。
- **优缺点：** 优点：覆盖多组（KV cache group）、ptr 间接寻址与 padding 清零语义。缺点：num_blocks 固定为满（max），未测部分块数/逐组不同块数的 gather；`BLOCK_SIZE=16` 固定。

### 2.9 test_gumbel_block_argmax.py
- **被测 helper/来源：** `vllm/v1/worker/gpu/sample/gumbel.py` 的 `gumbel_block_argmax`（JIT helper）。
- **测试标准：** 精确值断言（`==` idx）。
- **测试方式：** 定义本地 wrapper kernel（`_gumbel_block_argmax_wrapper`）调用 helper 并返回 (value, idx)，与边界语义断言对比。
- **测试流程：** setup（vocab=128,BLOCK_SIZE=64）；wrapper 单 block 调用；校验 idx。
- **用例：** `test_no_temperature_no_gumbel`（temp=0 取 max）、`test_temperature_applied`、`test_gumbel_noise`（大 margin 保证 argmax 不变）。
- **优缺点：** 优点：覆盖零噪声、温度应用、噪声下大 margin 三者。缺点：测试偏少（仅 3 个），未像 patch 版覆盖 processed_logits 输出路径；FP64/PER_TOKEN_COL 仅是参数存在未真正变化测试。

### 2.10 test_insert_resampled_kernel.py
- **被测 kernel/来源：** `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py` 的 `_insert_resampled_kernel`。
- **测试标准：** `assert_close(rtol=0,atol=0)` + 属性断言。
- **测试方式：** 直接 launch，与 CPU 参考 `_insert_resampled_ref` 对比。
- **测试流程：** 构造 sampled/num_sampled/resampled 等；launch `[(num_reqs,)]`；同步；对比。
- **用例：** `test_insert_resampled_basic`（num_reqs∈{1,2,4}×num_spec_steps∈{1,3}）、`test_greedy_non_bonus_skip`（早退，sampled==-1）、`test_bonus_token_greedy`（bonus 始终插入 42）。
- **优缺点：** 优点：覆盖 insert 的 normal/skip/bonus 三态。缺点：`test_greedy_non_bonus_skip` 因 Triton 编译"runtime early return 后的 load 仍需合法类型指针"，需传 dummy 有效指针（测试技巧，也说明边界早退路径复杂）。

### 2.11 test_load_ptr.py
- **被测 helper/来源：** `vllm/v1/worker/gpu/buffer_utils.py` 的 `_load_ptr`（JIT 间接寻址 helper）。
- **测试标准：** `assert` + `assert_close`。
- **测试方式：** 定义本地 wrapper kernel 调用 `_load_ptr` 装载/存储，校验 int32/float32/多值。
- **测试流程：** 分别定义 `_load_ptr_wrapper_kernel`/`_load_float_ptr_kernel`/`_load_and_store_multi_kernel`；launch；读取输出。
- **用例：** `test_load_ptr_int32`（load 42）、`test_load_ptr_float32`（int32 指针技巧 load 3.14159）、`test_load_ptr_multiple_values`（10 个 strided 值对 arange）。
- **优缺点：** 优点：对 `_load_ptr` 的 `tl.cast`/`tl.multiple_of(ptr,16)` 对齐语义有直接覆盖。缺点：helper 极基础，测试偏向"能运行且返回正确"的功能性验证而非精度；float 用 1e-5 容差。

### 2.12 test_prepare_dflash_inputs_kernel.py（上游 vanilla 版）
- **被测 kernel/来源：** `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` 的 `_prepare_dflash_inputs_kernel`。
- **测试标准：** 逐字段精确断言（`.item()`）。
- **测试方式：** 直接 launch，手写 inline 期望断言（query_start_loc/bonus 或非 bonus input_ids/seq_lens/context_positions/sample 索引）。
- **测试流程：** 构造大量输出/输入张量（9 输出+15 标量）；launch `[(num_reqs, num_blocks)]`；同步；逐字段校验。
- **用例：** `test_prepare_dflash_inputs`（num_reqs∈{1,2}×num_ctx∈{2,4}×num_query∈{2,3}）、`test_cuda_graph_padding`、`test_sample_from_anchor`。
- **优缺点：** 优点：覆盖 DFlash 输入构造的复杂多输出与 CUDA graph padding、anchor 采样。缺点：无独立整块 CPU 参考（逐字段 .item() 断言），可读性/覆盖连续性弱于 missing 版的大型参考实现；缺 import 时模块级 skip（旧包无 worker-v2 DFlash）。

### 2.13 test_prompt_logprobs_token_ids_kernel.py
- **被测 kernel/来源：** `vllm/v1/worker/gpu/sample/prompt_logprob.py` 的 `_prompt_logprobs_token_ids_kernel`。
- **测试标准：** `assert_close(rtol=0,atol=0)`。
- **测试方式：** 直接 launch，与 CPU 参考 `_prompt_logprobs_token_ids_ref` 对比。
- **测试流程：** 构造 all_token_ids/idx_mapping/query_start_loc/num_computed_tokens；launch `[(num_reqs,)]`，`BLOCK_SIZE=1024`；同步；对比。
- **用例：** `test_prompt_logprobs_token_ids`（num_reqs∈{1,2,4}×query_len∈{1,4,16}）、`test_nonzero_num_computed_tokens`（偏移 num_computed+1）。
- **优缺点：** 优点：覆盖非零 num_computed 偏移。缺点：query_len 偏小；参考实现对"shift by one"语义依赖注释清楚。

### 2.14 test_rejection_kernel.py（上游 vanilla）
- **被测 kernel/来源：** `_rejection_kernel` 及支持 `_compute_local_logits_stats_kernel`（rejection_sampler_utils.py）。
- **测试标准：** 行为/属性断言（rejected_steps 计数、token 相等）。
- **测试方式：** 直接 launch，用行为断言而非代数 CPU 参考。
- **测试流程：** 先 launch stats 再 launch rejection（`num_warps=1`，SYNTHETIC/USE_BLOCK_VERIFICATION constexpr）；同步；断言。
- **用例：** `test_greedy_rejection`（all accepted）、`test_non_greedy_rejection`、`test_greedy_rejection_with_rejected`、`test_synthetic_mode`。
- **优缺点：** 优点：覆盖 greedy/non-greedy/synthetic 三模式接受语义。缺点：非 greedy 只用范围断言（0≤rejected≤num_draft）而非统计匹配；`num_warps=1` 与生产运行有差异。

### 2.15 test_resample_kernel.py（上游 vanilla）
- **被测 kernel/来源：** `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py` 的 `_resample_kernel`。
- **测试标准：** argmax 精确 + max `rtol=1e-5,atol=1e-5`。
- **测试方式：** 直接 launch，与 CPU 参考 `_resample_ref` 对比。**版本自适应**：探测 `USE_BLOCK_VERIFICATION`/`cumulative_log_p_ptr` 是否在 arg_names，分发 legacy/block-verification 两种签名。
- **测试流程：** 构造输入；`_launch_resample` 按签名分发；同步；对比 argmax/max。
- **用例：** `test_resample_basic`（num_reqs∈{1,2,4}×num_spec_steps∈{1,3}×has_draft∈{F,T}×use_block_verif∈{F,T}，block 不可用则该组合 skip）、`test_greedy_no_resample`、`test_bonus_token`。
- **优缺点：** 优点：只用 temp=0（bonus/greedy）避免 Gumbel 噪声，使精确 argmax 可比成立（参考在 t≠0 时 raise ValueError 说明 PyTorch 与 Triton PRNG 流不同）；强大的双签名自适应。缺点：对非 bonus greedy 是 no-op 校验，未覆盖真实重采样随机路径。

### 2.16 test_scatter_num_accepted_kernel.py
- **被测 kernel/来源：** `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` 的 `_scatter_num_accepted_kernel`。
- **测试标准：** `assert_close(rtol=0,atol=0)`。
- **测试方式：** 直接 launch，与 CPU 参考 `_scatter_num_accepted_ref` 对比。
- **测试流程：** 构造 num_sampled/num_accepted/idx_mapping；launch `[(num_reqs,)]`；同步；对比。
- **用例：** `test_scatter_basic`（num_reqs∈{1,4,8,16}×max∈{16,32}）、`test_skip_negative`（idx_mapping<0 跳过）、`test_clamp_to_one`（num_sampled<1 置 1）。
- **优缺点：** 优点：覆盖 skip 负映射与 clamp≥1 两个易错语义，多批次 randperm 映射。缺点：kernel 简单（3 参数），覆盖偏功能型。

### 2.17 test_selective_scan_update_kernel.py
- **被测 kernel/来源：** `vllm/model_executor/layers/mamba/ops/mamba_ssm.py` 的 `_selective_scan_update_kernel`（Mamba 选择性扫描）。
- **测试标准：** `assert_close(rtol=1e-4,atol=1e-4)`（out 与 state）。
- **测试方式：** 直接 launch（`@triton.heuristics` 自动选 BLOCK_SIZE/num_warps），与纯 PyTorch CPU 参考 `_selective_scan_update_cpu`（离散化建模）对比。
- **测试流程：** 构造 D/z/dt_bias 等；launch（网格 `(cdiv(dim,BLOCK_SIZE_M), N, nheads)`）；同步；对比 out/state。
- **用例：** `test_basic_with_all_options`（全使能）、`test_no_z_no_d`、`test_varied_dstate`（dstate∈{2,8,16}，不同启发式）、`test_tie_hdim`、`test_dt_softplus_disabled`。
- **优缺点：** 优点：浮点 kernel 用 1e-4 容差并有独立 CPU 数学参考，覆盖多组合（Z/D/dt_bias/softplus/tie_hdim）；与 `try_get_optimal_ssm_config` 结合跟随启发式。缺点：大量 Tensor 传 None 与 stride=0（padding/constexpr 技巧）使可读性低；依赖 Mamba 特定配置。

### 2.18 test_tl_rand64.py
- **被测 helper/来源：** `tl_rand64`（`vllm/v1/worker/gpu/sample/gumbel.py`）→ 通过 A3 FP32 替代 `_tl_rand64_a3`（本地定义，用 `tl.rand`）与生产 wrapper `gumbel_sample` 测试。
- **测试标准：** 统计/范围断言 + `torch.equal`（生产路径）。
- **测试方式：** `TestTlRand64` 用本地 wrapper 统计校验（范围、均值均匀性、seed 差异）；`TestTlRand64ViaGumbelSample` 调生产 `gumbel_sample` 面向 dominant token（=7）断言 sampling 全为 7。
- **测试流程：** ①wrapper 生成 10000 样本校验范围/均匀；②不同 seed 输出不同；③生产 Gumbel 路径 dominant token 一定被采。
- **用例：** `TestTlRand64.test_range`（includes_zero F/T）、`test_statistical_uniformity`（mean∈(0.45,0.55)≈5σ）、`test_different_seeds`；`TestTlRand64ViaGumbelSample.test_fp32_business_path`。
- **优缺点：** 优点：不声称 FP64 逐位等价，只验证 A3 兼容的 FP32 随机均匀替代契约与生产 wrapper；统计与路径执行双重覆盖。缺点：注：上游 `tl_rand64` 在 Ascend 上无法编译（float64），因此权威参考并非逐位一致，判定标准是"行为契约"而非位级相等；均值检验只用 1 个种子。

### 2.19 test_topk_topp_kernel.py
- **被测 kernel/来源：** `vllm/v1/sample/ops/topk_topp_triton.py` 的 `_topk_topp_kernel`（top-k / top-p 掩码，含 NORMAL_CDF 查表）。
- **测试标准：** `assert_close(rtol=1e-5,atol=1e-5)` + 属性断言（幸存数、累计概率）。
- **测试方式：** 直接 launch，与纯 PyTorch CPU 参考 `_apply_topk_topp_cpu` 对比；并用独立用例外加 top-k 幸存数计数、top-p 累计概率校验。
- **测试流程：** 构造 logits/k/p 及查表；launch `[(1,)]`，MASK_VALUE=-inf，多 constexpr；同步；对比。
- **用例：** `test_topk_topp_combined`（batch∈{1,2}×vocab∈{64,128}×topk∈{F,T}×topp∈{F,T}，双禁用 skip）、`test_topk_only_exact_count`、`test_topk_duplicate_boundary`（k 边界重复值）、`test_topp_cumulative_probability`、`test_topk_topp_noop`。
- **优缺点：** 优点：覆盖 topk/topp 单独与组合、重复边界值、累计概率约束、no-op；对禁用 top-k 的"用 k=vocab_size 绕过"有专门处理。缺点：禁用 topk 时实际仍走 top-k 整理路径，语义绕行；查表内核本身精度用 1e-5 但 NORMAL 近似未独立校验。

### 2.20 test_update_min_larger_stats.py
- **被测 helper/来源：** `vllm/v1/sample/ops/topk_topp_triton.py` 的 `_update_min_larger_stats`（JIT helper，top-k/top-p pivot 之上最小值的合并逻辑）。
- **测试标准：** `assert_close`（min rtol=1e-5,atol=1e-5；count 精确 `==`）。
- **测试方式：** 定义本地 wrapper kernel `_update_min_larger_wrapper` 调用 helper，与 CPU 参考 `_update_min_larger_ref` 对比。
- **测试流程：** 构造 data/above_mask/min_larger/num_min_larger/sentinel；wrapper launch；同步；对比。
- **用例：** `test_basic_merge_new_min`（新 min<running 替换）、`test_merge_same_min`（相等累加）、`test_merge_larger_min`（更大保持不变）、`test_no_above_data`、`test_sentinel_filtering`（等于 sentinel 不参与）。
- **优缺点：** 优点：覆盖合并规则全部 3 种情形 + 无 above 数据 + sentinel 过滤，边界齐全。缺点：单 tile 测试，未在多 tile 串行合并的完整 topk 流程端到端验证；min 用容差、count 用精确，判定标准不一。

### 2.21 一组 `_patch.py` 测试（Ascend 改名/替换/再导出路径测试）
> 以下 patch 测试统一面向 **vLLM-Ascend 对上游 kernel 的改名、重写或再导出路径**，判定准则与 0.2 一致。多数以 CPU 参考 + `assert_close` 校验。

- **test_compute_block_max_and_sumexp_patch.py**：被测为 Ascend 模块导出的 `_compute_block_stats_kernel`（别名 `_compute_local_logits_stats_kernel`），经 parent kernel **间接**覆盖 `_compute_max_and_sumexp` helper。直接 launch，`test_compute_block_max_and_sumexp_via_ascend_parent`（vocab∈{15,31,65}，非 2 幂边界），对照 CPU 参考校验 max/sumexp（rtol=1e-5）。**优点**是覆盖非 2 幂 vocab 分块边界；**缺点**是间接覆盖、helper 本身不独立校验。
- **test_compute_block_stats_kernel_patch.py**：直接 launch Ascend 别名 `_compute_block_stats_kernel`，同时校验 non-greedy（target+draft stats）与 greedy（argmax/max）。`test_non_greedy_target_and_draft_stats`、`test_greedy_argmax_and_max`（block_size=16,vocab=37）。**优点**是两模式兼测；**缺点**是规模极小。
- **test_compute_global_logsumexp_patch.py**：与 2.5 同构，但导入 **Ascend 包导出的** `_compute_global_lse` 别名。wrapper launch，`_helper_name` 兼容 aliasing。README 确认其 `test_all_neg_inf_blocks` 也在 A3 复现 **NaN**（数值正确性问题，见 3 节英文确认发现）。
- **test_fill_logprob_token_ids_kernel_patch.py**：代表 Ascend 用**纯 tensor 拼装**（`compute_topk_logprobs` 内部）**替换**上游 Triton `_fill_logprob_token_ids_kernel`。直接调用 `compute_topk_logprobs` 返回结构，PyTorch 参考对比 token_ids/ranks（exact）与 logprobs（rtol=1e-4）。参数化 (48,1024,5)/(96,1024,0)/(24,1519,1)/(1,320,10)。**优点**是验证替换路径而非 Triton kernel；**缺点**是不再覆盖 Triton 算法（Ascend 已不调用该 kernel）。
- **test_gumbel_block_argmax_patch.py**：测试 Ascend `_npu_gumbel_block_argmax`（wrapper 单块 + inline 多块），与 missing 版 1.6 内容基本一致（零噪声/temp/noise 主导/processed_logits），但置于 from_vllm 的 patch 集。判定：idx 精确、processed_logits rtol=1e-5。
- **test_insert_resampled_kernel_patch.py**：确认 Ascend **再导出**上游 `_insert_resampled_kernel`（compatibility path，无算法改动）。测试与 2.10 基本一致（basic/greedy skip/bonus），对照 `_insert_resampled_ref`（rtol=0,atol=0）。
- **test_prepare_dflash_inputs_kernel_patch.py**：测试 Ascend `_prepare_dflash_inputs_kernel_ascend`（与 missing 版 1.10 的 legacy 分支一致），单参考路径、10 输出精确 `assert_close`。patch 差异包括 `tl.minimum` 截断、`tl.int64` cast、`tl.range` 简化、padded sample idx=-1 保护。**优点**补全该补丁 clamping/OOB 保护断言；**缺点**旧包缺 worker-v2 DFlash 时模块级 skip。
- **test_rejection_kernel_patch.py**：测试 Ascend `_probabilistic_rejection_kernel`（与 missing 版 1.15 大体相同）。覆盖 greedy 精确（与 CPU 参考）与 non-greedy NPU `u=0` 恒接受；处理两仓 helper 改名（`_load_rejection_kernels`）。判定：int 精确 + 行为断言。
- **test_tl_rand64_patch.py**：与 2.18 相同——验证 A3 FP32 随机均匀替代契约（`_tl_rand64_a3` 用 `tl.rand`+`tl.maximum` 下夹）+ 生产 `gumbel_sample` dominant token 路径。明确声明"不声称 FP64 逐位等价"。
- **test_resample_kernel_patch.py**：与 missing 版 1.16 相同——Ascend `_resample_kernel`（用 `_npu_gumbel_block_argmax`，`APPLY_TEMPERATURE=False`），覆盖 bonus（重采样）与 non-bonus greedy（no-op）。

**该组整体优缺点：**
- **优点：** 每个 patch 测试都明确绑定到 Ascend 的改名/替换源位置（注释含行号），验证 A3 **实际生效的实现路径**而非仅上游；对两仓 API 版本错位（`_compute_global_lse`/`_compute_local_logits_stats_kernel` 旧名）用兼容别名加载，鲁棒。
- **缺点：** 部分 patch 测试与 missing 版内容重复（同 kernel 两处测试，维护负担）；随机路径（Gumbel）只做行为契约而非统计/位级对比；替换路径（fill_logprob）不再覆盖 Triton 算法本身。

---

## 3. existing_accuracy_tests/from_vllm_ascend/（11 个从 vLLM-Ascend 搬运文件）

> 特点：保留 vLLM-Ascend 官方 A3/NPU 测试主体，仅增加来源注释。判定策略同 0.2。多数为 wrapper 公共函数测试（调用 apply_bad_words / apply_min_p / apply_penalties / apply_temperature / gumbel_sample / compute_topk_logprobs），或两 kernel 头对头对比。

### 3.1 test_bad_words.py
- **被测 kernel/来源：** `_bad_words_kernel`，经 wrapper `apply_bad_words`（vllm_ascend/worker/v2/sample/bad_words.py）。
- **测试标准：** `torch.allclose`（默认 rtol=1e-5,atol=1e-8）+ 行为断言（logits 是否变化）。
- **测试方式：** wrapper 级（不直接 launch kernel）；行为/一致性校验——有 bad words 时 logits 变化、无 bad words 时不变化。
- **测试流程：** module 级 `initialize_triton_device_properties` fixture；构造 bad_word_token_ids；调 `apply_bad_words`；对比前后 logits。
- **用例：** 参数化 `BAD_WORDS_TEST_CASES`（small/medium/large：num_tokens 512/1024/2048、vocab 50257、num_requests 16/32/64、bad word 3/5/8）；另加 no_bad_words、edge_case（128 bad words）、token_limit（1024 上限超限）。
- **优缺点：** 优点：覆盖规格大小与 1024 token 上限边界；缺点：无数值精度 oracle（只用"是否发生变化"判定，未校验 -inf 位置与参考的一致性），判定强度偏弱。

### 3.2 test_bincount.py
- **被测 kernel/来源：** `_bincount_kernel`（vllm_ascend/worker/v2/sample/penalties.py）。
- **测试标准：** `torch.equal`（位级精确）。
- **测试方式：** 直接 launch（非 wrapper），与纯 Python/NumPy 参考 `torch_bincount()` 位级对比 prompt_bin_mask 与 output_bin_counts。
- **测试流程：** 构造大用例（64 requests、40960 token buffer、151936 vocab、BLOCK_SIZE=1024）；launch；同步；torch.equal。
- **用例：** 单一大规模场景（seed 42），无参数化。
- **优缺点：** 优点：已确认的 A3 精度通过发现——2026-08-07 两个输出与 PyTorch 参考完全一致（1 passed in 19.12s），且历史 atomic_or 挂起未复现（见 3.11 诊断）；位级精确。缺点：规模固定单一；依赖大显存。

### 3.3 test_compute_slot_mapping.py
- **被测 kernel/来源：** Ascend `_compute_slot_mappings_kernel`（vllm_ascend/worker/v2/block_table.py）。
- **测试标准：** `torch.equal`（位级）。
- **测试方式：** 两 kernel 头对头——Ascend 版与上游 GPU `ref_compute_slot_mappings_kernel` 各自 launch，输出直接 equal。
- **测试流程：** 构造 block table/positions/query_start_loc；分别 launch 两 kernel；同步；equal。
- **用例：** 单一场景（seed 42，cp_rank=0 等），无参数化。
- **优缺点：** 优点：以上游 GPU kernel 为 oracle 的强交叉验证。缺点：整段包在 try/except Exception 中不 re-raise（打印后吞掉），若 launch 异常测试可能"空通过"（严重的潜在漏判）；只测单 rank。

### 3.4 test_compute_topk_logprobs.py
- **被测 kernel/来源：** `_topk_log_softmax_kernel` + `_ranks_kernel`，经 wrapper `compute_topk_logprobs`（vllm_ascend/worker/v2/sample/logprob.py）。
- **测试标准：** token_ids/ranks `torch.equal`；logprobs `allclose(rtol=1e-4,atol=1e-4)`。
- **测试方式：** wrapper 级，返回值结构比对 + PyTorch 参考（topk/log_softmax/gather/ranks）。
- **测试流程：** 构造 logits；调 `compute_topk_logprobs`；与 PyTorch 参考对比。
- **用例：** (48,1024,5)/(96,1024,0)/(24,1519,1)/(1,320,10)。
- **优缺点：** 优点：数值参考完整（log_softmax 与 rank 独立推导），覆盖 num_logprobs=0 边界。缺点：ranks 用 (logits>sampled).sum 参考与 kernel 排序语义可能因重复值有细微差异；等值边界未专测。

### 3.5 test_gumbel_sampling.py
- **被测 kernel/来源：** `_gumbel_sample_kernel`，经 wrapper `gumbel_sample`/`apply_temperature`（vllm_ascend/worker/v2/sample/gumbel.py）。
- **测试标准：** 混合——temperature `allclose(atol=1e-4,rtol=1e-5)`；greedy `torch.equal`；determinism `torch.equal`；分布用统计/属性校验（非直方图对比）。
- **测试方式：** wrapper 级；多角度：温度缩放（对照纯 Python 参考）、greedy（对照 argmax）、确定性（同 seed 相同/异 seed 不同）、分布倾向（256 次 trial 低 temp 胜率>90%）、processed_logits 存储（EAGLE 规格）。
- **测试流程：** 各测试构造相应输入调 gumbel_sample / apply_temperature，多路校验。
- **用例：** `TestGumbelSampling` 约 17 个方法，覆盖 (num_tokens,num_reqs,vocab) 多档（含 32000/102400/151936 大 vocab）、温度 0/1/0.01/100、混合温度、非连续 idx_mapping、processed_logits row/col 存储、单 token、EAGLE。
- **优缺点：** 优点：覆盖最全面之一（确定性/分布倾向/processed-logits/EAGLE/大词表/极端温度）；对随机采样用行为契约而非脆弱直方图。缺点：分布只统计"低 temp 胜率高"，未做严格分布检验；processed_logits 写入存在非 1:1 映射写竞争隐患（注释已注明）。

### 3.6 test_log_softmax.py
- **被测 kernel/来源：** `_topk_log_softmax_kernel`（vllm_ascend/worker/v2/sample/logprob.py），直接 launch。
- **测试标准：** `allclose(rtol=1e-3,atol=1e-3)`。
- **测试方式：** 直接 launch，与 `torch.gather(torch.log_softmax(...),1,token_ids)` 参考对比。
- **测试流程：** 构造 logits/token_ids/num_logprobs；launch [(batch_size,)]，PADDED_TOPK；同步；对比。
- **用例：** (48,102400,50)/(96,102400,1)/(24,151936,8)。
- **优缺点：** 优点：直接 kernel 校验；参考刻意重新计算 log_softmax 而非复用 Triton 输出格式，规避 NPU internal format 告警。缺点：容差 1e-3 偏大（float32 softmax）；无小 vocab 边界。

### 3.7 test_min_p.py
- **被测 kernel/来源：** `_min_p_kernel`，经 wrapper `apply_min_p`（vllm_ascend/worker/v2/sample/min_p.py）。
- **测试标准：** `torch.equal`（-inf mask）+ `allclose(rtol=1e-4,atol=1e-4)`（有效值）。
- **测试方式：** wrapper 级，与纯 PyTorch 参考 `torch_min_p_torch` 对比（mask + 值）。
- **测试流程：** 构造 logits/expanded_idx_mapping/min_p；调 apply_min_p；参考计算；分别比较 inf-mask 与有效 logits。
- **用例：** (48,102400)/(96,102400)/(24,151936)/(1,32000)。
- **优缺点：** 优点：分别校验"屏蔽位置"与"保留值"，min_p 随机覆盖。缺点：min_p==0 的跳过路径未专测；未测阈值边界精确相等。

### 3.8 test_penality.py
- **被测 kernel/来源：** `_penalties_kernel`，经 wrapper `apply_penalties`（vllm_ascend/worker/v2/sample/penalties.py）。
- **测试标准：** `allclose`（fp16/bf16：DEFAULT 1e-3，bf16 放宽 1e-2）。
- **测试方式：** wrapper 级，与独立 PyTorch 参考 `pytorch_apply_penalties`（含位解包/累计计数/draft counts）对比。
- **测试流程：** create_test_data 生成随机惩罚配置；apply_penalties；参考实现；allclose；gc.collect + empty_cache 释放。
- **用例：** num_tokens∈{1,4} x vocab 1000 x num_status∈{1,4} x num_spec_tokens∈{0,1,3} x dtype∈{bf16,fp16}。
- **优缺点：** 优点：较大规模参数化，独立参考覆盖 rep/freq/pres 三种惩罚与 draft counts；显存清理。缺点：参考实现与 kernel 语义高度对应、独立性中等；未测 fp32。

### 3.9 test_post_update.py
- **被测 kernel/来源：** 上游 `_post_update_kernel`（vLLM）与 Ascend `_post_update_kernel`（vllm_ascend/worker/v2/input_batch.py）头对头 + CPU oracle。
- **测试标准：** `assert_close(rtol=0,atol=0)`（int 精确）双验证。
- **测试方式：** 分别 launch 上游（num_warps=1）与 Ascend（grid=min(num_rows, vectorcore)），与独立串行 CPU 参考 post_update_ref 对比，并 upstream vs ascend 互较。
- **测试流程：** generate_test_data 生成随机；三分支（upstream/Ascend/reference）各自演进；对 5 个被改量逐项 assert_close。
- **用例：** (36,36,200,2)/(48,48,32000,5)/(128,128,32000,5)。
- **优缺点：** 优点：三重 oracle（上游、Ascend、CPU 参考）互证；Ascend 用 vectorcore grid、上游用行网格，对比覆盖 A3 网格策略。缺点：import 阶段有专门 bootstrapping 规避 cycle，若环境异常会 fail（非 skip）。

### 3.10 test_temperature.py
- **被测 kernel/来源：** `_temperature_kernel`，经 wrapper `apply_temperature`（vllm_ascend/worker/v2/sample/gumbel.py）。
- **测试标准：** `allclose(atol=1e-4,rtol=1e-5)`。
- **测试方式：** wrapper 级，与纯 PyTorch 参考 torch_apply_temperature 对比。
- **测试流程：** 构造 logits（主流词表 32000/50257/65024/128256/151936）；apply_temperature；参考；allclose。温度含 0.0/1.0 边界（不改变）。
- **用例：** [(random.randint(1,64), vocab) for vocab in VOCAB_SIZES]。
- **优缺点：** 优点：覆盖常见模型词表与 temp=0/1 边界。缺点：num_tokens 随机 1-64，无显式大 batch；参考直观但判定仅 allclose。

### 3.11 diagnose_bincount_atomic_or.py
- **被测对象/来源：** 非 pytest 的独立诊断脚本（`_bincount_kernel` 的 atomic_or 挂起排查）。
- **测试标准：** 每个探针在独立子进程运行，父进程侧超时（默认 60s）兜底，避免编译/设备挂起拖垮测试；断言 torch.testing.assert_close（精确）。
- **测试方式：** 8 个原子探针（store/atomic_add_unique/atomic_add_contended/atomic_or_single/atomic_or_unique/atomic_or_contended_same/atomic_or_contended_bits/bincount），逐步定位挂起根因。
- **测试流程：** python diagnose_bincount_atomic_or.py [--timeout N] [--only probe]；父进程对每个探针起子进程，超时 kill（返回 124）。
- **用例：** 8 个原子原语探针 + 1 个 bincount 全量探针。
- **优缺点：** 优点：进程级超时隔离能安全排查会挂死的 kernel——这是普通 pytest 无法做到的（挂起会无限阻塞）；逐原语定位原子操作兼容性。缺点：非 pytest，需手动/CI 单独调用；返回码/日志约定较自定义。

---

## 4. 共性优势与不足总结

### 4.1 总体优点
1. **判定策略先进**：conftest.py 用"后端兼容性失败→XFAIL（精度未知）"与"真实 assert 失败"分离，避免编译失败掩盖精度失败。
2. **直接 launch 为主、多参考来源**：绝大多数直接 launch NPU kernel，参考来自纯 Python CPU 参考 / PyTorch 参考 / 两 kernel 头对头 / 精确值断言，从位级到浮点容差梯度完整。
3. **Ascend patch 覆盖真实生效路径**：_patch.py 测试绑定到 vLLM-Ascend 改名/替换实现（含行号），并兼容两仓 API 版本错位。
4. **发现真实 bug**：test_compute_global_logsumexp（全 -inf 块 → NaN）与 test_fill_logprob_token_ids_kernel（非 custom 行 top-k 未写）均因测试足够敏感而复现并保留为 FAILED，价值高。
5. **随机/统计处理得当**：Gumbel 随机路径用"行为契约 + margin 确定性"或统计校验，避免脆弱。

### 4.2 总体不足
1. **随机采样路径无严格分布匹配**：非 greedy rejection（NPU u=0 恒接受）与 Gumbel 只测行为/分布倾向，未做 K-S 检验或分布差异对比。
2. **判据泄露风险**：部分参考实现与 kernel 逻辑高度同构，独立性有限；个别文件（test_compute_slot_mapping.py）有 try/except 吞异常导致"空通过"的隐患。
3. **覆盖不对称**：大量测试只测小规模/单一场景；scatter/flatten/load_ptr 等简单 kernel 覆盖偏"能运行"式。
4. **维护负担**：部分 patch 测试与 missing 版重复，且大量文件依赖 torch.manual_seed(42)、固定 BLOCK_SIZE，扩展性一般。
5. **环境强依赖**：多数测试需完整 Ascend A3 + vLLM + vLLM-Ascend + torch_npu 栈；本地只能做 AST 语法检查，真实执行受限。

### 4.3 报告要点速查（供引用）
- **48 个 pytest 文件**：missing 18 + from_vllm 28（含 10 个 patch）+ from_vllm_ascend 10（另 1 个诊断脚本）。
- **已确认 A3 通过**：test_bincount（2026-08-07，位级一致）。
- **已确认精度问题（保留 FAILED）**：_compute_global_lse 全 -inf→NaN（2026-08-04）；_fill_logprob_token_ids_kernel 非 custom 行未写（2026-08-04，上游已修）。
- **XFAIL/SKIP 政策**：编译/资源限制→precision unknown；参数无效/可能挂起→skip。
