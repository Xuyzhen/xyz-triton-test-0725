# strict_ut_027 — 双侧（CUDA GPU + 昇腾 NPU）算子精度测试集

## 1. 项目定位

`strict_ut_027` 是一个 **standalone 的算子精度测试集项目**，从 `easy_ut_026`（NPU 单侧）
迁移并扩展而来，结构参考 `strict_ut_028`。与 026 的区别：

| 维度 | easy_ut_026 | strict_ut_027 |
| --- | --- | --- |
| 运行侧 | 仅 NPU | GPU + NPU 双侧 |
| 用例规格 | 基础 shape | 基础 + 高规格（生产级）shape |
| 代码组织 | 单文件自包含 | `common/` 共享实现 + 双侧薄入口 |
| 双标杆比对 | 无 | **无（按要求暂不引入）** |

每个用例的验证方式与 026 一致：**设备侧 kernel 输出 vs 独立 CPU 参考实现**，全部为
bitwise 级断言（int32/索引类直接 `torch.equal`；float 拷贝类按原始位比较，NaN 哨兵
视为相等）。

### 排除项（不迁移）

- `test_num_nans_kernel.py`
- `test_topk_topp_kernel_a2a3.py`
- `test_topk_topp_kernel_a5.py`

## 2. 目录结构

```
strict_ut_027/
├── __init__.py            # 包说明
├── pytest.ini             # testpaths=gpu,npu + marker 注册（--strict-markers）
├── conftest.py            # 顶层 marker 分类（gpu/npu/accuracy_l0/accuracy_l1）
├── runtime_gpu.py         # CUDA 侧运行时（STRICT_DEVICE/synchronize/init_*）
├── runtime_npu.py         # NPU 侧运行时（含 Triton 3.2 的 vllm.triton_utils shim）
├── run_gpu.sh             # GPU 服务器一键运行
├── run_npu.sh             # NPU 服务器一键运行（调用隔离运行器）
├── run_npu_isolated.py    # NPU 按测试文件逐个起独立 pytest 进程
├── run_all.sh             # 自动检测 CUDA/NPU 并运行对应侧
├── common/                # 9 个算子的共享测试实现（设备无关）
│   ├── shift_input_ids_impl.py
│   ├── shift_input_embeds_impl.py
│   ├── pad_trailing_draft_slots_impl.py
│   ├── cache_inputs_impl.py
│   ├── prepare_input_buffers_impl.py
│   ├── prepare_input_hidden_states_and_embeddings_impl.py
│   ├── update_committed_marker_cache_impl.py
│   ├── thinking_budget_impl.py
│   └── preprocess_mamba_align_fused_impl.py
├── gpu/                   # GPU 侧入口（9 个 test_*.py + conftest 提供 rt=cuda runtime）
└── npu/                   # NPU 侧入口（9 个 test_*.py + conftest 提供 rt=npu runtime）
```

**注入机制**：`common/*_impl.py` 持有完整测试逻辑（kernel 导入、输入生成、CPU 参考、
launch、断言、pytest 参数化用例），通过 `rt` fixture 接收设备运行时
（`rt.STRICT_DEVICE` / `rt.init_device_properties_triton()` / `rt.synchronize()`）。
`gpu/conftest.py` 注入 `runtime_gpu`，`npu/conftest.py` 注入 `runtime_npu`，因此
**两侧运行完全相同的用例与断言**，仅设备不同。

## 3. 覆盖的 9 个算子

| # | kernel | 来源 | 类别 | 判定 |
| --- | --- | --- | --- | --- |
| 1 | `_shift_input_ids_kernel` | `vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py` | 非计算 shift（int32） | bitwise |
| 2 | `_shift_input_embeds_kernel` | 同上 | 非计算 shift（fp32 拷贝） | bitwise（按位） |
| 3 | `_pad_trailing_draft_slots_kernel` | 同上 | 非计算 pad（int32） | bitwise |
| 4 | `_cache_inputs_kernel` | 同上 | 非计算缓存写（fp32/int 混合） | bitwise（按位） |
| 5 | `_prepare_input_buffers_kernel` | 同上 | 非计算 buffer 装配 | bitwise |
| 6 | `_prepare_input_hidden_states_and_embeddings_kernel` | 同上 | float copy/gather 融合 | bitwise（按位） |
| 7 | `_update_committed_marker_cache_kernel` | `vllm/v1/worker/gpu/sample/thinking_budget.py` | 整数 marker 扫描 | bitwise |
| 8 | `_thinking_budget_kernel` | 同上 | 离散+字面量 1e9 写入 | bitwise（NaN 哨兵） |
| 9 | `preprocess_mamba_align_fused_kernel` | `vllm/v1/worker/mamba_utils.py` | 整数/索引计算 | bitwise |

所有 kernel 均为 upstream vLLM 实现，NPU 侧不修改直接复用（marker：
`npu_upstream_reuse`）。

## 4. 高规格 shape 扩展（相对 easy_ut_026）

在保留 026 全部基础用例的前提下，每个算子追加了生产级规格（对应 `*_impl.py` 中
`--- strict_ut_027 high-spec additions ---` 注释段）：

- **shift_input_ids**：`num_reqs=32/64` 大 batch；`BLOCK_SIZE=32/64` 大块边界；
  `long_query`（单请求 64~256 token）× `num_reqs=16/64`；大 batch + 1024 块。
- **shift_input_embeds**：`hidden_size=2048/5120/7168`（Llama / Kimi / Qwen 级）；
  `num_reqs=32/64`；长 query × 大 hidden 组合。
- **pad_trailing_draft_slots**：`num_reqs=64 + num_tokens=4096 + block=1024`
  生产级组合；`num_groups=8`；`full_pad` 极端场景。
- **cache_inputs**：`hidden_size=5120/7168` 生产级；`num_reqs=32`；
  `num_speculative_steps=5` 上限 + 大 hidden 组合。
- **prepare_input_buffers**：`max_num_tokens=8192`；`num_reqs=64/max_num_reqs=128`
  最大 batch；`num_speculative_steps=5` 上限。
- **prepare_input_hidden_states_and_embeddings**：`hidden_size=5120/7168`；
  `num_reqs=32/64`；`max_reprefill`/`tile_boundary` 场景规模化。
- **update_committed_marker_cache**：`BLOCK=256/2048/4096`（单 chunk 生产路径）；
  `start_len/natural_end_len=4/8` 多 token marker；`max_len=64/256`。
- **thinking_budget**：`(start_len, natural_end_len, end_len)` 扩展至
  `(8,4,8)`、`(4,8,6)` 等（多 token 推理 marker，DeepSeek-R1 风格）。
- **preprocess_mamba_align_fused**：`num_reqs=256/512` 生产级并发；
  `BLOCK_SIZE=128/256` 大块；`BLOCK_SIZE > num_reqs` 退化单 program 边界。

## 5. 运行方式

前置：Python 环境已安装 `torch`（+ CUDA 或 torch-npu）、`vllm`（≥ v0.26.0，
含 `multi_module_mtp`/`thinking_budget` 模块）、`pytest`；NPU 侧另需
`vllm-ascend ≥ v0.16.0rc1`。

```bash
# GPU 服务器上（在 strict_ut_027 目录下）
bash run_gpu.sh                          # 全部 GPU 侧用例
bash run_gpu.sh -k shift_input_ids       # 按关键字过滤
bash run_gpu.sh --tb=long -s             # 透传 pytest 参数

# NPU 服务器上
bash run_npu.sh                          # 每个测试文件独立进程（防设备上下文污染）
bash run_npu.sh -k mamba                 # 单进程过滤运行
bash run_npu.sh test_thinking_budget_kernel.py

# 任意机器：自动检测 CUDA / NPU 并运行对应侧（都可用则两侧都跑）
bash run_all.sh
```

脚本会自动 `cd` 到仓库根（`xyz-triton-test-0725`）并设置 `PYTHONPATH`，
保证 `from accuracy_test.strict_ut_027... import ...` 可解析；运行前做设备
可用性前置检查，不可用直接报错退出。

## 6. 已知说明

- **`_thinking_budget_kernel` NPU 侧限制**：该 kernel 通过 `tt.call` 调用
  `_load_effective_token` helper，昇腾 Triton 后端无法在
  `ConvertTritonIRToLinalgIR` 中合法化该调用，编译会失败。这是后端限制而非
  精度缺陷，UT 将其报告为 SKIP（`_is_ascend_tt_call_limitation` 判定）。
- **不含双标杆**：本测试集不做 GPU vs NPU 双标杆比对，只做 设备 vs CPU 参考。
  后续如需双标杆，可参考 strict_ut_028 的 `precision/` capture 框架另行叠加。
- **进程隔离策略**：NPU 侧沿用 easy_ut_026/strict_ut_026 的经验——昇腾
  vector-core 异常会污染进程级 device 上下文，因此 `run_npu_isolated.py`
  为每个测试模块单独起一个 pytest 子进程；GPU 侧无此问题，单进程直跑。
