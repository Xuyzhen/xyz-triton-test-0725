# strict_ut_026 — 自包含严格精度套件 + GPU/NPU 精度对比工具

> 原则：**不修改 `strict_ut/` 内任何内容**。本目录是 `strict_ut` 的完整独立快照
> （`gpu/`、`npu/` 双侧 pytest 套件 + 全部 runtime 基建 + 运行脚本，逐文件复制），
> 并在其上叠加 `precision/` 捕获-比对工具。**整个目录单独拷走即可运行**，
> 不 import 任何兄弟目录（strict_ut / easy_ut_026 等）。

## 1. 目录结构

```
strict_ut_026/                      # 整体自包含，可独立分发
├── gpu/                            # GPU 侧严格单测（与 strict_ut 同构，38 个 test_*.py）
├── npu/                            # NPU 侧严格单测（与 strict_ut 同构，44 个 test_*.py）
├── llm_configs/                    # 测试引用的模型 config
├── conftest.py                     # 分级 marker（accuracy_l0/l1/l2、npu_* 分类）
├── pytest.ini                      # testpaths = gpu npu
├── metrics.py                      # 单测精度标准（Accuracy Standard，pytest 侧用）
├── runtime_gpu.py                  # CUDA runtime 助手
├── runtime_npu.py                  # NPU runtime 助手 + vllm_ascend 导入 shim（完整版）
├── run_gpu.sh / run_gpu.ps1        # GPU 套件入口
├── run_npu.sh / run_npu.ps1        # NPU 套件入口
├── run_npu_isolated.py             # 每个测试模块独立进程跑 NPU 套件
├── run_static_checks.ps1 / check_npu_imports.py / generate_suite.py / validate_suite.py
├── npu_ut_shapes.md                # NPU 算子 shape 清单（对账依据）
├── precision/                      # ★ 026 新增：GPU-NPU 精度捕获与比对
│   ├── capture_runtime.py          #   CaseSpec / case_id / 输入与结果持久化
│   ├── compare_metrics.py          #   双侧比对指标（PASS/WARN/FAIL 分级）
│   ├── kernel_cases/               #   用例注册表（每 kernel 一个模块）
│   ├── run_capture.py              #   阶段入口 --side gpu / npu
│   ├── run_one_case.py             #   单用例执行器（NPU 侧子进程隔离）
│   ├── shape_audit.py              #   两侧 shape/case 对账
│   └── compare_results.py          #   比对 + 生成 report/compare_*.md
├── inputs/                         # (生成物) 共享输入  <kernel>/<case_id>.pt
├── results/                        # (生成物) gpu/ 与 npu/ 捕获结果
├── report/                         # (生成物) 比对报告
└── README.md
```

## 2. 用法 A：跑严格单测（与 strict_ut 完全相同的入口）

在 `strict_ut_026/` 目录下：

```bash
# GPU 机器
pytest gpu/ -m gpu -v          # 或 ./run_gpu.sh（Linux）/ run_gpu.ps1（Windows）
# NPU 机器（每模块独立进程，防设备上下文污染）
python run_npu_isolated.py     # 或 ./run_npu.sh
```

分级与分类 marker 见 `pytest.ini` 与 `conftest.py`（accuracy_l0/l1/l2、
npu_ascend_adapted / npu_upstream_reuse / npu_upstream_unwired、
deterministic / stochastic）。

## 3. 用法 B：GPU-NPU 精度对比（precision/ 三阶段）

单测回答"这一侧对不对"；precision 工具回答"**同一份输入**下 NPU 与 GPU 差多少"。

```
阶段 1: GPU 机器                阶段 2: NPU 机器                阶段 3: 任意机器
┌──────────────────┐          ┌──────────────────┐          ┌──────────────────┐
│ run_capture.py   │          │ run_capture.py   │          │ compare_results  │
│   --side gpu     │          │   --side npu     │          │ shape_audit.py   │
│ 生成+持久化 inputs│──同步──▶ │ 复用同一份 inputs │──同步──▶ │ 对账+比对+报告    │
│ 输出 results/gpu │ inputs/  │ 输出 results/npu │ results/ │ 输出 report/*.md │
└──────────────────┘          └──────────────────┘          └──────────────────┘
```

### 阶段 1 — GPU 机器上采标杆

```bash
python precision/run_capture.py --side gpu                                # 全部
python precision/run_capture.py --side gpu --kernels penalties,gumbel_sample   # 子集
```

进程内执行（快）；首次运行生成并持久化 `inputs/`（所有输入在 CPU 上用带种子的
`torch.Generator` 生成，两侧 bit 级一致；绝不使用设备 RNG）。

### 同步

把**整个 `strict_ut_026/`** 同步到 NPU 机器（至少 `precision/` + `inputs/` + `results/gpu/`）。
每次改完用例务必确认两端文件一致——历史多个 UT 失败源于跑了旧文件。

### 阶段 2 — NPU 机器上捕获

```bash
python precision/run_capture.py --side npu                  # 每用例一个子进程
python precision/run_one_case.py --side npu --kernel penalties --case-id <cid>   # 单个调试
```

- 每个用例独立子进程：单个 kernel 崩溃或设备上下文污染不拖垮整轮。
- NPU 侧直接 `import runtime_npu`（本套件完整版 shim，含
  `insert_slice/extract_slice/get_element` 解析），不再使用任何精简替身。

### 阶段 3 — 同步回来后：先对账，再比对

```bash
python precision/shape_audit.py            # 两侧 case 覆盖 + shape/dtype 一致性
python precision/compare_results.py        # 数值比对 -> report/compare_<时间戳>.md
python precision/shape_audit.py --list     # 查看注册表与 case_id
```

退出码非 0 = 有问题（shape_audit：缺失/形状不一致；compare_results：FAIL/ERROR/MISSING）。

### 判定标准（precision/compare_metrics.py）

| 比对模式 | 适用 | 标准 |
|---|---|---|
| `int_exact` | 整数张量 | 逐位相等 |
| `float32` / `float16` / `bfloat16` | 浮点输出 | atol=rtol= 1e-5 / 1e-3 / 1e-2 |
| `skip` | 随机采样结果（如 sampled token id） | 不比对 |

分级：**PASS**（全在容差内）/ **WARN**（超差占比 ≤0.1% 且 max_err ≤10×atol，人眼复核）/
**FAIL**（超限、NaN 位置不一致、shape 不一致）/ **MISSING / ERROR**（单侧缺失、输入 digest 不一致）。

两侧 API 语义分叉时用 case 级 `normalize` 钩子在比对前统一基准，例如
`gumbel_cases.py`：GPU 缓存**除温前** logits、NPU 缓存**除温后** logits，
NPU 侧乘回 temperature 后两侧才可比。

## 4. 新增一个捕获内核（三步）

1. 在 `precision/kernel_cases/` 新建 `<kernel>_cases.py`：
   - `build_inputs(params, seed)`：仅 CPU + 种子生成器；
   - `run(side, tensors, params)`：按 side 懒加载 GPU `vllm...` / NPU `vllm_ascend...` API；
   - `CASES`：声明 `params`、`stochastic`、每个输出的 `output_modes`，需要时挂 `normalize`。
2. 在 `precision/kernel_cases/__init__.py` 的 `REGISTRY` 注册一行。
3. 阶段 1 → 同步 → 阶段 2 → 同步 → 对账 + 比对。

注意生产不变式（历史踩坑）：penalties 的 `expanded_local_pos` 必须满足
`token_idx - pos >= 0`，否则内核越界读污染计数。

## 5. 与 strict_ut 的关系

| | strict_ut | strict_ut_026 |
|---|---|---|
| 严格单测（gpu/ npu/ + 基建） | 原版（不再修改） | 完整快照，随 026 演进 |
| GPU-NPU 同输入精度对比 | 无 | `precision/` 三阶段工具 |
| 形状对账 | 手工（docs/ csv） | `shape_audit.py` + `npu_ut_shapes.md` |

## 6. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `inputs for ... not found` | NPU 侧没同步到 `inputs/`；先跑 GPU 侧生成，或任一侧首跑会自动 bootstrap |
| compare 报 `input digests differ` | 两侧输入不一致：改过 `build_inputs` 却只重跑一侧；删旧 `inputs/` 与两侧 results 全量重跑 |
| shape_audit 报 MISSING | 单侧没跑该 case；用 `--kernels` / `--case-id` 补跑 |
| NPU 单个 case 卡死/崩溃 | `precision/run_one_case.py` 单独复现；子进程隔离已保证不拖垮整轮 |
| vllm 导入的是 dist-packages 而非源码树 | `python -c "import vllm; print(vllm.__file__)"` 确认路径，必要时以源码路径启动 |
| 想跑单测而非精度对比 | 见第 2 节，入口与 strict_ut 完全相同 |
