# 04 可复用 Skill

## Skill 1: GitHub Actions CI TPS 提取流程

### 适用场景
从 vllm-ascend nightly benchmark workflow 提取指定模型的 CI TPS 数据，与本地测试结果交叉验证。

### 前置条件
- GitHub token（repo scope），可从 `git credential fill` 读取
- Python 3 + requests 库

### 流程

```python
# 1. 读取 GitHub token (Windows 凭据管理器)
# 执行: echo "url=protocol=https\nhost=github.com\n" | git credential fill
# 输出包含 password=<token>

# 2. 分页获取 nightly runs
# GET /repos/vllm-project/vllm-ascend/actions/workflows/nightly-a3.yml/runs
#   ?per_page=30&page=N&created=>=2026-08-28&created=<=2026-09-13
# 注意: 每页 30 个 run，需要分页拉取

# 3. 对每个 run 获取 jobs
# GET /repos/vllm-project/vllm-ascend/actions/runs/{run_id}/jobs
# 定位 name 包含 "Kimi-K2.6-w4a8" 的 job

# 4. 获取 job annotations（TPS 仅在断言失败时输出）
# GET /repos/vllm-project/vllm-ascend/check-runs/{job_id}/annotations
# 解析 message 字段，正则提取 "current Output Token Throughput is (\d+\.\d+)"

# 5. 注意事项
# - 302 重定向到签名 URL 时，必须去掉 Authorization 头
# - nightly 每天 15:45 UTC 跑一次，每次 main HEAD
# - 我们选的 commit 中只有 daily HEAD 位置才有 CI 数据
```

### 关键正则
```python
# TPS 提取
tps_match = re.search(r'current Output Token Throughput is (\d+\.\d+) token/s', annotation_message)

# run 列表解析 (应对 JSON 截断)
# 用 4-space { + 6-space "id" 切块，而非 json.loads
chunks = response_text.split('    {')
for ch in chunks[1:]:
    if needle not in ch:  # needle = "Kimi-K2.6"
        continue
    m_id = re.search(r'"id": (\d+)', ch)
    m_sha = re.search(r'"head_sha": "([^"]+)"', ch)
    m_con = re.search(r'"conclusion": (null|"[^"]*")', ch)
```

---

## Skill 2: 性能 Bisect 框架

### 适用场景
在固定 vllm 版本的容器内，checkout 不同 vllm-ascend commit，定位性能劣化引入点。

### 框架设计

```bash
#!/bin/bash
# bisect2_runner.sh — 断点续跑的 bisect 框架

# 核心设计原则:
# 1. 锚点穿插: 每轮穿插已知好锚点 G，检测环境噪声
# 2. ABAB 交叉: 正序+倒序交替，排除时序漂移
# 3. 断点续跑: 结果逐行追加到文件，中断后自动跳过已完成项
# 4. NPU 空闲检查: 每次切换前确认 NPU 不忙
# 5. csrc 检查: checkout 后检查是否改了 C++ 代码

# 架构:
# 宿主机 runner
#   ├─ docker exec main: git checkout → serve (B组配置)
#   ├─ 轮询 :8000/health 就绪
#   ├─ docker exec aisbench: ais_bench 128 prompts
#   ├─ 收 TPS/TPOT/loadavg → results.txt
#   └─ pkill 三连 + NPU 空闲 → 下一点

# 判据:
# 好: 极差 ≤ 8% 且 均值 ≥ 1350
# 坏: 极差 ≥ 15% 或 均值 ≤ 1250
# 模糊: 两者之间 → 加测 2 轮
```

### 关键组件

**1. serve 脚本模板（B 组配置）**:
```bash
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export DYNAMIC_EPLB=true
vllm serve /mnt/weight/Kimi-K2.6-w4a8 \
  --quantization ascend --port 8000 \
  --tensor-parallel-size 8 --data-parallel-size 2 \
  --max-num-seqs 24 --max-model-len 6144 \
  --seed 42 --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_fused_mc2":1,"scheduler_config":{"enable_balance_scheduling":true},"enable_mlapo":true}' \
  --speculative-config '{"method":"dflash","model":"/mnt/weight/Kimi-K2.5-DFlash","num_speculative_tokens":7}'
```

**2. 交叉轮次设计**:
```bash
# R1 正序: G → C1 → C2 → C3 → C4 → C5
# R2 倒序: C5 → C4 → C3 → C2 → C1 → G
# 每轮穿插 G 锚点: 若 G < 1350 或极差 > 10% → 标记 ENV_NOISE
```

**3. 断点续跑**:
```bash
# 结果格式: "rN COMMIT | TPS=xxx | TPOT=xxx | load=xxx"
if grep -q "^$RN $COMMIT | TPS=" "$RESULTS" 2>/dev/null; then
    log "SKIP (already done)"
    continue
fi
```

**4. NPU 清理**:
```bash
kill_all() {
  for round in 1 2 3; do
    docker exec $CTN pkill -9 -f spawn_main 2>/dev/null
    docker exec $CTN pkill -9 -f resource_tracker 2>/dev/null
    docker exec $CTN pkill -9 -f WorkerProc 2>/dev/null
    docker exec $CTN pkill -9 -i -f vllm 2>/dev/null
    sleep 10
  done
  # NPU 空闲检查
  for i in $(seq 1 30); do
    busy=$(npu-smi info | grep -oE '[0-9]+ */ *65536' | awk -F'/' '{u=$1+0; if(u>5000) c++} END{print c+0}')
    [ "$busy" -eq 0 ] && return 0
    sleep 15
  done
}
```

---

## Skill 3: 稳定性判据体系

### 适用场景
判断性能测试结果是否可信，区分代码劣化与环境噪声。

### 判据矩阵

| 极差/均值 | 均值 ≥ 1350 | 均值 < 1350 | 判定 |
|-----------|------------|------------|------|
| ≤ 8% | ✅ 好 | ⚠️ 慢但稳 | 可信 |
| 8-15% | ⚠️ 轻微噪声 | ⚠️ 需加测 | 模糊 |
| ≥ 15% | ❌ 间歇性失效 | ❌ 坏 | 不可信单轮 |

### 关键经验值
- 环境噪声带: ±5~27%（共享环境宿主机 load 39~118）
- 不可下结论阈值: 差异 < 15%
- 可信差异阈值: 差异 ≥ 15% **且** 多轮一致
- CI 单轮落在低谷概率: ~1/3（间歇性失效）

### 多轮策略
- 最少 3 轮交叉（正序 + 倒序 + 正序）
- 锚点穿插检测环境噪声
- 差异 < 15% → 需 5+ 轮才能判别
- 间歇性触发 → 同代码连续 serve-restart 5 次统计触发率

---

## Skill 4: AISBench 测试配置

### 容器配置
```bash
# vllm serve 容器 (xyz_dts_3422_main 或 xyz_dts_3422_v023)
# - 共享 NPU /dev/davinci*
# - 挂载 /mnt/weight (模型权重)
# - 挂载 /mnt/share/x30084275/dts_3422 (代码)
# - 端口 8000

# aisbench 容器 (xyz_aisbench_dts3422)
# - 挂载 /mnt/share/x30084275/dts_3422/benchmark (ais_bench 代码)
# - 通过 localhost:8000 访问 vllm serve

# 环境变量
export no_proxy='127.0.0.1,0.0.0.0,localhost,local,.local,*.huawei.com'
export NO_PROXY=$no_proxy
```

### ais_bench 调用
```bash
cd /mnt/share/x30084275/dts_3422/benchmark
ais_bench \
  --models vllm_api_stream_chat \
  --datasets gsm8k_gen_0_shot_cot_str_perf \
  --mode perf \
  --num-prompts 128
```

### 结果提取
```bash
# TPS
grep -E 'Output Token Throughput' bench.log | grep -oE '[0-9]+\.[0-9]+' | head -1
# TPOT
grep -E 'TPOT ' bench.log | grep -oE '[0-9.]+ ms' | head -1
# 成功请求数
grep -E 'Success Requests' bench.log | grep -oE '[0-9]+' | tail -1
```
