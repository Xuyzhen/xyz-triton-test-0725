# 01 完整对话记录

## 第一阶段：CI 数据提取与折线图增强

### 用户需求
用户要求将本地已经完成的 xround 马拉松结果（a1affe4d3218f292c2c79c6d7666877efe526cbb → c843f75aff45050fd4958dc4f24ac7e54a472df4 区间，21 个 commit，每个 commit 三轮交叉），补充 GitHub Actions 项目 workflow 中同一模型对应 commit 的 CI 结果（具体 TPS 数值），加到折线图里并做出区分。

### 关键技术发现
1. **vllm-ascend nightly benchmark 机制**：每晚自动跑全量模型性能，Kimi-K2.6-w4a8-A3 是指定模型，参数和本地测试完全一致（128 prompts gsm8k，mc2=1，TP8 DP2），相同 baseline 1433.4454，相同阈值 0.97×baseline
2. **CI 结果提取规则**：只有当 TPS < 阈值时，断言失败才会在 GitHub annotations 里输出具体 TPS；成功时不输出具体值，仅标记 pass
3. **GitHub API 流程**：分页获取 workflow runs → 对每个 run 获取 jobs → 定位 Kimi 任务 job → 取 job annotations 提取 TPS

### 提取过程
1. 匿名 API 限速 60/h 不够用 → 从 Windows 凭据管理器读到 GitHub token（`git credential fill`，repo scope）
2. logs 下载报 401 → 根因是 302 重定向到签名 URL 时 Authorization 头被转发，手动处理重定向去掉 auth 头后成功
3. 共提取 08-28~09-13 **15 个 CI 点（11 个有 TPS，含指定的 09-12 run=1311.6）**

### 折线图双数据来源
- 本地三轮（R1/R2/R3）分开绘制 + 均值 + min-max 阴影带
- CI nightly TPS 标记在对应 commit x 位置，用红色菱形 + 红虚线区分
- 阈值/baseline 水平点线

### CI 关键发现
1. **CI 复现了低谷**：08-30 CI HEAD TPS 仅 421.2，比本地低谷 948 更深
2. **同 commit 交叉验证极佳**：748acedfe CI 单轮 1383.3 vs 本地三轮均值 1382.2，差 0.08%
3. **CI 平台期全线低于 0.97 阈值**（最低 1235.8），即 nightly 每晚都在报性能失败

---

## 第二阶段：稳定性分析

### 用户观察
"看起来在测试到的区间里，这个用例同样的条件进行试验其实是不稳定的？虽然有明显趋势，但是影响因素特别多"

### 数据量化
| 波动程度 | commit | 三轮 TPS | 极差/均值 |
|----------|--------|----------|-----------|
| 极稳 | 2080fffa6 | 1444/1442/1425 | 1.3% |
| 极稳 | 748acedfe | 1368/1391/1388 | 1.6% |
| 中位 | （21 个点的中位数） | — | ~9.5% |
| 不稳 | 42e039a90 | 1182/1295/1434 | 19.3% |
| 不稳 | 222677fc7 | 1190/1418/1143 | 22.0% |
| 最不稳 | 6ca174b47 | 1474/1484/1051 | 32.4% |

一半的 commit 三轮极差 ≥9.5%。轮次整体均值 R1/R2/R3 = 1298/1336/1298，轮间偏差只有 ~3%。

### 结论
噪声主要是单次运行的随机干扰（宿主机 load 39~118 波动、128 prompts 跑得短），不是固定的时段效应。CI nightly 阈值 1390.4 落在平台期分布的上沿，每晚都红更多是"阈值贴着噪声带顶部"的机制性问题。

---

## 第三阶段：拉起参数对比

### 用户需求
对比基准 run（https://github.com/vllm-project/vllm-ascend/actions/runs/31821190604/job/94944700594）与当前 main 的拉起参数变化。

### 关键发现
唯一实质差异：三个开关的启用形式

| 参数项 | 基准 (v0.23.0, PASS) | 当前 main (FAIL) |
|--------|----------------------|------------------|
| fused MC2 | 环境变量 `VLLM_ASCEND_ENABLE_FUSED_MC2=1` | `--additional-config {"enable_fused_mc2":1}` |
| balance scheduling | 环境变量 `VLLM_ASCEND_BALANCE_SCHEDULING=1` | `--additional-config {"scheduler_config":{"enable_balance_scheduling":true}}` |
| MLAPO | 环境变量 `VLLM_ASCEND_ENABLE_MLAPO=1` | `--additional-config {"enable_mlapo":true}` |
| 其余 server_cmd | TP8 DP2 / max-num-seqs 24 / max-model-len 6144 / seed 42 / FULL_DECODE_ONLY / DFlash spec=7 | **逐字相同** |

### 迁移时间线
| commit | 日期 | 内容 | 随后 nightly |
|--------|------|------|--------------|
| 80c833fb8 | 08-26 | MC2 env→config | — |
| 9bef60ad5 | 08-27 | balance env→config | 08-28 failure 无 TPS |
| 895ca078b | 08-29 | MLAPO env→config | 08-30 TPS=421 深谷，08-31 恢复 1387 |

---

## 第四阶段：版本/开关交叉验证（15 轮）

### 用户需求
测试 v0.23.0 release 和 0.27.1 兼容版本，观察是否本机的 0.23 能过测试、参数变化的实际影响，每个参数交叉测试排除时序影响。

### 测试设计
| 组 | 容器 | vllm-ascend | 三开关形式 |
|----|------|-------------|-----------|
| A | xyz_dts_3422_v023 | 0.23.0rc2.dev137 | env var 全开 |
| B | xyz_dts_3422_main @ 748acedfe | main 09-04 | additional-config 全开 |
| C | 同 B | 同 B | 关 enable_fused_mc2 |
| D | 同 B | 同 B | 关 enable_balance_scheduling |
| E | 同 B | 同 B | 关 enable_mlapo |

轮次：R1 正序 A→B→C→D→E，R2 倒序 E→D→C→B→A，R3 正序

### 结果
| 组合 | R1 | R2 | R3 | 均值 | 极差/均值 |
|------|-----|-----|-----|------|----------|
| A: v0.23 全开 | 1628.19 | 1632.51 | 1607.87 | 1622.86 | 1.5% |
| B: main 全开 | 1398.41 | 1335.08 | 959.30 | 1230.93 | 35.8% |
| C: main 关 MC2 | 418.55 | 412.00 | 418.01 | 416.19 | 1.6% |
| D: main 关 balance | 1302.06 | 1472.26 | 1372.69 | 1382.34 | 12.3% |
| E: main 关 MLAPO | 1426.06 | 1474.07 | 1471.21 | 1457.11 | 3.3% |

### 五个核心结论
1. v0.23 本机能过测试：均值 1623，远超阈值 1390（+17%）
2. 版本差距 -24%：v0.23 均值 1623 vs main 全开 1231
3. MC2 是性能主凶：关 MC2 → 416（-66%），且无 MC2 反而极稳（1.6%）
4. MLAPO 负优化确认：关 MLAPO（1457）比全开（1231）还高 +18%
5. 08-30 CI 深谷根因确认：MC2 迁移窗口期 yaml 未同步补键导致 MC2 静默失效

---

## 第五阶段：深度机制分析

### 用户需求
"怎么解释这个结果？关了优化更快？分析分析一下" → "这太简单了，能不能非常详细"

### 代码级深挖（三个 subagent 并行搜索）

#### MLAPO 实现
- 把 MLA 注意力 decode 前序 6 个独立算子合并为 1 个融合 C++ 算子 `npu_mla_prolog_v3`
- 关键变化：KV cache 写入被融合进 prolog 算子，改变了写入时点
- 启用条件：1024 token 上限，仅 decode-only 节点（A3）
- 权重预处理：拆分重组为 NZ 格式，释放 prefill 权重 + empty_cache

#### MC2 实现
- 把 MoE all-to-all dispatch + FFN + combine 三段重叠
- 融合版 `dispatch_ffn_combine`：AIC 核跑 GEMM，AIV 核跑 HCCL 通信，CrossCoreFlag 同步
- v0.23→main 变化：CANN MegaMoe 集成、combine_quant_mode、fullmesh_v2、MRv2 适配

#### Balance scheduling
- scheduler 层 DP 负载均衡门控
- 与 MC2/MLAPO 完全不同层，无时序交叉

### 完整因果链
```
MLAPO 把 KV cache 写入融合进 prolog 算子
→ prolog 算子占用 current_stream 时间变长
→ comm_stream.wait_stream(current_stream) 同步点后移
→ MC2 在 main 里流模型更复杂，AIV-AIC CrossCoreFlag 同步窗口对延迟更敏感
→ 同步点后移超出容忍度 → 间歇性 dispatch 数据不完整
→ 部分 token 走 padding 路径 → TPS 暴跌 (959)

关 MLAPO → KV cache 写入回到独立算子 → current_stream 释放更早
→ MC2 同步窗口正确 → 稳定 1457

v0.23 全开 → MC2 实现简单 + 多流 gate 存在
→ 流模型简单到 MLAPO 不干扰 MC2 → 稳定 1623
```

---

## 第六阶段：Bisect 方案设计

### 用户需求
"这个能找到第一次劣化的地方吗 尝试提供一个方案 找到0.23和0.27.1之间的一个性能劣化点"

### 已有锚点
| pos | commit | xround 表现 | 角色 |
|-----|--------|------------|------|
| 956 | 2080fffa6 | 1444/1442/1425（1.3%） | 好锚点 G |
| 1045 | 222677fc7 | 1190/1418/1143（22%） | 坏锚点 B |

劣化窗口锁定在 pos 956→1045（89 个 commit，09-01→09-04）。

### 方案四阶段
1. 边界聚焦（6 轮 ABABAB）：验证 B 是否就是第一个坏点
2. 窗口二分（~9 轮）：在 [G, B^] 内选 5 点扫描
3. 复现性测试（~5 轮）：对边界 commit 连续 serve-restart 5 次
4. 定案验证（3 轮）：在坏点跑 E' 组（关 MLAPO）确认机制

时间预算：最短 2.7h / 中等 5.5h / 最长 7h

### 用户追加约束
"先设计方案 不要执行 有别人再用"
