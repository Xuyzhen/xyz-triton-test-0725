# 08 Bisect 方案定稿

## 目标
在 main 容器内（vllm 0.27.1 固定），用 B 组配置（三开关全开），找到引入"均值下降 + 波动增大"的第一个 vllm-ascend commit。

## 已有锚点

| pos | commit | 日期 | xround 表现 | 角色 |
|-----|--------|------|------------|------|
| 920 | a1affe4d3 | 08-29 | 905/951/989 (8.8%) | 低谷（已定案） |
| 956 | 2080fffa6 | 09-01 | 1444/1442/1425 (1.3%) | **好锚点 G** |
| ~1000± | 42e039a90 | 09-03 | 1182/1295/1434 (19.3%) | 波动点 |
| 1045 | 222677fc7 | 09-04 | 1190/1418/1143 (22.0%) | **坏锚点 B** |
| 1070 | 748acedfe | 09-04 | xround 稳 / ver B 35.8% | 间歇性 |

劣化窗口: pos 956→1045（89 个 commit，09-01 16:11 → 09-04 00:06）

## 嫌疑 commit 排序

| pos | commit | PR | 内容 | 嫌疑度 |
|-----|--------|----|------|--------|
| 1045 | 222677fc7 | #15416 | [Performance][KDA] Compose and overlap gate projections | **最高** |
| 1049 | d1ce8a1c4 | #15305 | [Performance] Overlap DSA C4/C128 compressor tail with q proj | 高 |
| 992 | a563601e0 | #15014 | mega_moe_max_tokens 对称缓冲分配 | 高 |
| 975 | 30f54b5c3 | #14439 | Fix MegaMoe prefill buffer sizing | 中 |
| 1016 | 83f9ef38f | #15478 | Migrate MoE compilation to hw profiles | 中 |
| 1036 | 0a97c475a | #15609 | Move MLAPO_MAX_SUPPORTED_TOKENS | 低 |

## 四阶段方案

### 第 0 步: 离线分析（不占机器）
- xround 21 点按 pos 排序 + CI 点映射
- 确定最紧边界 [G, B]
- 检查窗口内每个 commit 是否改了 csrc/

### 第 1 步: 边界聚焦（6 轮 ABABAB, ~1.8h）
```
r1: B(222677fc7)  r2: B^（父commit）
r3: B             r4: B^
r5: B             r6: B^
```

| 结果 | 判定 | 下一步 |
|------|------|--------|
| B^ 稳 + B 波动 | B = 第一个劣化点 | 跳第 4 步 |
| B^ 也波动 | 坏点更早 | 第 2 步 |
| 两者都稳 | 间歇触发 | 第 2 步 + 加大轮次 |

### 第 2 步: 窗口二分（~9 轮, ~2.7h）
在 [G, B^] 内选 5 点: 975 / 992 / 1016 / 1036 / 中点
```
R1 正序: G → 975 → 992 → 1016 → 1036
R2 倒序: 1036 → 1016 → 992 → 975 → G
```
G 锚点穿插: 若 G < 1350 或极差 > 10% → 标记 ENV_NOISE 降权

### 第 3 步: 复现性测试（~5 轮, ~1.5h）
对边界 commit 同一代码连续 serve-restart 5 次单轮，区分:
- 代码触发但概率性 → 统计触发率
- 纯会话随机 → 无法定位到单 commit

### 第 4 步: 定案验证（3 轮, ~0.9h）
在坏点 X 上跑 E' 组（关 MLAPO）× 3 轮:
```
r1: X 三开  r2: X 关MLAPO  r3: X 三开
```
关 MLAPO 后波动消失 → 确认劣化机制 = MLAPO×MC2 交互

## 判据
| 判定 | 3 轮表现 |
|------|---------|
| 好 | 极差 ≤ 8% 且 均值 ≥ 1350 |
| 坏 | 极差 ≥ 15% 或 均值 ≤ 1250 |
| 模糊 | 两者之间 → 加测 2 轮 |

## 时间预算
| 路径 | 轮数 | 时间 |
|------|------|------|
| 最短 | 9 轮 | ~2.7h |
| 中等 | ~18 轮 | ~5.5h |
| 最长 | ~23 轮 | ~7h |

每轮成本: serve 启动 ~13min + bench ~3.5min + 清理 ~2min ≈ 18min

## 执行架构
```
宿主机 runner (bisect2_runner.sh)
  ├─ docker exec xyz_dts_3422_main: git checkout → serve (B组)
  ├─ 轮询 :8000/health 就绪（超时 20min）
  ├─ docker exec xyz_aisbench_dts3422: ais_bench 128 prompts
  ├─ 收 TPS/TPOT/loadavg → results.txt（断点续跑）
  └─ pkill 三连 + NPU 空闲检查 → 下一点
```

## 风险对策
| 风险 | 对策 |
|------|------|
| 间歇性触发漏检 | ABAB 交叉 + G 锚点穿插 + 第 3 步重启复现 |
| csrc 改动需重编 | 执行前 `git show --stat` 检查，需要时 `pip install -e .` |
| 宿主机负载噪声 | 极差为主判据 + loadavg 记录 + 锚点降权 |
| vllm 0.27.1 兼容性 | 窗口 956-1045 全 < 兼容边界 fd815467c (pos=1095) |
| git 状态污染 | checkout --force + submodule update + 结束恢复 748acedfe |

## 用户约束
- 机器: 仅 80.5.9.136，不允许其他机子
- 他人未知 vllm 进程可杀（已协调）
- 不允许改密码
- aisbench 在 xyz_aisbench_dts3422 容器内跑
- 有别人再用时先设计方案不执行
