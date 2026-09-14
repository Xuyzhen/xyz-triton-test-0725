# 03 核心结论与机制分析

## 一、五条核心结论

### 结论 1: v0.23 本机能过测试
v0.23 三轮均值 1622.86, 远超 CI 阈值 1390.44 (+16.7%), 三轮极差仅 1.5%。本机环境不是性能瓶颈来源。

### 结论 2: 版本差距 = -392 TPS (-24%)
vllm 0.23 → 0.27.1 的版本升级（含 vllm-ascend 0.23 → main）造成了 24% 的性能下降。

### 结论 3: enable_fused_mc2 是性能差距主凶
关掉 MC2 → TPS 从 1231 暴跌到 416 (-66%), TPOT 从 19ms 飙到 75ms。MC2 在 0.27.1 环境下贡献了至少 800+ TPS。

值得注意的是: 无 MC2 的 C 组三轮极差仅 1.6% (412~419), 而 MC2 全开的 B 组极差高达 35.8% (959~1398) — **0.27.1 的 MC2 实现既拉低均值又引入巨大波动**。

### 结论 4: MLAPO 在 main 分支是负优化
关掉 MLAPO 后 TPS 反而比全开高 226 (+18%), 且极差仅 3.3%。MLAPO 在此 workload 下是负优化, 关掉它既提升性能又减少波动。

### 结论 5: 08-30 CI 深谷根因确认
MC2 迁移窗口期（08-26~29）yaml 未同步补键导致 MC2 静默失效，421 ≈ 本次 C 组 416。

---

## 二、完整因果链

### 2.1 三个开关的本质

**MC2 (fused MoE communication-computation)**:
- 把 MoE 的 all-to-all dispatch + FFN + combine 三段重叠执行
- 融合版 `dispatch_ffn_combine`: AIC 核跑 GEMM, AIV 核跑 HCCL 通信, CrossCoreFlag 同步
- 对 MoE 模型是刚需（关掉 → 416 TPS）

**MLAPO (MLA Prolog Optimization)**:
- 把 MLA 注意力 decode 前序 6 个独立算子合并为 1 个融合算子 `npu_mla_prolog_v3`
- 关键变化: KV cache 写入被融合进 prolog 算子，改变了写入时点
- 有 1024 token 上限，仅 decode-only 节点启用

**Balance scheduling**:
- scheduler 层 DP 负载均衡门控
- 与 MC2/MLAPO 完全不同层，无时序交叉

### 2.2 Transformer Block 内执行时序

```
┌─── Attention 子层 ──────────────────────────┐
│  MLAPO 路径:                                 │
│    npu_mla_prolog_v3(...)  ← 一个算子完成:   │
│      q_a_layernorm + qkv_a_proj + RoPE       │
│      + KV cache 写入 ← 关键变化               │
│    attention(q, k, v)                        │
│  ──── hidden_states 输出 ────                │
│    comm_stream.wait_stream(current_stream)   │ ← 流同步点
└──────────┬──────────────────────────────────┘
           ▼
┌─── MoE/FFN 子层 ────────────────────────────┐
│  MC2 路径:                                   │
│    dispatch_ffn_combine(hidden_states, ...)   │
│      AIV: DispatchAndCombine (HCCL 通信)     │
│      AIC: GMM1 → GMM2 (MoE 计算)             │
│      CrossCoreFlag 同步                       │
└─────────────────────────────────────────────┘
```

### 2.3 MLAPO 干扰 MC2 的机制

```
v0.23 时代 (稳定 1623):
  MLAPO ──┐
           ├── 互不干扰 → 稳定 1.5%
  MC2   ──┘
  原因: v0.23 有 "shared-expert multistream overlap gate" (#12245)
        MC2 实现简单（无 MegaMoe/fullmesh_v2/combine_quant）
        MLAPO 权重处理路径简单
        → 流模型简单到不会冲突

0.27.1 时代 (不稳定 1231):
  MLAPO ──┬── 冲突! → 间歇性失效 35.8%
  MC2   ──┘
  
  机制:
  1. MLAPO 把 KV cache 写入融合进 prolog 算子
  2. prolog 算子占用 current_stream 时间变长
  3. comm_stream.wait_stream(current_stream) 同步点后移
  4. MC2 在 main 里流模型更复杂 (MegaMoe/combine_quant/fullmesh_v2)
  5. AIV-AIC CrossCoreFlag 同步窗口对延迟更敏感
  6. 同步点后移超出容忍度 → dispatch 数据不完整
  7. mc2_mask 标记部分真实 token 为 padding
  8. 部分 token 不参与 MoE 计算 → TPS 暴跌 (959)

  次要因素:
  • MLAPO empty_cache() 可能干扰 MC2 workspace 分配
  • 两者权重 NZ 格式处理独立执行，可能内存碎片化

关掉 MLAPO (稳定 1457):
  → KV cache 写入回到独立算子
  → current_stream 释放更早
  → MC2 同步窗口正确
  → 稳定 3.3%

关掉 MC2 (稳定 416):
  → 通信串行，无同步问题
  → 极慢但极稳 1.6%
```

### 2.4 版本间关键代码变更

**MLAPO 变化 (v0.23 → main)**:
| 变化 | 影响 |
|------|------|
| 多流重叠 gate 移除 (#11953) | 改变 MLAPO/MC2 流交互模型 |
| UNRESTRICTED_MLAPO | prefill 节点也走 MLAPO |
| MLAPO_NATIVE_WEIGHTS | 权重处理路径更复杂 |
| mlapo_keep_prefill_weights | 新增显存-稳定性 trade-off |
| 权重释放条件加严 | empty_cache 影响范围 |

**MC2 变化 (v0.23 → main)**:
| 变化 | 影响 |
|------|------|
| CANN MegaMoe 集成 | 新 mega_moe 算子路径 |
| combine_quant_mode | 强制 combine 通信量化 |
| mc2_comm_alg fullmesh_v2 | 新通信算法 |
| MegaMoe 对称缓冲 | 预分配通信缓冲 |
| shared-expert 多流重叠与 MC2 互斥 | 强制关闭重叠 |

---

## 三、CI nightly 每晚红的机制

main 全开 (B 组) 三轮 = 1398 / 1335 / 959。CI nightly 单轮测试:
- 落在 1398（R1）→ 过线（≥1390）
- 落在 1335（R2）→ 不过线
- 落在 959（R3）→ 远低于线

**单轮测试落在低谷的概率 ≈ 1/3**，这解释了为什么 CI nightly 时过时不过——不是代码在变，而是 MLAPO-MC2 冲突导致的间歇性失效。

---

## 四、工程建议

1. **短期止血**: 在 main 分支的 Kimi-K2.6-w4a8-A3 nightly yaml 中关闭 `enable_mlapo`，可立即恢复 1457 TPS 的稳定性能（过 CI 阈值 1390），同时消除 35.8% 的波动
2. **中期修复**: 排查 MLAPO prolog 算子占用 current_stream 的时长 vs MC2 AIV-AIC CrossCoreFlag 同步窗口的容忍度
3. **长期方案**: 在 MLAPO 和 MC2 之间建立显式的流依赖管理
4. **CI 阈值校准**: baseline 1433.4454 是 v0.23 时代校准的，0.27.1 的合理 baseline 应在 1450 左右
5. **CI 健壮性**: nightly 单轮测试对间歇性失效无判别力，建议改为 N 晚中位数
