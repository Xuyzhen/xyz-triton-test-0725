# 07 代码级分析: MLAPO / MC2 / Balance Scheduling

## 一、MLAPO (MLA Prolog Optimization)

### 1.1 本质
把 MLA 注意力 decode 前序 6 个独立算子合并为 1 个融合 C++ 算子 `npu_mla_prolog_v3`。

### 1.2 标准路径 (MLAPO 关闭)
文件: `vllm_ascend/attention/mla_v1.py:1968-2006`
```
1. fused_qkv_a_proj(hidden_states) → 拆出 q_c 和 kv_no_split
2. q_a_layernorm(q_c) → RMSNorm
3. mla_preprocess_decode(...) → 单独写 KV cache
4. apply_rope(q_pe, k_pe) → RoPE 旋转
5. kv_cache_update(...) → KV cache 写入
6. attention(q, k, v) → 标准 attention
```

### 1.3 MLAPO 融合路径 (MLAPO 开启)
文件: `vllm_ascend/attention/mla_v1.py:1808-1893`
```python
decode_q_nope, decode_q_pe, ... = prolog_op(  # npu_mla_prolog_v3
    kv_cache=decode_k_nope, kr_cache=decode_k_pe,
    token_x=quantized_x,
    weight_dq=self.weight_dq, weight_uq_qr=self.weight_uq_qr,
    weight_uk=self.mlapo_W_UK_T, weight_dkv_kr=self.weight_dkv_kr,
    rmsnorm_gamma_cq=..., rmsnorm_gamma_ckv=...,
    rope_sin=sin, rope_cos=cos,
    cache_index=cache_index,  # KV cache 写入融合进来
    ...
)
```
关键区别: KV cache 写入被合并进 prolog 算子。

### 1.4 启用条件
文件: `vllm_ascend/attention/utils.py:555-565`
```python
def enabling_mlapo(vllm_config) -> bool:
    config_val = get_ascend_config().enable_mlapo
    if get_current_hardware_profile().supports(UNRESTRICTED_MLAPO):
        return bool(config_val)          # A5/950: 所有节点
    is_decode_instance = (kv_consumer 且非 kv_producer)
    return bool(config_val and is_decode_instance)  # A2/A3: 仅 decode-only
```
- 1024 token 上限 (`MLAPO_MAX_SUPPORTED_TOKENS = 1024`)
- draft 模型一律禁用

### 1.5 权重预处理
文件: `vllm_ascend/attention/mla_v1.py:1049-1158`
- 拆分重组为 `weight_dq`, `weight_dkv_kr`, `weight_uq_qr`
- 转 `ACL_FORMAT_FRACTAL_NZ`
- decode 节点释放 prefill 权重 + `torch.npu.empty_cache()`

### 1.6 v0.23 → main 变化
| 变化 | 代码位置 |
|------|----------|
| 多流重叠 gate 移除 (#11953) | release_notes.md |
| UNRESTRICTED_MLAPO | hardware_profile.py:58,246 |
| MLAPO_NATIVE_WEIGHTS | hardware_profile.py:42,237 |
| mlapo_keep_prefill_weights | ascend_config.py:425 |
| 权重释放条件加严 | mla_v1.py:1144-1158 |
| 1024 token 上限检查 | mla_v1.py:2068 |

---

## 二、MC2 (Fused MoE Communication-Computation)

### 2.1 本质
把 MoE 专家并行的 all-to-all dispatch + FFN(W1+SwiGLU+W2) + all-to-all combine 三段重叠执行。

### 2.2 非融合 MC2
```
1. npu_moe_distribute_dispatch_v2() → all-to-all 分发 (通信)
2. grouped_matmul(W1) → SwiGLU → grouped_matmul(W2) → FFN 计算
3. npu_moe_distribute_combine_v2() → all-to-all 归约 (通信)
```

### 2.3 融合 MC2 (enable_fused_mc2=1)
文件: `vllm_ascend/ops/fused_moe/moe_comm_method.py:457-508`
调用 `torch.ops._C_ascend.dispatch_ffn_combine(...)`

C++ 内核 (`csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp:216-231`):
```cpp
// AIC 核 (计算核): 跑 MoE 计算
void operator()<AIC>(params) {
    GMM1(params);                                 // 第一层 GEMM
    AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGV2C);  // 等 AIV 完成 dispatch
    GMM2(params);                                 // 第二层 GEMM
}
// AIV 核 (向量核): 跑 all-to-all 通信
void operator()<AIV>(params) {
    DispatchAndCombine(params);
}
```

### 2.4 权重要求
- 必须是 `ACL_FORMAT_FRACTAL_NZ` 格式
- 文件: `vllm_ascend/ops/fused_moe/routed_experts.py:114-119`

### 2.5 forward 中的设置
文件: `vllm_ascend/ascend_forward_context.py:147-152`
- 每个 forward 开始调用 `select_moe_comm_method()` 选择通信方式
- 填充 `mc2_mask`（标记真实 token vs padding）

### 2.6 v0.23 → main 变化
| 变化 | 代码位置 |
|------|----------|
| CANN MegaMoe 集成 | moe_comm_method.py:372-455 |
| combine_quant_mode | ascend_config.py:398-401 |
| mc2_comm_alg "fullmesh_v2" | ascend_config.py:415 |
| MRv2 适配 | ascend_forward_context.py:33,67-84 |
| MegaMoe 对称缓冲 | moe_comm_method.py:292-370 |
| shared-expert 多流重叠与 MC2 互斥 | ascend_config.py:625-630 |
| LoRA 与 fused MC2 互斥 | ascend_forward_context.py:358-363 |

---

## 三、Balance Scheduling

### 3.1 本质
scheduler 层 DP 负载均衡门控: "任一 rank running 数达上限则全局冻结新请求准入"。

### 3.2 机制
文件: `vllm_ascend/patch/platform/patch_balance_schedule.py:410-411`
```python
if max(t.item() for t in self.balance_queue) == self.max_num_running_reqs:
    break  # 全局冻结
```
- 只用标量 `all_gather` 交换 running 数，不涉及 token/权重通信
- 与 MC2/MLAPO 完全不同层，无时序交叉

### 3.3 v0.23 → main 变化
| 变化 | 说明 |
|------|------|
| env→additional_config 迁移 | 删除 env var 读取 |
| 删除 run_busy_loop/run_engine_core 复制 | 改用 _has_global_unfinished_reqs hook |
| schedule() 复制对齐 v0.24.0 tag | 加逐字漂移守护测试 |
| 修复 gather 挂载点死锁 | 从 schedule() 内 → _has_global_unfinished_reqs 之后 |
| 新增校验 | 仅 PD-mixed / 互斥 profiling_chunk / 互斥 dyntra_lb |

---

## 四、关键文件路径

### MLAPO
- `vllm_ascend/attention/mla_v1.py` — 主实现
- `vllm_ascend/attention/utils.py` — 启用判定
- `vllm_ascend/ascend_config.py` — 配置定义
- `vllm_ascend/device/hardware_profile.py` — 硬件能力

### MC2
- `vllm_ascend/ascend_forward_context.py` — 通信方式选择
- `vllm_ascend/ops/fused_moe/moe_comm_method.py` — MC2CommImpl/FusedMC2CommImpl
- `vllm_ascend/ops/fused_moe/token_dispatcher.py` — dispatch/combine 算子
- `vllm_ascend/ops/fused_moe/routed_experts.py` — 权重处理
- `csrc/mc2/dispatch_ffn_combine/` — C++ 融合算子

### Balance
- `vllm_ascend/patch/platform/patch_balance_schedule.py` — 主实现
- `vllm_ascend/ascend_config.py` — 配置定义
- `vllm_ascend/platform.py` — 运行时校验

### 断言失败输出 (CI TPS 来源)
- `vllm_ascend/.github/workflows/_e2e_nightly_single_node.yaml` 中的 Python 断言:
```python
output_throughput = self.result_json["Output Token Throughput"]["total"].replace("token/s", "")
assert float(output_throughput) >= self.threshold * self.baseline, (
    "Performance verification failed. "
    f"The current Output Token Throughput is {output_throughput} token/s, "
    f"which is not greater than or equal to {self.threshold} * baseline {self.baseline}."
)
```
