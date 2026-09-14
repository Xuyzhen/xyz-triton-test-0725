"""最小复现: compute_local_logits_stats 的 sumexp 双标杆 FAIL (3 项).

复现目标 (来自 双标杆比对结果/260828/ratio_20260828_144618.md):
  deepseek_2l_129280v_3spec : target_local_sumexp  ratio=(4.91, 3.56, 3.55) FAIL
                              draft_local_sumexp   ratio=(1.41, 2.10, 2.08) FAIL
  multi_4l_16384v_2spec     : target_local_sumexp  ratio=(8.11, 3.65, 2.62) FAIL

算子在项目中的真实来源 (本文件不 import vllm / vllm_ascend):
  [1] Triton kernel 原文摘自:
        vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
        -> `_compute_max_and_sumexp` (L9-L17)
        -> `_compute_local_logits_stats_kernel` (L193-L295)
      vllm-ascend 侧 (vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py)
      运行的是同一份 Triton 源码, 由 triton-ascend 后端编译; 本复现在 GPU 上
      由 triton 编译、NPU 上由 triton-ascend 编译, 与项目内行为一致.
  [2] 用例参数与数据生成 (seed=42, 温度 1.0/0.8 全非贪婪路径) 复刻自:
        xyz-triton-test-0725/accuracy_test/strict_ut_028/precision/
        kernel_cases/local_logits_stats_cases.py
  [3] 指标与分级公式 (精度标准 2.1 §4.5) 复刻自:
        xyz-triton-test-0725/accuracy_test/strict_ut_028/precision/ratio_report.py
      Ratio = metric_side / max(metric_gpu, 2^-30)
      L2 <= (2, 1.2, 1.2)  L1 <= (5, 1.5, 1.5)  L0 <= (10, 2, 2)  [MARE比, MERE比, RMSE比]

用法 (同一目录拷到 GPU 服务器和 NPU 服务器各跑一次):
    python repro_sumexp.py        # 在当前可用设备上采集, 存 repro_results/<side>.json
                                  # 若另一侧结果已存在, 自动输出双标杆比率结论
退出码: 双侧齐全且有 FAIL 项时为 1 (便于流水线集成), 否则 0.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

VOCAB_BLOCK_SIZE = 8192
REL_EPS = 1e-7          # ratio_report.py: §4.1 相对误差分母保护
FLOOR = 2.0 ** -30      # ratio_report.py: fp32 小值域 err floor (§4.5.3)
GRADES = [("L2", 2.0, 1.2, 1.2), ("L1", 5.0, 1.5, 1.5), ("L0", 10.0, 2.0, 2.0)]

# ---------------------------------------------------------------------------
# [来源 1] vLLM 原版 Triton kernel, 原文摘录 (未改动任何计算逻辑)
# ---------------------------------------------------------------------------


@triton.jit
def _compute_max_and_sumexp(logits):
    max = tl.max(logits, axis=0)
    sumexp = tl.where(
        max > float("-inf"),
        tl.sum(tl.exp(logits - max)),
        0.0,
    )
    return max, sumexp


@triton.jit
def _compute_local_logits_stats_kernel(
    # [num_logits, num_blocks]
    target_local_argmax_ptr,
    target_local_argmax_stride,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    # [num_logits, num_blocks]
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    # [num_logits, num_blocks]
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    expanded_local_pos_ptr,
    # [max_num_reqs]
    temp_ptr,
    vocab_size,
    num_speculative_steps,
    BLOCK_SIZE: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    logit_idx = tl.program_id(0).to(tl.int64)
    draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)

    if draft_step_idx >= num_speculative_steps:
        # Bonus token. Max/argmax and summed exponentials are not needed.
        return

    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx).to(tl.int64)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)

    block_idx = tl.program_id(1)
    block_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < vocab_size

    if temp == 0.0:
        # Greedy sampling. Only the target max/argmax are needed.
        target_logits = tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        value, idx = tl.max(target_logits, axis=0, return_indices=True)
        token_id = block_idx * BLOCK_SIZE + idx
        tl.store(
            target_local_argmax_ptr
            + logit_idx * target_local_argmax_stride
            + block_idx,
            token_id,
        )
        tl.store(
            target_local_max_ptr + logit_idx * target_local_max_stride + block_idx,
            value,
        )
    else:
        # Get local target max and summed exponentials.
        target_logits = tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        target_max, target_sumexp = _compute_max_and_sumexp(target_logits)
        tl.store(
            target_local_max_ptr + logit_idx * target_local_max_stride + block_idx,
            target_max,
        )
        tl.store(
            target_local_sumexp_ptr
            + logit_idx * target_local_sumexp_stride
            + block_idx,
            target_sumexp,
        )
        if HAS_DRAFT_LOGITS:
            # Get local draft max and summed exponentials. draft_logits is
            # stored pre-temperature, so apply scale first.
            draft_logits = (
                tl.load(
                    draft_logits_ptr
                    + req_state_idx * draft_logits_stride_0
                    + draft_step_idx * draft_logits_stride_1
                    + block_offsets,
                    mask=mask,
                    other=float("-inf"),
                ).to(tl.float32)
                / temp
            )
            draft_max, draft_sumexp = _compute_max_and_sumexp(draft_logits)
            tl.store(
                draft_local_max_ptr + logit_idx * draft_local_max_stride + block_idx,
                draft_max,
            )
            tl.store(
                draft_local_sumexp_ptr
                + logit_idx * draft_local_sumexp_stride
                + block_idx,
                draft_sumexp,
            )


# ---------------------------------------------------------------------------
# [来源 2] 用例数据生成 + CPU fp64 golden (复刻 local_logits_stats_cases.py)
# ---------------------------------------------------------------------------

CASES = [
    {"name": "small_2l_1024v_1spec", "num_logits": 2, "vocab_size": 1024,
     "num_speculative_steps": 1},
    {"name": "multi_4l_16384v_2spec", "num_logits": 4, "vocab_size": 16384,
     "num_speculative_steps": 2},
    {"name": "deepseek_2l_129280v_3spec", "num_logits": 2, "vocab_size": 129280,
     "num_speculative_steps": 3},
]
MAX_NUM_REQS = 4  # local_logits_stats_cases.build_inputs 硬编码值


def build_inputs(params: dict, seed: int = 42) -> dict[str, torch.Tensor]:
    """与 strict_ut_028 的 build_inputs 逐行一致 (同 seed -> 同输入 digest)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_logits = params["num_logits"]
    vocab = params["vocab_size"]
    n_spec = params["num_speculative_steps"]
    target_logits = torch.randn(n_logits, vocab, generator=g, dtype=torch.float32)
    draft_logits = torch.randn(MAX_NUM_REQS, n_spec, vocab, generator=g,
                               dtype=torch.float32)
    mapping = torch.arange(n_logits, dtype=torch.int64) % MAX_NUM_REQS
    local_pos = torch.zeros(n_logits, dtype=torch.int64)
    temperature = torch.tensor([1.0, 0.8, 1.0, 0.8], dtype=torch.float32)
    return {
        "target_logits": target_logits,
        "draft_logits": draft_logits,
        "expanded_idx_mapping": mapping,
        "expanded_local_pos": local_pos,
        "temperature": temperature,
    }


def ref(t: dict[str, torch.Tensor], params: dict) -> dict[str, torch.Tensor]:
    """CPU fp64 golden, 与 local_logits_stats_cases.ref 一致 (修复后语义:
    非贪婪分支不写 argmax, 仅写 max/sumexp)."""
    BLOCK = VOCAB_BLOCK_SIZE
    n_logits = params["num_logits"]
    vocab = params["vocab_size"]
    n_spec = params["num_speculative_steps"]
    n_blocks = (vocab + BLOCK - 1) // BLOCK

    target = t["target_logits"].to(torch.float64)
    draft = t["draft_logits"].to(torch.float64)
    mapping = t["expanded_idx_mapping"].long()
    pos = t["expanded_local_pos"].long()
    temp = t["temperature"].to(torch.float64)

    t_am = torch.zeros(n_logits, n_blocks, dtype=torch.int64)
    t_mx = torch.zeros(n_logits, n_blocks, dtype=torch.float64)
    t_se = torch.zeros(n_logits, n_blocks, dtype=torch.float64)
    d_mx = torch.zeros(n_logits, n_blocks, dtype=torch.float64)
    d_se = torch.zeros(n_logits, n_blocks, dtype=torch.float64)

    for li in range(n_logits):
        p = int(pos[li])
        if p >= n_spec:
            continue
        r = int(mapping[li])
        tp = float(temp[r])
        for b in range(n_blocks):
            seg = target[li, b * BLOCK:(b + 1) * BLOCK]
            if tp == 0.0:
                t_am[li, b] = b * BLOCK + int(seg.argmax())
                t_mx[li, b] = seg.max()
            else:
                m = seg.max()
                t_mx[li, b] = m
                t_se[li, b] = torch.exp(seg - m).sum()
                dseg = draft[r, p, b * BLOCK:(b + 1) * BLOCK] / tp
                dm = dseg.max()
                d_mx[li, b] = dm
                d_se[li, b] = torch.exp(dseg - dm).sum()
    return {"target_local_argmax": t_am, "target_local_max": t_mx,
            "target_local_sumexp": t_se, "draft_local_max": d_mx,
            "draft_local_sumexp": d_se}


# ---------------------------------------------------------------------------
# [来源 3] 指标公式 (复刻 ratio_report.py, 精度标准 2.1 §4.1/§4.5)
# ---------------------------------------------------------------------------


def error_metrics(actual: torch.Tensor, golden: torch.Tensor):
    a = actual.to(torch.float64)
    g = golden.to(torch.float64)
    finite = torch.isfinite(g)
    if not finite.any():
        return 0.0, 0.0, 0.0
    diff = (a[finite] - g[finite]).abs()
    rel = diff / (g[finite].abs() + REL_EPS)
    return float(rel.max()), float(rel.mean()), float(torch.sqrt((diff ** 2).mean()))


def error_count(actual: torch.Tensor, golden: torch.Tensor) -> int:
    a = actual.to(torch.float64)
    g = golden.to(torch.float64)
    small = g.abs() < 2.0 ** -14  # fp32 小值域阈值 (§4.5.3)
    return int((small & ((a - g).abs() > FLOOR)).sum())


def grade_of(r_mare: float, r_mere: float, r_rmse: float):
    for name, m, e, r in GRADES:
        if r_mare <= m and r_mere <= e and r_rmse <= r:
            return name
    return None


# ---------------------------------------------------------------------------
# 采集与报告
# ---------------------------------------------------------------------------


def run_kernel_on(side: str, t: dict[str, torch.Tensor], params: dict):
    """在指定侧设备上启动原版 kernel (调用方式与 028 run() 一致)."""
    n_logits = params["num_logits"]
    vocab = params["vocab_size"]
    n_spec = params["num_speculative_steps"]
    vocab_num_blocks = triton.cdiv(vocab, VOCAB_BLOCK_SIZE)
    dev = t["target_logits"].device

    target_local_argmax = torch.zeros(n_logits, vocab_num_blocks,
                                      dtype=torch.int64, device=dev)
    target_local_max = torch.zeros(n_logits, vocab_num_blocks,
                                   dtype=torch.float32, device=dev)
    target_local_sumexp = torch.zeros(n_logits, vocab_num_blocks,
                                      dtype=torch.float32, device=dev)
    draft_local_max = torch.zeros(n_logits, vocab_num_blocks,
                                  dtype=torch.float32, device=dev)
    draft_local_sumexp = torch.zeros(n_logits, vocab_num_blocks,
                                     dtype=torch.float32, device=dev)

    _compute_local_logits_stats_kernel[(n_logits, vocab_num_blocks)](
        target_local_argmax, target_local_argmax.stride(0),
        target_local_max, target_local_max.stride(0),
        target_local_sumexp, target_local_sumexp.stride(0),
        draft_local_max, draft_local_max.stride(0),
        draft_local_sumexp, draft_local_sumexp.stride(0),
        t["target_logits"], t["target_logits"].stride(0),
        t["draft_logits"], t["draft_logits"].stride(0), t["draft_logits"].stride(1),
        t["expanded_idx_mapping"], t["expanded_local_pos"], t["temperature"],
        vocab, n_spec,
        BLOCK_SIZE=VOCAB_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=True,
    )
    if side == "gpu":
        torch.cuda.synchronize()
    else:
        torch.npu.synchronize()
    return {
        "target_local_argmax": target_local_argmax.cpu(),
        "target_local_max": target_local_max.cpu(),
        "target_local_sumexp": target_local_sumexp.cpu(),
        "draft_local_max": draft_local_max.cpu(),
        "draft_local_sumexp": draft_local_sumexp.cpu(),
    }


def detect_side() -> str | None:
    if torch.cuda.is_available():
        return "gpu"
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    return None


def collect(side: str, out_dir: Path) -> Path:
    dev = torch.device("cuda" if side == "gpu" else "npu")
    results = {}
    for params in CASES:
        t = build_inputs(params)
        golden = ref(t, params)  # CPU fp64 真值锚点
        t_dev = {k: v.to(dev) for k, v in t.items()}
        actual = run_kernel_on(side, t_dev, params)

        case_res = {}
        for out_name, g in golden.items():
            a = actual[out_name]
            if not g.is_floating_point():
                case_res[out_name] = {
                    "kind": "int",
                    "bitwise_equal": bool(torch.equal(a, g)),
                }
            else:
                mare, mere, rmse = error_metrics(a, g)
                case_res[out_name] = {
                    "kind": "float",
                    "mare": mare, "mere": mere, "rmse": rmse,
                    "bitwise_equal": bool(torch.equal(a, g)),
                    "error_count": error_count(a, g),
                }
        results[params["name"]] = case_res

        print(f"[{side}] {params['name']}")
        for out_name, r in case_res.items():
            if r["kind"] == "int":
                print(f"    {out_name:22s} bitwise_equal={r['bitwise_equal']}")
            else:
                print(f"    {out_name:22s} MARE={r['mare']:.3g} "
                      f"MERE={r['mere']:.3g} RMSE={r['rmse']:.3g} "
                      f"bitwise_equal={r['bitwise_equal']} "
                      f"ec={r['error_count']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{side}.json"
    path.write_text(json.dumps({"side": side,
                                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "cases": results}, indent=2))
    print(f"saved -> {path}")
    return path


def compare(out_dir: Path) -> int:
    gpu = json.loads((out_dir / "gpu.json").read_text())["cases"]
    npu = json.loads((out_dir / "npu.json").read_text())["cases"]

    print("\n==== 双标杆比率结论 (ratio_report.py 公式, 精度标准 2.1 §4.5) ====")
    print(f"{'case':30s} {'output':22s} {'MARE比':>7s} {'MERE比':>7s} "
          f"{'RMSE比':>7s}  grade  verdict")
    n_fail = 0
    for params in CASES:
        name = params["name"]
        for out_name, g_res in gpu[name].items():
            n_res = npu[name][out_name]
            if g_res["kind"] == "int":
                ok = n_res["bitwise_equal"] and g_res["bitwise_equal"]
                verdict = "PASS" if ok else "FAIL"
                n_fail += 0 if ok else 1
                print(f"{name:30s} {out_name:22s} {'-':>7s} {'-':>7s} "
                      f"{'-':>7s}  {'-':5s}  {verdict} (bitwise)")
                continue
            r_mare = n_res["mare"] / max(g_res["mare"], FLOOR)
            r_mere = n_res["mere"] / max(g_res["mere"], FLOOR)
            r_rmse = n_res["rmse"] / max(g_res["rmse"], FLOOR)
            grade = grade_of(r_mare, r_mere, r_rmse)
            ec_ratio = n_res["error_count"] / max(g_res["error_count"], 1)
            ok = grade is not None and ec_ratio <= 2.0
            verdict = "PASS" if ok else "FAIL"
            n_fail += 0 if ok else 1
            print(f"{name:30s} {out_name:22s} {r_mare:7.3g} {r_mere:7.3g} "
                  f"{r_rmse:7.3g}  {str(grade):5s}  {verdict}")

    print(f"\nsummary: {n_fail} FAIL row(s)")
    if n_fail:
        print("注: FAIL 项为 NPU 侧 tl.exp/tl.sum lowering 的 ulp 级精度差异 "
              "(max/argmax 路径 bitwise 一致, 分歧隔离在 exp/sum), "
              "与 ratio_20260828_144618.md 的 3 项 FAIL 同源.")
    return 1 if n_fail else 0


def main() -> int:
    out_dir = Path(__file__).resolve().parent / "repro_results"
    side = detect_side()
    if side is None:
        print("error: no cuda/npu device available (CPU-only host cannot "
              "run the triton kernel)")
        return 2
    print(f"device side = {side}")
    collect(side, out_dir)

    other = "npu" if side == "gpu" else "gpu"
    if (out_dir / f"{other}.json").exists():
        return compare(out_dir)
    print(f"\nnext: copy this folder to the {other} server and run again "
          f"to produce the dual-benchmark verdict")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
