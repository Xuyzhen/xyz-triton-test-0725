#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断探针：_compute_global_logsumexp 全 -inf 输入返回 nan 的根因定位。

背景
----
accuracy_test/acc_ut_260907/npu/test_compute_global_logsumexp_downstream.py
::TestComputeGlobalLogsumexp::test_all_neg_inf_blocks 断言"全 -inf block max
时全局 LSE 应为 -inf"，节点实际返回 nan。

本探针回答三个问题：
  Q1 是否为"精度"问题？
      对 4 个场景做三方对比：安装版 helper / 裸公式参考 kernel / CPU 参考。
      有限值域场景若 1e-5 内一致、仅全 -inf 场景发散 -> 非精度回归，
      而是退化输入的固有公式行为（IEEE754: -inf - (-inf) = NaN）。
  Q2 节点上实际执行的 kernel 是哪个、什么实现？
      打印 vllm / vllm-ascend 版本与 __file__（site-packages 还是源码目录）、
      符号同一性（vllm-ascend 是否纯 re-export）、并 dump 实际参与 JIT
      编译的 helper 源码。
  Q3 是否存在 vllm / vllm-ascend 版本错配？
      安装版 helper 与裸公式参考 kernel 全场景一致（含全 -inf 均 nan）
      -> 无错配，nan 为固有行为；若 helper 在全 -inf 场景返回 -inf
      -> 节点安装版与共享目录 checkout 不一致，版本错配实锤。

用法（在 NPU 节点）：
  cd accuracy_test/acc_ut_260907
  python probe/probe_global_lse.py               # NPU 全量探针
  python probe/probe_global_lse.py --device cpu  # 仅环境 + CPU 数学对照
"""

import argparse
import inspect
import math

NEG_INF = float("-inf")

# 安装版符号；延迟到函数内导入 vllm/vllm_ascend，保证报错可读。
_HELPER = {"obj": None, "err": None}


def _load_helper():
    if _HELPER["obj"] is None and _HELPER["err"] is None:
        try:
            from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import (
                _compute_global_lse,
            )
            _HELPER["obj"] = _compute_global_lse
        except Exception as exc:  # noqa: BLE001
            _HELPER["err"] = exc
    return _HELPER["obj"], _HELPER["err"]


# ---------------------------------------------------------------------------
# 参考实现
# ---------------------------------------------------------------------------

def _cpu_ref(maxes, sumexps):
    """CPU 参考：与 UT 内 _global_logsumexp_ref 相同（含 -inf 特判）。"""
    import torch

    m = torch.tensor(maxes, dtype=torch.float32)
    s = torch.tensor(sumexps, dtype=torch.float32)
    gmax = float(m.max().item())
    if gmax > NEG_INF:
        return gmax + float(torch.log(torch.sum(s * torch.exp(m - gmax))).item())
    return gmax


def _scenarios():
    """A 为失败用例场景；B/C/D 覆盖有限值域与 padding。"""
    return [
        ("A_all_neg_inf", [NEG_INF, NEG_INF, NEG_INF], [0.0, 0.0, 0.0]),
        ("B_mixed", [1.5, NEG_INF, NEG_INF], [2.0, 0.0, 0.0]),
        ("C_all_finite", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
        ("D_single_block", [1.0], [2.0]),
    ]


def _fmt(x):
    x = float(x)
    return "nan" if x != x else f"{x:g}"


def _same(a, b):
    a, b = float(a), float(b)
    if a != a and b != b:
        return True  # 双 nan 视为一致
    if a != a or b != b:
        return False
    return math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)


def _dump_source(fn, name):
    """dump 实际参与 JIT 编译的源码（triton.JITFunction.fn 为原始函数）。"""
    print(f"  --- {name} 实际源码 ---")
    raw = getattr(fn, "fn", None) or fn
    try:
        print(inspect.getsource(raw))
    except (OSError, TypeError):
        src = getattr(fn, "src", None)
        print(src if src else f"  (源码不可获取: {fn!r})")


def _where(path):
    p = str(path).replace("\\", "/")
    if "site-packages" in p or "dist-packages" in p:
        return "site-packages（安装版）"
    return "源码目录（editable/源码运行）"


# ---------------------------------------------------------------------------
# 探针分节
# ---------------------------------------------------------------------------

def section_env():
    print("=" * 72)
    print("[ENV] 运行环境与符号来源（回答 Q2/Q3 的静态部分）")
    print("=" * 72)
    try:
        import torch
        import vllm
    except Exception as exc:  # noqa: BLE001
        print(f"  vllm/torch 导入失败，探针需在装有 vllm 的节点上运行：{exc}")
        return
    print(f"  torch       : {torch.__version__}")
    print(f"  vllm        : {vllm.__version__}")
    print(f"               {vllm.__file__}  [{_where(vllm.__file__)}]")
    try:
        import vllm_ascend
        ver = getattr(vllm_ascend, "__version__", "?")
        print(f"  vllm_ascend : {ver}")
        print(f"               {vllm_ascend.__file__}  [{_where(vllm_ascend.__file__)}]")
    except Exception as exc:  # noqa: BLE001
        print(f"  vllm_ascend : 导入失败：{exc}")
        return

    from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as vllm_rsu
    from vllm_ascend.worker.v2.spec_decode import (
        rejection_sampler_utils as va_rsu,
    )
    print(f"  vllm  实现模块 : {vllm_rsu.__file__}")
    print(f"  va    实现模块 : {va_rsu.__file__}")

    helper, err = _load_helper()
    if err is not None:
        print(f"\n[IDENTITY] vllm_ascend._compute_global_lse 导入失败：{err}")
        return
    upstream = getattr(vllm_rsu, "_compute_global_logsumexp", None)
    if upstream is None:
        print("\n[IDENTITY] 安装版 vllm 无 _compute_global_logsumexp（< 06-30 版本），"
              "与 UT 可运行的事实矛盾，请人工核查。")
        return
    same = helper is upstream
    print(f"\n[IDENTITY] vllm_ascend._compute_global_lse is "
          f"vllm._compute_global_logsumexp : {same}")
    print("           True = 纯 re-export，实现在 vllm；False = vllm-ascend 有本地实现")
    _dump_source(upstream, "vllm._compute_global_logsumexp")


def section_cpu_math():
    print("=" * 72)
    print("[CPU] 公式逐项分解（纯 torch，与硬件无关，回答 Q1）")
    print("=" * 72)
    import torch

    for label, maxes, sumexps in [
        ("有限值域", [1.0, 2.0, 3.0], [1.0, 1.0, 1.0]),
        ("全 -inf ", [NEG_INF, NEG_INF, NEG_INF], [0.0, 0.0, 0.0]),
    ]:
        m = torch.tensor(maxes, dtype=torch.float32)
        s = torch.tensor(sumexps, dtype=torch.float32)
        gmax = m.max()
        shifted = m - gmax
        term = s * torch.exp(shifted)
        total = term.sum()
        lse = gmax + torch.log(total)
        print(f"  {label}: global_max={_fmt(gmax)}  maxes-gmax={[ _fmt(v) for v in shifted.tolist() ]}"
              f"  sum={_fmt(total)}  lse={_fmt(lse)}")
    print("  -> 全 -inf 时 maxes-gmax = -inf-(-inf) = nan（IEEE754），CPU 同样 nan，与 NPU 无关；")
    print("     UT 的 CPU 参考额外做了 global_max > -inf 特判，kernel 没有 -> 期望不匹配。")


def section_kernel_npu():
    print("=" * 72)
    print("[KERNEL] 安装版 helper vs 裸公式参考 kernel（NPU 实测，回答 Q1/Q3）")
    print("=" * 72)
    helper, err = _load_helper()
    if err is not None or helper is None:
        print(f"  helper 不可用（{err}），跳过 kernel 对比（与 UT 的 skip 分支一致）。")
        return None

    import torch
    if not getattr(torch, "npu", None) or not torch.npu.is_available():
        print("  torch.npu 不可用，请用 --device cpu 或在 NPU 节点运行。")
        return None

    from vllm.triton_utils import tl, triton
    from vllm_ascend.ops.triton.triton_utils import (
        init_device_properties_triton,
    )

    # 与 UT 完全相同的调用方式：wrapper 内引用模块全局 helper。
    global _compute_global_lse_helper
    _compute_global_lse_helper = helper

    @triton.jit
    def _helper_wrapper(
        local_max_ptr, local_max_stride,
        local_sumexp_ptr, local_sumexp_stride,
        output_ptr, logit_idx, vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    ):
        result = _compute_global_lse_helper(
            local_max_ptr, local_max_stride,
            local_sumexp_ptr, local_sumexp_stride,
            logit_idx, vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
        )
        tl.store(output_ptr, result)

    @triton.jit
    def _bare_formula(
        local_max_ptr, local_max_stride,
        local_sumexp_ptr, local_sumexp_stride,
        out_gmax_ptr, out_sum_ptr, out_lse_ptr,
        logit_idx, vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    ):
        # 与 vllm 上游 _compute_global_logsumexp 逐行一致的裸公式，
        # 额外存储中间量 global_max / sum_term 以观测 NaN 产生位置。
        blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
        blocks_mask = blocks < vocab_num_blocks
        maxes = tl.load(
            local_max_ptr + logit_idx * local_max_stride + blocks,
            mask=blocks_mask, other=float("-inf"),
        )
        sumexps = tl.load(
            local_sumexp_ptr + logit_idx * local_sumexp_stride + blocks,
            mask=blocks_mask, other=0.0,
        )
        global_max = tl.max(maxes, axis=0)
        sum_term = tl.sum(sumexps * tl.exp(maxes - global_max))
        lse = global_max + tl.log(sum_term)
        tl.store(out_gmax_ptr, global_max)
        tl.store(out_sum_ptr, sum_term)
        tl.store(out_lse_ptr, lse)

    init_device_properties_triton()
    device = torch.device("npu")
    results = {}
    print(f"  {'场景':<15}{'安装版helper':>14}{'裸公式lse':>12}"
          f"{'(gmax/sum)':>22}{'CPU参考':>10}")
    for name, maxes, sumexps in _scenarios():
        num_blocks = len(maxes)
        padded = triton.next_power_of_2(num_blocks)
        lm = torch.tensor([maxes], dtype=torch.float32, device=device)
        ls = torch.tensor([sumexps], dtype=torch.float32, device=device)
        out_h = torch.zeros(1, dtype=torch.float32, device=device)
        out_g = torch.zeros(1, dtype=torch.float32, device=device)
        out_s = torch.zeros(1, dtype=torch.float32, device=device)
        out_b = torch.zeros(1, dtype=torch.float32, device=device)

        _helper_wrapper[(1,)](
            lm, lm.stride(0), ls, ls.stride(0),
            out_h, 0, num_blocks, PADDED_VOCAB_NUM_BLOCKS=padded,
        )
        _bare_formula[(1,)](
            lm, lm.stride(0), ls, ls.stride(0),
            out_g, out_s, out_b, 0, num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=padded,
        )
        torch.npu.synchronize()

        ref = _cpu_ref(maxes, sumexps)
        h, g, s, b = (t.item() for t in (out_h, out_g, out_s, out_b))
        results[name] = (h, b, ref)
        print(f"  {name:<15}{_fmt(h):>14}{_fmt(b):>12}"
              f"  ({_fmt(g)}/{_fmt(s)})      {_fmt(ref):>10}")

    # ---- 自动判读 ----
    print("\n  [判读]")
    a = results.get("A_all_neg_inf")
    if a is not None:
        h, b, ref = a
        if _same(h, b):
            print("  1) 安装版 helper 与裸公式参考一致（全 -inf 场景同为 nan）")
            print("     -> 非 vllm/vllm-ascend 版本错配；nan 为公式固有行为")
            print("     -> UT 失败根因 = CPU 参考有 -inf 特判而 kernel 无（期望不匹配），")
            print("        非精度回归（有限值域场景见下）")
        else:
            print("  1) helper 与裸公式在全 -inf 场景不一致"
                  f"（helper={_fmt(h)}, 裸公式={_fmt(b)}）")
            print("     -> 节点安装版含 -inf 特判，与共享目录 checkout 存在版本错配！")
            print("     -> 需对齐节点 site-packages 与共享目录代码后复测")
    finite_ok = all(
        _same(results[n][0], results[n][2])
        for n in ("B_mixed", "C_all_finite", "D_single_block")
        if n in results
    )
    print(f"  2) 有限值域场景 helper vs CPU 参考 1e-5 一致："
          f"{'是 -> 精度正常' if finite_ok else '否 -> 存在实际精度问题，需进一步排查'}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="probe: _compute_global_logsumexp all -inf diagnostic",
    )
    parser.add_argument("--device", choices=["npu", "cpu"], default="npu",
                        help="npu=全量探针（默认）；cpu=仅环境+CPU 数学对照")
    args = parser.parse_args()

    section_env()
    print()
    section_cpu_math()
    print()
    if args.device == "npu":
        section_kernel_npu()
    else:
        print("[KERNEL] --device cpu：跳过 NPU kernel 对比。")
    print("\n探针结束。把以上完整输出发回即可完成判定。")


if __name__ == "__main__":
    main()
