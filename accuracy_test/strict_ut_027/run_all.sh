#!/usr/bin/env bash
#
# 一键运行 strict_ut_027 测试集：自动检测当前机器可用设备并运行对应侧。
#
#   - 检测到 CUDA        -> 运行 gpu/ 侧全部用例
#   - 检测到 昇腾 NPU    -> 运行 npu/ 侧全部用例（按文件隔离进程）
#   - 两者都可用          -> 依次运行两侧
#   - 都不可用            -> 报错退出
#
# 用法：
#   bash run_all.sh            # 自动检测并运行
#   bash run_all.sh --tb=long  # 透传任意 pytest 参数
#
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

HAS_CUDA=0
HAS_NPU=0

python - <<'PY' && HAS_CUDA=1 || true
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY

python - <<'PY' && HAS_NPU=1 || true
import torch
raise SystemExit(0 if (hasattr(torch, "npu") and torch.npu.is_available()) else 1)
PY

RAN_ANY=0

if [[ "$HAS_CUDA" == "1" ]]; then
  echo ">> 检测到 CUDA，运行 GPU 侧"
  bash "${SCRIPT_DIR}/run_gpu.sh" "$@"
  RAN_ANY=1
fi

if [[ "$HAS_NPU" == "1" ]]; then
  echo ">> 检测到昇腾 NPU，运行 NPU 侧"
  bash "${SCRIPT_DIR}/run_npu.sh" "$@"
  RAN_ANY=1
fi

if [[ "$RAN_ANY" == "0" ]]; then
  echo "[ERROR] 当前机器既无 CUDA GPU 也无昇腾 NPU，无法运行 strict_ut_027" >&2
  exit 1
fi
