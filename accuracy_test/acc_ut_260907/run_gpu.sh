#!/usr/bin/env bash
#
# 一键运行 acc_ut_260907 全部 GPU 侧算子精度测试。
#
# GPU 侧测试无跨文件 device-context 污染问题，单进程运行即可。
#
# 前置条件：已激活包含 vllm / torch (CUDA build) 的 Python 环境，
# 且目标机器 CUDA 可用（否则用例会统一 skip）。
#
# 用法：
#   bash run_gpu.sh                          # 跑全部 GPU 侧
#   bash run_gpu.sh -k shift                 # 任意 pytest 参数
#
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "=================================================="
echo "  acc_ut_260907 GPU 侧算子精度测试"
echo "=================================================="

python - <<'PY'
import sys
try:
    import torch
except Exception as exc:  # noqa: BLE001
    sys.exit(f"[ERROR] 无法导入 torch: {exc}")
if not torch.cuda.is_available():
    sys.exit("[ERROR] CUDA 不可用")
print(f"torch={torch.__version__}  CUDA 已就绪")
PY

python -m pytest "${SCRIPT_DIR}/gpu" -v --tb=short -ra "$@"
