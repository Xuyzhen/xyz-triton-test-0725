#!/usr/bin/env bash
#
# 一键运行 acc_ut_260907 全部 NPU 侧算子精度测试。
#
# 每个测试文件在独立子进程中运行（run_npu_isolated.py）：昇腾向量核异常
# 会污染当前进程的设备上下文，隔离运行可避免跨文件相互影响。
#
# 套件模式（无参）：每个文件以 pytest -x 运行，首个用例失败/出错即停止
# 该文件、直接进入下一个文件，避免坏文件的逐用例报错刷屏；需要完整执行
# 某文件全部用例时，单独执行：
#   pytest npu/test_xxx.py -v             # 直跑 pytest，不经本脚本
#   bash run_npu.sh npu/test_xxx.py       # 透传模式，同样不加 -x
#
# 前置条件：已激活包含 vllm / vllm-ascend / torch_npu 的 Python 环境。
#
# 用法：
#   bash run_npu.sh                          # 跑全部 NPU 侧（逐文件隔离，首错即停）
#   bash run_npu.sh npu/test_temperature.py  # 透传给 pytest 的部分运行（完整执行）
#   bash run_npu.sh -k "temperature"         # 关键字过滤（完整执行）
#
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 与 run_all.sh / run_gpu.sh 保持一致：优先 python，回退 python3。
if command -v python >/dev/null 2>&1; then
  PY=python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "[ERROR] 未找到 python / python3，请先激活含 vllm / vllm-ascend 的环境" >&2
  exit 1
fi

echo "=================================================="
echo "  acc_ut_260907 NPU 侧算子精度测试（逐文件隔离进程）"
echo "=================================================="

"${PY}" - <<'PY'
import sys
try:
    import torch
except Exception as exc:  # noqa: BLE001
    sys.exit(f"[ERROR] 无法导入 torch: {exc}")
try:
    import torch_npu  # noqa: F401
except Exception as exc:  # noqa: BLE001
    sys.exit(f"[ERROR] 无法导入 torch_npu（请先 source CANN set_env.sh 并激活环境）: {exc}")
if not (hasattr(torch, "npu") and torch.npu.is_available()):
    sys.exit("[ERROR] torch.npu.is_available() == False，NPU 设备不可用")
print(f"torch={torch.__version__}  NPU 已就绪")
PY

exec "${PY}" "${SCRIPT_DIR}/run_npu_isolated.py" "$@"
