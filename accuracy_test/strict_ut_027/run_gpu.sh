#!/usr/bin/env bash
#
# 一键运行 strict_ut_027 全部 GPU 侧算子精度测试（9 个算子）。
#
# 前置条件：已激活包含 vllm / torch(cuda) 的 Python 环境，
# 且目标机器 CUDA GPU 可用（否则用例会统一 skip）。
#
# 用法：
#   bash run_gpu.sh                              # 运行全部 GPU 侧用例
#   bash run_gpu.sh -k shift_input_ids           # 仅运行匹配用例
#   bash run_gpu.sh --tb=long -s                 # 透传任意 pytest 参数
#
set -euo pipefail

# 脚本所在目录 = strict_ut_027；项目根目录 = 上两级
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

# 保证 `from accuracy_test.strict_ut_027.* import ...` 可被解析
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "=================================================="
echo "  strict_ut_027 GPU 侧算子精度测试（CUDA）"
echo "=================================================="

# 前置检查：Python 环境与 CUDA 可用性
python - <<'PY'
import sys
try:
    import torch
except Exception as exc:  # noqa: BLE001
    sys.exit(f"[ERROR] 无法导入 torch，请先激活 vllm/CUDA 环境: {exc}")
if not torch.cuda.is_available():
    sys.exit("[ERROR] torch.cuda 不可用，检测不到 CUDA GPU")
print(f"torch={torch.__version__}  CUDA 已就绪")
PY

# GPU 侧无 Ascend 那样的跨用例设备上下文污染问题，整目录单进程直跑。
python -m pytest accuracy_test/strict_ut_027/gpu -v --tb=short -ra "$@"
