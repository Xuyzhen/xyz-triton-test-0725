#!/usr/bin/env bash
#
# 一键运行 easy_ut_026 全部 NPU 算子精度测试（9 个算子）。
#
# 前置条件：已激活包含 vllm / vllm-ascend / torch-npu 的 Python 环境，
# 且目标机器昇腾 NPU 可用（否则用例会统一 skip）。
#
# 用法：
#   bash run_npu.sh                                   # 运行全部用例
#   bash run_npu.sh -k shift_input_ids                # 按关键字筛选
#   bash run_npu.sh test_shift_input_ids_kernel.py    # 只跑指定文件
#   bash run_npu.sh --tb=long -s                      # 透传任意 pytest 参数
#
set -euo pipefail

# 脚本所在目录 = easy_ut_026；项目根目录 = 上两级
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

# 保证 `from accuracy_test.easy_ut_026.* import ...` 可被解析
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "=============================================="
echo "  easy_ut_026 NPU 算子精度测试"
echo "=============================================="

# 前置检查：Python 环境与 NPU 可用性
python - <<'PY'
import sys
try:
    import torch
except Exception as exc:  # noqa: BLE001
    sys.exit(f"[ERROR] 无法导入 torch，请先激活 vllm-ascend/NPU 环境: {exc}")
if not (hasattr(torch, "npu") and torch.npu.is_available()):
    sys.exit("[ERROR] torch.npu 不可用，检测不到昇腾 NPU")
print(f"torch={torch.__version__}  NPU 已就绪")
PY

# 若第一个参数为 .py 文件名，则仅运行该文件；否则运行整个目录。
TARGET="accuracy_test/easy_ut_026"
if [ $# -gt 0 ]; then
    case "$1" in
        *.py)
            TARGET="${SCRIPT_DIR}/$1"
            shift
            ;;
    esac
fi

echo "执行: python -m pytest ${TARGET} -v --tb=short -ra $*"
python -m pytest "${TARGET}" -v --tb=short -ra "$@"