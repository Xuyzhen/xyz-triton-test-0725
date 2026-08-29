#!/usr/bin/env bash
#
# 一键运行 strict_ut_027 全部 NPU 侧算子精度测试（9 个算子）。
#
# 关键设计：与 easy_ut_026 一致，每个测试文件在一个独立 pytest 进程里运行。
# 昇腾 vector-core 异常会污染当前进程的 device 上下文，导致后续算子在被测
# 前的张量创建阶段就失败；按文件隔离进程可以避免相互污染，也便于逐个算子
# 定位与重跑。
#
# 前置条件：已激活包含 vllm / vllm-ascend / torch-npu 的 Python 环境，
# 且目标机器昇腾 NPU 可用（否则用例会统一 skip）。
#
# 用法：
#   bash run_npu.sh                                   # 每个文件独立进程逐个跑全部
#   bash run_npu.sh -k shift_input_ids                # 仅新进程内跑匹配用例
#   bash run_npu.sh test_shift_input_ids_kernel.py    # 仅新进程内跑指定文件
#   bash run_npu.sh --tb=long -s                      # 透传任意 pytest 参数
#
set -euo pipefail

# 脚本所在目录 = strict_ut_027；项目根目录 = 上两级
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

# 保证 `from accuracy_test.strict_ut_027.* import ...` 可被解析
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "=================================================="
echo "  strict_ut_027 NPU 侧算子精度测试（按文件隔离进程）"
echo "=================================================="

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

# 调用隔离运行器：每个测试模块一个独立 pytest 进程。
python "${SCRIPT_DIR}/run_npu_isolated.py" "$@"
