#!/bin/bash
# ============================================================
# 一键运行所有 Triton 算子精度测试
# 用法: cd xyz-triton-test-0725/acc_test && bash run_all_tests.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  vLLM v0.24 Triton 算子精度测试"
echo "  共 28 个测试文件"
echo "========================================"
echo ""

# 颜色
GREEN='\033[32m'
RED='\033[31m'
NC='\033[0m'
pass=0
fail=0
failed_tests=""

run_test() {
    local file=$1
    local name=$(basename "$file" .py)
    echo -n "  [RUNNING] $name ... "
    if python -m pytest "$file" -v --tb=short 2>&1 | tail -20; then
        echo -e "  ${GREEN}[PASS]${NC} $name"
        pass=$((pass + 1))
    else
        echo -e "  ${RED}[FAIL]${NC} $name"
        fail=$((fail + 1))
        failed_tests="$failed_tests $name"
    fi
    echo "----------------------------------------"
}

# 逐个运行测试（避免相互影响）
for tf in test_*.py; do
    run_test "$tf"
done

echo ""
echo "========================================"
echo -e "  结果: ${GREEN}$pass 通过${NC}, ${RED}$fail 失败${NC}"
echo "========================================"

if [ $fail -ne 0 ]; then
    echo "失败测试:"
    for ft in $failed_tests; do
        echo "  - $ft"
    done
fi
