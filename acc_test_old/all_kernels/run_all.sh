#!/bin/bash
# ============================================================
# 一键运行所有 Triton 算子精度测试（vllm 原版 + vllm-ascend patch）
# 共计 60 个测试文件
#
# 用法: cd xyz-triton-test-0725/acc_test/all_kernels && bash run_all.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================"
echo "  vLLM v0.24 + vLLM-Ascend Triton 算子精度测试全集"
echo "  覆盖: $({ for f in test_*.py; do echo; done; } | wc -l) 个核函数"
echo "================================================"
echo ""

GREEN='\033[32m'
RED='\033[31m'
YELLOW='\033[33m'
NC='\033[0m'
pass=0
fail=0
failed_tests=""

run_test() {
    local file=$1
    local name=$(basename "$file" .py)
    echo -ne "  ${YELLOW}[RUNNING]${NC} $name ... "
    output=$(python -m pytest "$file" -v --tb=short 2>&1)
    if echo "$output" | grep -q "passed"; then
        echo -e "  ${GREEN}[PASS]${NC}"
        pass=$((pass + 1))
    else
        echo -e "  ${RED}[FAIL]${NC}"
        fail=$((fail + 1))
        failed_tests="$failed_tests $name"
        echo "$output" | tail -30
    fi
    echo "----------------------------------------"
}

# 先跑 vanilla 原版（按名称排序）
echo "---------- vLLM vanilla kernels ----------"
for tf in $(ls test_*.py | grep -v '_patch\.py$' | sort); do
    run_test "$tf"
done

echo ""
echo "---------- vLLM-Ascend patched kernels ----------"
for tf in $(ls test_*_patch.py 2>/dev/null | sort); do
    run_test "$tf"
done

echo ""
echo "================================================"
echo -e "  结果: ${GREEN}$pass 通过${NC}, ${RED}$fail 失败${NC}"
echo "================================================"

if [ $fail -ne 0 ]; then
    echo ""
    echo "失败测试:"
    for ft in $failed_tests; do
        echo "  - $ft"
    done
    echo ""
    echo "单独运行失败测试查看详情:"
    for ft in $failed_tests; do
        echo "  python -m pytest ${ft}.py -v -s --tb=long"
    done
fi
