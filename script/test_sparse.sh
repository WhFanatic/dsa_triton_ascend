#!/bin/bash
# ============================================================================
# Sparse 算子功能与精度测试
# 用法: ./script/test_sparse.sh [test_type]
#   test_type: smoke | golden | accuracy | basic | all (默认all)
#
# 环境要求: Ascend NPU, mindspore 2.9.0, triton-ascend 3.2.1
# ============================================================================
set -euo pipefail

export ASCEND_RT_VISIBLE_DEVICES=0
export TRITON_END=mindspore
export TRITON_BACKEND=mindspore
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TRITON_CACHE_DIR=./my_triton_cache

TEST_TYPE="${1:-all}"
TEST_FILE="test_sli_grad_kl_loss_triton.py"

echo "================================================"
echo "Sparse 算子测试"
echo "测试类型: ${TEST_TYPE}"
echo "测试文件: ${TEST_FILE}"
echo "NPU 设备: ${ASCEND_RT_VISIBLE_DEVICES}"
echo "================================================"

run_smoke() {
    echo ""
    echo ">>> [smoke] 快速冒烟 (CANN 兼容 shape, triton vs CANN)"
    python -m pytest "${TEST_FILE}" -v -k "test_sparse_grad_kl_loss_large_precision[1-4096-4096-64-512-64-128-2048-fp16]" "$@"
    echo ">>> [smoke] 通过"
}

run_golden() {
    echo ""
    echo ">>> [golden] 算法正确性 (triton vs numpy)"
    python -m pytest "${TEST_FILE}" -v -k "test_golden" "$@"
    echo ">>> [golden] 通过"
}

run_accuracy() {
    echo ""
    echo ">>> [accuracy] CANN 精度对齐 (triton vs CANN)"
    python -m pytest "${TEST_FILE}" -v -k "test_accuracy" "$@"
    echo ">>> [accuracy] 通过"
}

run_basic() {
    echo ""
    echo ">>> [basic] 功能自检 (shape/dtype, 多处超 CANN 约束)"
    python -m pytest "${TEST_FILE}" -v -k "test_basic" "$@"
    echo ">>> [basic] 通过"
}

run_all() {
    run_smoke
    run_golden
    run_accuracy
    run_basic
}

case "${TEST_TYPE}" in
    smoke)   run_smoke ;;
    golden)  run_golden ;;
    accuracy) run_accuracy ;;
    basic)   run_basic ;;
    all)     run_all ;;
    *)
        echo "用法: $0 {smoke|golden|accuracy|basic|all}"
        exit 1
        ;;
esac
