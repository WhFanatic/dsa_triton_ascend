#!/bin/bash
# ============================================================================
# Sparse operator function and accuracy tests
# Usage: ./script/test_sparse.sh [test_type]
#   test_type: smoke | golden | accuracy | basic | all (default all)
#
# Requirements: Ascend NPU, mindspore 2.9.0, triton-ascend 3.2.1
# ============================================================================

export ASCEND_RT_VISIBLE_DEVICES=0
export TRITON_END=mindspore
export TRITON_BACKEND=mindspore
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TRITON_CACHE_DIR=./my_triton_cache

TEST_TYPE="${1:-all}"
TEST_FILE="test_sli_grad_kl_loss_triton.py"

echo "================================================"
echo "Sparse Operator Test"
echo "Test type: ${TEST_TYPE}"
echo "Test file: ${TEST_FILE}"
echo "NPU device: ${ASCEND_RT_VISIBLE_DEVICES}"
echo "================================================"

run_smoke() {
    echo ""
    echo ">>> [smoke] Quick smoke test (CANN-compatible shape, triton vs CANN)"
    python -m pytest "${TEST_FILE}" -v -k "test_sparse_grad_kl_loss_large_precision[1-4096-4096-64-512-64-128-2048-fp16]" "$@"
    echo ">>> [smoke] PASSED"
}

run_golden() {
    echo ""
    echo ">>> [golden] Algorithm correctness (triton vs numpy)"
    python -m pytest "${TEST_FILE}" -v -k "test_golden" "$@"
    echo ">>> [golden] PASSED"
}

run_accuracy() {
    echo ""
    echo ">>> [accuracy] CANN accuracy alignment (triton vs CANN)"
    python -m pytest "${TEST_FILE}" -v -k "test_accuracy" "$@"
    echo ">>> [accuracy] PASSED"
}

run_basic() {
    echo ""
    echo ">>> [basic] Self-check (shape/dtype, beyond CANN constraints)"
    python -m pytest "${TEST_FILE}" -v -k "test_basic" "$@"
    echo ">>> [basic] PASSED"
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
        echo "Usage: $0 {smoke|golden|accuracy|basic|all}"
        exit 1
        ;;
esac
