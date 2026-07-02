#!/bin/bash
# ============================================================================
# Sparse operator function and accuracy tests
# Usage: ./script/test_sparse.sh [test_type]
#   test_type: smoke | accuracy | cann_triton | triton_numpy | cann_numpy | all (default all)
#
# Requirements: Ascend NPU, mindspore 2.9.0, triton-ascend 3.2.1
# ============================================================================

export ASCEND_RT_VISIBLE_DEVICES=6
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
    python -m pytest "${TEST_FILE}" -v -m "smoke" "$@"
    echo ">>> [smoke] PASSED"
}

run_cann_triton() {
    echo ""
    echo ">>> [cann_triton] Triton vs CANN (CANN-supported shapes)"
    python -m pytest "${TEST_FILE}" -v \
        -k "test_sparse_grad_kl_loss_precision_cann_triton" "$@"
    echo ">>> [cann_triton] PASSED"
}

run_triton_numpy() {
    echo ""
    echo ">>> [triton_numpy] Triton vs numpy reference (triton-only shapes)"
    python -m pytest "${TEST_FILE}" -v \
        -k "test_sparse_grad_kl_loss_precision_triton_numpy" "$@"
    echo ">>> [triton_numpy] PASSED"
}

run_cann_numpy() {
    echo ""
    echo ">>> [cann_numpy] CANN vs numpy reference (CANN-supported shapes)"
    python -m pytest "${TEST_FILE}" -v \
        -k "test_sparse_grad_kl_loss_precision_cann_numpy" "$@"
    echo ">>> [cann_numpy] PASSED"
}

run_accuracy() {
    echo ""
    echo ">>> [accuracy] All accuracy tests (cann + numpy references)"
    python -m pytest "${TEST_FILE}" -v -m "accuracy" "$@"
    echo ">>> [accuracy] PASSED"
}

run_all() {
    run_smoke
    run_accuracy
}

case "${TEST_TYPE}" in
    smoke)        run_smoke ;;
    cann_triton)  run_cann_triton ;;
    triton_numpy) run_triton_numpy ;;
    cann_numpy)   run_cann_numpy ;;
    accuracy)     run_accuracy ;;
    all)          run_all ;;
    *)
        echo "Usage: $0 {smoke|cann_triton|triton_numpy|cann_numpy|accuracy|all}"
        exit 1
        ;;
esac
