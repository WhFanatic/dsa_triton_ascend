#!/bin/bash
# ============================================================================
# Sparse operator profiling data collection (timing + triton + CANN)
# Usage: ./script/profile_sparse.sh [mode]
#
#   mode: timing | triton | cann | all (default: all)
#     timing  -- triton vs CANN timing benchmark (run_timing)
#     triton  -- triton kernel profiling via msprof (run_profiling)
#     cann    -- CANN operator profiling via msprof (run_profiling_cann)
#     all     -- timing + triton + cann
#
# Output:
#   ./profiler_data_sli_grad_kl_loss/       -- triton profiling data
#   ./profiler_data_sli_grad_kl_loss_cann/  -- CANN profiling data
#
# Requirements: Ascend NPU, mindspore 2.9.0, triton-ascend 3.2.1
# ============================================================================

export ASCEND_RT_VISIBLE_DEVICES=6
export TRITON_END=mindspore
export TRITON_BACKEND=mindspore
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TRITON_CACHE_DIR=./my_triton_cache

MODE="${1:-all}"
PROFILER_SCRIPT="perf_sli_grad_kl_loss_triton.py"
TRITON_OUT_DIR="./profiler_data_sli_grad_kl_loss"
CANN_OUT_DIR="./profiler_data_sli_grad_kl_loss_cann"

echo "================================================"
echo "Sparse Operator Profiling"
echo "Mode:      ${MODE}"
echo "NPU device: ${ASCEND_RT_VISIBLE_DEVICES}"
echo "================================================"
echo ""

run_timing() {
    echo ">>> Triton vs CANN timing benchmark ..."
    python "${PROFILER_SCRIPT}" --timing-only
}

run_triton() {
    rm -rf "${TRITON_OUT_DIR}"
    mkdir -p "${TRITON_OUT_DIR}"
    echo ">>> Triton profiling ..."
    python "${PROFILER_SCRIPT}" --triton-only
    echo ">>> Triton profiling saved to ${TRITON_OUT_DIR}"
}

run_cann() {
    rm -rf "${CANN_OUT_DIR}"
    mkdir -p "${CANN_OUT_DIR}"
    echo ">>> CANN profiling ..."
    python "${PROFILER_SCRIPT}" --cann-only
    echo ">>> CANN profiling saved to ${CANN_OUT_DIR}"
}

case "${MODE}" in
    timing)
        run_timing
        ;;
    triton)
        run_triton
        ;;
    cann)
        run_cann
        ;;
    all)
        run_timing
        run_triton
        run_cann
        ;;
    *)
        echo "Usage: $0 {timing|triton|cann|all}"
        exit 1
        ;;
esac

echo ""
echo "================================================"
echo "Profiling complete"
echo "================================================"
