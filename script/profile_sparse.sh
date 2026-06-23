#!/bin/bash
# ============================================================================
# Sparse operator profiling data collection (triton + CANN)
# Usage: ./script/profile_sparse.sh [device_id]
#
# Output:
#   ./profiler_data_sli_grad_kl_loss/       -- triton profiling data
#   ./profiler_data_sli_grad_kl_loss_cann/  -- CANN profiling data
#
# Requirements: Ascend NPU, mindspore 2.9.0, triton-ascend 3.2.1
# ============================================================================

DEVICE_ID="${1:-0}"
export ASCEND_RT_VISIBLE_DEVICES="${DEVICE_ID}"
export TRITON_END=mindspore
export TRITON_BACKEND=mindspore
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TRITON_CACHE_DIR=./my_triton_cache

PROFILER_SCRIPT="perf_sli_grad_kl_loss_triton.py"
TRITON_OUT_DIR="./profiler_data_sli_grad_kl_loss"
CANN_OUT_DIR="./profiler_data_sli_grad_kl_loss_cann"

echo "================================================"
echo "Sparse Operator Profiling"
echo "NPU device: ${DEVICE_ID}"
echo "================================================"

rm -rf "${TRITON_OUT_DIR}" "${CANN_OUT_DIR}"

echo ""
echo ">>> Step 1/3: Performance timing (triton vs CANN)"
python "${PROFILER_SCRIPT}"

echo ""
echo ">>> Step 2/3: Triton profiling saved to ${TRITON_OUT_DIR}"
echo ">>> Step 3/3: CANN profiling saved to ${CANN_OUT_DIR}"

echo ""
echo "================================================"
echo "Profiling complete"
echo "  Triton: ${TRITON_OUT_DIR}"
echo "  CANN:   ${CANN_OUT_DIR}"
echo "================================================"
