#!/bin/bash
# ============================================================================
# Sparse 算子 Profiling 数据采集 (triton + CANN 双端)
# 用法: ./script/profile_sparse.sh [device_id]
#
# 输出:
#   ./profiler_data_sli_grad_kl_loss/       -- triton profiling 数据
#   ./profiler_data_sli_grad_kl_loss_cann/  -- CANN profiling 数据
#
# 环境要求: Ascend NPU, mindspore 2.9.0, triton-ascend 3.2.1
# ============================================================================
set -euo pipefail

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
echo "Sparse 算子 Profiling 数据采集"
echo "NPU 设备: ${DEVICE_ID}"
echo "================================================"

# 清理旧数据
rm -rf "${TRITON_OUT_DIR}" "${CANN_OUT_DIR}"

echo ""
echo ">>> Step 1/3: 性能计时 (triton vs CANN)"
python "${PROFILER_SCRIPT}"

echo ""
echo ">>> Step 2/3: Triton profiling 数据已保存到 ${TRITON_OUT_DIR}"
echo ">>> Step 3/3: CANN profiling 数据已保存到 ${CANN_OUT_DIR}"

echo ""
echo "================================================"
echo "Profiling 完成"
echo "  Triton: ${TRITON_OUT_DIR}"
echo "  CANN:   ${CANN_OUT_DIR}"
echo "================================================"
