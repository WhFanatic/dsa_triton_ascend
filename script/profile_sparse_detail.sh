#!/bin/bash
# ============================================================================
# Sparse operator Triton detailed profiling (msprof op)
#
# Usage: ./script/profile_sparse_detail.sh [device_id] [output_dir]
#
# Single pass captures:
#   - Full kernel timing (OpBasicInfo.csv)
#   - Arithmetic utilization (ArithmeticUtilization.csv)
#   - Pipeline utilization (PipeUtilization.csv)
#   - Memory bandwidth (Memory.csv)
#   - UB usage (MemoryUB.csv)
#
# Requires: msprof (CANN toolkit)
# ============================================================================

DEVICE_ID="${1:-0}"
OUT_DIR="${2:-./profiler_data_sli_detail}"
export ASCEND_RT_VISIBLE_DEVICES="${DEVICE_ID}"
export TRITON_END=mindspore
export TRITON_BACKEND=mindspore
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TRITON_CACHE_DIR=./my_triton_cache

export SLI_SYNC=1
export SPARSE_GRAD_PROFILE_MARKERS=1

KERNEL_ONLY_SCRIPT="perf_sli_grad_kl_loss_triton.py"

echo "================================================"
echo "Sparse Operator Detailed Profiling (msprof op)"
echo "NPU device: ${DEVICE_ID}"
echo "Output dir: ${OUT_DIR}"
echo "================================================"

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

echo ""
echo ">>> Full kernel profiling ..."
msprof op --output="${OUT_DIR}" \
    python "${KERNEL_ONLY_SCRIPT}" --kernel-only

echo ""
echo "================================================"
echo "Profiling complete"
echo "Output dir: ${OUT_DIR}"
echo ""
echo "Key files (under OPPROF_* subdir):"
echo "  OpBasicInfo.csv           -- kernel name/duration/block dim"
echo "  ArithmeticUtilization.csv -- arithmetic utilization (per block)"
echo "  PipeUtilization.csv       -- pipeline utilization (per block)"
echo "  Memory.csv                -- memory bandwidth"
echo "  MemoryUB.csv              -- UB usage"
echo "  L2Cache.csv               -- L2 cache hit rate"
echo "================================================"
