#!/bin/bash
# ============================================================================
# Sparse operator Triton detailed profiling (msprof op) - all-kernel mode
#
# Usage: ./script/profile_sparse_detail.sh [device_id] [output_dir]
#
# Uses --kernel-name="prefix1|prefix2|..." to capture all 5 triton computation
# kernels in a single msprof op run.
#
# Output: ${OUT_DIR}/OPPROF_*/
#   OpBasicInfo.csv           -- kernel name / duration / block dim
#   PipeUtilization.csv       -- pipeline utilization
#   ArithmeticUtilization.csv -- arithmetic utilization
#   Memory.csv / MemoryUB.csv -- memory / UB usage
#   L2Cache.csv               -- L2 cache
#
# Requires: msprof (CANN toolkit)
# ============================================================================

DEVICE_ID="${1:-6}"
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
echo "Sparse Operator All-Kernel Profiling"
echo "NPU device: ${DEVICE_ID}"
echo "Output dir: ${OUT_DIR}"
echo "================================================"

rm -rf "${OUT_DIR}" ./my_triton_cache
mkdir -p "${OUT_DIR}"

echo ""
echo ">>> Profiling 5 kernels in one pass ..."

# --kernel-name 支持 | 拼接多个前缀，一次采集全部 5 个 kernel
# --launch-count=50 采集每个 kernel 前 50 次 launch（10次slis × 4chunks = 40次 + 余量）
msprof op --output="${OUT_DIR}" \
    --kernel-name="_gather_kv_fused|_teacher_distribution|_indexer_grad_kl_loss|_query_index_weight_grad|_scatter_dkey_index" \
    --launch-count=50 \
    python "${KERNEL_ONLY_SCRIPT}" --kernel-only

# 解析结果
OPPROF_DIR=$(find "${OUT_DIR}" -maxdepth 1 -type d -name "OPPROF_*" 2>/dev/null | head -1)
if [ -z "${OPPROF_DIR}" ]; then
    echo "FAILED: no OPPROF data"
    exit 1
fi

CSV="${OPPROF_DIR}/OpBasicInfo.csv"
if [ ! -f "${CSV}" ]; then
    echo "FAILED: no OpBasicInfo.csv"
    exit 1
fi

echo ""
echo "================================================"
echo "  Per-Kernel Profiling Summary"
echo "================================================"
printf "  %-40s %10s %10s\n" "kernel" "duration(us)" "block_dim"
printf "  %-40s %10s %10s\n" "----------------------------------------" "----------" "----------"

# 解析 csv：跳过 marker/framework kernel，聚合同名 kernel 取平均
awk -F',' '
NR>1 {
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3)
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4)

    name = $1
    dur  = $3 + 0
    blk  = $4

    if (name ~ /profile_marker|Cast_|Add_|StridedSlice|ZerosLike|ReduceSum|ConcatD/)
        next

    sum[name] += dur
    cnt[name]++
    blkdim[name] = blk
    total += dur
}
END {
    for (name in sum) {
        avg = sum[name] / cnt[name]
        printf "  %-40s %10.2f %10s\n", name, avg, blkdim[name]
    }
    if (total > 0)
        printf "  %-40s %10.2f\n", "--- per-chunk sum ---", total
}' "${CSV}"

echo "================================================"
echo "Per-kernel csv files under: ${OPPROF_DIR}/"
echo "================================================"
