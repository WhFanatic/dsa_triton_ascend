#!/bin/bash
# ============================================================================
# Sparse operator Triton detailed profiling (msprof op) - per-kernel mode
#
# Usage: ./script/profile_sparse_detail.sh [device_id] [output_dir]
#
# 每个 kernel 单独跑一次 msprof op，结果落到 ${OUT_DIR}/<kernel>/ 子目录，
# 避免共享 OPPROF 输出时的合并语义影响（不同 kernel 的 PipeUtilization 等
# CSV 不会再被混在同一份里）。
#
# Output: ${OUT_DIR}/<kernel>/OPPROF_*/
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

# 要分别采集的 kernel 前缀（msprof op --kernel-name 接受 |，但这里每个独立跑）
KERNELS=(
    "_teacher_indexer_kl"
    "_indexer_grad"
)

echo "================================================"
echo "Sparse Operator Per-Kernel Profiling"
echo "NPU device: ${DEVICE_ID}"
echo "Output dir: ${OUT_DIR}"
echo "Kernels   : ${#KERNELS[@]}"
echo "================================================"

rm -rf "${OUT_DIR}" ./my_triton_cache
mkdir -p "${OUT_DIR}"

for kernel in "${KERNELS[@]}"; do
    SUB_OUT="${OUT_DIR}/${kernel}"
    mkdir -p "${SUB_OUT}"

    echo ""
    echo ">>> [${kernel}] profiling ..."

    msprof op --output="${SUB_OUT}" \
        --kernel-name="${kernel}" \
        python "${KERNEL_ONLY_SCRIPT}" --kernel-only
done

# 解析结果：遍历每个 kernel 子目录下的 OPPROF_*/OpBasicInfo.csv
echo ""
echo "================================================"
echo "  Per-Kernel Profiling Summary"
echo "================================================"
printf "  %-40s %10s %10s\n" "kernel" "duration(us)" "block_dim"
printf "  %-40s %10s %10s\n" "----------------------------------------" "----------" "----------"

TOTAL_AVG_FILE=$(mktemp)

for kernel in "${KERNELS[@]}"; do
    SUB_OUT="${OUT_DIR}/${kernel}"
    OPPROF_DIR=$(find "${SUB_OUT}" -maxdepth 1 -type d -name "OPPROF_*" 2>/dev/null | head -1)
    if [ -z "${OPPROF_DIR}" ]; then
        printf "  %-40s %10s %10s\n" "${kernel}" "MISSING" "-"
        continue
    fi

    CSV="${OPPROF_DIR}/OpBasicInfo.csv"
    if [ ! -f "${CSV}" ]; then
        printf "  %-40s %10s %10s\n" "${kernel}" "NO_CSV" "-"
        continue
    fi

    # 每个子目录内可能包含多次 launch / mix 后缀变体（如 _mix_aic），
    # 同名聚合取平均，并把每行作为一行单独打印
    awk -F',' -v out="${TOTAL_AVG_FILE}" '
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
    }
    END {
        for (name in sum) {
            avg = sum[name] / cnt[name]
            printf "  %-40s %10.2f %10s\n", name, avg, blkdim[name]
            printf "%.6f\n", avg >> out
        }
    }' "${CSV}"
done

# 汇总每个 kernel 平均时长之和（用于 per-chunk 总耗时估算）
TOTAL=$(awk '{s+=$1} END {printf "%.2f", s}' "${TOTAL_AVG_FILE}")
rm -f "${TOTAL_AVG_FILE}"

printf "  %-40s %10s\n" "----------------------------------------" "----------"
printf "  %-40s %10s us\n" "--- per-chunk sum (avg) ---" "${TOTAL}"

echo "================================================"
echo "Per-kernel csv directories under: ${OUT_DIR}/<kernel>/OPPROF_*/"
echo "================================================"
