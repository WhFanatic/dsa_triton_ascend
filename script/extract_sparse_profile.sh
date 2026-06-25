#!/bin/bash
# ============================================================================
# Sparse operator profiling full text extraction
#
# Usage: ./script/extract_sparse_profile.sh [triton_dir] [cann_dir] [msprof_dir] [output_file]
#
# Extracts from profile_sparse.sh and profile_sparse_detail.sh output:
#   1. Per-kernel name, duration, call count (kernel_details.csv)
#   2. API call statistics (api_statistic.csv)
#   3. Operator-level timing ranking (step_trace / op_range)
#   4. AICore hardware metrics (sqlite database)
#   5. Triton vs CANN timing comparison (perf script output)
#   6. msprof per-kernel sampling results
#   7. Full log keywords (memory H2D/D2H tasks etc.)
#
# Output: single text file organized by sections
# ============================================================================

TRITON_DIR="${1:-./profiler_data_sli_grad_kl_loss}"
CANN_DIR="${2:-./profiler_data_sli_grad_kl_loss_cann}"
MSPROF_DIR="${3:-./profiler_data_sli_detail}"
OUTPUT_FILE="${4:-./sparse_profile_report.txt}"

echo "================================================"
echo "Sparse Operator Profiling Text Extraction"
echo "Triton  profiler: ${TRITON_DIR}"
echo "CANN    profiler: ${CANN_DIR}"
echo "msprof  detailed: ${MSPROF_DIR}"
echo "Output file:      ${OUTPUT_FILE}"
echo "================================================"

> "${OUTPUT_FILE}"

section() {
    echo "" >> "${OUTPUT_FILE}"
    echo "################################################################################" >> "${OUTPUT_FILE}"
    echo "# $*" >> "${OUTPUT_FILE}"
    echo "################################################################################" >> "${OUTPUT_FILE}"
}

latest_parse_dir() {
    local base="$1"
    # triton profiler uses "syn-*", cann profiler uses "{hostname}_*_ascend_ms"
    local d
    d=$(find "${base}" -maxdepth 1 -name "syn-*" -type d 2>/dev/null | sort -r | head -1 | tr -d '\n')
    if [ -z "${d}" ]; then
        d=$(find "${base}" -maxdepth 1 -name "*_ascend_ms" -type d 2>/dev/null | sort -r | head -1 | tr -d '\n')
    fi
    echo "${d}"
}

extract_kernel_summary() {
    local label="$1"
    local parse_dir="$2"
    local csv_file="${parse_dir}/ASCEND_PROFILER_OUTPUT/kernel_details.csv"

    if [ ! -f "${csv_file}" ]; then
        echo "[${label}] kernel_details.csv not found" >> "${OUTPUT_FILE}"
        return
    fi

    echo "[${label}] Kernel execution details (kernel_details.csv)" >> "${OUTPUT_FILE}"
    echo "" >> "${OUTPUT_FILE}"
    echo "  Top 30 by duration:" >> "${OUTPUT_FILE}"
    echo "  ---------------------------------------------------------------------------" >> "${OUTPUT_FILE}"

    {
        head -1 "${csv_file}"
        grep -iE "sparse|lightning|indexer|gather|scatter|teacher|kl_loss|grad" "${csv_file}" || true
    } | awk -F',' '
    NR==1 {
        for(i=1;i<=NF;i++) {
            gsub(/^[ \t]+|[ \t]+$/, "", $i)
            col[$i]=i
        }
        printf "  %-70s %12s\n", "Name", "Duration(us)"
        next
    }
    {
        name=$col["Name"]
        dur=$col["Duration(us)"]
        gsub(/^[ \t]+|[ \t]+$/, "", name)
        gsub(/^[ \t]+|[ \t]+$/, "", dur)
        if (dur+0 > 0 && name != "") {
            printf "  %-70s %12s\n", name, dur
        }
    }' | sort -t$'\t' -k2 -rn 2>/dev/null | head -30 >> "${OUTPUT_FILE}"
}

extract_api_stat() {
    local label="$1"
    local parse_dir="$2"
    local csv_file="${parse_dir}/ASCEND_PROFILER_OUTPUT/api_statistic.csv"

    if [ ! -f "${csv_file}" ]; then
        echo "[${label}] api_statistic.csv not found" >> "${OUTPUT_FILE}"
        return
    fi

    echo "" >> "${OUTPUT_FILE}"
    echo "[${label}] API call statistics (by total time)" >> "${OUTPUT_FILE}"
    echo "  ---------------------------------------------------------------------------" >> "${OUTPUT_FILE}"
    {
        head -1 "${csv_file}"
        tail -n +2 "${csv_file}"
    } | awk -F',' '
    NR==1 {
        for(i=1;i<=NF;i++) { col[$i]=i }
        printf "  %-50s %8s %10s %10s\n", "API Name", "Count", "Total(us)", "Avg(us)"
        next
    }
    {
        name=$col["API Name"]; cnt=$col["Count"]; total=$col["Time(us)"]; avg=$col["Avg(us)"]
        gsub(/^[ \t]+|[ \t]+$/, "", name)
        if (total+0 > 0) printf "  %-50s %8s %10s %10s\n", name, cnt, total, avg
    }' 2>/dev/null | sort -t$'\t' -k3 -rn | head -20 >> "${OUTPUT_FILE}"
}

extract_aicore_summary() {
    local label="$1"
    local parse_dir="$2"
    local db_file

    db_file=$(find "${parse_dir}" -name "time.db" -path "*/sqlite/*" 2>/dev/null | head -1)
    if [ -z "${db_file}" ]; then
        echo "[${label}] time.db not found" >> "${OUTPUT_FILE}"
        return
    fi

    echo "" >> "${OUTPUT_FILE}"
    echo "[${label}] AICore hardware metrics (time.db)" >> "${OUTPUT_FILE}"

    sqlite3 "${db_file}" "
        SELECT '  Model ID: ' || model_id, '  Task count: ' || count(*), '  Total time(us): ' || sum(task_time)
        FROM task_time_info
        GROUP BY model_id;
    " 2>/dev/null >> "${OUTPUT_FILE}" || echo "  (sqlite query failed)" >> "${OUTPUT_FILE}"
}

extract_op_range() {
    local label="$1"
    local parse_dir="$2"
    local op_file="${parse_dir}/FRAMEWORK/mindspore.op_range"

    if [ ! -f "${op_file}" ]; then
        echo "[${label}] mindspore.op_range not found" >> "${OUTPUT_FILE}"
        return
    fi

    echo "" >> "${OUTPUT_FILE}"
    echo "[${label}] MindSpore operator timing range (op_range)" >> "${OUTPUT_FILE}"
    echo "  ---------------------------------------------------------------------------" >> "${OUTPUT_FILE}"
    { grep -iE "sparse|lightning|indexer|gather|scatter|teacher|kl_loss|grad|loss" "${op_file}" 2>/dev/null || true; } | head -30 >> "${OUTPUT_FILE}"
}

extract_logs() {
    local label="$1"
    local parse_dir="$2"

    echo "" >> "${OUTPUT_FILE}"
    echo "[${label}] Key log messages" >> "${OUTPUT_FILE}"
    echo "  ---------------------------------------------------------------------------" >> "${OUTPUT_FILE}"

    for f in "${parse_dir}"/logs/profiler_*.log; do
        [ -f "${f}" ] || continue
        echo "  --- ${f} ---" >> "${OUTPUT_FILE}"
        grep -iE "error|warning|sparse|lightning|indexer|OOM|memory|allocat" "${f}" 2>/dev/null | head -20 >> "${OUTPUT_FILE}" || true
    done
}

section "Sparse operator (sparse_lightning_indexer_grad_kl_loss) profiling full text report"
echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')" >> "${OUTPUT_FILE}"

# ---- Part 1: Triton full profiling ----
section "Part 1: Triton full profiling (perf script)"
TRITON_PARSE=$(latest_parse_dir "${TRITON_DIR}")
if [ -n "${TRITON_PARSE}" ]; then
    extract_kernel_summary "triton_full" "${TRITON_PARSE}"
    extract_api_stat "triton_full" "${TRITON_PARSE}"
    extract_aicore_summary "triton_full" "${TRITON_PARSE}"
    extract_op_range "triton_full" "${TRITON_PARSE}"
    extract_logs "triton_full" "${TRITON_PARSE}"

    st_csv="${TRITON_PARSE}/ASCEND_PROFILER_OUTPUT/step_trace_time.csv"
    if [ -f "${st_csv}" ]; then
        echo "" >> "${OUTPUT_FILE}"
        echo "[triton_full] Step-level timing (step_trace_time.csv)" >> "${OUTPUT_FILE}"
        echo "  ---------------------------------------------------------------------------" >> "${OUTPUT_FILE}"
        tail -n +2 "${st_csv}" | awk -F',' '{
            computing=$2
            gsub(/^[ \t\r]+|[ \t\r]+$/, "", computing)
            if (computing+0 > 0) { cnt++; sum += computing }
        }
        END {
            if (cnt > 0) printf "  Active steps: %d, avg computing=%.1f ms/step\n", cnt, sum/cnt/1000
        }' >> "${OUTPUT_FILE}"
    fi
else
    echo "(no triton profiling data in ${TRITON_DIR})" >> "${OUTPUT_FILE}"
fi

# ---- Part 2: CANN full profiling ----
section "Part 2: CANN full profiling (perf script)"
CANN_PARSE=$(latest_parse_dir "${CANN_DIR}")
if [ -n "${CANN_PARSE}" ]; then
    extract_kernel_summary "cann_full" "${CANN_PARSE}"
    extract_api_stat "cann_full" "${CANN_PARSE}"
    extract_aicore_summary "cann_full" "${CANN_PARSE}"
    extract_op_range "cann_full" "${CANN_PARSE}"
    extract_logs "cann_full" "${CANN_PARSE}"

    st_csv="${CANN_PARSE}/ASCEND_PROFILER_OUTPUT/step_trace_time.csv"
    if [ -f "${st_csv}" ]; then
        echo "" >> "${OUTPUT_FILE}"
        echo "[cann_full] Step-level timing (step_trace_time.csv)" >> "${OUTPUT_FILE}"
        echo "  ---------------------------------------------------------------------------" >> "${OUTPUT_FILE}"
        tail -n +2 "${st_csv}" | awk -F',' '{
            computing=$2
            gsub(/^[ \t\r]+|[ \t\r]+$/, "", computing)
            if (computing+0 > 0) { cnt++; sum += computing }
        }
        END {
            if (cnt > 0) printf "  Active steps: %d, avg computing=%.1f ms/step\n", cnt, sum/cnt/1000
        }' >> "${OUTPUT_FILE}"
    fi
else
    echo "(no CANN profiling data in ${CANN_DIR})" >> "${OUTPUT_FILE}"
fi

# ---- Part 3: msprof per-kernel detailed profiling ----
section "Part 3: msprof per-kernel profiling (msprof op, 5 kernels)"
MSPROF_OPPROF_DIR=$(find "${MSPROF_DIR}" -maxdepth 2 -type d -name "OPPROF_*" 2>/dev/null | head -1)
if [ -n "${MSPROF_OPPROF_DIR}" ]; then
    echo "[msprof] Data directory: ${MSPROF_OPPROF_DIR}" >> "${OUTPUT_FILE}"

    # 收集所有 kernel 的 OpBasicInfo
    echo "" >> "${OUTPUT_FILE}"
    echo "[msprof] Per-kernel timing (OpBasicInfo, avg over all instances)" >> "${OUTPUT_FILE}"
    echo "  ---------------------------------------------------------------" >> "${OUTPUT_FILE}"
    printf "  %-45s %12s %10s %8s %6s\n" "Kernel Name" "AvgDur(us)" "Block Dim" "Type" "N" >> "${OUTPUT_FILE}"

    {
        for csv in $(find "${MSPROF_OPPROF_DIR}" -name "OpBasicInfo_*.csv" -type f 2>/dev/null | sort); do
            tail -1 "${csv}"
        done
    } | awk -F',' '
    {
        name=$1; type=$2; dur=$3; blk=$4
        gsub(/^[ \t]+|[ \t]+$/, "", name)
        gsub(/^[ \t]+|[ \t]+$/, "", dur)
        gsub(/^[ \t]+|[ \t]+$/, "", blk)
        gsub(/^[ \t]+|[ \t]+$/, "", type)
        dur_num = dur + 0
        if (dur_num <= 0) next
        sum[name] += dur_num
        cnt[name]++
        blkdim[name] = blk
        typeinfo[name] = type
    }
    END {
        for (name in sum) {
            avg = sum[name] / cnt[name]
            printf "  %-45s %12.2f %10s %8s %6d\n", name, avg, blkdim[name], typeinfo[name], cnt[name]
        }
    }' >> "${OUTPUT_FILE}"

    # 汇总
    echo "" >> "${OUTPUT_FILE}"
    echo "[msprof] Kernel stage breakdown (single launch, per-chunk)" >> "${OUTPUT_FILE}"
    echo "  ---------------------------------------------------------------" >> "${OUTPUT_FILE}"
    {
        for csv in $(find "${MSPROF_OPPROF_DIR}" -name "OpBasicInfo_*.csv" -type f 2>/dev/null | sort); do
            tail -1 "${csv}"
        done
    } | awk -F',' '
    {
        name=$1; dur=$3
        gsub(/^[ \t]+|[ \t]+$/, "", name)
        gsub(/^[ \t]+|[ \t]+$/, "", dur)
        dur_num = dur + 0
        if (dur_num <= 0) next
        if (name ~ /gather_kv/)       { gs += dur_num; gc++ }
        else if (name ~ /teacher/)    { ts += dur_num; tc++ }
        else if (name ~ /indexer/)    { is += dur_num; ic++ }
        else if (name ~ /query_index/){ qs += dur_num; qc++ }
        else if (name ~ /scatter/)    { ss += dur_num; sc++ }
    }
    END {
        if (gc > 0) ga = gs / gc; if (tc > 0) ta = ts / tc
        if (ic > 0) ia = is / ic; if (qc > 0) qa = qs / qc
        if (sc > 0) sa = ss / sc
        sum_avg = ga + ta + ia + qa + sa
        if (sum_avg <= 0) exit
        printf "  %-35s %8.1f us  (%5.1f%%)\n", "gather_kv", ga, ga*100/sum_avg
        if (tc > 0) printf "  %-35s %8.1f us  (%5.1f%%)\n", "teacher_dist", ta, ta*100/sum_avg
        else         printf "  %-35s %8s  (not captured)\n", "teacher_dist", "-"
        printf "  %-35s %8.1f us  (%5.1f%%)\n", "indexer_grad_kl", ia, ia*100/sum_avg
        printf "  %-35s %8.1f us  (%5.1f%%)\n", "query_index_weight", qa, qa*100/sum_avg
        printf "  %-35s %8.1f us  (%5.1f%%)\n", "scatter_dkey", sa, sa*100/sum_avg
        printf "  %-35s %8.1f us\n", "--- per-chunk avg ---", sum_avg
    }' >> "${OUTPUT_FILE}"

    # 硬件指标（每个 kernel 的 PipeUtilization / ArithmeticUtilization）
    for csv in $(find "${MSPROF_OPPROF_DIR}" -name "OpBasicInfo_*.csv" -type f 2>/dev/null | sort); do
        kdir=$(dirname "${csv}")
        kname=$(basename "${kdir}")
        echo "" >> "${OUTPUT_FILE}"
        echo "[msprof] ${kname} hardware metrics" >> "${OUTPUT_FILE}"

        for metric in PipeUtilization ArithmeticUtilization MemoryUB Memory L2Cache; do
            mfile="${kdir}/${metric}_*.csv"
            mfile=$(ls ${mfile} 2>/dev/null | head -1)
            if [ -f "${mfile}" ]; then
                echo "  --- ${metric} (first 2 rows) ---" >> "${OUTPUT_FILE}"
                head -2 "${mfile}" | awk -F',' '{printf "    "; for(i=1;i<=NF;i++) printf "%s%s", $i, (i==NF?"\n":" | ")}' >> "${OUTPUT_FILE}" 2>/dev/null || true
            fi
        done
    done
else
    echo "(no msprof op data in ${MSPROF_DIR})" >> "${OUTPUT_FILE}"
fi

# ---- Part 4: triton vs CANN time comparison ----
section "Part 4: Triton vs CANN timing comparison"
echo "  (from perf_sli_grad_kl_loss_triton.py standard output)" >> "${OUTPUT_FILE}"
echo "  Production shape: B=1, S1/S2=4096, N1=64, D=512, Nidx1=64, D_idx=128, topK=2048" >> "${OUTPUT_FILE}"

# ---- Part 5: summary statistics ----
section "Part 5: Summary statistics"
echo "  Triton kernel stage breakdown:" >> "${OUTPUT_FILE}"

if [ -n "${MSPROF_OPPROF_DIR}" ]; then
    echo "  (from msprof op per-kernel profiling, avg over all instances)" >> "${OUTPUT_FILE}"
    echo "" >> "${OUTPUT_FILE}"
    find "${MSPROF_OPPROF_DIR}" -name "OpBasicInfo_*.csv" -type f 2>/dev/null | sort | while read -r csv; do
        tail -1 "${csv}"
    done | awk -F',' '
    {
        name=$1; dur=$3
        gsub(/^[ \t]+|[ \t]+$/, "", name)
        gsub(/^[ \t]+|[ \t]+$/, "", dur)
        dur_num = dur + 0
        if (dur_num <= 0) next

        if (name ~ /gather_kv/)            { gs += dur_num; gc++ }
        else if (name ~ /teacher/)         { ts += dur_num; tc++ }
        else if (name ~ /indexer/)         { is += dur_num; ic++ }
        else if (name ~ /query_index/)     { qs += dur_num; qc++ }
        else if (name ~ /scatter/)         { ss += dur_num; sc++ }
        total_instances++
    }
    END {
        if (gc > 0) ga = gs / gc; if (tc > 0) ta = ts / tc
        if (ic > 0) ia = is / ic; if (qc > 0) qa = qs / qc
        if (sc > 0) sa = ss / sc
        sum_avg = ga + ta + ia + qa + sa
        if (sum_avg <= 0) exit
        if (gc > 0) printf "  gather_kv:          %8.1f us  (%5.1f%%)  n=%d\n", ga, ga*100/sum_avg, gc
        if (tc > 0) printf "  teacher_dist:       %8.1f us  (%5.1f%%)  n=%d\n", ta, ta*100/sum_avg, tc
        else         printf "  teacher_dist:       (not captured)\n"
        if (ic > 0) printf "  indexer_grad_kl:    %8.1f us  (%5.1f%%)  n=%d\n", ia, ia*100/sum_avg, ic
        if (qc > 0) printf "  query_idx_weight:   %8.1f us  (%5.1f%%)  n=%d\n", qa, qa*100/sum_avg, qc
        if (sc > 0) printf "  scatter_dkey:       %8.1f us  (%5.1f%%)  n=%d\n", sa, sa*100/sum_avg, sc
        printf "  --- per-chunk avg:  %8.1f us  (single launch)\n", sum_avg
        printf "  --- 4 chunks est:   %8.1f ms  (NPU-only, per-chunk avg × 4)\n", sum_avg*4/1000
        printf "  instances: %d  (1 row per kernel launch)\n", total_instances
    }' >> "${OUTPUT_FILE}"

else
    echo "  (from kernel_details.csv, no msprof data)" >> "${OUTPUT_FILE}"
    kd="${TRITON_PARSE}/ASCEND_PROFILER_OUTPUT/kernel_details.csv"
    if [ -f "${kd}" ]; then
        echo "" >> "${OUTPUT_FILE}"
        { grep -iE "sparse|lightning|indexer|gather|scatter|teacher|kl_loss|grad|loss" "${kd}" 2>/dev/null || true; } | \
        awk -F',' '{
            name=$5; dur=$10
            gsub(/^[ \t\r]+|[ \t\r]+$/, "", name)
            gsub(/^[ \t\r]+|[ \t\r]+$/, "", dur)
            dur_num = dur + 0
            name_lower = tolower(name)
            if (dur_num > 0) {
                if (name_lower ~ /gather_kv/) total_gather += dur_num
                else if (name_lower ~ /teacher/) total_teacher += dur_num
                else if (name_lower ~ /scatter_dkey/) total_scatter += dur_num
                else if (name_lower ~ /query_index_weight/) total_qw += dur_num
                else if (name_lower ~ /indexer_grad_kl/) total_main += dur_num
                else if (name_lower ~ /cast_dkey/) total_cast += dur_num
                else total_other += dur_num
                total_all += dur_num
            }
        }
        END {
            if (total_all > 0) {
                printf "  gather_kv:          %10.1f ms  (%5.1f%%)\n", total_gather/1000, total_gather*100/total_all
                printf "  teacher_dist:       %10.1f ms  (%5.1f%%)\n", total_teacher/1000, total_teacher*100/total_all
                printf "  indexer_grad:       %10.1f ms  (%5.1f%%)\n", total_main/1000, total_main*100/total_all
                printf "  scatter_dkey:       %10.1f ms  (%5.1f%%)\n", total_scatter/1000, total_scatter*100/total_all
                printf "  query_idx_weight:   %10.1f ms  (%5.1f%%)\n", total_qw/1000, total_qw*100/total_all
                printf "  cast_dkey:          %10.1f ms  (%5.1f%%)\n", total_cast/1000, total_cast*100/total_all
                printf "  other:              %10.1f ms  (%5.1f%%)\n", total_other/1000, total_other*100/total_all
                printf "  TOTAL:              %10.1f ms\n", total_all/1000
            }
        }' >> "${OUTPUT_FILE}"
    fi
fi

echo "" >> "${OUTPUT_FILE}"
echo "Report complete" >> "${OUTPUT_FILE}"

echo ""
echo "================================================"
echo "Full text report written to: ${OUTPUT_FILE}"
echo "================================================"
