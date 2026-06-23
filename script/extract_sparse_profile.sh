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
    find "${base}" -maxdepth 1 -name "syn-*" -type d 2>/dev/null | sort -r | head -1 | tr -d '\n'
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

# ---- Part 3: msprof full detailed profiling ----
section "Part 3: msprof full detailed profiling (msprof op)"
MSPROF_OPPROF=$(find "${MSPROF_DIR}" -maxdepth 2 -name "OpBasicInfo.csv" -type f 2>/dev/null | head -1)
if [ -n "${MSPROF_OPPROF}" ]; then
    opprof_dir=$(dirname "${MSPROF_OPPROF}")
    echo "[msprof] Data directory: ${opprof_dir}" >> "${OUTPUT_FILE}"

    echo "" >> "${OUTPUT_FILE}"
    echo "[msprof] Kernel timing (OpBasicInfo.csv)" >> "${OUTPUT_FILE}"
    echo "  ---------------------------------------------------------------------------" >> "${OUTPUT_FILE}"
    {
        head -1 "${MSPROF_OPPROF}"
        tail -n +2 "${MSPROF_OPPROF}"
    } | awk -F',' '
    NR==1 {
        for(i=1;i<=NF;i++) {
            gsub(/^[ \t]+|[ \t]+$/, "", $i)
            col[$i]=i
        }
        printf "  %-50s %12s\n", "Op Name", "Duration(us)"
        next
    }
    {
        name=$col["Op Name"]; dur=$col["Task Duration(us)"]
        gsub(/^[ \t]+|[ \t]+$/, "", name)
        gsub(/^[ \t]+|[ \t]+$/, "", dur)
        if (dur+0 > 0) printf "  %-50s %12s\n", name, dur
    }' 2>/dev/null | sort -t$'\t' -k2 -rn | head -30 >> "${OUTPUT_FILE}"

    for csv in ArithmeticUtilization PipeUtilization MemoryUB Memory; do
        csv_path="${opprof_dir}/${csv}.csv"
        if [ -f "${csv_path}" ]; then
            echo "" >> "${OUTPUT_FILE}"
            echo "[msprof] ${csv}.csv overview (first block)" >> "${OUTPUT_FILE}"
            head -2 "${csv_path}" >> "${OUTPUT_FILE}" || true
        fi
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
if [ -n "${TRITON_PARSE}" ]; then
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
