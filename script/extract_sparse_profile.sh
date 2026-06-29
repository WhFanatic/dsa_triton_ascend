#!/bin/bash
# ============================================================================
# Sparse operator profiling full text extraction
#
# Usage: ./script/extract_sparse_profile.sh [triton_dir] [cann_dir] [msprof_dir] [output_file]
#
# Consumes the output of:
#   - script/profile_sparse.sh         -> ${TRITON_DIR}, ${CANN_DIR}
#       contains syn-*/ASCEND_PROFILER_OUTPUT/{kernel_details,api_statistic,
#       step_trace_time}.csv
#   - script/profile_sparse_detail.sh  -> ${MSPROF_DIR}
#       contains <kernel>/OPPROF_*/{OpBasicInfo,PipeUtilization,
#       ArithmeticUtilization,MemoryUB,L2Cache}.csv (one OPPROF dir per kernel)
#
# Produces a single plain-text report grouped into:
#   Part 1 Triton kernel timing (aggregated by kernel name, top by total)
#   Part 2 CANN  kernel timing
#   Part 3 Triton vs CANN side-by-side (step / chunk / sum)
#   Part 4 Host-side API statistics (acl* / aclnn* etc.)
#   Part 5 msprof per-kernel hardware metrics (pipe / arithmetic / memory)
#   Part 6 Final summary (per-chunk stage breakdown + total)
# ============================================================================

set -u

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

: > "${OUTPUT_FILE}"

section() {
    {
        echo ""
        echo "################################################################################"
        echo "# $*"
        echo "################################################################################"
    } >> "${OUTPUT_FILE}"
}

# Locate the latest ASCEND_PROFILER_OUTPUT dir under a profile_sparse.sh output root.
latest_parse_dir() {
    local base="$1"
    local d
    d=$(find "${base}" -maxdepth 1 -mindepth 1 -type d -name "syn-*"        2>/dev/null | sort -r | head -1)
    [ -z "${d}" ] && d=$(find "${base}" -maxdepth 1 -mindepth 1 -type d -name "*_ascend_ms" 2>/dev/null | sort -r | head -1)
    echo "${d}"
}

# Aggregate kernel_details.csv by kernel basename:
#   - bare kernel name from "Name" column (strip 'Kernel::KernelLaunch::Default/.../')
#   - sum / count / avg by name
#   - sort desc by total
extract_kernel_details() {
    local label="$1"
    local parse_dir="$2"
    local csv="${parse_dir}/ASCEND_PROFILER_OUTPUT/kernel_details.csv"

    if [ ! -f "${csv}" ]; then
        echo "[${label}] kernel_details.csv not found at ${csv}" >> "${OUTPUT_FILE}"
        return
    fi

    {
        echo "[${label}] Kernel timing (aggregated by kernel name)"
        echo "  source: ${csv}"
        echo "  ----------------------------------------------------------------------------------"
        printf "  %-50s %4s %12s %12s\n" "Kernel" "N" "TotalDur(us)" "AvgDur(us)"
    } >> "${OUTPUT_FILE}"

    awk -F',' '
    NR==1 {
        for (i=1; i<=NF; i++) {
            v=$i; gsub(/^[ \t\r]+|[ \t\r]+$/, "", v); col[v]=i
        }
        nameC = col["Name"]; durC = col["Duration(us)"]
        next
    }
    {
        name=$nameC; dur=$durC
        gsub(/^[ \t\r]+|[ \t\r]+$/, "", name)
        gsub(/^[ \t\r]+|[ \t\r]+$/, "", dur)
        # strip "Kernel::KernelLaunch::Default/.../" prefix to keep bare kernel name
        sub(/.*\//, "", name)
        d = dur + 0
        if (name == "" || d <= 0) next
        sum[name] += d
        cnt[name]++
    }
    END {
        # print as sortable lines: <total>\t<avg>\t<n>\t<name>
        for (n in sum)
            printf "%.3f\t%.3f\t%d\t%s\n", sum[n], sum[n]/cnt[n], cnt[n], n
    }' "${csv}" \
    | sort -t$'\t' -k1 -rn \
    | awk -F'\t' '{ printf "  %-50s %4d %12.2f %12.2f\n", $4, $3, $1, $2 }' \
    >> "${OUTPUT_FILE}"
}

# api_statistic.csv columns:
#   Device_id,Level,API Name,Time(us),Count,Avg(us),Min(us),Max(us),Variance
extract_api_stat() {
    local label="$1"
    local parse_dir="$2"
    local csv="${parse_dir}/ASCEND_PROFILER_OUTPUT/api_statistic.csv"

    if [ ! -f "${csv}" ]; then
        echo "[${label}] api_statistic.csv not found" >> "${OUTPUT_FILE}"
        return
    fi

    {
        echo ""
        echo "[${label}] Host API statistics (top 15 by total time)"
        echo "  source: ${csv}"
        echo "  --------------------------------------------------------------------------"
        printf "  %-45s %6s %14s %10s\n" "API Name" "Count" "Total(us)" "Avg(us)"
    } >> "${OUTPUT_FILE}"

    awk -F',' '
    NR==1 {
        for (i=1; i<=NF; i++) {
            v=$i; gsub(/^[ \t\r]+|[ \t\r]+$/, "", v); col[v]=i
        }
        nameC=col["API Name"]; totC=col["Time(us)"]; cntC=col["Count"]; avgC=col["Avg(us)"]
        next
    }
    {
        name=$nameC; tot=$totC; cnt=$cntC; avg=$avgC
        gsub(/^[ \t\r]+|[ \t\r]+$/, "", name)
        gsub(/^[ \t\r]+|[ \t\r]+$/, "", tot)
        gsub(/^[ \t\r]+|[ \t\r]+$/, "", cnt)
        gsub(/^[ \t\r]+|[ \t\r]+$/, "", avg)
        if (name == "" || (tot + 0) <= 0) next
        printf "%.3f\t%d\t%.3f\t%s\n", tot+0, cnt+0, avg+0, name
    }' "${csv}" \
    | sort -t$'\t' -k1 -rn \
    | head -15 \
    | awk -F'\t' '{ printf "  %-45s %6d %14.2f %10.2f\n", $4, $2, $1, $3 }' \
    >> "${OUTPUT_FILE}"
}

# step_trace_time.csv columns (looked up by header):
#   Step,Computing,Communication(Not Overlapped),Overlapped,Communication,Free,...
# Aggregate "Computing > 0" rows.
#
# Output mode (selected by $3):
#   "block"  -> indented multi-line, used for Part 1/2 detail blocks
#   "row"    -> single fixed-width table row, used for Part 3 side-by-side
#
# Note on `avg free`: profiler reports Free of the last active step as the
# residual gap until profile stop, which is often much shorter than the true
# inter-step idle time. We compute two means: one over all active steps (raw),
# one excluding the last active step when there are >=2 active steps (stable).
# When only 1 active step exists we report the raw value with a note.
aggregate_step_trace() {
    local label="$1"
    local parse_dir="$2"
    local mode="${3:-block}"
    local csv="${parse_dir}/ASCEND_PROFILER_OUTPUT/step_trace_time.csv"

    if [ ! -f "${csv}" ]; then
        if [ "${mode}" = "row" ]; then
            printf "  %-10s %16s %16s %16s\n" "${label}" "-" "-" "-" >> "${OUTPUT_FILE}"
        fi
        return
    fi

    awk -F',' -v lab="${label}" -v mode="${mode}" '
    NR==1 {
        for (i=1; i<=NF; i++) {
            v=$i; gsub(/^[ \t\r]+|[ \t\r]+$/, "", v); col[v]=i
        }
        cC = col["Computing"]; fC = col["Free"]
        if (!cC || !fC) {
            printf "  %-10s (Computing/Free columns missing in step_trace_time.csv)\n", lab
            exit
        }
        next
    }
    {
        c=$cC; f=$fC
        gsub(/^[ \t\r]+|[ \t\r]+$/, "", c)
        gsub(/^[ \t\r]+|[ \t\r]+$/, "", f)
        if ((c + 0) > 0) {
            cnt++
            comp[cnt] = c + 0
            freev[cnt] = f + 0
            csum += c + 0
            fsum += f + 0
        }
    }
    END {
        if (cnt == 0) {
            if (mode == "row")
                printf "  %-10s %16s %16s %16s\n", lab, "-", "-", "-"
            else
                printf "  Active steps: 0\n"
            exit
        }

        avg_c = csum / cnt / 1000     # ms
        avg_f = fsum / cnt / 1000

        # Stable mean of Free: drop last active step when we have >=2 steps,
        # because the last steps Free is typically truncated by profile stop.
        if (cnt >= 2) {
            fsum_stable = fsum - freev[cnt]
            avg_f_stable = fsum_stable / (cnt - 1) / 1000
            free_note = sprintf("avg free=%.2f ms/step (stable, last step %.2f ms dropped)",
                                avg_f_stable, freev[cnt]/1000)
        } else {
            avg_f_stable = avg_f
            free_note = sprintf("avg free=%.2f ms/step (1 step only; may be truncated)", avg_f)
        }

        if (mode == "row") {
            printf "  %-10s %16d %16.2f %16.2f\n",
                   lab, cnt, avg_c, avg_f_stable
        } else {
            printf "\n[%s] Step-level timing (step_trace_time.csv)\n", lab
            printf "  --------------------------------------------------------------------------\n"
            printf "  Active steps: %d   avg computing=%.2f ms/step   %s\n",
                   cnt, avg_c, free_note
        }
    }' "${csv}" >> "${OUTPUT_FILE}"
}

extract_step_trace() { aggregate_step_trace "$1" "$2" "block"; }
print_step_row()     { aggregate_step_trace "$1" "$2" "row"; }

# msprof per-kernel: ${MSPROF_DIR}/<kernel>/OPPROF_*/{OpBasicInfo,PipeUtilization,...}.csv
# Returns list of "kernel_subdir<TAB>opprof_dir" pairs on stdout.
list_msprof_runs() {
    local base="$1"
    [ -d "${base}" ] || return

    find "${base}" -mindepth 2 -maxdepth 2 -type d -name "OPPROF_*" 2>/dev/null | sort \
    | while read -r d; do
        # parent of OPPROF_* is the kernel subdir
        printf "%s\t%s\n" "$(basename "$(dirname "${d}")")" "${d}"
    done
}

# OpBasicInfo.csv columns:
#   Op Name,Op Type,Task Duration(us),Block Dim,Mix Block Dim,Device Id,...
extract_msprof_basic() {
    local opprof="$1"
    local csv="${opprof}/OpBasicInfo.csv"
    [ -f "${csv}" ] || return

    awk -F',' 'NR>1 {
        name=$1; type=$2; dur=$3; blk=$4; mix=$5
        gsub(/^[ \t\r]+|[ \t\r]+$/, "", name)
        gsub(/^[ \t\r]+|[ \t\r]+$/, "", type)
        gsub(/^[ \t\r]+|[ \t\r]+$/, "", dur)
        gsub(/^[ \t\r]+|[ \t\r]+$/, "", blk)
        gsub(/^[ \t\r]+|[ \t\r]+$/, "", mix)
        if (name == "" || (dur + 0) <= 0) next
        printf "%s\t%s\t%.3f\t%s\t%s\n", name, type, dur+0, blk, mix
    }' "${csv}"
}

# Average a column (by header name) across all data rows of a CSV.
# Empty / NA / non-numeric cells are skipped. Optional `scale` multiplier
# (defaults to 1.0) is applied to the final mean — pass 100 to render
# 0-1 ratios as percentages.
csv_avg_col() {
    local csv="$1"
    local colname="$2"
    local scale="${3:-1}"
    [ -f "${csv}" ] || { echo "-"; return; }
    awk -F',' -v want="${colname}" -v scale="${scale}" '
    NR==1 {
        for (i=1; i<=NF; i++) {
            v=$i; gsub(/^[ \t\r]+|[ \t\r]+$/, "", v)
            if (v == want) { ci = i; break }
        }
        if (!ci) { print "-"; exit }
        next
    }
    {
        v=$ci; gsub(/^[ \t\r]+|[ \t\r]+$/, "", v)
        if (v == "" || v == "NA") next
        if (v ~ /^-?[0-9.]+([eE][+-]?[0-9]+)?$/) { s += v + 0; n++ }
    }
    END {
        if (n > 0) printf "%.2f", (s/n) * scale
        else        print "-"
    }' "${csv}"
}


########################################################################
# Body
########################################################################

section "Sparse operator profiling report"
echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')" >> "${OUTPUT_FILE}"
echo "TRITON_DIR=${TRITON_DIR}"  >> "${OUTPUT_FILE}"
echo "CANN_DIR=${CANN_DIR}"      >> "${OUTPUT_FILE}"
echo "MSPROF_DIR=${MSPROF_DIR}"  >> "${OUTPUT_FILE}"

# Detect the number of S1 chunks per backward call. This drives the EST
# multiplier in Part 6 (one msprof per-kernel Duration covers one chunk;
# a full backward issues N chunks worth of kernel launches).
#
# Priority:
#   1. env SPARSE_GRAD_S1_CHUNK_OVERRIDE  -> trust user
#   2. source .py SPARSE_GRAD_S1_CHUNK    -> ceil(S1 / chunk)
#   3. Part 1 kernel_details: most-frequent N among 5 SLI grad kernels
#   4. fallback: 1
SLI_PY="$(dirname "$0")/../sparse_lightning_indexer_grad_kl_loss_triton.py"
PRODUCTION_S1="${PRODUCTION_S1:-4096}"

NUM_CHUNKS=""
if [ -n "${SPARSE_GRAD_S1_CHUNK_OVERRIDE:-}" ]; then
    NUM_CHUNKS="${SPARSE_GRAD_S1_CHUNK_OVERRIDE}"
    CHUNK_SRC="env override"
elif [ -f "${SLI_PY}" ]; then
    CHUNK_SIZE=$(awk -F'=' '/^SPARSE_GRAD_S1_CHUNK[[:space:]]*=/ {
        gsub(/[[:space:]]/, "", $2); print $2; exit
    }' "${SLI_PY}")
    if [ -n "${CHUNK_SIZE}" ] && [ "${CHUNK_SIZE}" -gt 0 ] 2>/dev/null; then
        NUM_CHUNKS=$(( (PRODUCTION_S1 + CHUNK_SIZE - 1) / CHUNK_SIZE ))
        CHUNK_SRC="SPARSE_GRAD_S1_CHUNK=${CHUNK_SIZE} from ${SLI_PY}, S1=${PRODUCTION_S1}"
    fi
fi
echo "EST chunk factor: NUM_CHUNKS=${NUM_CHUNKS:-(deferred to Part 1)}  source=${CHUNK_SRC:-(none yet)}" >> "${OUTPUT_FILE}"

# --- Part 1: Triton (perf script) -------------------------------------
section "Part 1: Triton end-to-end profiling (profile_sparse.sh triton)"
TRITON_PARSE=$(latest_parse_dir "${TRITON_DIR}")
if [ -n "${TRITON_PARSE}" ]; then
    extract_kernel_details "triton" "${TRITON_PARSE}"
    extract_step_trace     "triton" "${TRITON_PARSE}"
else
    echo "(no triton profiling data in ${TRITON_DIR})" >> "${OUTPUT_FILE}"
fi

# Fallback chunk detection: if .py probe failed, count how many times any of
# the 5 SLI grad kernels was launched in kernel_details (one launch per chunk).
if [ -z "${NUM_CHUNKS}" ] && [ -n "${TRITON_PARSE}" ]; then
    KCSV="${TRITON_PARSE}/ASCEND_PROFILER_OUTPUT/kernel_details.csv"
    if [ -f "${KCSV}" ]; then
        NUM_CHUNKS=$(awk -F',' '
        NR==1 { for (i=1; i<=NF; i++) { v=$i; gsub(/^[ \t\r]+|[ \t\r]+$/, "", v); col[v]=i }
                nameC=col["Name"]; next }
        {
            name=$nameC; sub(/.*\//, "", name)
            if (name ~ /_teacher_indexer_kl_kernel/)         cnt["teacher_indexer_kl"]++
            else if (name ~ /_indexer_grad_kernel/)          cnt["indexer_grad"]++
        }
        END {
            n = 0
            for (k in cnt) if (cnt[k] > n) n = cnt[k]   # most frequent
            print n+0
        }' "${KCSV}")
        if [ -n "${NUM_CHUNKS}" ] && [ "${NUM_CHUNKS}" -gt 0 ] 2>/dev/null; then
            CHUNK_SRC="kernel_details.csv max(launches per SLI grad kernel)"
        else
            NUM_CHUNKS=1
            CHUNK_SRC="fallback (no SLI grad kernel launches found)"
        fi
    fi
fi
[ -z "${NUM_CHUNKS}" ] && { NUM_CHUNKS=1; CHUNK_SRC="fallback (no source)"; }

# --- Part 2: CANN (perf script) ---------------------------------------
section "Part 2: CANN end-to-end profiling (profile_sparse.sh cann)"
CANN_PARSE=$(latest_parse_dir "${CANN_DIR}")
if [ -n "${CANN_PARSE}" ]; then
    extract_kernel_details "cann" "${CANN_PARSE}"
    extract_step_trace     "cann" "${CANN_PARSE}"
else
    echo "(no CANN profiling data in ${CANN_DIR})" >> "${OUTPUT_FILE}"
fi

# --- Part 3: side-by-side timing --------------------------------------
section "Part 3: Triton vs CANN side-by-side"
{
    echo "  Production shape: B=1, S1/S2=4096, N1=64, D=512, Nidx1=64, D_idx=128, topK=2048"
    echo "  (active steps from step_trace_time.csv; chunk = SPARSE_GRAD_S1_CHUNK split of S1)"
    echo "  -------------------------------------------------------------------------------"
    printf "  %-10s %16s %16s %16s\n" "side" "active steps" "avg comp(ms)" "avg free(ms)"
} >> "${OUTPUT_FILE}"

print_step_row() { aggregate_step_trace "$1" "$2" "row"; }
[ -n "${TRITON_PARSE}" ] && print_step_row "triton" "${TRITON_PARSE}"
[ -n "${CANN_PARSE}"   ] && print_step_row "cann"   "${CANN_PARSE}"

# --- Part 4: host-side API statistics ---------------------------------
section "Part 4: Host-side API statistics"
[ -n "${TRITON_PARSE}" ] && extract_api_stat "triton" "${TRITON_PARSE}"
[ -n "${CANN_PARSE}"   ] && extract_api_stat "cann"   "${CANN_PARSE}"

# --- Part 5: msprof per-kernel ----------------------------------------
section "Part 5: msprof per-kernel profiling (profile_sparse_detail.sh)"

MSPROF_RUNS=$(list_msprof_runs "${MSPROF_DIR}")
if [ -z "${MSPROF_RUNS}" ]; then
    echo "(no msprof op data under ${MSPROF_DIR})" >> "${OUTPUT_FILE}"
else
    {
        echo "[msprof] Per-kernel basic info (one msprof op run per kernel)"
        echo "  ------------------------------------------------------------------------------------------"
        printf "  %-45s %8s %12s %10s %10s\n" "Op Name" "Type" "Dur(us)" "BlockDim" "MixBlk"
    } >> "${OUTPUT_FILE}"

    # collect basic rows so we can also use them for the summary section
    BASIC_TSV=$(mktemp)
    printf "%s\n" "${MSPROF_RUNS}" | while IFS=$'\t' read -r sub opprof; do
        extract_msprof_basic "${opprof}" | while IFS=$'\t' read -r name type dur blk mix; do
            printf "%s\t%s\t%s\t%s\t%s\n" "${name}" "${type}" "${dur}" "${blk}" "${mix}" >> "${BASIC_TSV}"
            printf "  %-45s %8s %12.2f %10s %10s\n" "${name}" "${type}" "${dur}" "${blk}" "${mix}" >> "${OUTPUT_FILE}"
        done
    done

    # per-kernel hardware metrics (avg ratios across all blocks)
    {
        echo ""
        echo "[msprof] Per-kernel hardware utilization (avg over all blocks)"
        echo "  ------------------------------------------------------------------------------------------"
        printf "  %-42s %8s %8s %8s %8s %8s %8s %8s\n" \
            "Op Name" "cube%" "aic_sca%" "aic_mte2%" "vec%" "aiv_sca%" "aiv_mte2%" "aiv_mte3%"
    } >> "${OUTPUT_FILE}"

    printf "%s\n" "${MSPROF_RUNS}" | while IFS=$'\t' read -r sub opprof; do
        pipe="${opprof}/PipeUtilization.csv"
        [ -f "${pipe}" ] || continue
        # bare kernel name from OpBasicInfo
        kname=$(awk -F',' 'NR==2 {gsub(/^[ \t\r]+|[ \t\r]+$/, "", $1); print $1}' "${opprof}/OpBasicInfo.csv")
        [ -z "${kname}" ] && kname="${sub}"
        # ratios are stored as 0-1 fractions, multiply by 100 for percent display
        cube=$(csv_avg_col   "${pipe}" "aic_cube_ratio"    100)
        ascal=$(csv_avg_col  "${pipe}" "aic_scalar_ratio"  100)
        amte2=$(csv_avg_col  "${pipe}" "aic_mte2_ratio"    100)
        vec=$(csv_avg_col    "${pipe}" "aiv_vec_ratio"     100)
        vscal=$(csv_avg_col  "${pipe}" "aiv_scalar_ratio"  100)
        vmte2=$(csv_avg_col  "${pipe}" "aiv_mte2_ratio"    100)
        vmte3=$(csv_avg_col  "${pipe}" "aiv_mte3_ratio"    100)
        printf "  %-42s %8s %8s %8s %8s %8s %8s %8s\n" \
            "${kname}" "${cube}" "${ascal}" "${amte2}" \
            "${vec}" "${vscal}" "${vmte2}" "${vmte3}" >> "${OUTPUT_FILE}"
    done

    # NOTE: 更细粒度的全 CSV 采集（ArithmeticUtilization / Memory* / L2Cache /
    # ResourceConflictRatio 等）由 script/extract_one_kernel_profile.sh 提供，
    # 针对单一 kernel 输出完整原始数据。Part 5 只给所有 kernel 的概览。

    # --- Part 6: summary ----------------------------------------------
    section "Part 6: Summary"
    {
        echo "  Per-chunk stage breakdown (msprof single-launch duration per kernel)"
        echo "  -------------------------------------------------------------------"
    } >> "${OUTPUT_FILE}"

    # categorise kernels into 5 logical stages by name substring match
    awk -F'\t' -v num_chunks="${NUM_CHUNKS}" -v chunk_src="${CHUNK_SRC}" '
    function tag(n) {
        if (n ~ /teacher_indexer_kl/) return "teacher_indexer_kl"
        if (n ~ /indexer_grad/)       return "indexer_grad"
        return ""
    }
    {
        t = tag($1)
        if (t == "") next
        sum[t] += $3
        cnt[t]++
        name[t] = $1
    }
    END {
        order[1]="teacher_indexer_kl"; order[2]="indexer_grad"
        total = 0
        for (i=1; i<=2; i++) if (cnt[order[i]] > 0) total += sum[order[i]]/cnt[order[i]]
        if (total <= 0) { print "  (no kernels matched)"; exit }
        for (i=1; i<=2; i++) {
            t = order[i]
            if (cnt[t] > 0) {
                avg = sum[t]/cnt[t]
                printf "  %-22s %10.2f us  (%5.1f%%)  n=%d  name=%s\n",
                       t, avg, avg*100/total, cnt[t], name[t]
            } else {
                printf "  %-22s %10s  (not captured)\n", t, "-"
            }
        }
        nc = num_chunks + 0; if (nc < 1) nc = 1
        printf "  %-22s %10.2f us\n", "PER-CHUNK SUM", total
        printf "  %-22s %10.2f ms  (per-chunk sum x %d chunks; source: %s)\n",
               "EST " nc "-CHUNK NPU-TIME", total*nc/1000, nc, chunk_src
        printf "  note: EST excludes inter-kernel launch gaps and host sync;\n"
        printf "        compare with step_trace avg/step below for the true device-busy time per backward.\n"
    }' "${BASIC_TSV}" >> "${OUTPUT_FILE}"

    # End-to-end comparison (using kernel_details totals from Part 1/2)
    if [ -n "${TRITON_PARSE}" ] && [ -n "${CANN_PARSE}" ]; then
        {
            echo ""
            echo "  End-to-end NPU time (sum of kernel Duration(us) per active step)"
            echo "  -------------------------------------------------------------------"
        } >> "${OUTPUT_FILE}"

        for label in triton cann; do
            if [ "${label}" = "triton" ]; then csv="${TRITON_PARSE}/ASCEND_PROFILER_OUTPUT/kernel_details.csv"
            else                                csv="${CANN_PARSE}/ASCEND_PROFILER_OUTPUT/kernel_details.csv"
            fi
            [ -f "${csv}" ] || continue
            awk -F',' -v lab="${label}" '
            NR==1 {
                for (i=1; i<=NF; i++) { v=$i; gsub(/^[ \t\r]+|[ \t\r]+$/, "", v); col[v]=i }
                durC=col["Duration(us)"]; stepC=col["Step ID"]
                next
            }
            {
                d=$durC; s=$stepC
                gsub(/^[ \t\r]+|[ \t\r]+$/, "", d)
                gsub(/^[ \t\r]+|[ \t\r]+$/, "", s)
                if ((d + 0) > 0) {
                    total += d + 0
                    if (s != "") {
                        if (!(s in steps)) { steps[s]=1; sn++ }
                        per_step[s] += d + 0
                    }
                }
            }
            END {
                if (sn == 0) sn = 1
                pmin = -1; pmax = -1
                for (k in per_step) {
                    v = per_step[k]
                    if (pmin < 0 || v < pmin) pmin = v
                    if (pmax < 0 || v > pmax) pmax = v
                }
                printf "  %-10s sum=%10.2f ms   steps=%d   avg/step=%10.2f ms   min/max per step=%.2f / %.2f ms\n",
                       lab, total/1000, sn, total/sn/1000,
                       (pmin<0?0:pmin/1000), (pmax<0?0:pmax/1000)
            }' "${csv}" >> "${OUTPUT_FILE}"
        done
    fi

    rm -f "${BASIC_TSV}"
fi

echo "" >> "${OUTPUT_FILE}"
echo "Report complete" >> "${OUTPUT_FILE}"

echo ""
echo "================================================"
echo "Full text report written to: ${OUTPUT_FILE}"
echo "================================================"
