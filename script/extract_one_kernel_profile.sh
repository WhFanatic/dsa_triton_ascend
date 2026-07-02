#!/bin/bash
# ============================================================================
# 单算子 profiling 全量 CSV 提取
#
# Usage:
#   ./script/extract_one_kernel_profile.sh <kernel_name> [msprof_dir] [output_file]
#
# 输入：profile_sparse_detail.sh 产出的 ${MSPROF_DIR}/<kernel>/OPPROF_*/
#   或直接传单个 OPPROF_* 目录路径作为 msprof_dir。
#
# 输出：把该 kernel 的所有 profiling CSV 原样落到一个文本文件，便于后续优化
#   分析时检索 block 级原始指标。覆盖 msprof op 默认产出的全部 CSV：
#     OpBasicInfo / PipeUtilization / ArithmeticUtilization /
#     MemoryUB / Memory / MemoryL0 / L2Cache / ResourceConflictRatio
#   其他 CSV（若存在）也会一并落盘。
# ============================================================================

set -u

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <kernel_name> [msprof_dir] [output_file]"
    echo "  kernel_name : 如 _scatter_dkey_index_kernel"
    echo "  msprof_dir  : profile_sparse_detail.sh 输出根目录"
    echo "                （默认 ./profiler_data_sli_detail），"
    echo "                或直接传单个 OPPROF_* 目录"
    echo "  output_file : 默认 ./<kernel>_profile_raw.txt"
    exit 1
fi

KERNEL="$1"
MSPROF_DIR="${2:-./profiler_data_sli_detail}"
OUTPUT_FILE="${3:-./${KERNEL}_profile_raw.txt}"

# 定位 OPPROF_* 目录：
#   1) 若 MSPROF_DIR 自身就是 OPPROF_*，直接用
#   2) 否则尝试 MSPROF_DIR/<kernel>/OPPROF_*
#   3) 再退一步在 MSPROF_DIR 下全局搜含 kernel 名字的 OPPROF_*
locate_opprof() {
    local root="$1"
    local kn="$2"
    if [ -d "${root}" ] && basename "${root}" | grep -q "^OPPROF_"; then
        echo "${root}"
        return
    fi
    local d
    d=$(find "${root}/${kn}" -maxdepth 1 -type d -name "OPPROF_*" 2>/dev/null | sort | head -1)
    if [ -n "${d}" ]; then echo "${d}"; return; fi
    d=$(find "${root}" -type d -name "OPPROF_*" 2>/dev/null \
        | while read -r p; do
            if grep -q "${kn}" "${p}/OpBasicInfo.csv" 2>/dev/null; then echo "${p}"; fi
        done | head -1)
    echo "${d}"
}

OPPROF=$(locate_opprof "${MSPROF_DIR}" "${KERNEL}")
if [ -z "${OPPROF}" ] || [ ! -d "${OPPROF}" ]; then
    echo "ERROR: 未在 ${MSPROF_DIR} 下找到 ${KERNEL} 对应的 OPPROF_* 目录"
    echo "  建议先跑 ./script/profile_sparse_detail.sh 重新采集"
    exit 2
fi

echo "================================================"
echo "Kernel       : ${KERNEL}"
echo "OPPROF dir   : ${OPPROF}"
echo "Output file  : ${OUTPUT_FILE}"
echo "================================================"

: > "${OUTPUT_FILE}"
{
    echo "################################################################################"
    echo "# Kernel profiling raw CSV dump"
    echo "# kernel    : ${KERNEL}"
    echo "# opprof    : ${OPPROF}"
    echo "# generated : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################################"
} >> "${OUTPUT_FILE}"

# 已知的 msprof op 标准 CSV，按习惯顺序优先；其余 *.csv 兜底追加，保证不漏
ORDERED=(
    OpBasicInfo
    PipeUtilization
    ArithmeticUtilization
    MemoryUB
    Memory
    MemoryL0
    L2Cache
    ResourceConflictRatio
)

dumped=""
for name in "${ORDERED[@]}"; do
    f="${OPPROF}/${name}.csv"
    if [ -f "${f}" ]; then
        {
            echo ""
            echo "################################################################################"
            echo "# ${name}.csv"
            echo "################################################################################"
            cat "${f}"
        } >> "${OUTPUT_FILE}"
        dumped="${dumped} ${name}.csv"
    else
        {
            echo ""
            echo "# ${name}.csv (missing)"
        } >> "${OUTPUT_FILE}"
    fi
done

# 兜底：捕获不在 ORDERED 列表中的其它 CSV
find "${OPPROF}" -maxdepth 1 -type f -name "*.csv" | sort | while read -r f; do
    base=$(basename "${f}")
    skip=0
    for name in "${ORDERED[@]}"; do
        if [ "${base}" = "${name}.csv" ]; then skip=1; break; fi
    done
    [ "${skip}" -eq 1 ] && continue
    {
        echo ""
        echo "################################################################################"
        echo "# ${base}"
        echo "################################################################################"
        cat "${f}"
    } >> "${OUTPUT_FILE}"
    dumped="${dumped} ${base}"
done

echo ""
echo "dumped:${dumped}"
echo "Raw profile written to: ${OUTPUT_FILE}"
