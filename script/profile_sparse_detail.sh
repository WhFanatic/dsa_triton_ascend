#!/bin/bash
# ============================================================================
# Sparse 算子 Triton 详细 Profiling
#
# 用法: ./script/profile_sparse_detail.sh [device_id] [output_dir]
#
# 一次采集即可获取:
#   - 全量 kernel timing (kernel_details.csv)
#   - API 调用统计 (api_statistic.csv)
#   - MindSpore 算子耗时范围 (op_range)
#   - AICore 硬件指标 (需要在 Python 侧配置 aic_metrics)
#
# 依赖: msprof (CANN toolkit)
# ============================================================================
set -euo pipefail

DEVICE_ID="${1:-0}"
OUT_DIR="${2:-./profiler_data_sli_detail}"
export ASCEND_RT_VISIBLE_DEVICES="${DEVICE_ID}"
export TRITON_END=mindspore
export TRITON_BACKEND=mindspore
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TRITON_CACHE_DIR=./my_triton_cache

# 启用 stage 级同步，使 msprof 能准确拆分各 kernel 耗时
export SLI_SYNC=1
# 启用 profiling 标记 kernel，用于定位各 stage
export SPARSE_GRAD_PROFILE_MARKERS=1

KERNEL_ONLY_SCRIPT="perf_sli_grad_kl_loss_triton.py"

echo "================================================"
echo "Sparse 算子 Triton 详细 Profiling"
echo "NPU 设备: ${DEVICE_ID}"
echo "输出目录: ${OUT_DIR}"
echo "================================================"

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

# 一次全量采集，获取所有 kernel 的 timing + API 统计
echo ""
echo ">>> 全量 kernel profiling ..."
msprof op --output="${OUT_DIR}" \
    python "${KERNEL_ONLY_SCRIPT}" --kernel-only

echo ""
echo "================================================"
echo "Profiling 完成"
echo "输出目录: ${OUT_DIR}"
echo ""
echo "关键文件 (在 OPPROF_* 子目录下):"
echo "  OpBasicInfo.csv           — 各 kernel 名称/耗时/block dim"
echo "  ArithmeticUtilization.csv — 算术利用率 (按 block)"
echo "  PipeUtilization.csv       — 流水线利用率 (按 block)"
echo "  Memory.csv                — 内存带宽"
echo "  MemoryUB.csv              — UB 使用量"
echo "  L2Cache.csv               — L2 缓存命中率"
echo "================================================"
