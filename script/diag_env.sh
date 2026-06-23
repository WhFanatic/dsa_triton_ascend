#!/bin/bash
# ============================================================================
# NPU memory environment diagnostic script
# Usage: ./script/diag_env.sh [device_id]
#
# Collects information to diagnose VMM handle exhaustion / OOM on Ascend NPU.
# Works on both A2 and A3 servers.
# ============================================================================

DEVICE_ID="${1:-0}"
OUTPUT="diag_env_device${DEVICE_ID}.log"

echo "NPU Memory Diagnostic - device ${DEVICE_ID}" | tee "${OUTPUT}"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${OUTPUT}"
echo "Host: $(cat /proc/sys/kernel/hostname 2>/dev/null || uname -n)" | tee -a "${OUTPUT}"
echo "" | tee -a "${OUTPUT}"

section() {
    echo "" | tee -a "${OUTPUT}"
    echo "========================================" | tee -a "${OUTPUT}"
    echo "=== $*" | tee -a "${OUTPUT}"
    echo "========================================" | tee -a "${OUTPUT}"
}

section "1. NPU hardware info"
npu-smi info -t board -i "${DEVICE_ID}" 2>&1 | tee -a "${OUTPUT}"
echo "" | tee -a "${OUTPUT}"
npu-smi info -t product -i "${DEVICE_ID}" 2>&1 | tee -a "${OUTPUT}"

section "2. NPU memory usage"
echo ">>> HBM usage:" | tee -a "${OUTPUT}"
npu-smi info -t usages -i "${DEVICE_ID}" 2>&1 | tee -a "${OUTPUT}"
echo "" | tee -a "${OUTPUT}"
echo ">>> Memory detail:" | tee -a "${OUTPUT}"
npu-smi info -t memory -i "${DEVICE_ID}" 2>&1 | tee -a "${OUTPUT}"
echo "" | tee -a "${OUTPUT}"
echo ">>> Process memory:" | tee -a "${OUTPUT}"
npu-smi info -t proc-mem -i "${DEVICE_ID}" 2>&1 | tee -a "${OUTPUT}"

section "3. NPU process list"
npu-smi info 2>&1 | grep -A30 "Process id" | tee -a "${OUTPUT}"

section "4. Driver and toolkit versions"
echo ">>> Driver:" | tee -a "${OUTPUT}"
cat /usr/local/Ascend/driver/version.info 2>/dev/null | tee -a "${OUTPUT}" || echo "  not found" | tee -a "${OUTPUT}"
echo "" | tee -a "${OUTPUT}"
echo ">>> CANN:" | tee -a "${OUTPUT}"
cat /usr/local/Ascend/ascend-toolkit/latest/arm64-linux/version.info 2>/dev/null | tee -a "${OUTPUT}" || echo "  not found" | tee -a "${OUTPUT}"
echo "" | tee -a "${OUTPUT}"
echo ">>> npu-smi version:" | tee -a "${OUTPUT}"
npu-smi info -t common -i "${DEVICE_ID}" 2>&1 | head -5 | tee -a "${OUTPUT}"

section "5. Device nodes"
ls -la /dev/davinci* 2>&1 | tee -a "${OUTPUT}"
echo "" | tee -a "${OUTPUT}"
echo ">>> ascend_install.info:" | tee -a "${OUTPUT}"
cat /etc/ascend_install.info 2>/dev/null | tee -a "${OUTPUT}" || echo "  not found" | tee -a "${OUTPUT}"

section "6. VMM / memory mapping info (kernel-level)"
echo ">>> /proc/davinci:" | tee -a "${OUTPUT}"
find /proc/davinci -name "vmm_info" -type f 2>/dev/null | while read p; do
    echo "--- ${p} ---" | tee -a "${OUTPUT}"
    cat "${p}" 2>/dev/null | tee -a "${OUTPUT}" || echo "  cannot read" | tee -a "${OUTPUT}"
done
if [ -z "$(find /proc/davinci -name 'vmm_info' 2>/dev/null)" ]; then
    echo "  /proc/davinci not available" | tee -a "${OUTPUT}"
fi

echo "" | tee -a "${OUTPUT}"
echo ">>> /sys/kernel/debug:" | tee -a "${OUTPUT}"
find /sys/kernel/debug -maxdepth 1 -name "davinci*" -type d 2>/dev/null | while read d; do
    echo "--- ${d} ---" | tee -a "${OUTPUT}"
    find "${d}" -name "*vmm*" -o -name "*mem*" 2>/dev/null | while read f; do
        echo "  ${f}:" | tee -a "${OUTPUT}"
        cat "${f}" 2>/dev/null | head -5 | tee -a "${OUTPUT}"
    done
done
if [ -z "$(find /sys/kernel/debug -maxdepth 1 -name 'davinci*' 2>/dev/null)" ]; then
    echo "  /sys/kernel/debug/davinci not available" | tee -a "${OUTPUT}"
fi

section "7. MindSpore memory reporting"
python -c "
import mindspore as ms
ms.set_context(mode=ms.GRAPH_MODE)
print('MindSpore version:', ms.__version__)

# Check memory stats API
for attr in ['memory_stats', 'memory_summary', 'get_device_memory']:
    if hasattr(ms.runtime, attr):
        try:
            result = getattr(ms.runtime, attr)()
            print(f'ms.runtime.{attr}(): {result}')
        except Exception as e:
            print(f'ms.runtime.{attr}(): error - {e}')
    else:
        print(f'ms.runtime.{attr}: not available')
" 2>&1 | tee -a "${OUTPUT}"

section "8. Quick memory pressure test (small allocation)"
python -c "
import mindspore as ms
import numpy as np
ms.set_context(mode=ms.PYNATIVE_MODE)

# Try a small allocation to check basic functionality
try:
    t = ms.Tensor(np.random.randn(128, 128).astype(np.float16)).to('Ascend')
    print(f'Small allocation OK: shape={t.shape}, dtype={t.dtype}, device={t.device}')
    del t
except Exception as e:
    print(f'Small allocation FAILED: {e}')

# Check available device count
from mindspore import runtime
try:
    count = runtime.device_count()
    print(f'Available Ascend devices: {count}')
except Exception as e:
    print(f'device_count FAILED: {e}')
" 2>&1 | tee -a "${OUTPUT}"

section "9. Kernel memory limits"
echo ">>> Checking system limits:" | tee -a "${OUTPUT}"
ulimit -a 2>&1 | grep -E "memory|virtual|locked|data" | tee -a "${OUTPUT}"
echo "" | tee -a "${OUTPUT}"
echo ">>> HugePages:" | tee -a "${OUTPUT}"
cat /proc/meminfo 2>/dev/null | grep -iE "huge|memtotal|memfree|memavailable" | tee -a "${OUTPUT}"

echo "" | tee -a "${OUTPUT}"
echo "========================================" | tee -a "${OUTPUT}"
echo "Diagnostic complete." | tee -a "${OUTPUT}"
echo "Output saved to: ${OUTPUT}" | tee -a "${OUTPUT}"
echo "========================================" | tee -a "${OUTPUT}"
