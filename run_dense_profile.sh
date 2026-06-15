#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"
export TRITON_END=mindspore
export TRITON_BACKEND=mindspore
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-./my_triton_cache}"

cmd="${1:-all}"
shift || true

run_prof_triton() {
    rm -rf profiler_data_dense_loss_backward
    echo ">>> [dense_loss] ms.profiler triton"
    python - <<'PY'
import numpy as np
import mindspore as ms
from perf_dense_loss_backward_triton import run_profiling

ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
np.random.seed(42)
ms.set_seed(42)

run_profiling()
PY
}

run_prof_cann() {
    rm -rf profiler_data_dense_loss_backward_cann
    echo ">>> [dense_loss] ms.profiler cann"
    python - <<'PY'
import numpy as np
import mindspore as ms
from perf_dense_loss_backward_triton import run_profiling_cann

ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
np.random.seed(42)
ms.set_seed(42)

run_profiling_cann()
PY
}

run_msprof_kernel() {
    local kernel_name="$1"
    local out_dir="$2"
    rm -rf "$out_dir"
    echo ">>> [dense_loss] msprof op ${kernel_name}"
    msprof op --kernel-name="${kernel_name}" \
        --output="${out_dir}" \
        python perf_dense_loss_backward_triton.py --kernel-only
}

run_msprof_kernel_optional() {
    local kernel_name="$1"
    local out_dir="$2"
    if ! run_msprof_kernel "$kernel_name" "$out_dir"; then
        echo ">>> WARNING: msprof op replay failed for ${kernel_name}; continue." >&2
        echo ">>> WARNING: Use './run_dense_profile.sh prof-triton' and inspect profiler CSV for executed-kernel timing." >&2
    fi
}

run_op_stats() {
    run_msprof_kernel "_dense_indexer_stats_kernel" "./profilers_dense_stats"
}

run_op_loss() {
    run_msprof_kernel "_dense_loss_kernel" "./profilers_dense_loss"
}

run_op_main() {
    run_msprof_kernel "_dense_main_grad_kernel" "./profilers_dense_main"
}

run_op_dkey() {
    run_msprof_kernel "_dense_dkey_index_kernel" "./profilers_dense_dkey"
}

run_op_triton() {
    run_msprof_kernel_optional "_dense_indexer_stats_kernel" "./profilers_dense_stats"
    run_msprof_kernel_optional "_dense_loss_kernel" "./profilers_dense_loss"
    run_msprof_kernel_optional "_dense_main_grad_kernel" "./profilers_dense_main"
    run_msprof_kernel_optional "_dense_dkey_index_kernel" "./profilers_dense_dkey"
}

run_op_cann() {
    local kernel_name="${1:-}"
    if [ -z "$kernel_name" ]; then
        echo "Usage: $0 op-cann <cann_kernel_name>" >&2
        echo "First run: $0 prof-cann, then grep profiler dump for the real CANN kernel name." >&2
        exit 1
    fi

    rm -rf ./profilers_dense_cann
    echo ">>> [dense_loss] msprof op CANN ${kernel_name}"
    msprof op --kernel-name="${kernel_name}" \
        --output=./profilers_dense_cann \
        python - <<'PY'
import numpy as np
import mindspore as ms
from mindspore import runtime
from perf_dense_loss_backward_triton import (
    DENSE_PROFILE_CONFIG,
    DROPE,
    _call_official_grad,
    _call_official_lse,
    _make_inputs,
    _resolve_official_dense_ops,
)

ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
np.random.seed(42)
ms.set_seed(42)

B, S1, S2, N1, N2, D, Nidx1, D_idx = DENSE_PROFILE_CONFIG
scale_value = 1.0 / np.sqrt(D + DROPE)
q, k, qi, ki, w, qr, kr, softmax_max, softmax_sum = _make_inputs(
    B, S1, S2, N1, N2, D, Nidx1, D_idx
)

official_lse, official_grad, official_source = _resolve_official_dense_ops()
if official_lse is None or official_grad is None:
    raise RuntimeError(f"CANN dense ops unavailable: {official_source}")

for _ in range(10):
    max_index, sum_index = _call_official_lse(official_lse, qi, ki, w)
    out = _call_official_grad(
        official_grad, q, k, qi, ki, w, softmax_max, softmax_sum,
        max_index, sum_index, scale_value, qr, kr,
    )
    runtime.synchronize()
    del max_index, sum_index, out
PY
}

case "$cmd" in
    prof-triton)
        run_prof_triton
        ;;
    prof-cann)
        run_prof_cann
        ;;
    prof)
        run_prof_triton
        run_prof_cann
        ;;
    op-stats)
        run_op_stats
        ;;
    op-loss)
        run_op_loss
        ;;
    op-main)
        run_op_main
        ;;
    op-dkey)
        run_op_dkey
        ;;
    op-triton)
        run_op_triton
        ;;
    op-cann)
        run_op_cann "${1:-}"
        ;;
    all)
        run_prof_triton
        run_prof_cann
        run_op_triton
        ;;
    *)
        echo "Usage: $0 {prof-triton|prof-cann|prof|op-stats|op-loss|op-main|op-dkey|op-triton|op-cann <kernel>|all}" >&2
        exit 1
        ;;
esac
