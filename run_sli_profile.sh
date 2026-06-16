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
    rm -rf profiler_data_sli_grad_kl_loss
    echo ">>> [sparse_loss] ms.profiler triton"
    python - <<'PY'
import numpy as np
import mindspore as ms
from perf_sli_grad_kl_loss_triton import run_profiling

ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
np.random.seed(42)
ms.set_seed(42)

run_profiling()
PY
}

run_prof_triton_markers() {
    rm -rf profiler_data_sli_grad_kl_loss
    echo ">>> [sparse_loss] ms.profiler triton with stage markers"
    SPARSE_GRAD_PROFILE_MARKERS=1 python - <<'PY'
import numpy as np
import mindspore as ms
from perf_sli_grad_kl_loss_triton import run_profiling

ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
np.random.seed(42)
ms.set_seed(42)

run_profiling()
PY
}

run_prof_cann() {
    rm -rf profiler_data_sli_grad_kl_loss_cann
    echo ">>> [sparse_loss] ms.profiler cann"
    python - <<'PY'
import numpy as np
import mindspore as ms
from perf_sli_grad_kl_loss_triton import run_profiling_cann

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
    echo ">>> [sparse_loss] msprof op ${kernel_name}"
    msprof op --kernel-name="${kernel_name}" \
        --output="${out_dir}" \
        python perf_sli_grad_kl_loss_triton.py --kernel-only
}

run_msprof_kernel_optional() {
    local kernel_name="$1"
    local out_dir="$2"
    if ! run_msprof_kernel "$kernel_name" "$out_dir"; then
        echo ">>> WARNING: msprof op replay failed for ${kernel_name}; continue." >&2
        echo ">>> WARNING: Use './run_sli_profile.sh prof-triton' and inspect profiler CSV for executed-kernel timing." >&2
    fi
}

run_op_gather() {
    run_msprof_kernel "_gather_kv_kernel" "./profilers_sli_gather"
}

run_op_main() {
    run_msprof_kernel "_indexer_grad_kl_loss_kernel" "./profilers_sli_main"
}

run_op_scatter() {
    run_msprof_kernel "_scatter_dkey_index_kernel" "./profilers_sli_scatter"
}

run_op_cast() {
    echo ">>> [sparse_loss] op-cast skipped"
    echo ">>> Current sparse_loss graph uses MindSpore ops.cast for d_key_index,"
    echo ">>> not the unused _cast_dkey_index_kernel symbol."
    echo ">>> Use './run_sli_profile.sh prof-triton' and summarize task_time Cast_* rows if needed."
}

run_op_triton() {
    run_op_gather
    run_op_main
    run_msprof_kernel_optional "_scatter_dkey_index_kernel" "./profilers_sli_scatter"
}

run_op_triton_replayable() {
    run_op_gather
    run_op_main
}

run_op_cann() {
    local kernel_name="${1:-}"
    if [ -z "$kernel_name" ]; then
        echo "Usage: $0 op-cann <cann_kernel_name>" >&2
        echo "First run: $0 prof-cann, then grep profiler dump for the real CANN kernel name." >&2
        exit 1
    fi

    rm -rf ./profilers_sli_cann
    echo ">>> [sparse_loss] msprof op CANN ${kernel_name}"
    msprof op --kernel-name="${kernel_name}" \
        --output=./profilers_sli_cann \
        python - <<'PY'
import numpy as np
import mindspore as ms
from mindspore import runtime
from perf_sli_grad_kl_loss_triton import _make_inputs, SparseLightningIndexerGradKLLoss

ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
np.random.seed(42)
ms.set_seed(42)

B, S1, S2, N1, D, Nidx1, D_idx, topK = 1, 4096, 4096, 64, 512, 64, 128, 2048
scale_value = 1.0 / np.sqrt(D)
q, k, qr, kr, qi, ki, w, si, softmax_max, softmax_sum = _make_inputs(
    B, S1, S2, N1, D, Nidx1, D_idx, topK
)

op = SparseLightningIndexerGradKLLoss()
for _ in range(10):
    out = op(
        q, k, qi, ki, w, si, softmax_max, softmax_sum,
        scale_value=scale_value,
        query_rope=qr,
        key_rope=kr,
        layout="BSND",
        sparse_mode=3,
    )
    runtime.synchronize()
    del out
PY
}

case "$cmd" in
    prof-triton)
        run_prof_triton
        ;;
    prof-triton-markers)
        run_prof_triton_markers
        ;;
    prof-cann)
        run_prof_cann
        ;;
    prof)
        run_prof_triton
        run_prof_cann
        ;;
    op-gather)
        run_op_gather
        ;;
    op-main)
        run_op_main
        ;;
    op-scatter)
        run_op_scatter
        ;;
    op-cast)
        run_op_cast
        ;;
    op-triton)
        run_op_triton
        ;;
    op-triton-replayable)
        run_op_triton_replayable
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
        echo "Usage: $0 {prof-triton|prof-triton-markers|prof-cann|prof|op-gather|op-main|op-scatter|op-cast|op-triton|op-triton-replayable|op-cann <kernel>|all}" >&2
        exit 1
        ;;
esac
