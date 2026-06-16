#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

target="${1:-all}"
cmd="${2:-prof}"
if [ "$#" -ge 2 ]; then
    shift 2
else
    shift "$#" || true
fi

run_dense() {
    bash ./run_dense_profile.sh "$cmd" "$@"
}

run_sparse() {
    bash ./run_sli_profile.sh "$cmd" "$@"
}

case "$target" in
    dense|dense_loss)
        run_dense
        ;;
    sparse|sparse_loss|sli)
        run_sparse
        ;;
    all)
        run_dense
        run_sparse
        ;;
    *)
        echo "Usage: $0 {dense|sparse|all} {prof-triton|prof-cann|prof|op-triton|all}" >&2
        exit 1
        ;;
esac
