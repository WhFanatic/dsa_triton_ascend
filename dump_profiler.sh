#!/bin/bash
set -euo pipefail

CMD="${1:-./run_test.sh}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DUMP_DIR="$SCRIPT_DIR/dump"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DUMP_FILE="$DUMP_DIR/profiler_dump_${TIMESTAMP}.txt"
OP_DUMP_FILE="$DUMP_DIR/msprof_op_${TIMESTAMP}.txt"
LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT

mkdir -p "$DUMP_DIR"

echo ">>> Executing: $CMD" >&2
eval "$CMD" 2>&1 | tee "$LOG"
EXIT_CODE=${PIPESTATUS[0]}

if [ "$EXIT_CODE" -ne 0 ]; then
    echo ">>> ERROR: Command failed with exit code $EXIT_CODE" >&2
    exit 1
fi

WORKDIR=$(pwd)
COUNT1=0
COUNT2=0

DIRS1=$(grep "Start parsing profiling data" "$LOG" | sed 's/.*at: //' || true)
DIRS2=$(grep "Profiling results saved in" "$LOG" | sed 's/.*saved in //' || true)

if [ -z "$DIRS1" ] && [ -z "$DIRS2" ]; then
    echo ">>> WARNING: No profiler directories found in output" >&2
    exit 1
fi

: > "$DUMP_FILE"
: > "$OP_DUMP_FILE"

if [ -n "$DIRS1" ]; then
    while IFS= read -r dir; do
        dir=$(echo "$dir" | tr -d '\r')
        if [ ! -d "$dir" ]; then
            echo ">>> WARNING: Directory not found: $dir" >&2
            continue
        fi
        while IFS= read -r csv; do
            REL_PATH="${csv#$WORKDIR/}"
            echo "######## DUMP_FILE_START ########"
            echo "PATH: $REL_PATH"
            echo "######## DUMP_CONTENT ########"
            cat "$csv"
            echo ""
            echo "######## DUMP_FILE_END ########"
            COUNT1=$((COUNT1 + 1))
        done < <(find "$dir" -name "*.csv" -type f | sort)
    done <<< "$DIRS1" >> "$DUMP_FILE"
fi

if [ -n "$DIRS2" ]; then
    while IFS= read -r dir; do
        dir=$(echo "$dir" | tr -d '\r')
        if [ ! -d "$dir" ]; then
            echo ">>> WARNING: Directory not found: $dir" >&2
            continue
        fi
        while IFS= read -r csv; do
            REL_PATH="${csv#$WORKDIR/}"
            echo "######## DUMP_FILE_START ########"
            echo "PATH: $REL_PATH"
            echo "######## DUMP_CONTENT ########"
            cat "$csv"
            echo ""
            echo "######## DUMP_FILE_END ########"
            COUNT2=$((COUNT2 + 1))
        done < <(find "$dir" -name "*.csv" -type f | sort)
    done <<< "$DIRS2" >> "$OP_DUMP_FILE"
fi

echo ">>> Dumped $COUNT1 CSV files to $DUMP_FILE" >&2
echo ">>> Dumped $COUNT2 CSV files to $OP_DUMP_FILE" >&2
