import argparse
import csv
import glob
import os


MARKERS = {
    "teacher": (
        "_profile_marker_teacher_start_kernel",
        "_profile_marker_teacher_end_kernel",
    ),
    "main": (
        "_profile_marker_main_start_kernel",
        "_profile_marker_main_end_kernel",
    ),
    "query": (
        "_profile_marker_query_start_kernel",
        "_profile_marker_query_end_kernel",
    ),
    "scatter": (
        "_profile_marker_scatter_start_kernel",
        "_profile_marker_scatter_end_kernel",
    ),
}


def _safe_float(value, default=0.0):
    try:
        return float(str(value).strip())
    except Exception:
        return default


def _latest_task_time(root):
    files = glob.glob(os.path.join(root, "**", "task_time_*.csv"), recursive=True)
    if not files:
        raise FileNotFoundError(f"no task_time_*.csv found under {root}")
    return max(files, key=os.path.getmtime)


def _read_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            name = row.get("kernel_name", "")
            ktype = row.get("kernel_type", "")
            dur = _safe_float(row.get("task_time(us)", "0"))
            start = _safe_float(row.get("task_start(us)", "0"))
            stop = _safe_float(row.get("task_stop(us)", "0"))
            rows.append(
                {
                    "row_idx": row_idx,
                    "name": name,
                    "type": ktype,
                    "dur": dur,
                    "start": start,
                    "stop": stop,
                }
            )
    rows.sort(key=lambda r: (r["start"], r["row_idx"]))
    return rows


def _find_marker_rows(rows, marker):
    return [r for r in rows if marker in r["name"]]


def _pair_intervals(rows, start_marker, end_marker):
    intervals = []
    pending = None
    for row in rows:
        name = row["name"]
        if start_marker in name:
            pending = row
            continue
        if end_marker in name and pending is not None:
            gap = max(row["start"] - pending["stop"], 0.0)
            inclusive = max(row["stop"] - pending["start"], 0.0)
            intervals.append(
                {
                    "start_row": pending["row_idx"],
                    "end_row": row["row_idx"],
                    "start_us": pending["start"],
                    "end_us": row["stop"],
                    "gap_us": gap,
                    "inclusive_us": inclusive,
                }
            )
            pending = None
    return intervals


def _print_stats(name, intervals):
    if not intervals:
        print(f"{name}: no marker pairs found")
        return
    gaps = [item["gap_us"] for item in intervals]
    total = sum(gaps)
    avg = total / len(gaps)
    print(
        f"{name}: calls={len(gaps)} total_ms={total / 1000.0:.3f} "
        f"avg_ms={avg / 1000.0:.3f} max_ms={max(gaps) / 1000.0:.3f} "
        f"min_ms={min(gaps) / 1000.0:.3f}"
    )


def _print_top(name, intervals, topn):
    if not intervals:
        return
    print(f"top {name} intervals:")
    print("rank gap_ms inclusive_ms start_row end_row")
    ordered = sorted(intervals, key=lambda x: x["gap_us"], reverse=True)
    for rank, item in enumerate(ordered[:topn], 1):
        print(
            rank,
            f"{item['gap_us'] / 1000.0:.3f}",
            f"{item['inclusive_us'] / 1000.0:.3f}",
            item["start_row"],
            item["end_row"],
        )


def _print_mix_aic(rows, topn):
    mix = [
        r for r in rows
        if r["name"] == "N/A" and "MIX_AIC" in r["type"]
    ]
    if not mix:
        print("N/A MIX_AIC: none")
        return
    total = sum(r["dur"] for r in mix)
    print(
        f"N/A MIX_AIC: calls={len(mix)} total_ms={total / 1000.0:.3f} "
        f"avg_ms={total / len(mix) / 1000.0:.3f} "
        f"max_ms={max(r['dur'] for r in mix) / 1000.0:.3f} "
        f"min_ms={min(r['dur'] for r in mix) / 1000.0:.3f}"
    )
    print("top N/A MIX_AIC rows:")
    print("rank row_idx duration_ms")
    for rank, row in enumerate(sorted(mix, key=lambda r: r["dur"], reverse=True)[:topn], 1):
        print(rank, row["row_idx"], f"{row['dur'] / 1000.0:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate hidden sparse grad KL query/scatter durations from "
            "profile marker kernels in task_time_*.csv."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="profiler_data_sli_grad_kl_loss",
        help="Profiler root or a task_time_*.csv path.",
    )
    parser.add_argument("--topn", type=int, default=16)
    args = parser.parse_args()

    path = args.root
    if os.path.isdir(path):
        path = _latest_task_time(path)
    rows = _read_rows(path)

    print(f"file: {path}")
    for stage, (start_marker, end_marker) in MARKERS.items():
        start_rows = _find_marker_rows(rows, start_marker)
        end_rows = _find_marker_rows(rows, end_marker)
        print(
            f"{stage} markers: start={len(start_rows)} end={len(end_rows)} "
            f"start_marker={start_marker} end_marker={end_marker}"
        )
        intervals = _pair_intervals(rows, start_marker, end_marker)
        _print_stats(stage, intervals)
        _print_top(stage, intervals, args.topn)
    _print_mix_aic(rows, args.topn)


if __name__ == "__main__":
    main()
