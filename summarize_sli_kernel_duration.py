import argparse
import csv
import glob
import os
from collections import defaultdict


TRITON_TAG = "profiler_data_sli_grad_kl_loss"
CANN_TAG = "profiler_data_sli_grad_kl_loss_cann"
TASK_TIME = "task_time"
KERNEL_DETAILS = "kernel_details.csv"
EXPECTED_TRITON_KERNELS = (
    "_gather_kv_kernel",
    "_indexer_grad_kl_loss_kernel",
    "_scatter_dkey_index_kernel",
)


def _latest_dump():
    files = glob.glob(os.path.join("dump", "profiler_dump_*.txt"))
    if not files:
        raise FileNotFoundError("no dump/profiler_dump_*.txt found")
    return max(files, key=os.path.getmtime)


def _read_sections(path):
    current_path = None
    current_lines = []
    in_content = False

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line == "######## DUMP_FILE_START ########":
                current_path = None
                current_lines = []
                in_content = False
                continue
            if line.startswith("PATH: "):
                current_path = line[len("PATH: "):].strip()
                continue
            if line == "######## DUMP_CONTENT ########":
                in_content = True
                continue
            if line == "######## DUMP_FILE_END ########":
                if current_path is not None:
                    yield current_path, current_lines
                current_path = None
                current_lines = []
                in_content = False
                continue
            if in_content:
                current_lines.append(line)


def _source_from_path(path):
    if CANN_TAG in path:
        return "cann"
    if TRITON_TAG in path:
        return "triton"
    return None


def _safe_float(value):
    try:
        return float(value.strip().replace("\t", ""))
    except (AttributeError, ValueError):
        return None


def _normalize_name(name):
    name = name.strip()
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    return name


def _is_task_time(path):
    filename = os.path.basename(path)
    return filename.startswith(TASK_TIME) and filename.endswith(".csv")


def _parse_task_time(lines):
    """Parse mindstudio task_time_*.csv rows.

    Expected rows:
      device_id,op_name,task_type,stream_id,task_id,duration_us,start,end
    """
    for row in csv.reader(lines):
        if len(row) < 6:
            continue
        duration_us = _safe_float(row[5])
        if duration_us is None:
            continue
        name = row[1].strip()
        task_type = row[2].strip()
        if not name or name.lower() in ("op name", "name"):
            continue
        yield _normalize_name(name), task_type, duration_us


def _parse_kernel_details(lines):
    """Parse ASCEND_PROFILER_OUTPUT/kernel_details.csv rows.

    This is kept as a fallback. task_time_*.csv is preferred because it is the
    table you inspect with sed and it lists actual NPU tasks directly.
    """
    for row in csv.reader(lines):
        if len(row) < 10:
            continue
        duration_us = _safe_float(row[9])
        if duration_us is None:
            continue
        name = row[4].strip()
        if not name or name.lower() in ("op name", "name"):
            continue
        yield _normalize_name(name), "KERNEL", duration_us


def _is_counted_task(task_type, include_device_tasks):
    if task_type.startswith("KERNEL"):
        return True
    if include_device_tasks and task_type in ("MEMCPY_ASYNC", "EVENT_WAIT"):
        return True
    return False


def _preferred_kernel_task_types(rows):
    """Avoid double-counting generic and hardware-specific kernel rows."""
    specific = {
        task_type
        for _, task_type, _ in rows
        if task_type.startswith("KERNEL_")
    }
    if specific:
        return specific
    if any(task_type == "KERNEL" for _, task_type, _ in rows):
        return {"KERNEL"}
    return set()


def parse_dump(path, include_device_tasks=False):
    by_source_total = defaultdict(float)
    by_source_kernel = defaultdict(lambda: defaultdict(float))
    by_source_calls = defaultdict(lambda: defaultdict(int))
    source_task_files = defaultdict(int)
    source_all_task_types = defaultdict(lambda: defaultdict(int))
    task_sections = []
    kernel_detail_sections = []

    for rel_path, lines in _read_sections(path):
        source = _source_from_path(rel_path)
        if source is None:
            continue

        if _is_task_time(rel_path):
            source_task_files[source] += 1
            task_sections.append((source, lines))
        elif rel_path.endswith(KERNEL_DETAILS):
            kernel_detail_sections.append((source, lines))
        else:
            continue

    sections = task_sections if task_sections else kernel_detail_sections

    for source, lines in sections:
        if task_sections:
            rows = list(_parse_task_time(lines))
            preferred_kernel_types = _preferred_kernel_task_types(rows)
        else:
            rows = list(_parse_kernel_details(lines))
            preferred_kernel_types = {"KERNEL"}

        for name, task_type, duration_us in rows:
            source_all_task_types[source][task_type] += 1
            if task_type.startswith("KERNEL"):
                if task_type not in preferred_kernel_types:
                    continue
            elif not _is_counted_task(task_type, include_device_tasks):
                continue
            by_source_total[source] += duration_us
            by_source_kernel[source][name] += duration_us
            by_source_calls[source][name] += 1

    return by_source_total, by_source_kernel, by_source_calls, source_task_files, source_all_task_types


def _infer_triton_calls(kernel_calls):
    main_calls = kernel_calls.get("_indexer_grad_kl_loss_kernel_0", 0)
    if main_calls == 0:
        main_calls = sum(
            count for name, count in kernel_calls.items()
            if name.startswith("_indexer_grad_kl_loss_kernel")
        )
    # The current perf shape has S1=4096 and SPARSE_GRAD_S1_CHUNK=512.
    # Therefore one sparse_loss call launches 8 main kernels.
    if main_calls and main_calls % 8 == 0:
        return main_calls // 8
    return None


def _infer_cann_calls(kernel_calls):
    calls = sum(
        count for name, count in kernel_calls.items()
        if "SparseLightningIndexerGradKLLoss" in name
    )
    return calls or None


def _print_source_summary(source, total_us, kernel_calls, active_steps):
    if total_us <= 0:
        print(f"{source}: no counted task rows found")
        return None

    inferred_calls = (
        _infer_triton_calls(kernel_calls)
        if source == "triton"
        else _infer_cann_calls(kernel_calls)
    )
    calls = inferred_calls or active_steps
    avg_us = total_us / calls
    call_note = "inferred" if inferred_calls else "configured"
    print(
        f"{source:6s} total_active={total_us / 1000.0:10.3f} ms "
        f"calls={calls:2d}({call_note}) "
        f"per_call={avg_us / 1000.0:10.3f} ms"
    )
    return avg_us


def _print_breakdown(source, total_us, kernel_durations, kernel_calls, calls, topn):
    if total_us <= 0:
        return
    print(f"\n{source} full-chain breakdown, averaged per call:")
    print(f"{'avg_ms':>12s} {'calls/call':>12s} {'pct':>8s}  task/kernel")
    rows = []
    for name, duration_us in kernel_durations.items():
        avg_us = duration_us / calls
        calls_per_call = kernel_calls[name] / calls
        pct = duration_us / total_us * 100.0
        rows.append((avg_us, calls_per_call, pct, name))
    rows.sort(reverse=True)
    for avg_us, calls_per_call, pct, name in rows[:topn]:
        print(f"{avg_us / 1000.0:12.3f} {calls_per_call:12.2f} {pct:7.2f}%  {name}")


def _warn_missing_expected(kernel_calls):
    names = "\n".join(kernel_calls)
    missing = [
        expected for expected in EXPECTED_TRITON_KERNELS
        if expected not in names
    ]
    if not missing:
        return
    print("\nWARNING: expected Triton kernel name(s) not found in counted tasks:")
    for name in missing:
        print(f"  - {name}")
    print("If a required kernel is missing, the full-chain number is incomplete.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Sparse Grad KL Loss full-chain NPU kernel duration from "
            "dump/profiler_dump_*.txt. This sums actual ms.profiler task_time "
            "kernel durations, not msprof single-kernel replay and not Python "
            "end-to-end time."
        )
    )
    parser.add_argument(
        "--dump",
        default=None,
        help="Path to profiler_dump_*.txt. Defaults to the newest dump/profiler_dump_*.txt.",
    )
    parser.add_argument(
        "--active-steps",
        type=int,
        default=4,
        help="Profiler active steps. perf_sli_grad_kl_loss_triton.py currently uses active=4.",
    )
    parser.add_argument(
        "--include-device-tasks",
        action="store_true",
        help="Also include MEMCPY_ASYNC and EVENT_WAIT task durations.",
    )
    parser.add_argument("--topn", type=int, default=20)
    args = parser.parse_args()

    dump_path = args.dump or _latest_dump()
    totals, durations, calls, task_files, task_types = parse_dump(
        dump_path,
        include_device_tasks=args.include_device_tasks,
    )

    print(f"dump: {dump_path}")
    print("Full-chain duration from actual ms.profiler task_time*.csv")
    if args.include_device_tasks:
        print("counted tasks: KERNEL_* + MEMCPY_ASYNC + EVENT_WAIT")
    else:
        print("counted tasks: KERNEL_* only")
    print("not counted: Python overhead / host API time / end-to-end wall time")
    print()

    per_call = {}
    for source in ("triton", "cann"):
        per_call[source] = _print_source_summary(
            source,
            totals.get(source, 0.0),
            calls.get(source, {}),
            args.active_steps,
        )

    if per_call.get("triton") and per_call.get("cann"):
        print(f"\nratio triton/cann per_call: {per_call['triton'] / per_call['cann']:.2f}x")

    if calls.get("triton"):
        _warn_missing_expected(calls["triton"])

    print("\ntask_time files parsed:")
    for source in ("triton", "cann"):
        print(f"  {source}: {task_files.get(source, 0)}")

    print("\ntask types observed:")
    for source in ("triton", "cann"):
        if not task_types.get(source):
            continue
        items = ", ".join(
            f"{name}={count}" for name, count in sorted(task_types[source].items())
        )
        print(f"  {source}: {items}")

    for source in ("triton", "cann"):
        kernel_calls = calls.get(source, {})
        source_calls = (
            _infer_triton_calls(kernel_calls)
            if source == "triton"
            else _infer_cann_calls(kernel_calls)
        ) or args.active_steps
        _print_breakdown(
            source,
            totals.get(source, 0.0),
            durations.get(source, {}),
            kernel_calls,
            source_calls,
            args.topn,
        )


if __name__ == "__main__":
    main()
