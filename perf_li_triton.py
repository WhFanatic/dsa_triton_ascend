import time
import numpy as np
import mindspore as ms
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics, ExportType
from mindspore import Profiler, runtime


def _do_bench(fn, warmup=10, rep=100):
    """Simple benchmarking with manual timing (Ascend-compatible)."""
    for _ in range(warmup):
        fn()
    runtime.synchronize()

    times = []
    for _ in range(rep):
        runtime.synchronize()
        t0 = time.perf_counter()
        fn()
        runtime.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    n = len(times)
    return times[n // 2], times[n // 5], times[4 * n // 5]


def run_timing():
    configs = [
        (1, 4, 128, 8, 128, 32),
    ]

    for B, S1, S2, N1, D, k in configs:
        print(f"\nB={B}, S1={S1}, S2={S2}, N1={N1}, D={D}, topk={k}")

        q = ms.Tensor(np.random.randn(B, S1, N1, D).astype(np.float16))
        k_t = ms.Tensor(np.random.randn(B, S2, 1, D).astype(np.float16))
        w = ms.Tensor(np.random.randn(B, S1, N1).astype(np.float32))

        cell = LightningIndexerTriton(sparse_count=k)
        t_med, t_p20, t_p80 = _do_bench(
            lambda cell=cell: cell(q, k_t, w)
        )
        print(f"triton:  median={t_med:.2f}ms, p20={t_p20:.2f}ms, p80={t_p80:.2f}ms")

        o_med, o_p20, o_p80 = _do_bench(
            lambda q=q, k_t=k_t, w=w, k=k: ms.ops.lightning_indexer(q, k_t, w, sparse_count=k)
        )
        print(f"ms.ops:  median={o_med:.2f}ms, p20={o_p20:.2f}ms, p80={o_p80:.2f}ms")

        speedup = o_med / t_med if t_med > 0 else float('inf')
        print(f"speedup: {speedup:.2f}x")


def run_profiling():
    total_steps = 10
    out_dir = './profiler_data'
    B, S1, S2, N1, D, k = 1, 4, 128, 8, 128, 32
    q = ms.Tensor(np.random.randn(B, S1, N1, D).astype(np.float16))
    k_t = ms.Tensor(np.random.randn(B, S2, 1, D).astype(np.float16))
    w = ms.Tensor(np.random.randn(B, S1, N1).astype(np.float32))

    cell = LightningIndexerTriton(sparse_count=k)

    experimental_config = ms.profiler._ExperimentalConfig(
        profiler_level=ProfilerLevel.Level0,
        aic_metrics=AicoreMetrics.AiCoreNone,
        l2_cache=False,
        mstx=False,
        data_simplification=False,
        )

    with ms.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        with_stack=True,
        schedule=ms.profiler.schedule(wait=2, warmup=2, active=2, repeat=1, skip_first=2),
        on_trace_ready=ms.profiler.tensorboard_trace_handler(out_dir),
        profile_memory=False,
        experimental_config=experimental_config
        ) as prof:

        for _ in range(total_steps):
            cell(q, k_t, w)
            prof.step()

    print(f"Profiler data saved to {out_dir}")


if __name__ == "__main__":
    from lightning_indexer_triton import LightningIndexerTriton

    ms.set_context(mode=ms.GRAPH_MODE)
    np.random.seed(42)
    ms.set_seed(42)

    run_timing()
    run_profiling()
