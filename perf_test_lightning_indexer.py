import time
import numpy as np
import mindspore as ms
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
        (1, 1024, 1024, 8, 128, 2048),
        (1, 2048, 2048, 8, 128, 2048),
        (4, 1024, 1024, 8, 128, 2048),
    ]

    for B, S1, S2, N1, D, k in configs:
        print(f"\nB={B}, S1={S1}, S2={S2}, N1={N1}, D={D}, topk={k}")

        q = ms.Tensor(np.random.randn(B, S1, N1, D).astype(np.float16))
        k_t = ms.Tensor(np.random.randn(B, S2, 1, D).astype(np.float16))
        w = ms.Tensor(np.random.randn(B, S1, N1).astype(np.float32))

        t_med, t_p20, t_p80 = _do_bench(
            lambda q=q, k_t=k_t, w=w, k=k: lightning_indexer_triton(q, k_t, w, sparse_count=k)
        )
        print(f"triton:  median={t_med:.2f}ms, p20={t_p20:.2f}ms, p80={t_p80:.2f}ms")

        o_med, o_p20, o_p80 = _do_bench(
            lambda q=q, k_t=k_t, w=w, k=k: ms.ops.lightning_indexer(q, k_t, w, sparse_count=k)
        )
        print(f"ms.ops:  median={o_med:.2f}ms, p20={o_p20:.2f}ms, p80={o_p80:.2f}ms")

        speedup = o_med / t_med if t_med > 0 else float('inf')
        print(f"speedup: {speedup:.2f}x")


def run_profiling():
    B, S1, S2, N1, D, k = 1, 1024, 1024, 8, 128, 2048
    q = ms.Tensor(np.random.randn(B, S1, N1, D).astype(np.float16))
    k_t = ms.Tensor(np.random.randn(B, S2, 1, D).astype(np.float16))
    w = ms.Tensor(np.random.randn(B, S1, N1).astype(np.float32))

    total_steps = 20
    profiler = Profiler(
        schedule=ms.profiler.ProfilerSchedule(skip_first=4, skip_last=4),
        output_path="./profiler_data",
    )
    profiler.start()
    for _ in range(total_steps):
        lightning_indexer_triton(q, k_t, w, sparse_count=k)
        profiler.step()
    profiler.stop()
    profiler.analyse()
    print("Profiler data saved to ./profiler_data")


if __name__ == "__main__":
    from lightning_indexer_triton import lightning_indexer_triton

    ms.set_context(mode=ms.GRAPH_MODE)
    np.random.seed(42)
    ms.set_seed(42)

    run_timing()
    run_profiling()
