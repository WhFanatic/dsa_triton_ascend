import time
import numpy as np
import mindspore as ms
from mindspore import runtime
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics


def _do_bench(fn, warmup=10, rep=50):
    """Simple benchmarking with manual timing (Ascend-compatible)."""
    for _ in range(warmup):
        out = fn()
        runtime.synchronize()
        del out
        runtime.empty_cache()

    times = []
    for _ in range(rep):
        runtime.synchronize()
        t0 = time.perf_counter()
        out = fn()
        runtime.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        del out
        runtime.empty_cache()

    times.sort()
    n = len(times)
    return times[n // 2], times[n // 5], times[4 * n // 5]


def _make_inputs(B, S1, S2, N1, N2, D):
    q = ms.Tensor(np.random.randn(B, S1, N1, D).astype(np.float16)).to('Ascend')
    k_t = ms.Tensor(np.random.randn(B, S2, N2, D).astype(np.float16)).to('Ascend')
    w = ms.Tensor(np.random.randn(B, S1, N1).astype(np.float32)).to('Ascend')
    return q, k_t, w


def run_timing():
    configs = [
        # (1, 128, 128, 16, 1, 128, 32),
        # (1, 1024, 1024, 64, 1, 128, 512),
        (1, 4096, 4096, 64, 1, 128, 2048),
    ]

    for B, S1, S2, N1, N2, D, k in configs:
        print(f"\nB={B}, S1={S1}, S2={S2}, N1={N1}, N2={N2}, D={D}, topk={k}")

        q, k_t, w = _make_inputs(B, S1, S2, N1, N2, D)
        cell = LightningIndexerTriton(sparse_count=k)

        t_med, t_p20, t_p80 = _do_bench(lambda cell=cell: cell(q, k_t, w))
        o_med, o_p20, o_p80 = _do_bench(lambda q=q, k_t=k_t, w=w, k=k: ms.ops.lightning_indexer(q, k_t, w, sparse_count=k))
        speedup = o_med / t_med if t_med > 0 else float('inf')

        print(f"triton:  median={t_med:.2f}ms, p20={t_p20:.2f}ms, p80={t_p80:.2f}ms")
        print(f"ms.ops:  median={o_med:.2f}ms, p20={o_p20:.2f}ms, p80={o_p80:.2f}ms")
        print(f"speedup: {speedup:.2f}x")

        del q, k_t, w, cell
        runtime.synchronize()
        runtime.empty_cache()


def run_profiling():
    total_steps = 10
    out_dir = './profiler_data'

    B, S1, S2, N1, N2, D, k = 1, 4096, 4096, 64, 1, 128, 2048

    q, k_t, w = _make_inputs(B, S1, S2, N1, N2, D)
    cell = LightningIndexerTriton(sparse_count=k)

    experimental_config = ms.profiler._ExperimentalConfig(
        profiler_level=ProfilerLevel.Level0,
        aic_metrics=AicoreMetrics.AiCoreNone,
        # aic_metrics=AicoreMetrics.ArithmeticUtilization,
        # aic_metrics=AicoreMetrics.PipeUtilization,
        # aic_metrics=AicoreMetrics.Memory,
        # aic_metrics=AicoreMetrics.MemoryUB,
        l2_cache=False,
        mstx=False,
        data_simplification=False,
        )

    with ms.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        with_stack=True,
        schedule=ms.profiler.schedule(wait=2, warmup=2, active=4, repeat=1, skip_first=2),
        on_trace_ready=ms.profiler.tensorboard_trace_handler(out_dir),
        profile_memory=False,
        experimental_config=experimental_config
        ) as prof:

        for _ in range(total_steps):
            cell(q, k_t, w)
            prof.step()

    print(f"Profiler data saved to {out_dir}")


def run_profiling_cann():
    total_steps = 10
    out_dir = './profiler_data_cann'

    B, S1, S2, N1, N2, D, k = 1, 4096, 4096, 64, 1, 128, 2048

    q, k_t, w = _make_inputs(B, S1, S2, N1, N2, D)

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
        schedule=ms.profiler.schedule(wait=2, warmup=2, active=4, repeat=1, skip_first=2),
        on_trace_ready=ms.profiler.tensorboard_trace_handler(out_dir),
        profile_memory=False,
        experimental_config=experimental_config,
    ) as prof:

        for _ in range(total_steps):
            ms.ops.lightning_indexer(q, k_t, w, sparse_count=k)
            prof.step()

    print(f"Profiler data saved to {out_dir}")


def run_kernel_only():
    from lightning_indexer_triton import LightningIndexerTriton

    B, S1, S2, N1, N2, D, k = 1, 4096, 4096, 64, 1, 128, 2048

    q, k_t, w = _make_inputs(B, S1, S2, N1, N2, D)
    cell = LightningIndexerTriton(sparse_count=k)

    for _ in range(10):
        cell(q, k_t, w)
    print("kernel-only run finished")


if __name__ == "__main__":
    import sys

    from lightning_indexer_triton import LightningIndexerTriton

    ms.set_context(mode=ms.GRAPH_MODE)
    np.random.seed(42)
    ms.set_seed(42)

    if len(sys.argv) > 1 and sys.argv[1] == "--kernel-only":
        run_kernel_only()
    else:
        run_timing()
        run_profiling()
        run_profiling_cann()
