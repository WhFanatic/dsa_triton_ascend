import time
import numpy as np
import mindspore as ms
from mindspore import ops, runtime
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics

D_NOPE = 512
D_ROPE = 64
D_IDX = 128


def _do_bench(fn, warmup=10, rep=100):
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


def _make_sparse_indices(B, S1, S2, topK):
    rng = np.random.RandomState(7)
    si = np.full((B, S1, 1, topK), -1, dtype=np.int32)
    for b in range(B):
        for s1 in range(S1):
            threshold = S2 - S1 + s1 + 1
            if threshold <= 0:
                continue
            n = min(topK, threshold)
            perm = rng.permutation(threshold)[:n]
            si[b, s1, 0, :n] = np.sort(perm).astype(np.int32)
    return ms.Tensor(si, dtype=ms.int32).to('Ascend')


def _make_inputs(B, S1, S2, N1, Nidx1, topK, dtype=ms.float16):
    rng = np.random.RandomState(42)
    N2, Nidx2 = 1, 1

    def _t(shape, d=dtype):
        return ms.Tensor(rng.randn(*shape).astype(np.float16), dtype=d).to('Ascend')

    q = _t((B, S1, N1, D_NOPE))
    k = _t((B, S2, N2, D_NOPE))
    qr = _t((B, S1, N1, D_ROPE))
    kr = _t((B, S2, N2, D_ROPE))
    qi = _t((B, S1, Nidx1, D_IDX))
    ki = _t((B, S2, Nidx2, D_IDX))
    w = ms.Tensor(np.abs(rng.randn(B, S1, Nidx1)).astype(np.float16), dtype=dtype).to('Ascend')
    si = _make_sparse_indices(B, S1, S2, topK)
    sm_max = ms.Tensor(np.abs(rng.randn(B, N2, S1, N1)).astype(np.float32), dtype=ms.float32).to('Ascend')
    sm_sum = ms.Tensor(np.abs(rng.randn(B, N2, S1, N1)).astype(np.float32), dtype=ms.float32).to('Ascend')
    return q, k, qr, kr, qi, ki, w, si, sm_max, sm_sum


def run_timing():
    from sparse_lightning_indexer_grad_kl_loss_triton import SparseLightningIndexerGradKLLossTriton
    from sli_grad_kl_loss_cann import SparseLightningIndexerGradKLLoss

    configs = [
        (1, 128, 512, 64, 8, 1024),
        (1, 256, 2048, 64, 8, 2048),
        (1, 1024, 4096, 64, 8, 2048),
        (1, 4096, 4096, 64, 8, 2048),
        (1, 4096, 8192, 64, 8, 2048),
    ]

    for B, S1, S2, N1, Nidx1, topK in configs:
        print(f"\nB={B}, S1={S1}, S2={S2}, N1={N1}, Nidx1={Nidx1}, topK={topK}")

        q, k, qr, kr, qi, ki, w, si, sm_max, sm_sum = _make_inputs(B, S1, S2, N1, Nidx1, topK)
        scale = 1.0 / np.sqrt(D_NOPE + D_ROPE)

        cell_tri = SparseLightningIndexerGradKLLossTriton(scale_value=scale)
        cell_cann = SparseLightningIndexerGradKLLoss()

        try:
            t_med, t_p20, t_p80 = _do_bench(
                lambda: cell_tri(q, k, qi, ki, w, si, sm_max, sm_sum,
                                 query_rope=qr, key_rope=kr))
            print(f"triton:  median={t_med:.2f}ms, p20={t_p20:.2f}ms, p80={t_p80:.2f}ms")
        except Exception as e:
            print(f"triton:  FAILED - {e}")
            t_med = None

        try:
            o_med, o_p20, o_p80 = _do_bench(
                lambda: cell_cann(q, k, qi, ki, w, si, sm_max, sm_sum,
                                  scale_value=scale, query_rope=qr, key_rope=kr,
                                  layout="BSND", sparse_mode=3))
            print(f"cann:    median={o_med:.2f}ms, p20={o_p20:.2f}ms, p80={o_p80:.2f}ms")
        except Exception as e:
            print(f"cann:    FAILED - {e}")
            o_med = None

        if t_med is not None and o_med is not None and t_med > 0:
            print(f"speedup: {o_med / t_med:.2f}x")


def run_profiling():
    from sparse_lightning_indexer_grad_kl_loss_triton import SparseLightningIndexerGradKLLossTriton

    total_steps = 10
    out_dir = './profiler_data_sli'

    B, S1, S2, N1, Nidx1, topK = 1, 4096, 4096, 64, 8, 2048

    q, k, qr, kr, qi, ki, w, si, sm_max, sm_sum = _make_inputs(B, S1, S2, N1, Nidx1, topK)
    scale = 1.0 / np.sqrt(D_NOPE + D_ROPE)

    cell = SparseLightningIndexerGradKLLossTriton(scale_value=scale)

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
        experimental_config=experimental_config,
    ) as prof:

        for _ in range(total_steps):
            cell(q, k, qi, ki, w, si, sm_max, sm_sum, query_rope=qr, key_rope=kr)
            prof.step()

    print(f"Profiler data saved to {out_dir}")


def run_profiling_cann():
    from sli_grad_kl_loss_cann import SparseLightningIndexerGradKLLoss

    total_steps = 10
    out_dir = './profiler_data_sli_cann'

    B, S1, S2, N1, Nidx1, topK = 1, 4096, 4096, 64, 8, 2048

    q, k, qr, kr, qi, ki, w, si, sm_max, sm_sum = _make_inputs(B, S1, S2, N1, Nidx1, topK)
    scale = 1.0 / np.sqrt(D_NOPE + D_ROPE)

    cell = SparseLightningIndexerGradKLLoss()

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
            cell(q, k, qi, ki, w, si, sm_max, sm_sum,
                 scale_value=scale, query_rope=qr, key_rope=kr,
                 layout="BSND", sparse_mode=3)
            prof.step()

    print(f"Profiler data saved to {out_dir}")


def run_kernel_only():
    from sparse_lightning_indexer_grad_kl_loss_triton import SparseLightningIndexerGradKLLossTriton

    B, S1, S2, N1, Nidx1, topK = 1, 4096, 4096, 64, 8, 2048

    q, k, qr, kr, qi, ki, w, si, sm_max, sm_sum = _make_inputs(B, S1, S2, N1, Nidx1, topK)
    scale = 1.0 / np.sqrt(D_NOPE + D_ROPE)

    cell = SparseLightningIndexerGradKLLossTriton(scale_value=scale)

    for _ in range(10):
        cell(q, k, qi, ki, w, si, sm_max, sm_sum, query_rope=qr, key_rope=kr)
    print("kernel-only run finished")


if __name__ == "__main__":
    import sys

    ms.set_context(mode=ms.GRAPH_MODE)
    np.random.seed(42)
    ms.set_seed(42)

    if len(sys.argv) > 1 and sys.argv[1] == "--kernel-only":
        run_kernel_only()
    else:
        run_timing()
        run_profiling()
        run_profiling_cann()