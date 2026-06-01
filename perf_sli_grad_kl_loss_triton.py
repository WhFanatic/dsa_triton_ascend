import time
import numpy as np
import mindspore as ms
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics
from mindspore import runtime

from sparse_lightning_indexer_grad_kl_loss_triton import (
    SparseLightningIndexerGradKLLossTriton,
)
from sli_grad_kl_loss_cann import SparseLightningIndexerGradKLLoss


DROPE = 64


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


def _make_inputs(B, S1, S2, N1, D, Nidx1, D_idx, topK, dtype=ms.float16):
    """Create legal random inputs without doing the expensive reference FA pass."""
    q = ms.Tensor(np.random.randn(B, S1, N1, D).astype(np.float16), dtype=dtype).to("Ascend")
    k = ms.Tensor(np.random.randn(B, S2, 1, D).astype(np.float16), dtype=dtype).to("Ascend")
    qr = ms.Tensor(np.random.randn(B, S1, N1, DROPE).astype(np.float16), dtype=dtype).to("Ascend")
    kr = ms.Tensor(np.random.randn(B, S2, 1, DROPE).astype(np.float16), dtype=dtype).to("Ascend")
    qi = ms.Tensor(np.random.randn(B, S1, Nidx1, D_idx).astype(np.float16), dtype=dtype).to("Ascend")
    ki = ms.Tensor(np.random.randn(B, S2, 1, D_idx).astype(np.float16), dtype=dtype).to("Ascend")
    w = ms.Tensor(np.abs(np.random.randn(B, S1, Nidx1)).astype(np.float16), dtype=dtype).to("Ascend")
    si = ms.Tensor(np.random.randint(0, S2, (B, S1, 1, topK)).astype(np.int32), dtype=ms.int32).to("Ascend")

    softmax_max = ms.Tensor(
        np.random.randn(B, 1, S1, N1).astype(np.float32), dtype=ms.float32
    ).to("Ascend")
    softmax_sum = ms.Tensor(
        np.random.uniform(1.0, 32.0, (B, 1, S1, N1)).astype(np.float32), dtype=ms.float32
    ).to("Ascend")

    return q, k, qr, kr, qi, ki, w, si, softmax_max, softmax_sum


def run_timing():
    cann_comparable_configs = [
    # B, S1, S2, N1, D, Nidx1, D_idx, topK

    # S1 == S2
    (1, 128, 128, 32, 512, 8, 128, 1024),
    (1, 256, 256, 32, 512, 8, 128, 1024),
    (1, 512, 512, 32, 512, 8, 128, 1024),

    # S1 < S2
    (1, 128, 256, 32, 512, 8, 128, 1024),
    (1, 128, 512, 32, 512, 8, 128, 1024),
    (1, 256, 512, 32, 512, 8, 128, 1024),
    ]

    for label, configs in (
        ("cann-comparable", cann_comparable_configs),
    ):
        print(f"\n=== {label} ===")
        for B, S1, S2, N1, D, Nidx1, D_idx, topK in configs:
            _run_one_config(label, B, S1, S2, N1, D, Nidx1, D_idx, topK)


def _cann_supports_config(D, topK):
    return D == 512 and topK < 8192 and topK % 1024 == 0


def _run_one_config(label, B, S1, S2, N1, D, Nidx1, D_idx, topK):
    print(
        f"\nB={B}, S1={S1}, S2={S2}, N1={N1}, D={D}, "
        f"Nidx1={Nidx1}, D_idx={D_idx}, topK={topK}"
    )

    scale_value = 1.0 / np.sqrt(D)
    q, k, qr, kr, qi, ki, w, si, softmax_max, softmax_sum = _make_inputs(
        B, S1, S2, N1, D, Nidx1, D_idx, topK
    )

    cell = SparseLightningIndexerGradKLLossTriton(
        scale_value=scale_value, layout="BSND", sparse_mode=3,
    )
    t_med, t_p20, t_p80 = _do_bench(
        lambda cell=cell: cell(
            q, k, qi, ki, w, si, softmax_max, softmax_sum,
            query_rope=qr, key_rope=kr,
        )
    )
    print(f"triton:  median={t_med:.2f}ms, p20={t_p20:.2f}ms, p80={t_p80:.2f}ms")

    if label != "cann-comparable":
        print("cann:    skipped (triton-only config)")
        return
    if not _cann_supports_config(D, topK):
        print("cann:    skipped (unsupported by CANN: D must be 512 and topK must be a 1024 multiple)")
        return

    op = SparseLightningIndexerGradKLLoss()
    o_med, o_p20, o_p80 = _do_bench(
        lambda op=op: op(
            q, k, qi, ki, w, si, softmax_max, softmax_sum,
            scale_value=scale_value,
            query_rope=qr, key_rope=kr,
            layout="BSND",
            sparse_mode=3,
        )
    )
    print(f"cann:    median={o_med:.2f}ms, p20={o_p20:.2f}ms, p80={o_p80:.2f}ms")

    speedup = o_med / t_med if t_med > 0 else float("inf")
    print(f"speedup: {speedup:.2f}x")


def run_profiling():
    total_steps = 10
    out_dir = "./profiler_data_sli_grad_kl_loss"
    B, S1, S2, N1, D, Nidx1, D_idx, topK = 1, 512, 128, 32, 512, 8, 128, 1024
    scale_value = 1.0 / np.sqrt(D)
    q, k, qr, kr, qi, ki, w, si, softmax_max, softmax_sum = _make_inputs(
        B, S1, S2, N1, D, Nidx1, D_idx, topK
    )

    cell = SparseLightningIndexerGradKLLossTriton(
        scale_value=scale_value, layout="BSND", sparse_mode=3,
    )

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
            cell(
                q, k, qi, ki, w, si, softmax_max, softmax_sum,
                query_rope=qr, key_rope=kr,
            )
            prof.step()

    print(f"Profiler data saved to {out_dir}")


if __name__ == "__main__":
    ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
    np.random.seed(42)
    ms.set_seed(42)

    run_timing()
    # run_profiling()
