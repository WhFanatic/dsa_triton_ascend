import time
import numpy as np
import mindspore as ms
from mindspore import ops, runtime
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics

D_NOPE = 512  #来源kv_lora_rank:256
D_ROPE = 64
DEFAULT_CONFIG = (1, 512, 4096, 64, 2048)


def _do_bench(fn, warmup=3, rep=10):
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


def _make_sparse_indices(B, S1, S2, sparse_count, sparse_mode):
    rng = np.random.RandomState(7)
    si = np.full((B, S1, 1, sparse_count), -1, dtype=np.int32)
    act_q, act_k = S1, S2
    for b in range(B):
        for s1 in range(S1):
            threshold = act_k if sparse_mode == 0 else act_k - act_q + s1 + 1
            if threshold <= 0:
                continue
            n = min(sparse_count, threshold)
            perm = rng.permutation(threshold)[:n]
            si[b, s1, 0, :n] = np.sort(perm).astype(np.int32)
    return ms.Tensor(si, dtype=ms.int32)


def _make_inputs(B, S1, S2, N1, sparse_count, dtype=ms.bfloat16, D=D_NOPE):
    rng = np.random.RandomState(42)

    def _t(shape):
        return ms.Tensor(rng.randn(*shape).astype(np.float16), dtype=dtype)

    q = _t((B, S1, N1, D))
    k = _t((B, S2, 1, D))
    v = _t((B, S2, 1, D))
    qr = _t((B, S1, N1, D_ROPE))
    kr = _t((B, S2, 1, D_ROPE))
    si = _make_sparse_indices(B, S1, S2, sparse_count, sparse_mode=3)
    return q, k, v, qr, kr, si


def run_autotune_confirm():
    from sparse_flash_attention_triton import SparseFlashAttentionTriton

    B, S1, S2, N1, sparse_count = DEFAULT_CONFIG
    q, k, v, qr, kr, si = _make_inputs(B, S1, S2, N1, sparse_count)
    scale = 1.0 / np.sqrt(D_NOPE + D_ROPE)

    print(f"\nShape: B={B}, S1={S1}, S2={S2}, N1={N1}, topK={sparse_count}, D={D_NOPE}")
    print("Running 1 invocation to trigger autotune — TRITON_PRINT_AUTOTUNING=1 output in stderr")
    print("=" * 80)

    cell = SparseFlashAttentionTriton(
        scale_value=scale, sparse_mode=3, return_softmax_lse=True,
    )
    result = cell(q, k, v, si, query_rope=qr, key_rope=kr)
    runtime.synchronize()
    print(f"Autotune done. Output[0] shape: {result[0].shape}")

    del q, k, v, qr, kr, si, cell, result
    runtime.synchronize()
    runtime.empty_cache()


def run_timing():
    from sparse_flash_attention_triton import SparseFlashAttentionTriton

    configs = [
        DEFAULT_CONFIG,
    ]

    for B, S1, S2, N1, sparse_count in configs:
        print(f"\nB={B}, S1={S1}, S2={S2}, N1={N1}, topk={sparse_count}")

        q, k, v, qr, kr, si = _make_inputs(B, S1, S2, N1, sparse_count)
        scale = 1.0 / np.sqrt(D_NOPE + D_ROPE)

        cell = SparseFlashAttentionTriton(
            scale_value=scale, sparse_mode=3, return_softmax_lse=True,
        )

        try:
            t_med, t_p20, t_p80 = _do_bench(
                lambda: cell(q, k, v, si, query_rope=qr, key_rope=kr))
            print(f"triton:  median={t_med:.2f}ms, p20={t_p20:.2f}ms, p80={t_p80:.2f}ms")
        except Exception as e:
            print(f"triton:  FAILED - {e}")
            t_med = None

        try:
            o_med, o_p20, o_p80 = _do_bench(
                lambda: ops.sparse_flash_attention(
                    q, k, v, si, scale,
                    query_rope=qr, key_rope=kr,
                    layout_query="BSND", layout_kv="BSND",
                    sparse_block_size=1, sparse_mode=3,
                    attention_mode=2, return_softmax_lse=True,
                ))
            print(f"cann:    median={o_med:.2f}ms, p20={o_p20:.2f}ms, p80={o_p80:.2f}ms")
        except Exception as e:
            print(f"cann:    FAILED - {e}")
            o_med = None

        if t_med is not None and o_med is not None and t_med > 0:
            print(f"speedup: {o_med / t_med:.2f}x")

        del q, k, v, qr, kr, si, cell
        runtime.synchronize()
        runtime.empty_cache()


def run_profiling():
    from sparse_flash_attention_triton import SparseFlashAttentionTriton

    total_steps = 10
    out_dir = './profiler_data_sfa'

    B, S1, S2, N1, sparse_count = DEFAULT_CONFIG

    q, k, v, qr, kr, si = _make_inputs(B, S1, S2, N1, sparse_count)
    scale = 1.0 / np.sqrt(D_NOPE + D_ROPE)

    cell = SparseFlashAttentionTriton(
        scale_value=scale, sparse_mode=3, return_softmax_lse=True,
    )

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
            cell(q, k, v, si, query_rope=qr, key_rope=kr)
            prof.step()

    print(f"Profiler data saved to {out_dir}")


def run_profiling_cann():
    total_steps = 10
    out_dir = './profiler_data_sfa_cann'

    B, S1, S2, N1, sparse_count = DEFAULT_CONFIG

    q, k, v, qr, kr, si = _make_inputs(B, S1, S2, N1, sparse_count)
    scale = 1.0 / np.sqrt(D_NOPE + D_ROPE)

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
            ops.sparse_flash_attention(
                q, k, v, si, scale,
                query_rope=qr, key_rope=kr,
                layout_query="BSND", layout_kv="BSND",
                sparse_block_size=1, sparse_mode=3,
                attention_mode=2, return_softmax_lse=True,
            )
            prof.step()

    print(f"Profiler data saved to {out_dir}")


def run_kernel_only():
    from sparse_flash_attention_triton import SparseFlashAttentionTriton

    B, S1, S2, N1, sparse_count = DEFAULT_CONFIG

    q, k, v, qr, kr, si = _make_inputs(B, S1, S2, N1, sparse_count)
    scale = 1.0 / np.sqrt(D_NOPE + D_ROPE)

    cell = SparseFlashAttentionTriton(
        scale_value=scale, sparse_mode=3, return_softmax_lse=True,
    )

    for _ in range(10):
        cell(q, k, v, si, query_rope=qr, key_rope=kr)
    print("kernel-only run finished")


if __name__ == "__main__":
    import sys

    ms.set_context(mode=ms.GRAPH_MODE)
    np.random.seed(42)
    ms.set_seed(42)

    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "--kernel-only":
        run_kernel_only()
    elif mode == "--autotune-confirm":
        run_autotune_confirm()
    else:
        run_timing()
        run_profiling()
        run_profiling_cann()
