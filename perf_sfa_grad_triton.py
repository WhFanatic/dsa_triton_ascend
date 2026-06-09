"""SFA backward (sparse_flash_attention_grad) performance benchmark.

Usage (Ascend NPU required):
    export ASCEND_RT_VISIBLE_DEVICES=8 TRITON_END=mindspore TRITON_BACKEND=mindspore TORCH_DEVICE_BACKEND_AUTOLOAD=0

    # timing: triton bwd vs CANN bwd (pure backward kernel, forward stats pre-computed)
    python perf_sfa_grad_triton.py

    # msprof op — single-kernel profiling (target: _sfa_grad_kernel)
    msprof op --kernel-name="_sfa_grad_kernel" --output=./profilers python perf_sfa_grad_triton.py --kernel-only
"""
import time
import numpy as np
import mindspore as ms
from mindspore import ops, runtime
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics

D_NOPE = 512  #客户建议优先测试数值256，来源kv_lora_rank:256
D_ROPE = 64

PROF_SHAPE = (1, 512, 4096, 64, 64)


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


def _make_inputs(B, S1, S2, N1, sparse_count, dtype=ms.float16, D=D_NOPE):
    rng = np.random.RandomState(42)

    def _t(shape):
        return ms.Tensor(rng.randn(*shape).astype(np.float16), dtype=dtype)

    q = _t((B, S1, N1, D))
    k = _t((B, S2, 1, D))
    v = _t((B, S2, 1, D))
    qr = _t((B, S1, N1, D_ROPE))
    kr = _t((B, S2, 1, D_ROPE))
    do = _t((B, S1, N1, D))
    si = _make_sparse_indices(B, S1, S2, sparse_count, sparse_mode=3)
    return q, k, v, qr, kr, do, si


def run_timing():
    from sparse_flash_attention_triton import SparseFlashAttentionTriton
    from sparse_flash_attention_grad_triton import SparseFlashAttentionGradTriton
    from sfa_grad_cann import SparseFlashAttentionGradCANN

    configs = [
        (1, 128, 1024, 64, 16),
        (1, 256, 2048, 64, 32),
        (1, 512, 4096, 64, 64),
    ]

    for B, S1, S2, N1, sparse_count in configs:
        print(f"\nB={B}, S1={S1}, S2={S2}, N1={N1}, topk={sparse_count}")

        q, k, v, qr, kr, do, si = _make_inputs(B, S1, S2, N1, sparse_count)
        scale = 1.0 / np.sqrt(D_NOPE + D_ROPE)
        act_q = ms.Tensor([S1] * B, dtype=ms.int32)
        act_k = ms.Tensor([S2] * B, dtype=ms.int32)

        fwd_tri = SparseFlashAttentionTriton(
            scale_value=scale, sparse_mode=3, return_softmax_lse=True)
        out_tri, smax_tri, ssum_tri = fwd_tri(q, k, v, si, query_rope=qr, key_rope=kr)

        tri_grad = SparseFlashAttentionGradTriton(scale_value=scale, sparse_mode=3)

        try:
            t_med, t_p20, t_p80 = _do_bench(
                lambda: tri_grad(q, k, v, si, do, out_tri, smax_tri, ssum_tri,
                                 query_rope=qr, key_rope=kr))
            print(f"triton:  median={t_med:.2f}ms, p20={t_p20:.2f}ms, p80={t_p80:.2f}ms")
        except Exception as e:
            print(f"triton:  FAILED - {e}")
            t_med = None

        out_cann, smax_cann, ssum_cann = ops.sparse_flash_attention(
            q, k, v, si, scale,
            query_rope=qr, key_rope=kr,
            layout_query="BSND", layout_kv="BSND",
            sparse_block_size=1, sparse_mode=3,
            attention_mode=2, return_softmax_lse=True)

        cann_grad = SparseFlashAttentionGradCANN(scale_value=scale, sparse_mode=3)

        try:
            o_med, o_p20, o_p80 = _do_bench(
                lambda: cann_grad(q, k, v, si, do, out_cann, smax_cann, ssum_cann,
                                  query_rope=qr, key_rope=kr,
                                  actual_seq_lengths_query=act_q,
                                  actual_seq_lengths_kv=act_k))
            print(f"cann:    median={o_med:.2f}ms, p20={o_p20:.2f}ms, p80={o_p80:.2f}ms")
        except Exception as e:
            print(f"cann:    FAILED - {e}")
            o_med = None

        if t_med is not None and o_med is not None and t_med > 0:
            print(f"speedup: {o_med / t_med:.2f}x")


def run_profiling():
    from sparse_flash_attention_triton import SparseFlashAttentionTriton
    from sparse_flash_attention_grad_triton import SparseFlashAttentionGradTriton

    total_steps = 10
    out_dir = './profiler_data_sfa_grad'

    B, S1, S2, N1, sparse_count = PROF_SHAPE

    q, k, v, qr, kr, do, si = _make_inputs(B, S1, S2, N1, sparse_count)
    scale = 1.0 / np.sqrt(D_NOPE + D_ROPE)

    fwd = SparseFlashAttentionTriton(
        scale_value=scale, sparse_mode=3, return_softmax_lse=True)
    out, smax, ssum = fwd(q, k, v, si, query_rope=qr, key_rope=kr)

    grad = SparseFlashAttentionGradTriton(scale_value=scale, sparse_mode=3)

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
            grad(q, k, v, si, do, out, smax, ssum, query_rope=qr, key_rope=kr)
            prof.step()

    print(f"Profiler data saved to {out_dir}")


def run_profiling_cann():
    from sfa_grad_cann import SparseFlashAttentionGradCANN

    total_steps = 10
    out_dir = './profiler_data_sfa_grad_cann'

    B, S1, S2, N1, sparse_count = PROF_SHAPE

    q, k, v, qr, kr, do, si = _make_inputs(B, S1, S2, N1, sparse_count)
    scale = 1.0 / np.sqrt(D_NOPE + D_ROPE)
    act_q = ms.Tensor([S1] * B, dtype=ms.int32)
    act_k = ms.Tensor([S2] * B, dtype=ms.int32)

    out, smax, ssum = ops.sparse_flash_attention(
        q, k, v, si, scale,
        query_rope=qr, key_rope=kr,
        layout_query="BSND", layout_kv="BSND",
        sparse_block_size=1, sparse_mode=3,
        attention_mode=2, return_softmax_lse=True)

    cann_grad = SparseFlashAttentionGradCANN(scale_value=scale, sparse_mode=3)

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
            cann_grad(q, k, v, si, do, out, smax, ssum,
                       query_rope=qr, key_rope=kr,
                       actual_seq_lengths_query=act_q,
                       actual_seq_lengths_kv=act_k)
            prof.step()

    print(f"Profiler data saved to {out_dir}")


def run_kernel_only():
    from sparse_flash_attention_triton import SparseFlashAttentionTriton
    from sparse_flash_attention_grad_triton import SparseFlashAttentionGradTriton

    B, S1, S2, N1, sparse_count = PROF_SHAPE

    q, k, v, qr, kr, do, si = _make_inputs(B, S1, S2, N1, sparse_count)
    scale = 1.0 / np.sqrt(D_NOPE + D_ROPE)

    fwd = SparseFlashAttentionTriton(
        scale_value=scale, sparse_mode=3, return_softmax_lse=True)
    out, smax, ssum = fwd(q, k, v, si, query_rope=qr, key_rope=kr)

    grad = SparseFlashAttentionGradTriton(scale_value=scale, sparse_mode=3)

    for _ in range(10):
        grad(q, k, v, si, do, out, smax, ssum, query_rope=qr, key_rope=kr)
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