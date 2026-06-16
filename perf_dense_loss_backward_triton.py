import time
import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore import runtime
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics

from dense_loss_backward_triton import (
    DenseLightningIndexerSoftmaxLseTriton,
    DenseLightningIndexerGradKLLossTriton,
)

INT64_MAX = 9223372036854775807
DROPE = 64
# Customer-priority shape while keeping CANN DenseLightningIndexerGradKLLoss comparable:
# B, S1, S2, N1, N2, D, Nidx1, D_idx
# Dense CANN grad currently supports D=128 and D_idx=128; use N1/Nidx1=64.
DENSE_PROFILE_CONFIG = (1, 512, 4096, 64, 64, 128, 64, 128)


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


def _make_inputs(B, S1, S2, N1, N2, D, Nidx1, D_idx, dtype=ms.float16):
    q = ms.Tensor(np.random.randn(B, S1, N1, D).astype(np.float16), dtype=dtype).to("Ascend")
    k = ms.Tensor(np.random.randn(B, S2, N2, D).astype(np.float16), dtype=dtype).to("Ascend")
    qi = ms.Tensor(np.random.randn(B, S1, Nidx1, D_idx).astype(np.float16), dtype=dtype).to("Ascend")
    ki = ms.Tensor(np.random.randn(B, S2, 1, D_idx).astype(np.float16), dtype=dtype).to("Ascend")
    w = ms.Tensor(np.abs(np.random.randn(B, S1, Nidx1)).astype(np.float16), dtype=dtype).to("Ascend")
    qr = ms.Tensor(np.random.randn(B, S1, N1, DROPE).astype(np.float16), dtype=dtype).to("Ascend")
    kr = ms.Tensor(np.random.randn(B, S2, N2, DROPE).astype(np.float16), dtype=dtype).to("Ascend")
    softmax_max = ms.Tensor(
        np.random.randn(B, N2, S1, N1 // N2).astype(np.float32), dtype=ms.float32
    ).to("Ascend")
    softmax_sum = ms.Tensor(
        np.random.uniform(1.0, 32.0, (B, N2, S1, N1 // N2)).astype(np.float32), dtype=ms.float32
    ).to("Ascend")
    return q, k, qi, ki, w, qr, kr, softmax_max, softmax_sum


def _force_grad_outputs(outputs):
    """Create control dependencies on all grad outputs for fair profiling.

    Without consuming d_qi/d_ki/d_w, graph optimization may keep only the loss
    path in task_time, hiding the actual backward kernels from profiling.
    """
    d_qi, d_ki, d_w, loss = outputs
    token = ops.depend(loss, d_qi)
    token = ops.depend(token, d_ki)
    token = ops.depend(token, d_w)
    return token


def _materialize_grad_outputs(outputs):
    """Force all grad outputs to be materialized for profiler full-chain runs."""
    d_qi, d_ki, d_w, loss = outputs
    d_qi.asnumpy()
    d_ki.asnumpy()
    d_w.asnumpy()
    loss.asnumpy()
    return outputs


def _resolve_official_dense_ops():
    errors = []
    try:
        from hyper_parallel.custom_ops.experimental import experimental_ops
        lse = getattr(experimental_ops, "npu_dense_lightning_indexer_softmax_lse", None)
        grad = getattr(experimental_ops, "npu_dense_lightning_indexer_grad_kl_loss", None)
        if lse is not None and grad is not None:
            return lse, grad, "hyper_parallel.custom_ops.experimental.experimental_ops"
        errors.append("hyper_parallel experimental ops missing dense lightning functions")
    except Exception as exc:
        errors.append(f"hyper_parallel import failed: {type(exc).__name__}: {exc}")

    lse = getattr(ops, "npu_dense_lightning_indexer_softmax_lse", None)
    grad = getattr(ops, "npu_dense_lightning_indexer_grad_kl_loss", None)
    if lse is not None and grad is not None:
        return lse, grad, "mindspore.ops"
    errors.append("mindspore.ops missing npu_dense_lightning_indexer_* functions")

    try:
        from dense_loss_backward_cann import (
            DenseLightningIndexerSoftmaxLse,
            DenseLightningIndexerGradKLLoss,
        )
        return (
            DenseLightningIndexerSoftmaxLse(),
            DenseLightningIndexerGradKLLoss(),
            "dense_loss_backward_cann local CustomRegOp wrappers",
        )
    except Exception as exc:
        errors.append(f"dense_loss_backward_cann import failed: {type(exc).__name__}: {exc}")

    return None, None, "; ".join(errors)


def _call_official_lse(lse_op, qi, ki, w):
    return lse_op(
        qi, ki, w,
        actual_seq_qlen=None,
        actual_seq_klen=None,
        layout="BSND",
        sparse_mode=3,
        pre_tokens=INT64_MAX,
        next_tokens=INT64_MAX,
    )


def _call_official_grad(grad_op, q, k, qi, ki, w,
                        softmax_max, softmax_sum, max_index, sum_index,
                        scale_value, query_rope, key_rope):
    return grad_op(
        q, k, qi, ki, w,
        softmax_max, softmax_sum,
        max_index, sum_index,
        scale_value=scale_value,
        query_rope=query_rope,
        key_rope=key_rope,
        actual_seq_qlen=None,
        actual_seq_klen=None,
        layout="BSND",
        sparse_mode=3,
        pre_tokens=INT64_MAX,
        next_tokens=INT64_MAX,
    )


def run_timing():
    configs = [
        # B, S1, S2, N1, N2, D, Nidx1, D_idx
        DENSE_PROFILE_CONFIG,
    ]
    official_lse, official_grad, official_source = _resolve_official_dense_ops()
    print(f"official dense DLI source: {official_source}")

    for B, S1, S2, N1, N2, D, Nidx1, D_idx in configs:
        print(
            f"\nB={B}, S1={S1}, S2={S2}, N1={N1}, N2={N2}, "
            f"D={D}, Nidx1={Nidx1}, D_idx={D_idx}, D_rope={DROPE}"
        )
        scale_value = 1.0 / np.sqrt(D + DROPE)
        q, k, qi, ki, w, qr, kr, softmax_max, softmax_sum = _make_inputs(
            B, S1, S2, N1, N2, D, Nidx1, D_idx)

        stats_cell = DenseLightningIndexerSoftmaxLseTriton(layout="BSND", sparse_mode=3)
        grad_cell = DenseLightningIndexerGradKLLossTriton(scale_value=scale_value, layout="BSND", sparse_mode=3)
        max_index, sum_index = stats_cell(qi, ki, w)

        t_med, t_p20, t_p80 = _do_bench(lambda stats_cell=stats_cell: stats_cell(qi, ki, w))
        print(f"triton softmax_lse:  median={t_med:.2f}ms, p20={t_p20:.2f}ms, p80={t_p80:.2f}ms")
        triton_lse_med = t_med
        official_max_index = None
        official_sum_index = None
        if official_lse is None:
            print(f"official softmax_lse: skipped ({official_source})")
        else:
            official_lse_call = lambda: _call_official_lse(official_lse, qi, ki, w)
            official_max_index, official_sum_index = official_lse_call()
            o_med, o_p20, o_p80 = _do_bench(official_lse_call)
            speedup = o_med / triton_lse_med if triton_lse_med > 0 else float("inf")
            print(
                f"official softmax_lse: median={o_med:.2f}ms, "
                f"p20={o_p20:.2f}ms, p80={o_p80:.2f}ms, speedup={speedup:.2f}x"
            )

        def _triton_grad():
            return _force_grad_outputs(grad_cell(
                q, k, qi, ki, w, softmax_max, softmax_sum, max_index, sum_index,
                query_rope=qr, key_rope=kr,
            ))

        t_med, t_p20, t_p80 = _do_bench(_triton_grad)
        print(f"triton grad_kl_loss: median={t_med:.2f}ms, p20={t_p20:.2f}ms, p80={t_p80:.2f}ms")
        triton_grad_med = t_med
        if official_grad is None or official_max_index is None:
            print("official grad_kl_loss: skipped (official softmax_lse baseline unavailable)")
            official_grad_call = None
        else:
            def official_grad_call():
                return _force_grad_outputs(_call_official_grad(
                    official_grad, q, k, qi, ki, w, softmax_max, softmax_sum,
                    official_max_index, official_sum_index, scale_value, qr, kr,
                ))
            o_med, o_p20, o_p80 = _do_bench(official_grad_call)
            speedup = o_med / triton_grad_med if triton_grad_med > 0 else float("inf")
            print(
                f"official grad_kl_loss: median={o_med:.2f}ms, "
                f"p20={o_p20:.2f}ms, p80={o_p80:.2f}ms, speedup={speedup:.2f}x"
            )

        def _combined():
            mi, si = stats_cell(qi, ki, w)
            return _force_grad_outputs(grad_cell(
                q, k, qi, ki, w, softmax_max, softmax_sum, mi, si,
                query_rope=qr, key_rope=kr,
            ))

        t_med, t_p20, t_p80 = _do_bench(_combined)
        print(f"triton combined:     median={t_med:.2f}ms, p20={t_p20:.2f}ms, p80={t_p80:.2f}ms")
        triton_combined_med = t_med

        if official_lse is None or official_grad is None:
            print("official combined:   skipped (official dense baseline unavailable)")
        elif official_grad_call is None:
            print("official combined:   skipped (official grad_kl_loss baseline unavailable)")
        else:
            def _official_combined():
                mi, si = _call_official_lse(official_lse, qi, ki, w)
                return _force_grad_outputs(_call_official_grad(
                    official_grad, q, k, qi, ki, w, softmax_max, softmax_sum,
                    mi, si, scale_value, qr, kr,
                ))

            o_med, o_p20, o_p80 = _do_bench(_official_combined)
            speedup = o_med / triton_combined_med if triton_combined_med > 0 else float("inf")
            print(
                f"official combined:   median={o_med:.2f}ms, "
                f"p20={o_p20:.2f}ms, p80={o_p80:.2f}ms, speedup={speedup:.2f}x"
            )


def run_profiling():
    total_steps = 10
    out_dir = "./profiler_data_dense_loss_backward"
    B, S1, S2, N1, N2, D, Nidx1, D_idx = DENSE_PROFILE_CONFIG
    scale_value = 1.0 / np.sqrt(D + DROPE)
    q, k, qi, ki, w, qr, kr, softmax_max, softmax_sum = _make_inputs(
        B, S1, S2, N1, N2, D, Nidx1, D_idx)

    stats_cell = DenseLightningIndexerSoftmaxLseTriton(layout="BSND", sparse_mode=3)
    grad_cell = DenseLightningIndexerGradKLLossTriton(scale_value=scale_value, layout="BSND", sparse_mode=3)

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
            max_index, sum_index = stats_cell(qi, ki, w)
            out = grad_cell(
                q, k, qi, ki, w, softmax_max, softmax_sum, max_index, sum_index,
                query_rope=qr, key_rope=kr,
            )
            _materialize_grad_outputs(out)
            runtime.synchronize()
            prof.step()
            del max_index, sum_index, out

    print(f"Profiler data saved to {out_dir}")


def run_profiling_cann():
    total_steps = 10
    out_dir = "./profiler_data_dense_loss_backward_cann"
    B, S1, S2, N1, N2, D, Nidx1, D_idx = DENSE_PROFILE_CONFIG
    scale_value = 1.0 / np.sqrt(D + DROPE)
    q, k, qi, ki, w, qr, kr, softmax_max, softmax_sum = _make_inputs(
        B, S1, S2, N1, N2, D, Nidx1, D_idx)

    official_lse, official_grad, official_source = _resolve_official_dense_ops()
    if official_lse is None or official_grad is None:
        print(f"CANN dense profiling skipped: {official_source}")
        return
    print(f"CANN dense DLI source: {official_source}")

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
            max_index, sum_index = _call_official_lse(official_lse, qi, ki, w)
            out = _call_official_grad(
                official_grad, q, k, qi, ki, w, softmax_max, softmax_sum,
                max_index, sum_index, scale_value, qr, kr,
            )
            _materialize_grad_outputs(out)
            runtime.synchronize()
            prof.step()
            del max_index, sum_index, out

    print(f"Profiler data saved to {out_dir}")


def run_kernel_only():
    B, S1, S2, N1, N2, D, Nidx1, D_idx = DENSE_PROFILE_CONFIG
    scale_value = 1.0 / np.sqrt(D + DROPE)
    q, k, qi, ki, w, qr, kr, softmax_max, softmax_sum = _make_inputs(
        B, S1, S2, N1, N2, D, Nidx1, D_idx)

    stats_cell = DenseLightningIndexerSoftmaxLseTriton(layout="BSND", sparse_mode=3)
    grad_cell = DenseLightningIndexerGradKLLossTriton(
        scale_value=scale_value, layout="BSND", sparse_mode=3)

    for _ in range(10):
        max_index, sum_index = stats_cell(qi, ki, w)
        out = grad_cell(
            q, k, qi, ki, w, softmax_max, softmax_sum, max_index, sum_index,
            query_rope=qr, key_rope=kr,
        )
        _materialize_grad_outputs(out)
        runtime.synchronize()
        del max_index, sum_index, out
    print("kernel-only run finished")


if __name__ == "__main__":
    import sys

    ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
    np.random.seed(42)
    ms.set_seed(42)

    if len(sys.argv) > 1 and sys.argv[1] == "--kernel-only":
        run_kernel_only()
    else:
        run_timing()
        run_profiling()
        run_profiling_cann()
