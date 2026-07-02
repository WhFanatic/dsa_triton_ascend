import time
import numpy as np
import mindspore as ms
from mindspore import runtime, ops
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics

from sparse_lightning_indexer_grad_kl_loss_triton import (
    SparseLightningIndexerGradKLLossTriton,
)
from sli_grad_kl_loss_cann import SparseLightningIndexerGradKLLoss


DROPE = 64


def _cann_supports_config(D, topK):
    return D == 512 and topK % 1024 == 0


def _do_bench(fn, warmup=5, rep=3):
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


def _make_sparse_indices(B, S1, S2, topK):
    si = np.zeros((B, S1, 1, topK), dtype=np.int32)
    for s1 in range(S1):
        visible = min(max(S2 - S1 + s1 + 1, 1), S2)
        valid_k = min(topK, visible)
        si[:, s1, 0, :valid_k] = np.arange(valid_k, dtype=np.int32)
    return ms.Tensor(si, dtype=ms.int32).to("Ascend")


def _make_inputs(B, S1, S2, N1, D, Nidx1, D_idx, topK, dtype=ms.bfloat16):
    q = ms.Tensor(np.random.randn(B, S1, N1, D).astype(np.float32), dtype=dtype).to("Ascend")
    k = ms.Tensor(np.random.randn(B, S2, 1, D).astype(np.float32), dtype=dtype).to("Ascend")
    qr = ms.Tensor(np.random.randn(B, S1, N1, DROPE).astype(np.float32), dtype=dtype).to("Ascend")
    kr = ms.Tensor(np.random.randn(B, S2, 1, DROPE).astype(np.float32), dtype=dtype).to("Ascend")
    qi = ms.Tensor(np.random.randn(B, S1, Nidx1, D_idx).astype(np.float32), dtype=dtype).to("Ascend")
    ki = ms.Tensor(np.random.randn(B, S2, 1, D_idx).astype(np.float32), dtype=dtype).to("Ascend")
    w = ms.Tensor(np.abs(np.random.randn(B, S1, Nidx1)).astype(np.float32), dtype=dtype).to("Ascend")
    si = _make_sparse_indices(B, S1, S2, topK)
    softmax_max = ms.Tensor(
        np.random.randn(B, 1, S1, N1).astype(np.float32), dtype=ms.float32
    ).to("Ascend")
    softmax_sum = ms.Tensor(
        np.random.uniform(1.0, 32.0, (B, 1, S1, N1)).astype(np.float32), dtype=ms.float32
    ).to("Ascend")
    return q, k, qr, kr, qi, ki, w, si, softmax_max, softmax_sum


def _force_grad_outputs(outputs):
    """Keep all returned gradients live for fair profiling.

    The perf path does not inspect returned tensors. After loss accumulation was
    changed from an in-kernel atomic side effect to loss_parts + ReduceSum, graph
    optimization can otherwise remove the main backward kernel from task_time.
    """
    d_qi, d_ki, d_w, loss = outputs
    token = ops.depend(loss, d_qi)
    token = ops.depend(token, d_ki)
    token = ops.depend(token, d_w)
    return token


def _materialize_grad_outputs(outputs):
    """Force all returned tensors to be materialized in profiler runs."""
    d_qi, d_ki, d_w, loss = outputs
    d_qi.asnumpy()
    d_ki.asnumpy()
    d_w.asnumpy()
    loss.asnumpy()
    return outputs


def run_timing():
    configs = [
        (1, 512, 4096, 64, 512, 64, 128, 2048),
    ]

    for B, S1, S2, N1, D, Nidx1, D_idx, topK in configs:
        print(
            f"\nB={B}, S1={S1}, S2={S2}, N1={N1}, N2=1, D={D}, "
            f"Nidx1={Nidx1}, Nidx2=1, D_idx={D_idx}, topK={topK}"
        )

        scale_value = 1.0 / np.sqrt(D)
        q, k, qr, kr, qi, ki, w, si, softmax_max, softmax_sum = _make_inputs(
            B, S1, S2, N1, D, Nidx1, D_idx, topK
        )
        cell = SparseLightningIndexerGradKLLossTriton(
            scale_value=scale_value, layout="BSND", sparse_mode=3,
        )

        def _triton_call(cell=cell):
            return _force_grad_outputs(cell(
                q, k, qi, ki, w, si, softmax_max, softmax_sum,
                query_rope=qr, key_rope=kr,
            ))

        t_med, t_p20, t_p80 = _do_bench(_triton_call)
        print(f"triton:  median={t_med:.2f}ms, p20={t_p20:.2f}ms, p80={t_p80:.2f}ms")

        if not _cann_supports_config(D, topK):
            print("cann:    skipped (CANN requires D=512 for this op)")
            del q, k, qr, kr, qi, ki, w, si, softmax_max, softmax_sum, cell
            runtime.synchronize()
            runtime.empty_cache()
            continue

        op = SparseLightningIndexerGradKLLoss()
        def _cann_call(op=op):
            return _force_grad_outputs(op(
                q, k, qi, ki, w, si, softmax_max, softmax_sum,
                scale_value=scale_value,
                query_rope=qr, key_rope=kr,
                layout="BSND",
                sparse_mode=3,
            ))

        o_med, o_p20, o_p80 = _do_bench(_cann_call)
        speedup = o_med / t_med if t_med > 0 else float("inf")

        print(f"cann:    median={o_med:.2f}ms, p20={o_p20:.2f}ms, p80={o_p80:.2f}ms")
        if t_med > 0:
            print(f"speedup: {speedup:.2f}x")

        del q, k, qr, kr, qi, ki, w, si, softmax_max, softmax_sum, cell, op
        runtime.synchronize()
        runtime.empty_cache()


def run_profiling():
    total_steps = 8
    out_dir = "./profiler_data_sli_grad_kl_loss"

    B, S1, S2, N1, D, Nidx1, D_idx, topK = 1, 512, 4096, 64, 512, 64, 128, 2048

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
        schedule=ms.profiler.schedule(wait=0, warmup=5, active=3, repeat=1, skip_first=0),
        on_trace_ready=ms.profiler.tensorboard_trace_handler(out_dir),
        profile_memory=False,
        experimental_config=experimental_config
        ) as prof:

        for _ in range(total_steps):
            out = cell(
                q, k, qi, ki, w, si, softmax_max, softmax_sum,
                query_rope=qr, key_rope=kr,
            )
            _materialize_grad_outputs(out)
            runtime.synchronize()
            prof.step()
            del out

    print(f"Profiler data saved to {out_dir}")


def run_profiling_cann():
    total_steps = 10
    out_dir = "./profiler_data_sli_grad_kl_loss_cann"

    B, S1, S2, N1, D, Nidx1, D_idx, topK = 1, 512, 4096, 64, 512, 64, 128, 2048
    if not _cann_supports_config(D, topK):
        print(
            "CANN profiling skipped: CANN requires D=512 for "
            f"this op, got D={D}, topK={topK}"
        )
        return

    scale_value = 1.0 / np.sqrt(D)
    q, k, qr, kr, qi, ki, w, si, softmax_max, softmax_sum = _make_inputs(
        B, S1, S2, N1, D, Nidx1, D_idx, topK
    )
    op = SparseLightningIndexerGradKLLoss()

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
            out = op(
                q, k, qi, ki, w, si, softmax_max, softmax_sum,
                scale_value=scale_value,
                query_rope=qr, key_rope=kr,
                layout="BSND",
                sparse_mode=3,
            )
            _materialize_grad_outputs(out)
            runtime.synchronize()
            prof.step()
            del out

    print(f"Profiler data saved to {out_dir}")


def run_kernel_only():
    B, S1, S2, N1, D, Nidx1, D_idx, topK = 1, 4096, 4096, 64, 512, 64, 128, 2048

    scale_value = 1.0 / np.sqrt(D)
    q, k, qr, kr, qi, ki, w, si, softmax_max, softmax_sum = _make_inputs(
        B, S1, S2, N1, D, Nidx1, D_idx, topK
    )
    cell = SparseLightningIndexerGradKLLossTriton(
        scale_value=scale_value, layout="BSND", sparse_mode=3,
    )

    for _ in range(1):
        out = cell(
            q, k, qi, ki, w, si, softmax_max, softmax_sum,
            query_rope=qr, key_rope=kr,
        )
        _materialize_grad_outputs(out)
        runtime.synchronize()
        del out
    print("kernel-only run finished")


if __name__ == "__main__":
    import sys

    ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
    np.random.seed(42)
    ms.set_seed(42)

    if len(sys.argv) > 1 and sys.argv[1] == "--timing-only":
        run_timing()
    elif len(sys.argv) > 1 and sys.argv[1] == "--kernel-only":
        run_kernel_only()
    elif len(sys.argv) > 1 and sys.argv[1] == "--triton-only":
        run_profiling()
    elif len(sys.argv) > 1 and sys.argv[1] == "--cann-only":
        run_profiling_cann()
    else:
        run_timing()
        run_profiling()
        run_profiling_cann()