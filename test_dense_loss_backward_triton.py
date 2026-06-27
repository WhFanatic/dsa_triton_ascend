"""Test dense_loss_backward_triton."""

import pytest
import numpy as np
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})

INT64_MAX = 9223372036854775807
DROPE = 64
ATOL = 1e-2
RTOL = 1e-2
FP32_ATOL = 1e-3
FP32_RTOL = 1e-3
MAX_PRINT_VALUES = 8
SUPPORTED_D = (128, 256, 512)
SUPPORTED_NIDX1 = (32, 64, 128)
SUPPORTED_D_IDX = (128, 256, 512)

DENSE_CORE_TEST_CONFIGS = [
    (1, 2, 64, 32, 32, d, nidx1, d_idx)
    for d in SUPPORTED_D
    for nidx1 in SUPPORTED_NIDX1
    for d_idx in SUPPORTED_D_IDX
]

EXTRA_DENSE_TEST_CONFIGS = [
    # B, S1, S2, N1, N2, D, Nidx1, D_idx
    (1, 3, 96, 32, 32, 128, 32, 128),  # tail block on S2
    (1, 4, 128, 32, 32, 128, 32, 128),  # long key sequence, S2 is block-aligned
]

DENSE_LSE_TEST_CONFIGS = DENSE_CORE_TEST_CONFIGS + [
    config for config in EXTRA_DENSE_TEST_CONFIGS
    if config not in DENSE_CORE_TEST_CONFIGS
]

DENSE_GRAD_CANN_TEST_CONFIGS = [
    # CANN aclnnDenseLightningIndexerGradKLLoss only supports D=128,
    # D_idx=128, and Nidx1 in 8/16/32/64. Keep the independent grad
    # baseline inside that official range.
    (1, 2, 64, 32, 32, 128, 32, 128),
    (1, 2, 64, 32, 32, 128, 64, 128),
    (1, 3, 96, 32, 32, 128, 32, 128),
    (1, 4, 128, 32, 32, 128, 32, 128),
]

DENSE_TEST_CONFIGS = DENSE_GRAD_CANN_TEST_CONFIGS

DENSE_CANN_LSE_CHECK_CONFIGS = [
    # (1, 4, 128, 32, 32, 128, 32, 128),
    (1, 3, 96, 32, 32, 128, 64, 128),
]

DENSE_FP16_SMOKE = [
    (1, 2, 64, 32, 32, 128, 32, 128),
    (1, 2, 64, 32, 32, 256, 64, 256),
    (1, 2, 64, 32, 32, 512, 128, 512),
    (1, 3, 96, 32, 32, 128, 32, 128),
]

DENSE_DTYPE_TEST_CONFIGS = (
    [(cfg + (ms.bfloat16,)) for cfg in DENSE_LSE_TEST_CONFIGS]
    + [(cfg + (ms.float16,)) for cfg in DENSE_FP16_SMOKE]
)


def _log(message):
    print(f"[dense-test] {message}", flush=True)


def _as_numpy(value):
    if hasattr(value, "asnumpy"):
        return value.asnumpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _assert_allclose_with_values(name, expected_value, actual_value,
                                 expected_label="baseline", actual_label="triton",
                                 atol=None, rtol=None):
    actual_raw = _as_numpy(actual_value)
    if atol is None:
        atol = FP32_ATOL if actual_raw.dtype == np.float32 else ATOL
    if rtol is None:
        rtol = FP32_RTOL if actual_raw.dtype == np.float32 else RTOL
    expected_np = _as_numpy(expected_value).astype(np.float32)
    actual_np = actual_raw.astype(np.float32)
    if np.allclose(expected_np, actual_np, atol=atol, rtol=rtol):
        return
    error = np.abs(expected_np - actual_np)
    greater = np.greater(error, atol + np.abs(actual_np) * rtol)
    loss_count = np.count_nonzero(greater)
    total_count = error.size
    ratio = loss_count / total_count
    if ratio < rtol:
        _log(f"{name} {actual_label} vs {expected_label}: {loss_count}/{total_count} "
             f"({ratio*100:.4f}%) exceed tol(atol={atol},rtol={rtol}), within ratio budget")
        return
    expected_flat = expected_np.reshape(-1)
    actual_flat = actual_np.reshape(-1)
    print(f"[dense-test] {name} {expected_label} shape={expected_np.shape}", flush=True)
    print(expected_flat[:MAX_PRINT_VALUES], flush=True)
    print(f"[dense-test] {name} {actual_label} shape={actual_np.shape}", flush=True)
    print(actual_flat[:MAX_PRINT_VALUES], flush=True)
    raise AssertionError(
        f"{name}: {loss_count}/{total_count} ({ratio*100:.4f}%) exceed tol(atol={atol},rtol={rtol})")


def _actual_seq_to_numpy(actual_seq, seq_len, batch_size):
    if actual_seq is None:
        return np.full((batch_size,), seq_len, dtype=np.int64)
    if hasattr(actual_seq, "asnumpy"):
        actual_seq = actual_seq.asnumpy()
    actual_seq = np.asarray(actual_seq, dtype=np.int64)
    if actual_seq.shape != (batch_size,):
        raise ValueError(
            f"actual seq length must have shape ({batch_size},), got {actual_seq.shape}"
        )
    return actual_seq


def _dense_indexer_lse_numpy(qi, ki, w, actual_seq_qlen=None, actual_seq_klen=None):
    """Official dense indexer LSE formula, implemented as a stable NumPy baseline."""
    qi_np = qi.asnumpy().astype(np.float32)
    ki_np = ki.asnumpy().astype(np.float32)
    w_np = w.asnumpy().astype(np.float32)

    B, S1, Nidx1, D_idx = qi_np.shape
    if ki_np.shape[0] != B or ki_np.shape[2] != 1 or ki_np.shape[3] != D_idx:
        raise ValueError(f"ki must be [B, S2, 1, D_idx], got {ki_np.shape}")
    if w_np.shape != (B, S1, Nidx1):
        raise ValueError(f"w must be [B, S1, Nidx1], got {w_np.shape}")

    S2 = ki_np.shape[1]
    actual_q = _actual_seq_to_numpy(actual_seq_qlen, S1, B)
    actual_k = _actual_seq_to_numpy(actual_seq_klen, S2, B)

    max_index = np.full((B, 1, S1), -np.inf, dtype=np.float32)
    sum_index = np.zeros((B, 1, S1), dtype=np.float32)

    for b in range(B):
        act_q = int(actual_q[b])
        act_k = int(actual_k[b])
        for s1 in range(S1):
            if s1 >= act_q:
                continue
            visible_k_num = min(max(act_k - act_q + s1 + 1, 0), act_k, S2)
            if visible_k_num <= 0:
                continue

            dots = np.matmul(qi_np[b, s1], ki_np[b, :visible_k_num, 0].T)
            scores = np.sum(w_np[b, s1, :, None] * np.maximum(dots, 0.0), axis=0)
            score_max = np.float32(np.max(scores))
            max_index[b, 0, s1] = score_max
            sum_index[b, 0, s1] = np.float32(np.sum(np.exp(scores - score_max)))

    return ms.Tensor(max_index, dtype=ms.float32), ms.Tensor(sum_index, dtype=ms.float32)


def _assert_tensor_metadata(name, value, shape, dtype):
    if value.shape != shape:
        raise AssertionError(f"{name} shape must be {shape}, got {value.shape}")
    if value.dtype != dtype:
        raise AssertionError(f"{name} dtype must be {dtype}, got {value.dtype}")


def _make_inputs(B, S1, S2, N1, N2, D, Nidx1, D_idx, dtype=ms.float16):
    rng = np.random.RandomState(42)
    q = ms.Tensor(rng.randn(B, S1, N1, D).astype(np.float16), dtype=dtype)
    k = ms.Tensor(rng.randn(B, S2, N2, D).astype(np.float16), dtype=dtype)
    qr = ms.Tensor(rng.randn(B, S1, N1, DROPE).astype(np.float16), dtype=dtype)
    kr = ms.Tensor(rng.randn(B, S2, N2, DROPE).astype(np.float16), dtype=dtype)
    qi = ms.Tensor(rng.randn(B, S1, Nidx1, D_idx).astype(np.float16), dtype=dtype)
    ki = ms.Tensor(rng.randn(B, S2, 1, D_idx).astype(np.float16), dtype=dtype)
    w = ms.Tensor(np.abs(rng.randn(B, S1, Nidx1)).astype(np.float16), dtype=dtype)
    return q, k, qr, kr, qi, ki, w


def _compute_softmax_stats(q, k, qr, kr, scale_value):
    """softmaxMax/Sum from FULL forward attention: (B, N2, S1, G)."""
    B, S1, N1, _ = q.shape
    S2, N2 = k.shape[1], k.shape[2]
    G = N1 // N2
    q_np = q.asnumpy().astype(np.float32)
    k_np = k.asnumpy().astype(np.float32)
    qr_np = qr.asnumpy().astype(np.float32)
    kr_np = kr.asnumpy().astype(np.float32)

    sm_max = np.full((B, N2, S1, G), -np.inf, dtype=np.float32)
    sm_sum = np.zeros((B, N2, S1, G), dtype=np.float32)

    for b in range(B):
        for s1 in range(S1):
            causal_limit = min(max(S2 - S1 + s1 + 1, 0), S2)
            if causal_limit == 0:
                continue
            for h in range(N1):
                n2 = h // G
                g = h - n2 * G
                scores = (
                    np.dot(q_np[b, s1, h], k_np[b, :causal_limit, n2].T)
                    + np.dot(qr_np[b, s1, h], kr_np[b, :causal_limit, n2].T)
                ) * scale_value
                s_max = np.max(scores)
                sm_max[b, n2, s1, g] = s_max
                sm_sum[b, n2, s1, g] = np.sum(np.exp(scores - s_max))

    return (
        ms.Tensor(sm_max, dtype=ms.float32),
        ms.Tensor(sm_sum, dtype=ms.float32),
    )


def _dense_cann_supported(D_idx, Nidx1):
    """Whether a config falls in the CANN operator's supported range."""
    return D_idx == 128 and Nidx1 in (32, 64)


def _dense_grad_cann_supported(D, D_idx, Nidx1):
    """Whether a grad config falls in the CANN grad operator's supported range.

    aclnnDenseLightningIndexerGradKLLoss additionally requires D == 128
    (the LSE op does not take query/key, so D is irrelevant there).
    """
    return D == 128 and D_idx == 128 and Nidx1 in (32, 64)


def _dense_loss_backward_numpy(q, k, qr, kr, qi, ki, w,
                               softmax_max, softmax_sum,
                               max_index, sum_index,
                               scale_value,
                               actual_seq_qlen=None, actual_seq_klen=None):
    """Pure NumPy golden reference for dense LightningIndexer KL loss backward.

    Mirrors the CANN/Triton operator (rightDownCausal, sparse_mode=3): the
    indexer branch reuses the externally-provided max_index/sum_index to
    normalize softmax(I), and the target p is reconstructed from the main
    attention softmax_max/sum over the same causal visible-k window.
    """
    EPS = 1e-8
    q_np = q.asnumpy().astype(np.float32)
    k_np = k.asnumpy().astype(np.float32)
    qr_np = qr.asnumpy().astype(np.float32)
    kr_np = kr.asnumpy().astype(np.float32)
    qi_np = qi.asnumpy().astype(np.float32)
    ki_np = ki.asnumpy().astype(np.float32)
    w_np = w.asnumpy().astype(np.float32)
    sm_max = softmax_max.asnumpy().astype(np.float32)
    sm_sum = softmax_sum.asnumpy().astype(np.float32)
    max_idx = max_index.asnumpy().astype(np.float32).reshape(-1)
    sum_idx = sum_index.asnumpy().astype(np.float32).reshape(-1)

    B, S1, N1, D = q_np.shape
    S2 = k_np.shape[1]
    N2 = k_np.shape[2]
    G = N1 // N2
    Nidx1, D_idx = qi_np.shape[2], qi_np.shape[3]

    actual_q = _actual_seq_to_numpy(actual_seq_qlen, S1, B)
    actual_k = _actual_seq_to_numpy(actual_seq_klen, S2, B)

    d_qi = np.zeros((B, S1, Nidx1, D_idx), dtype=np.float32)
    d_ki = np.zeros((B, S2, 1, D_idx), dtype=np.float32)
    d_w = np.zeros((B, S1, Nidx1), dtype=np.float32)
    loss_val = 0.0

    for b in range(B):
        act_q = int(actual_q[b])
        act_k = int(actual_k[b])
        for s1 in range(S1):
            if s1 >= act_q:
                continue
            n = min(max(act_k - act_q + s1 + 1, 0), S2)
            if n <= 0:
                continue
            stat_pos = b * S1 + s1
            i_max = float(max_idx[stat_pos])
            i_sum = float(sum_idx[stat_pos])
            if i_sum <= 0.0:
                continue
            ki_vis = ki_np[b, :n, 0, :]
            dots = np.matmul(qi_np[b, s1], ki_vis.T)
            relu = np.maximum(dots, 0.0)
            i_scores = np.sum(w_np[b, s1, :, None] * relu, axis=0)
            student = np.exp(i_scores - i_max) / max(i_sum, EPS)
            p = np.zeros((n,), dtype=np.float32)
            for h in range(N1):
                n2 = h // G
                g = h - n2 * G
                scores = (
                    np.dot(q_np[b, s1, h], k_np[b, :n, n2].T)
                    + np.dot(qr_np[b, s1, h], kr_np[b, :n, n2].T)
                ) * scale_value
                probs = np.exp(scores - sm_max[b, n2, s1, g]) / (sm_sum[b, n2, s1, g] + EPS)
                p += probs
            p *= (1.0 / N1)
            d_i = student - p
            p_safe = np.maximum(p, EPS)
            q_safe = np.maximum(student, EPS)
            loss_val += float(np.sum(p_safe * (np.log(p_safe) - np.log(q_safe))))
            relu_mask = (relu > 0.0).astype(np.float32)
            for g in range(Nidx1):
                d_w[b, s1, g] = np.sum(d_i * relu[g])
                ds_g = d_i * w_np[b, s1, g] * relu_mask[g]
                d_qi[b, s1, g] = np.matmul(ds_g, ki_vis)
                d_ki[b, :n, 0] += (ds_g.reshape(n, 1)) * qi_np[b, s1, g]

    return (
        ms.Tensor(d_qi, dtype=ms.float32),
        ms.Tensor(d_ki, dtype=ms.float32),
        ms.Tensor(d_w, dtype=ms.float32),
        ms.Tensor(np.array([loss_val], dtype=np.float32), dtype=ms.float32),
    )


@pytest.mark.parametrize("B,S1,S2,N1,N2,D,Nidx1,D_idx,dtype", DENSE_DTYPE_TEST_CONFIGS)
def test_dense_softmax_lse_precision(B, S1, S2, N1, N2, D, Nidx1, D_idx, dtype):
    from dense_loss_backward_triton import DenseLightningIndexerSoftmaxLseTriton

    _log(
        f"building LSE inputs: B={B}, S1={S1}, S2={S2}, Nidx1={Nidx1}, D_idx={D_idx}, dtype={dtype}"
    )
    _, _, _, _, qi, ki, w = _make_inputs(B, S1, S2, N1, N2, D, Nidx1, D_idx, dtype=dtype)

    _log("running Triton dense softmax_lse")
    stats_cell = DenseLightningIndexerSoftmaxLseTriton()
    max_index_tri, sum_index_tri = stats_cell(qi, ki, w, layout="BSND", sparse_mode=3)
    _assert_tensor_metadata("max_index_tri", max_index_tri, (B, 1, S1), ms.float32)
    _assert_tensor_metadata("sum_index_tri", sum_index_tri, (B, 1, S1), ms.float32)

    if _dense_cann_supported(D_idx, Nidx1):
        from dense_loss_backward_cann import DenseLightningIndexerSoftmaxLse
        _log("running CANN dense softmax_lse baseline")
        lse_op = DenseLightningIndexerSoftmaxLse()
        max_index_base, sum_index_base = lse_op(
            qi, ki, w, layout="BSND", sparse_mode=3)
        base_label = "cann"
    else:
        _log("running NumPy dense softmax_lse baseline")
        max_index_base, sum_index_base = _dense_indexer_lse_numpy(qi, ki, w)
        base_label = "numpy"

    _log("checking dense softmax_lse outputs")
    _assert_allclose_with_values(
        "max_index", max_index_base, max_index_tri,
        expected_label=base_label, actual_label="triton")
    _assert_allclose_with_values(
        "sum_index", sum_index_base, sum_index_tri,
        expected_label=base_label, actual_label="triton")
    _log("passed")


def test_dense_softmax_lse_guards():
    """Interface guards: unsupported params must raise ValueError at the host gate."""
    from dense_loss_backward_triton import (
        npu_dense_lightning_indexer_softmax_lse_triton as lse_fn, INT64_MAX)

    B, S1, S2, N1, N2, D, Nidx1, D_idx = 1, 2, 64, 32, 32, 128, 32, 128
    _, _, _, _, qi, ki, w = _make_inputs(B, S1, S2, N1, N2, D, Nidx1, D_idx)

    def _call(**kw):
        base = dict(layout="BSND", sparse_mode=3,
                    pre_tokens=INT64_MAX, next_tokens=INT64_MAX)
        base.update(kw)
        return lse_fn(qi, ki, w, **base)

    with pytest.raises(ValueError):
        _call(layout="TND")
    with pytest.raises(ValueError):
        _call(sparse_mode=0)
    with pytest.raises(ValueError):
        _call(pre_tokens=128)
    with pytest.raises(ValueError):
        _call(next_tokens=128)

    rng = np.random.RandomState(0)
    qi_bad = ms.Tensor(rng.randn(B, S1, 8, D_idx).astype(np.float16), dtype=ms.float16)
    w_bad = ms.Tensor(np.abs(rng.randn(B, S1, 8)).astype(np.float16), dtype=ms.float16)
    with pytest.raises(ValueError):
        lse_fn(qi_bad, ki, w_bad, layout="BSND", sparse_mode=3)
    qi_bad2 = ms.Tensor(rng.randn(B, S1, Nidx1, 64).astype(np.float16), dtype=ms.float16)
    ki_bad2 = ms.Tensor(rng.randn(B, S2, 1, 64).astype(np.float16), dtype=ms.float16)
    with pytest.raises(ValueError):
        lse_fn(qi_bad2, ki_bad2, w, layout="BSND", sparse_mode=3)
    qi_3d = ms.Tensor(rng.randn(S1, Nidx1, D_idx).astype(np.float16), dtype=ms.float16)
    with pytest.raises(ValueError):
        lse_fn(qi_3d, ki, w, layout="BSND", sparse_mode=3)
    w_2d = ms.Tensor(np.abs(rng.randn(B, S1)).astype(np.float16), dtype=ms.float16)
    with pytest.raises(ValueError):
        lse_fn(qi, ki, w_2d, layout="BSND", sparse_mode=3)


@pytest.mark.parametrize("B,S1,S2,N1,N2,D,Nidx1,D_idx,dtype", DENSE_DTYPE_TEST_CONFIGS)
def test_dense_grad_kl_loss_triton_supported_shapes(B, S1, S2, N1, N2, D, Nidx1, D_idx, dtype):
    from dense_loss_backward_triton import (
        DenseLightningIndexerSoftmaxLseTriton,
        DenseLightningIndexerGradKLLossTriton,
        dense_loss_backward_triton,
    )

    scale_value = 1.0 / np.sqrt(D + DROPE)

    _log(
        f"building Triton grad inputs: B={B}, S1={S1}, S2={S2}, N1={N1}, N2={N2}, "
        f"D={D}, Nidx1={Nidx1}, D_idx={D_idx}, dtype={dtype}"
    )
    q, k, qr, kr, qi, ki, w = _make_inputs(B, S1, S2, N1, N2, D, Nidx1, D_idx, dtype=dtype)
    _log("computing attention softmax stats")
    softmax_max, softmax_sum = _compute_softmax_stats(q, k, qr, kr, scale_value)

    _log("running Triton dense softmax_lse")
    stats_cell = DenseLightningIndexerSoftmaxLseTriton()
    max_index_tri, sum_index_tri = stats_cell(qi, ki, w, layout="BSND", sparse_mode=3)

    _log("running Triton dense grad_kl_loss")
    grad_cell = DenseLightningIndexerGradKLLossTriton()
    d_qi_tri, d_ki_tri, d_w_tri, loss_tri = grad_cell(
        q, k, qi, ki, w,
        softmax_max, softmax_sum,
        max_index_tri, sum_index_tri,
        scale_value=scale_value, layout="BSND", sparse_mode=3,
        query_rope=qr, key_rope=kr,
    )

    if _dense_grad_cann_supported(D, D_idx, Nidx1):
        from dense_loss_backward_cann import DenseLightningIndexerGradKLLoss
        _log("running CANN dense grad_kl_loss baseline")
        grad_op = DenseLightningIndexerGradKLLoss()
        d_qi_base, d_ki_base, d_w_base, loss_base = grad_op(
            q, k, qi, ki, w,
            softmax_max, softmax_sum,
            max_index_tri, sum_index_tri,
            scale_value=scale_value,
            query_rope=qr, key_rope=kr,
            layout="BSND", sparse_mode=3,
        )
        base_label = "cann"
    else:
        _log("running NumPy dense grad_kl_loss baseline")
        d_qi_base, d_ki_base, d_w_base, loss_base = _dense_loss_backward_numpy(
            q, k, qr, kr, qi, ki, w,
            softmax_max, softmax_sum,
            max_index_tri, sum_index_tri,
            scale_value,
        )
        base_label = "numpy"

    _log("checking Triton supported-shape outputs")
    _assert_tensor_metadata("d_qi_tri", d_qi_tri, qi.shape, qi.dtype)
    _assert_tensor_metadata("d_ki_tri", d_ki_tri, ki.shape, ki.dtype)
    _assert_tensor_metadata("d_w_tri", d_w_tri, w.shape, w.dtype)
    _assert_tensor_metadata("loss_tri", loss_tri, (1,), ms.float32)
    _assert_allclose_with_values("d_qi", d_qi_base, d_qi_tri,
                                 expected_label=base_label, actual_label="triton")
    _assert_allclose_with_values("d_ki", d_ki_base, d_ki_tri,
                                 expected_label=base_label, actual_label="triton")
    _assert_allclose_with_values("d_w", d_w_base, d_w_tri,
                                 expected_label=base_label, actual_label="triton")
    _assert_allclose_with_values("loss", loss_base, loss_tri,
                                 expected_label=base_label, actual_label="triton")

    _log("running Triton compatibility path")
    d_qi_auto, d_ki_auto, d_w_auto, loss_auto = dense_loss_backward_triton(
        q, k, qi, ki, w,
        softmax_max, softmax_sum,
        scale_value=scale_value,
        query_rope=qr, key_rope=kr,
        layout="BSND", sparse_mode=3,
    )
    _assert_allclose_with_values("d_qi_auto", d_qi_tri, d_qi_auto)
    _assert_allclose_with_values("d_ki_auto", d_ki_tri, d_ki_auto)
    _assert_allclose_with_values("d_w_auto", d_w_tri, d_w_auto)
    _assert_allclose_with_values("loss_auto", loss_tri, loss_auto)
    _log("passed")


@pytest.mark.parametrize("B,S1,S2,N1,N2,D,Nidx1,D_idx", DENSE_GRAD_CANN_TEST_CONFIGS)
def test_dense_grad_kl_loss_precision(B, S1, S2, N1, N2, D, Nidx1, D_idx):
    from dense_loss_backward_triton import (
        DenseLightningIndexerSoftmaxLseTriton,
        DenseLightningIndexerGradKLLossTriton,
        dense_loss_backward_triton,
    )
    from dense_loss_backward_cann import DenseLightningIndexerGradKLLoss

    scale_value = 1.0 / np.sqrt(D + DROPE)

    _log(
        f"building inputs: B={B}, S1={S1}, S2={S2}, N1={N1}, N2={N2}, "
        f"D={D}, Nidx1={Nidx1}, D_idx={D_idx}"
    )
    q, k, qr, kr, qi, ki, w = _make_inputs(B, S1, S2, N1, N2, D, Nidx1, D_idx)
    _log("computing attention softmax stats")
    softmax_max, softmax_sum = _compute_softmax_stats(q, k, qr, kr, scale_value)

    _log("running NumPy dense softmax_lse baseline")
    max_index_base, sum_index_base = _dense_indexer_lse_numpy(qi, ki, w)

    _log("running CANN dense grad_kl_loss")
    op = DenseLightningIndexerGradKLLoss()
    d_qi_base, d_ki_base, d_w_base, loss_base = op(
        q, k, qi, ki, w,
        softmax_max, softmax_sum, max_index_base, sum_index_base,
        scale_value=scale_value,
        query_rope=qr, key_rope=kr,
        layout="BSND", sparse_mode=3,
    )

    # Triton two-stage path.
    _log("running Triton dense softmax_lse")
    stats_cell = DenseLightningIndexerSoftmaxLseTriton()
    max_index_tri, sum_index_tri = stats_cell(qi, ki, w, layout="BSND", sparse_mode=3)

    _log("running Triton dense grad_kl_loss")
    grad_cell = DenseLightningIndexerGradKLLossTriton()
    d_qi_tri, d_ki_tri, d_w_tri, loss_tri = grad_cell(
        q, k, qi, ki, w,
        softmax_max, softmax_sum,
        max_index_tri, sum_index_tri,
        scale_value=scale_value, layout="BSND", sparse_mode=3,
        query_rope=qr, key_rope=kr,
    )

    _log("checking two-stage outputs")
    _assert_allclose_with_values("max_index", max_index_base, max_index_tri)
    _assert_allclose_with_values("sum_index", sum_index_base, sum_index_tri)
    _assert_allclose_with_values("d_qi", d_qi_base, d_qi_tri)
    _assert_allclose_with_values("d_ki", d_ki_base, d_ki_tri)
    _assert_allclose_with_values("d_w", d_w_base, d_w_tri)
    _assert_allclose_with_values("loss", loss_base, loss_tri)

    # Compatibility path: dense_loss_backward_triton computes index stats internally.
    _log("running Triton compatibility path")
    d_qi_auto, d_ki_auto, d_w_auto, loss_auto = dense_loss_backward_triton(
        q, k, qi, ki, w,
        softmax_max, softmax_sum,
        scale_value=scale_value,
        query_rope=qr, key_rope=kr,
        layout="BSND", sparse_mode=3,
    )
    _assert_allclose_with_values("d_qi_auto", d_qi_base, d_qi_auto)
    _assert_allclose_with_values("d_ki_auto", d_ki_base, d_ki_auto)
    _assert_allclose_with_values("d_w_auto", d_w_base, d_w_auto)
    _assert_allclose_with_values("loss_auto", loss_base, loss_auto)
    _log("passed")

@pytest.mark.parametrize("B,S1,S2,N1,N2,D,Nidx1,D_idx", DENSE_CANN_LSE_CHECK_CONFIGS)
def test_dense_lse_numpy_matches_cann(B, S1, S2, N1, N2, D, Nidx1, D_idx):
    from dense_loss_backward_cann import DenseLightningIndexerSoftmaxLse

    _log(
        f"building CANN LSE calibration inputs: B={B}, S1={S1}, S2={S2}, "
        f"Nidx1={Nidx1}, D_idx={D_idx}"
    )
    _, _, _, _, qi, ki, w = _make_inputs(B, S1, S2, N1, N2, D, Nidx1, D_idx)

    _log("running NumPy dense softmax_lse baseline")
    max_index_base, sum_index_base = _dense_indexer_lse_numpy(qi, ki, w)

    _log("running optional CANN dense softmax_lse calibration")
    lse_op = DenseLightningIndexerSoftmaxLse()
    _log("guod.....000111")
    max_index_cann, sum_index_cann = lse_op(
        qi, ki, w,
        layout="BSND", sparse_mode=3,
    )
    _log("guod.....000222")
    _assert_allclose_with_values(
        "max_index_cann_lse", max_index_base, max_index_cann,
        expected_label="numpy", actual_label="cann",
    )
    _assert_allclose_with_values(
        "sum_index_cann_lse", sum_index_base, sum_index_cann,
        expected_label="numpy", actual_label="cann",
    )


if __name__ == "__main__":
    # for config in DENSE_TEST_CONFIGS:
    #     test_dense_grad_kl_loss_triton_supported_shapes(*config, ms.bfloat16)
    # print("dense grad kl loss triton test passed!")

    for config in DENSE_CANN_LSE_CHECK_CONFIGS:
        test_dense_lse_numpy_matches_cann(*config)
    print("dense lse numpy matches cann test passed!")

