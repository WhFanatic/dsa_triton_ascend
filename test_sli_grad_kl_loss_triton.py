"""Test sparse_lightning_indexer_grad_kl_loss_triton."""
import pytest
import numpy as np
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})

DROPE = 64
_TOLS = {ms.float16: (2e-3, 2e-3), ms.bfloat16: (1e-2, 1e-2)}
NEAR_ZERO = 1e-2

# 每个参数取值至少覆盖一次：
#   N1∈{32,64,128}, Nidx1∈{8,16,32,64}, S1∈{1,4}, S2∈{2048,4096}, topK∈{1024,2048}
SPARSE_GRAD_CANN_TRITON_STRICT_CHECK_TEST_CONFIGS = [
    (1, 1, 2048, 32, 512, 8, 128, 1024),
    (1, 4, 2048, 64, 512, 16, 128, 1024),
    (1, 4, 2048, 128, 512, 32, 128, 1024),
    (1, 4, 4096, 32, 512, 64, 128, 2048),
]

# 大 shape：跳过 d_ki 元素级比较（累加顺序 + fp16 舍入使得 1-ULP 差异不可避免）。
# 覆盖 S1∈{512,1024,4096}, S2∈{1024,4096}, N1∈{32,64}, Nidx1∈{8,64}, topK∈{1024,2048}
SPARSE_GRAD_CANN_TRITON_LOOSE_CHECK_TEST_CONFIGS = [
    (1, 512, 4096, 64, 512, 64, 128, 2048), # 目标shape，后续性能基于此shape
    (1, 1024, 1024, 32, 512, 8, 128, 1024),
    (1, 4096, 4096, 64, 512, 64, 128, 2048),
]

SPARSE_GRAD_CANN_TRITON_TEST_CONFIGS = (
    [(*c, "causal_random", False) for c in SPARSE_GRAD_CANN_TRITON_STRICT_CHECK_TEST_CONFIGS] +
    [(*c, "causal_continuous", True) for c in SPARSE_GRAD_CANN_TRITON_LOOSE_CHECK_TEST_CONFIGS]
)

# CANN A3 不支持但需求要 Triton 支持的 shape：
#   N1∈{32,64,128}, Nidx1∈{32,64,128}, D∈{128,256,512}, D_idx∈{128,256,512}, topK∈{1024,2048}
# 每个二元组合(N1×D, N1×Nidx1, D×D_idx, Nidx1×D_idx)至少覆盖一次
SPARSE_GRAD_TRITON_NUMPY_TEST_CONFIGS = [
    # 同值对角线：确认每维每值可独立工作
    (1, 4, 2048, 64, 512, 64, 128, 2048),
    (1, 4, 2048, 32, 128, 32, 128, 1024),
    (1, 4, 2048, 64, 256, 64, 256, 1024),
    (1, 4, 2048, 128, 512, 128, 512, 2048),
    # 交叉组合：填补对角线之外的二元组合空白
    (1, 4, 2048, 32, 256, 128, 512, 1024),   # N1=32×D=256, N1=32×Nidx1=128, D=256×D_idx=512, Nidx1=128×D_idx=512
    (1, 4, 2048, 32, 512, 64, 256, 2048),    # N1=32×D=512, N1=32×Nidx1=64, D=512×D_idx=256, Nidx1=64×D_idx=256
    (1, 4, 2048, 64, 128, 32, 256, 1024),    # N1=64×D=128, N1=64×Nidx1=32, D=128×D_idx=256, Nidx1=32×D_idx=256
    (1, 4, 2048, 128, 128, 32, 512, 2048),   # N1=128×D=128, N1=128×Nidx1=32, D=128×D_idx=512, Nidx1=32×D_idx=512
    (1, 4, 2048, 128, 256, 64, 128, 1024),   # N1=128×D=256, N1=128×Nidx1=64, D=256×D_idx=128, Nidx1=64×D_idx=128
    (1, 4, 2048, 128, 512, 128, 128, 2048),  # Nidx1=128×D_idx=128
    (1, 4, 2048, 64, 512, 128, 512, 1024),   # N1=64×Nidx1=128, Nidx1=128×D_idx=512, D=512×topK=1024
    (1, 4, 2048, 32, 256, 64, 512, 2048),    # Nidx1=64×D_idx=512, D=256×topK=2048
    (1, 4, 2048, 64, 512, 128, 256, 1024),   # Nidx1=128×D_idx=256
]

# 覆盖 N1∈{32,64,128}, Nidx1∈{8,16,32}, S1∈{1,4}, S2=2048, topK=1024
SPARSE_GRAD_CANN_NUMPY_STRICT_CHECK_TEST_CONFIGS = [
    (1, 1, 2048, 32, 512, 8, 128, 1024),
    (1, 4, 2048, 64, 512, 16, 128, 1024),
    (1, 4, 2048, 128, 512, 32, 128, 1024),
]

# 大 shape / 大 topK：跳过 d_ki 元素级比较（atomic_add 累加顺序在 CANN 与 numpy 之间不一致）。
# 覆盖 Nidx1=64, S2=4096, topK=2048
SPARSE_GRAD_CANN_NUMPY_LOOSE_CHECK_TEST_CONFIGS = [
    (1, 4, 2048, 64, 512, 64, 128, 2048),
    (1, 4, 4096, 32, 512, 64, 128, 2048),
]

SPARSE_GRAD_CANN_NUMPY_TEST_CONFIGS = (
    [(*c, "causal_random", False) for c in SPARSE_GRAD_CANN_NUMPY_STRICT_CHECK_TEST_CONFIGS] +
    [(*c, "causal_random", True) for c in SPARSE_GRAD_CANN_NUMPY_LOOSE_CHECK_TEST_CONFIGS]
)


def _make_inputs(B, S1, S2, N1, D, Nidx1, D_idx, topK, dtype=ms.float16):
    rng = np.random.RandomState(42)
    N2, Nidx2 = 1, 1
    q = ms.Tensor(rng.randn(B, S1, N1, D).astype(np.float16), dtype=ms.float16)
    k = ms.Tensor(rng.randn(B, S2, N2, D).astype(np.float16), dtype=ms.float16)
    qr = ms.Tensor(rng.randn(B, S1, N1, DROPE).astype(np.float16), dtype=ms.float16)
    kr = ms.Tensor(rng.randn(B, S2, N2, DROPE).astype(np.float16), dtype=ms.float16)
    qi = ms.Tensor(rng.randn(B, S1, Nidx1, D_idx).astype(np.float16), dtype=ms.float16)
    ki = ms.Tensor(rng.randn(B, S2, Nidx2, D_idx).astype(np.float16), dtype=ms.float16)
    w = ms.Tensor(np.abs(rng.randn(B, S1, Nidx1)).astype(np.float16), dtype=ms.float16)
    si = ms.Tensor(rng.randint(0, S2, (B, S1, Nidx2, topK)).astype(np.int32), dtype=ms.int32)
    if dtype != ms.float16:
        q, k, qr, kr, qi, ki, w = (
            q.astype(dtype), k.astype(dtype), qr.astype(dtype),
            kr.astype(dtype), qi.astype(dtype), ki.astype(dtype),
            w.astype(dtype),
        )
    return q, k, qr, kr, qi, ki, w, si


def _make_sparse_indices(B, S1, S2, topK, pattern):
    if pattern == "random":
        rng = np.random.RandomState(42)
        si = rng.randint(0, S2, (B, S1, 1, topK)).astype(np.int32)
    elif pattern == "causal_random":
        rng = np.random.RandomState(42)
        si = np.zeros((B, S1, 1, topK), dtype=np.int32)
        for b in range(B):
            for s1 in range(S1):
                visible = min(max(S2 - S1 + s1 + 1, 1), S2)
                valid_k = min(topK, visible)
                si[b, s1, 0, :valid_k] = rng.choice(
                    visible, size=valid_k, replace=False).astype(np.int32)
    elif pattern == "continuous":
        row = np.arange(topK, dtype=np.int32) % S2
        si = np.broadcast_to(row.reshape(1, 1, 1, topK),
                             (B, S1, 1, topK)).copy()
    elif pattern == "repeated":
        row = np.zeros((topK,), dtype=np.int32)
        row[1::2] = min(S2 - 1, 7)
        si = np.broadcast_to(row.reshape(1, 1, 1, topK),
                             (B, S1, 1, topK)).copy()
    elif pattern == "causal_continuous":
        si = np.zeros((B, S1, 1, topK), dtype=np.int32)
        for s1 in range(S1):
            visible = min(max(S2 - S1 + s1 + 1, 1), S2)
            valid_k = min(topK, visible)
            si[:, s1, 0, :valid_k] = np.arange(valid_k, dtype=np.int32)
    else:
        raise ValueError(f"Unknown sparse index pattern: {pattern}")
    return ms.Tensor(si, dtype=ms.int32)


def _actual_seq_to_numpy(actual_seq, seq_len, batch_size):
    if actual_seq is None:
        return np.full((batch_size,), seq_len, dtype=np.int32)
    return np.asarray(actual_seq, dtype=np.int32)


def _compute_softmax_stats(q, k, qr, kr, scale_value,
                           actual_seq_qlen=None, actual_seq_klen=None):
    """softmaxMax/Sum from FULL forward FlashAttention: (B, 1, S1, N1).

    Must compute over ALL S2 keys, not just topK gathered subset.
    """
    B, S1, N1, D = q.shape
    S2 = k.shape[1]
    actual_q = _actual_seq_to_numpy(actual_seq_qlen, S1, B)
    actual_k = _actual_seq_to_numpy(actual_seq_klen, S2, B)
    q_np = q.asnumpy().astype(np.float32)
    k_np = k.asnumpy().astype(np.float32)
    qr_np = qr.asnumpy().astype(np.float32)
    kr_np = kr.asnumpy().astype(np.float32)

    sm_max = np.full((B, 1, S1, N1), -np.inf, dtype=np.float32)
    sm_sum = np.zeros((B, 1, S1, N1), dtype=np.float32)

    for b in range(B):
        all_k = k_np[b, :, 0, :]      # (S2, D)
        all_kr = kr_np[b, :, 0, :]    # (S2, DRope)
        for s1 in range(S1):
            if s1 >= actual_q[b]:
                continue
            causal_limit = min(max(actual_k[b] - actual_q[b] + s1 + 1, 0), S2)
            if causal_limit <= 0:
                continue
            for h in range(N1):
                scores = (np.dot(q_np[b, s1, h], all_k.T)
                          + np.dot(qr_np[b, s1, h], all_kr.T)) * scale_value
                scores[causal_limit:] = float('-inf')
                s_max = np.max(scores)
                sm_max[b, 0, s1, h] = s_max
                sm_sum[b, 0, s1, h] = np.sum(np.exp(scores - s_max))

    return (ms.Tensor(sm_max, dtype=ms.float32),
            ms.Tensor(sm_sum, dtype=ms.float32))


def _to_np(t):
    return t.asnumpy().astype(np.float32)


def _assert_close_skip_nearzero(a, b, atol, rtol):
    a_np, b_np = np.asarray(a, np.float32), np.asarray(b, np.float32)
    near_zero = (np.abs(a_np) < NEAR_ZERO) & (np.abs(b_np) < NEAR_ZERO)
    a_f = np.where(near_zero, 0.0, a_np)
    b_f = np.where(near_zero, 0.0, b_np)
    np.testing.assert_allclose(a_f, b_f, atol=atol, rtol=rtol)


def _assert_outputs_close_strict(base_outputs, tri_outputs, dtype):
    d_qi_base, d_ki_base, d_w_base, loss_base = base_outputs
    d_qi_tri, d_ki_tri, d_w_tri, loss_tri = tri_outputs
    atol, rtol = _TOLS[dtype]
    _assert_close_skip_nearzero(_to_np(d_qi_base), _to_np(d_qi_tri), atol, rtol)
    _assert_close_skip_nearzero(_to_np(d_ki_base), _to_np(d_ki_tri), atol, rtol)
    _assert_close_skip_nearzero(_to_np(d_w_base), _to_np(d_w_tri), atol, rtol)
    _assert_close_skip_nearzero(_to_np(loss_base), _to_np(loss_tri), atol, rtol)


def _assert_outputs_close_loose(base_outputs, tri_outputs, dtype):
    d_qi_base, d_ki_base, d_w_base, loss_base = base_outputs
    d_qi_tri, d_ki_tri, d_w_tri, loss_tri = tri_outputs
    atol, rtol = _TOLS[dtype]

    _assert_close_skip_nearzero(_to_np(d_qi_base), _to_np(d_qi_tri), atol, rtol)
    _assert_close_skip_nearzero(_to_np(d_w_base), _to_np(d_w_tri), atol, rtol)
    _assert_close_skip_nearzero(_to_np(loss_base), _to_np(loss_tri), atol, rtol)

    assert d_ki_tri.shape == d_ki_base.shape, (
        f"d_ki shape mismatch: {d_ki_tri.shape} vs {d_ki_base.shape}"
    )
    assert d_ki_tri.dtype == d_ki_base.dtype, (
        f"d_ki dtype mismatch: {d_ki_tri.dtype} vs {d_ki_base.dtype}"
    )
    assert np.isfinite(d_ki_tri.asnumpy().astype(np.float32)).all()


def _run_cann_triton_precision(B, S1, S2, N1, D, Nidx1, D_idx, topK,
                               actual_seq_qlen=None, actual_seq_klen=None,
                               sparse_pattern="causal_random", large_check=False,
                               dtype=ms.float16):
    from sparse_lightning_indexer_grad_kl_loss_triton import (
        SparseLightningIndexerGradKLLossTriton,
    )
    from sli_grad_kl_loss_cann import (
        SparseLightningIndexerGradKLLoss,
    )
    scale_value = 1.0 / np.sqrt(D)

    q, k, qr, kr, qi, ki, w, si = _make_inputs(B, S1, S2, N1, D, Nidx1, D_idx, topK, dtype=dtype)
    if sparse_pattern != "random":
        si = _make_sparse_indices(B, S1, S2, topK, sparse_pattern)
    softmax_max, softmax_sum = _compute_softmax_stats(
        q, k, qr, kr, scale_value,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_klen=actual_seq_klen,
    )

    # CANN baseline
    op = SparseLightningIndexerGradKLLoss()
    base_outputs = op(
        q, k, qi, ki, w, si, softmax_max, softmax_sum,
        query_rope=qr, key_rope=kr,
        scale_value=scale_value, layout="BSND", sparse_mode=3,
        actual_seq_qlen=actual_seq_qlen, actual_seq_klen=actual_seq_klen,
    )

    # Triton
    cell = SparseLightningIndexerGradKLLossTriton(
        scale_value=scale_value, layout="BSND", sparse_mode=3,
    )
    tri_outputs = cell(
        q, k, qi, ki, w, si, softmax_max, softmax_sum,
        query_rope=qr, key_rope=kr,
        actual_seq_qlen=actual_seq_qlen, actual_seq_klen=actual_seq_klen,
    )

    if large_check:
        _assert_outputs_close_loose(base_outputs, tri_outputs, dtype)
    else:
        _assert_outputs_close_strict(base_outputs, tri_outputs, dtype)


def _run_triton_numpy_precision(B, S1, S2, N1, D, Nidx1, D_idx, topK,
                                sparse_pattern="causal_random"):
    """Compare Triton output against pure numpy reference.

    Used for shapes CANN A3 does not support (Nidx1=128 或 D∈{128,256}).
    numpy_reference 内部按 fp16 语义 round，因此这里固定 dtype=fp16。
    """
    from sparse_lightning_indexer_grad_kl_loss_triton import (
        SparseLightningIndexerGradKLLossTriton,
    )
    from sli_grad_kl_loss_numpy import numpy_reference

    dtype = ms.float16
    scale_value = 1.0 / np.sqrt(D)

    q, k, qr, kr, qi, ki, w, si = _make_inputs(
        B, S1, S2, N1, D, Nidx1, D_idx, topK, dtype=dtype)
    if sparse_pattern != "random":
        si = _make_sparse_indices(B, S1, S2, topK, sparse_pattern)
    softmax_max, softmax_sum = _compute_softmax_stats(
        q, k, qr, kr, scale_value)

    ref = numpy_reference(
        q.asnumpy(), k.asnumpy(), qr.asnumpy(), kr.asnumpy(),
        qi.asnumpy(), ki.asnumpy(), w.asnumpy(), si.asnumpy(),
        softmax_max.asnumpy(), softmax_sum.asnumpy(), scale_value,
    )

    cell = SparseLightningIndexerGradKLLossTriton(
        scale_value=scale_value, layout="BSND", sparse_mode=3,
    )
    d_qi_tri, d_ki_tri, d_w_tri, loss_tri = cell(
        q, k, qi, ki, w, si, softmax_max, softmax_sum,
        query_rope=qr, key_rope=kr,
    )

    atol, rtol = _TOLS[dtype]
    _assert_close_skip_nearzero(ref['dQueryIndex'], _to_np(d_qi_tri), atol, rtol)
    _assert_close_skip_nearzero(ref['dKeyIndex'], _to_np(d_ki_tri), atol, rtol)
    _assert_close_skip_nearzero(ref['dW'], _to_np(d_w_tri), atol, rtol)
    _assert_close_skip_nearzero(ref['loss'], _to_np(loss_tri), atol, rtol)


def _run_cann_numpy_precision(B, S1, S2, N1, D, Nidx1, D_idx, topK,
                              sparse_pattern="causal_random", large_check=False):
    """Compare CANN baseline against pure numpy reference.

    Sanity-check that CANN and numpy_reference agree on the CANN-supported
    subset; dtype fixed to fp16 to match numpy_reference's fp16-round semantics.
    """
    from sli_grad_kl_loss_cann import SparseLightningIndexerGradKLLoss
    from sli_grad_kl_loss_numpy import numpy_reference

    dtype = ms.float16
    scale_value = 1.0 / np.sqrt(D)

    q, k, qr, kr, qi, ki, w, si = _make_inputs(
        B, S1, S2, N1, D, Nidx1, D_idx, topK, dtype=dtype)
    if sparse_pattern != "random":
        si = _make_sparse_indices(B, S1, S2, topK, sparse_pattern)
    softmax_max, softmax_sum = _compute_softmax_stats(
        q, k, qr, kr, scale_value)

    ref = numpy_reference(
        q.asnumpy(), k.asnumpy(), qr.asnumpy(), kr.asnumpy(),
        qi.asnumpy(), ki.asnumpy(), w.asnumpy(), si.asnumpy(),
        softmax_max.asnumpy(), softmax_sum.asnumpy(), scale_value,
    )

    op = SparseLightningIndexerGradKLLoss()
    d_qi_cann, d_ki_cann, d_w_cann, loss_cann = op(
        q, k, qi, ki, w, si, softmax_max, softmax_sum,
        query_rope=qr, key_rope=kr,
        scale_value=scale_value, layout="BSND", sparse_mode=3,
    )

    base_outputs = (
        ms.Tensor(ref['dQueryIndex'], dtype=d_qi_cann.dtype),
        ms.Tensor(ref['dKeyIndex'], dtype=d_ki_cann.dtype),
        ms.Tensor(ref['dW'], dtype=d_w_cann.dtype),
        ms.Tensor(ref['loss'], dtype=loss_cann.dtype),
    )
    cann_outputs = (d_qi_cann, d_ki_cann, d_w_cann, loss_cann)
    if large_check:
        _assert_outputs_close_loose(base_outputs, cann_outputs, dtype)
    else:
        _assert_outputs_close_strict(base_outputs, cann_outputs, dtype)


# ============================================================================
# Test cases
# ============================================================================
@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [ms.float16, ms.bfloat16], ids=["fp16", "bf16"])
def test_sparse_grad_kl_loss_smoke(dtype):
    _run_cann_triton_precision(
        1, 512, 4096, 64, 512, 64, 128, 2048,
        sparse_pattern="causal_continuous", large_check=True, dtype=dtype,
    )


@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", [ms.float16, ms.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "B,S1,S2,N1,D,Nidx1,D_idx,topK,sparse_pattern,large_check",
    SPARSE_GRAD_CANN_TRITON_TEST_CONFIGS,
)
def test_sparse_grad_kl_loss_precision_cann_triton(
        B, S1, S2, N1, D, Nidx1, D_idx, topK,
        sparse_pattern, large_check, dtype):
    _run_cann_triton_precision(
        B, S1, S2, N1, D, Nidx1, D_idx, topK,
        sparse_pattern=sparse_pattern, large_check=large_check, dtype=dtype,
    )


@pytest.mark.accuracy
@pytest.mark.parametrize(
    "B,S1,S2,N1,D,Nidx1,D_idx,topK",
    SPARSE_GRAD_TRITON_NUMPY_TEST_CONFIGS,
)
def test_sparse_grad_kl_loss_precision_triton_numpy(
        B, S1, S2, N1, D, Nidx1, D_idx, topK):
    _run_triton_numpy_precision(
        B, S1, S2, N1, D, Nidx1, D_idx, topK,
        sparse_pattern="causal_random",
    )


@pytest.mark.accuracy
@pytest.mark.parametrize(
    "B,S1,S2,N1,D,Nidx1,D_idx,topK,sparse_pattern,large_check",
    SPARSE_GRAD_CANN_NUMPY_TEST_CONFIGS,
)
def test_sparse_grad_kl_loss_precision_cann_numpy(
        B, S1, S2, N1, D, Nidx1, D_idx, topK,
        sparse_pattern, large_check):
    _run_cann_numpy_precision(
        B, S1, S2, N1, D, Nidx1, D_idx, topK,
        sparse_pattern=sparse_pattern, large_check=large_check,
    )


@pytest.mark.parametrize("sparse_pattern", ["causal_random", "continuous", "repeated"])
def test_sparse_grad_kl_loss_sparse_indices_patterns(sparse_pattern):
    _run_cann_triton_precision(
        B=1, S1=4, S2=2048, N1=32, D=512,
        Nidx1=8, D_idx=128, topK=1024,
        sparse_pattern=sparse_pattern,
    )


if __name__ == "__main__":
    test_sparse_grad_kl_loss_precision_cann_triton(
        1, 4, 2048, 32, 512, 8, 128, 1024,
        "causal_random", False, ms.float16)
    print("fp16 cann_triton precision test passed!")
    test_sparse_grad_kl_loss_precision_cann_triton(
        1, 4, 2048, 32, 512, 8, 128, 1024,
        "causal_random", False, ms.bfloat16)
    print("bf16 cann_triton precision test passed!")
    test_sparse_grad_kl_loss_precision_triton_numpy(
        1, 4, 2048, 128, 512, 128, 128, 1024)
    print("triton_numpy precision test passed!")
    test_sparse_grad_kl_loss_precision_cann_numpy(
        1, 4, 2048, 128, 512, 32, 128, 1024, "causal_random", False)
    print("cann_numpy precision test passed!")
