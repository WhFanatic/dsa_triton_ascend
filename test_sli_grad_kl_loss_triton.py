"""Test sparse_lightning_indexer_grad_kl_loss_triton."""
import pytest
import numpy as np
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})

DROPE = 64
ATOL = 1e-2
RTOL = 1e-2

SPARSE_GRAD_CANN_TEST_CONFIGS = [
    (1, 1, 128, 32, 512, 8, 128, 1024),
    (1, 4, 128, 32, 512, 8, 128, 1024),
    (1, 4, 128, 64, 512, 16, 128, 1024),
    (1, 4, 128, 128, 512, 32, 128, 1024),
    (1, 3, 96, 32, 512, 16, 128, 1024),
    (1, 4, 256, 32, 512, 64, 128, 2048),
]

SPARSE_GRAD_LARGE_TEST_CONFIGS = [
    (1, 1024, 1024, 32, 512, 8, 128, 1024),
    (1, 4096, 4096, 32, 512, 8, 128, 1024),
]

SPARSE_GRAD_ACTUAL_SEQ_TEST_CONFIGS = [
    (1, 4, 128, 32, 512, 8, 128, 1024, [3], [112]),
    (2, 4, 128, 32, 512, 8, 128, 1024, [4, 3], [128, 112]),
]


def _make_inputs(B, S1, S2, N1, D, Nidx1, D_idx, topK, dtype=ms.float16):
    rng = np.random.RandomState(42)
    N2, Nidx2 = 1, 1
    q = ms.Tensor(rng.randn(B, S1, N1, D).astype(np.float16), dtype=dtype)
    k = ms.Tensor(rng.randn(B, S2, N2, D).astype(np.float16), dtype=dtype)
    qr = ms.Tensor(rng.randn(B, S1, N1, DROPE).astype(np.float16), dtype=dtype)
    kr = ms.Tensor(rng.randn(B, S2, N2, DROPE).astype(np.float16), dtype=dtype)
    qi = ms.Tensor(rng.randn(B, S1, Nidx1, D_idx).astype(np.float16), dtype=dtype)
    ki = ms.Tensor(rng.randn(B, S2, Nidx2, D_idx).astype(np.float16), dtype=dtype)
    w = ms.Tensor(np.abs(rng.randn(B, S1, Nidx1)).astype(np.float16), dtype=dtype)
    si = ms.Tensor(rng.randint(0, S2, (B, S1, Nidx2, topK)).astype(np.int32), dtype=ms.int32)
    return q, k, qr, kr, qi, ki, w, si


def _make_sparse_indices(B, S1, S2, topK, pattern):
    if pattern == "random":
        rng = np.random.RandomState(42)
        si = rng.randint(0, S2, (B, S1, 1, topK)).astype(np.int32)
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
                scores = scores.astype(np.float16).astype(np.float32)
                scores[causal_limit:] = float('-inf')
                s_max = np.max(scores)
                sm_max[b, 0, s1, h] = s_max
                sm_sum[b, 0, s1, h] = np.sum(np.exp(scores - s_max))

    return (ms.Tensor(sm_max, dtype=ms.float32),
            ms.Tensor(sm_sum, dtype=ms.float32))


def _assert_outputs_close(base_outputs, tri_outputs):
    d_qi_base, d_ki_base, d_w_base, loss_base = base_outputs
    d_qi_tri, d_ki_tri, d_w_tri, loss_tri = tri_outputs
    np.testing.assert_allclose(d_qi_base.asnumpy(), d_qi_tri.asnumpy(), atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(d_ki_base.asnumpy(), d_ki_tri.asnumpy(), atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(d_w_base.asnumpy(), d_w_tri.asnumpy(), atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(loss_base.asnumpy(), loss_tri.asnumpy(), atol=ATOL, rtol=RTOL)


def _assert_large_outputs(base_outputs, tri_outputs):
    d_qi_base, d_ki_base, d_w_base, loss_base = base_outputs
    d_qi_tri, d_ki_tri, d_w_tri, loss_tri = tri_outputs

    np.testing.assert_allclose(d_qi_base.asnumpy(), d_qi_tri.asnumpy(), atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(d_w_base.asnumpy(), d_w_tri.asnumpy(), atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(loss_base.asnumpy(), loss_tri.asnumpy(), atol=ATOL, rtol=RTOL)

    assert d_ki_tri.shape == d_ki_base.shape, (
        f"d_ki shape mismatch: {d_ki_tri.shape} vs {d_ki_base.shape}"
    )
    assert d_ki_tri.dtype == d_ki_base.dtype, (
        f"d_ki dtype mismatch: {d_ki_tri.dtype} vs {d_ki_base.dtype}"
    )
    assert np.isfinite(d_ki_tri.asnumpy().astype(np.float32)).all()


def _run_cann_triton_precision(B, S1, S2, N1, D, Nidx1, D_idx, topK,
                               actual_seq_qlen=None, actual_seq_klen=None,
                               sparse_pattern="random", large_check=False):
    from sparse_lightning_indexer_grad_kl_loss_triton import (
        SparseLightningIndexerGradKLLossTriton,
    )
    from sli_grad_kl_loss_cann import (
        SparseLightningIndexerGradKLLoss,
    )
    scale_value = 1.0 / np.sqrt(D)

    q, k, qr, kr, qi, ki, w, si = _make_inputs(B, S1, S2, N1, D, Nidx1, D_idx, topK)
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
        _assert_large_outputs(base_outputs, tri_outputs)
    else:
        _assert_outputs_close(base_outputs, tri_outputs)


@pytest.mark.parametrize("B,S1,S2,N1,D,Nidx1,D_idx,topK",
                         SPARSE_GRAD_CANN_TEST_CONFIGS)
def test_sparse_grad_kl_loss_precision(B, S1, S2, N1, D, Nidx1, D_idx, topK):
    _run_cann_triton_precision(B, S1, S2, N1, D, Nidx1, D_idx, topK)


@pytest.mark.large
@pytest.mark.parametrize("B,S1,S2,N1,D,Nidx1,D_idx,topK",
                         SPARSE_GRAD_LARGE_TEST_CONFIGS)
def test_sparse_grad_kl_loss_large_precision(B, S1, S2, N1, D, Nidx1, D_idx, topK):
    _run_cann_triton_precision(
        B, S1, S2, N1, D, Nidx1, D_idx, topK,
        sparse_pattern="causal_continuous", large_check=True,
    )


@pytest.mark.parametrize(
    "B,S1,S2,N1,D,Nidx1,D_idx,topK,actual_seq_qlen,actual_seq_klen",
    SPARSE_GRAD_ACTUAL_SEQ_TEST_CONFIGS,
)
def test_sparse_grad_kl_loss_actual_seq_precision(
        B, S1, S2, N1, D, Nidx1, D_idx, topK,
        actual_seq_qlen, actual_seq_klen):
    _run_cann_triton_precision(
        B, S1, S2, N1, D, Nidx1, D_idx, topK,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_klen=actual_seq_klen,
    )


@pytest.mark.parametrize("sparse_pattern", ["random", "continuous", "repeated"])
def test_sparse_grad_kl_loss_sparse_indices_patterns(sparse_pattern):
    _run_cann_triton_precision(
        B=1, S1=4, S2=128, N1=32, D=512,
        Nidx1=8, D_idx=128, topK=1024,
        sparse_pattern=sparse_pattern,
    )


if __name__ == "__main__":
    test_sparse_grad_kl_loss_precision(1, 4, 128, 32, 512, 8, 128, 1024)
    print("precision test passed!")
