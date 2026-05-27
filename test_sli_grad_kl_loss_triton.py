"""Test sparse_lightning_indexer_grad_kl_loss_triton."""
import pytest
import numpy as np
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})

DROPE = 64


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


def _compute_softmax_stats(q, k, qr, kr, si, scale_value):
    """softmaxMax/Sum from FULL forward FlashAttention: (B, 1, S1, N1).
    
    Must compute over ALL S2 keys, not just topK gathered subset.
    """
    B, S1, N1, D = q.shape
    S2 = k.shape[1]
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
            causal_limit = S2 - S1 + s1 + 1
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


@pytest.mark.parametrize("B,S1,S2,N1,D,Nidx1,D_idx,topK", [
    (1, 4, 128, 32, 512, 8, 128, 1024),
])
def test_sparse_grad_kl_loss_precision(B, S1, S2, N1, D, Nidx1, D_idx, topK):
    from sparse_lightning_indexer_grad_kl_loss_triton import (
        SparseLightningIndexerGradKLLossTriton,
    )
    from sli_grad_kl_loss_cann import (
        SparseLightningIndexerGradKLLoss,
    )
    scale_value = 1.0 / np.sqrt(D)

    q, k, qr, kr, qi, ki, w, si = _make_inputs(B, S1, S2, N1, D, Nidx1, D_idx, topK)
    softmax_max, softmax_sum = _compute_softmax_stats(q, k, qr, kr, si, scale_value)

    # CANN baseline
    op = SparseLightningIndexerGradKLLoss()
    d_qi_base, d_ki_base, d_w_base, loss_base = op(
        q, k, qi, ki, w, si, softmax_max, softmax_sum,
        query_rope=qr, key_rope=kr,
        scale_value=scale_value, layout="BSND", sparse_mode=3,
    )

    # Triton
    cell = SparseLightningIndexerGradKLLossTriton(
        scale_value=scale_value, layout="BSND", sparse_mode=3,
    )
    d_qi_tri, d_ki_tri, d_w_tri, loss_tri = cell(
        q, k, qi, ki, w, si, softmax_max, softmax_sum,
        query_rope=qr, key_rope=kr,
    )

    np.testing.assert_allclose(d_qi_base.asnumpy(), d_qi_tri.asnumpy(), atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(d_ki_base.asnumpy(), d_ki_tri.asnumpy(), atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(d_w_base.asnumpy(), d_w_tri.asnumpy(), atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(loss_base.asnumpy(), loss_tri.asnumpy(), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    test_sparse_grad_kl_loss_precision(1, 4, 128, 32, 512, 8, 128, 1024)
    print("precision test passed!")
