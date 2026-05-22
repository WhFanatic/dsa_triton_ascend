"""Test sparse_lightning_indexer_grad_kl_loss_triton.
Run with:
    pytest test_sparse_grad_kl_loss_triton.py -v
"""
import pytest
import numpy as np
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE)


def _make_inputs(B, S1, S2, N1, D, Nidx1, D_idx, topK, dtype=ms.float16):
    rng = np.random.RandomState(42)
    N2 = 1
    Nidx2 = 1
    q = ms.Tensor(rng.randn(B, S1, N1, D).astype(np.float32).astype(np.float16), dtype=dtype)
    k = ms.Tensor(rng.randn(B, S2, N2, D).astype(np.float32).astype(np.float16), dtype=dtype)
    qi = ms.Tensor(rng.randn(B, S1, Nidx1, D_idx).astype(np.float32).astype(np.float16), dtype=dtype)
    ki = ms.Tensor(rng.randn(B, S2, Nidx2, D_idx).astype(np.float32).astype(np.float16), dtype=dtype)
    w = ms.Tensor(np.abs(rng.randn(B, S1, Nidx1)).astype(np.float32).astype(np.float16), dtype=dtype)
    si = ms.Tensor(rng.randint(0, S2, (B, S1, Nidx2, topK)).astype(np.int32), dtype=ms.int32)
    sm = ms.Tensor(np.zeros((B, S1, N2, topK)).astype(np.float16), dtype=dtype)
    return q, k, qi, ki, w, si, sm, sm


@pytest.mark.parametrize("B,S1,S2,N1,D,Nidx1,D_idx,topK", [
    (1, 4, 128, 8, 128, 4, 64, 32),
    (2, 8, 256, 8, 128, 4, 64, 32),
])
def test_sparse_grad_kl_loss(B, S1, S2, N1, D, Nidx1, D_idx, topK):
    from sparse_lightning_indexer_grad_kl_loss_triton import (
        sparse_lightning_indexer_grad_kl_loss_triton,
    )

    q, k, qi, ki, w, si, sm, ss = _make_inputs(B, S1, S2, N1, D, Nidx1, D_idx, topK)

    d_qi, d_ki, d_w, loss = sparse_lightning_indexer_grad_kl_loss_triton(
        q, k, qi, ki, w, si, sm, ss,
        scale_value=1.0,
    )

    assert d_qi.shape == qi.shape
    assert d_ki.shape == ki.shape
    assert d_w.shape == w.shape
    assert loss.shape == (1,)


if __name__ == "__main__":
    test_sparse_grad_kl_loss(1, 4, 128, 8, 128, 4, 64, 32)
