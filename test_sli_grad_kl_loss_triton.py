"""Test sparse_lightning_indexer_grad_kl_loss_triton."""
import pytest
import numpy as np
import mindspore as ms
from sparse_flash_attention_numpy import BF16 as _BF16

ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})

DROPE = 64
# _TOLS is for the large-shape smoke group only (d_ki finite-checked, not
# numerically compared). The 1e-3 target from the original TODO is now met by
# the precision group via _PRECISION_TOLS below.
_TOLS = {ms.float16: (1e-2, 1e-2), ms.bfloat16: (2e-1, 2e-2)}
NEAR_ZERO = 1e-2

# ms dtype -> numpy golden dtype (bf16 uses round-to-nearest-even sentinel)
_NP_DTYPE = {ms.float16: np.float16, ms.bfloat16: _BF16}
# Tolerances for Triton-vs-numpy (golden aligned with CANN fp32 internals,
# tighter than the CANN-vs-Triton _TOLS which stays loose for bf16).
_GEN_TOLS = {ms.float16: (1e-3, 1e-3), ms.bfloat16: (1e-2, 1e-2)}
# dKeyIndex: scatter-add atomic noise (same rationale as CANN-vs-numpy).
_GEN_DKI_TOLS = {ms.float16: (1e-2, 1e-2), ms.bfloat16: (7e-2, 7e-2)}

# Precision group (Triton vs CANN, small shapes): tightened to the numpy-vs-CANN
# leg (fp32 internals). dKeyIndex stays loose for scatter-add atomic noise.
_PRECISION_TOLS = {ms.float16: (1e-3, 1e-3), ms.bfloat16: (1e-2, 1e-2)}
_PRECISION_DKI_TOLS = {ms.float16: (1e-2, 1e-2), ms.bfloat16: (7e-2, 7e-2)}

SPARSE_GRAD_CANN_TEST_CONFIGS = [
    (1, 1, 2048, 32, 512, 8, 128, 1024),
    (1, 4, 2048, 32, 512, 8, 128, 1024),
    (1, 4, 2048, 64, 512, 16, 128, 1024),
    (1, 4, 2048, 128, 512, 32, 128, 1024),
    (1, 3, 2048, 32, 512, 16, 128, 1024),
    (1, 4, 4096, 32, 512, 64, 128, 2048),
]

SPARSE_GRAD_LARGE_TEST_CONFIGS = [
    (1, 1024, 1024, 32, 512, 8, 128, 1024),
    (1, 4096, 4096, 32, 512, 8, 128, 1024),
    (1, 4096, 4096, 64, 512, 64, 128, 2048),
]

# Generalization configs: D (DQuery) and Nidx1 values beyond CANN spec.
# CANN baseline only supports D=512 and Nidx1 in {8,16,32,64}; these cases
# compare Triton against the NumPy golden reference (fp16) instead.
# D_idx stays 128 (256/512 not supported by Triton yet).
SPARSE_GRAD_GENERALIZATION_TEST_CONFIGS = [
    (1, 4, 2048, 32, 128, 8, 128, 1024),    # D=128
    (1, 4, 2048, 32, 256, 8, 128, 1024),    # D=256
    (1, 4, 2048, 64, 256, 16, 128, 1024),   # D=256, N1=64
    (1, 4, 2048, 32, 512, 128, 128, 1024),  # Nidx1=128
    (1, 4, 2048, 64, 512, 128, 128, 1024),  # Nidx1=128, N1=64
]

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
    # bf16 asnumpy() is unreliable; cast on-device to fp32 first
    return t.astype(ms.float32).asnumpy()


def _assert_close_skip_nearzero(a, b, atol, rtol, name=None):
    a_np, b_np = np.asarray(a, np.float32), np.asarray(b, np.float32)
    near_zero = (np.abs(a_np) < NEAR_ZERO) & (np.abs(b_np) < NEAR_ZERO)
    a_f = np.where(near_zero, 0.0, a_np)
    b_f = np.where(near_zero, 0.0, b_np)
    diff = np.abs(a_f - b_f)
    max_abs = float(diff.max()) if diff.size else 0.0
    ok = np.allclose(a_f, b_f, atol=atol, rtol=rtol)
    if name:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}  max_abs_diff={max_abs:.6f}")
    assert ok, f"{name or 'tensor'}: max_abs_diff={max_abs} exceeds atol={atol}"


def _assert_outputs_close(base_outputs, tri_outputs, dtype):
    d_qi_base, d_ki_base, d_w_base, loss_base = base_outputs
    d_qi_tri, d_ki_tri, d_w_tri, loss_tri = tri_outputs
    satol, srtol = _PRECISION_TOLS[dtype]
    datol, drtol = _PRECISION_DKI_TOLS[dtype]
    _assert_close_skip_nearzero(_to_np(d_qi_base), _to_np(d_qi_tri), satol, srtol, "dQueryIndex")
    _assert_close_skip_nearzero(_to_np(d_ki_base), _to_np(d_ki_tri), datol, drtol, "dKeyIndex")
    _assert_close_skip_nearzero(_to_np(d_w_base), _to_np(d_w_tri), satol, srtol, "dW")
    _assert_close_skip_nearzero(_to_np(loss_base), _to_np(loss_tri), satol, srtol, "loss")


def _assert_large_outputs(base_outputs, tri_outputs, dtype):
    d_qi_base, d_ki_base, d_w_base, loss_base = base_outputs
    d_qi_tri, d_ki_tri, d_w_tri, loss_tri = tri_outputs
    atol, rtol = _TOLS[dtype]

    _assert_close_skip_nearzero(_to_np(d_qi_base), _to_np(d_qi_tri), atol, rtol, "dQueryIndex")
    _assert_close_skip_nearzero(_to_np(d_w_base), _to_np(d_w_tri), atol, rtol, "dW")
    _assert_close_skip_nearzero(_to_np(loss_base), _to_np(loss_tri), atol, rtol, "loss")

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
        _assert_large_outputs(base_outputs, tri_outputs, dtype)
    else:
        _assert_outputs_close(base_outputs, tri_outputs, dtype)


def _run_triton_numpy_precision(B, S1, S2, N1, D, Nidx1, D_idx, topK,
                                sparse_pattern="causal_random", dtype=ms.float16):
    """Triton vs NumPy golden for generalization configs (CANN unsupported).

    NumPy golden is aligned with CANN fp32 internals (inputs quantized to
    fp16/bf16, intermediates fp32), so both fp16 and bf16 are meaningful.
    """
    from sparse_lightning_indexer_grad_kl_loss_triton import (
        SparseLightningIndexerGradKLLossTriton,
    )
    from sli_grad_kl_loss_numpy import numpy_reference, _compute_sm_stats
    scale_value = 1.0 / np.sqrt(D)
    np_dtype = _NP_DTYPE[dtype]

    q, k, qr, kr, qi, ki, w, si = _make_inputs(
        B, S1, S2, N1, D, Nidx1, D_idx, topK, dtype=dtype)
    if sparse_pattern != "random":
        si = _make_sparse_indices(B, S1, S2, topK, sparse_pattern)

    q_np = _to_np(q)
    k_np = _to_np(k)
    qr_np = _to_np(qr)
    kr_np = _to_np(kr)
    qi_np = _to_np(qi)
    ki_np = _to_np(ki)
    w_np = _to_np(w)
    si_np = si.asnumpy()

    sm_max_np, sm_sum_np = _compute_sm_stats(
        q_np, k_np, qr_np, kr_np, scale_value, np_dtype)
    softmax_max = ms.Tensor(sm_max_np, dtype=ms.float32)
    softmax_sum = ms.Tensor(sm_sum_np, dtype=ms.float32)

    # Triton
    cell = SparseLightningIndexerGradKLLossTriton(
        scale_value=scale_value, layout="BSND", sparse_mode=3,
    )
    tri_outputs = cell(
        q, k, qi, ki, w, si, softmax_max, softmax_sum,
        query_rope=qr, key_rope=kr,
    )

    # NumPy golden
    ref = numpy_reference(
        q_np, k_np, qr_np, kr_np, qi_np, ki_np, w_np, si_np,
        sm_max_np, sm_sum_np, scale_value, np_dtype,
    )

    d_qi_tri, d_ki_tri, d_w_tri, loss_tri = tri_outputs
    strict_atol, strict_rtol = _GEN_TOLS[dtype]
    dki_atol, dki_rtol = _GEN_DKI_TOLS[dtype]
    _assert_close_skip_nearzero(_to_np(d_qi_tri), ref['dQueryIndex'], strict_atol, strict_rtol)
    _assert_close_skip_nearzero(_to_np(d_ki_tri), ref['dKeyIndex'], dki_atol, dki_rtol)
    _assert_close_skip_nearzero(_to_np(d_w_tri), ref['dW'], strict_atol, strict_rtol)
    _assert_close_skip_nearzero(_to_np(loss_tri), ref['loss'], strict_atol, strict_rtol)


@pytest.mark.parametrize("dtype", [ms.float16, ms.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("B,S1,S2,N1,D,Nidx1,D_idx,topK",
                         SPARSE_GRAD_CANN_TEST_CONFIGS)
def test_sparse_grad_kl_loss_precision(B, S1, S2, N1, D, Nidx1, D_idx, topK, dtype):
    _run_cann_triton_precision(B, S1, S2, N1, D, Nidx1, D_idx, topK, dtype=dtype)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [ms.float16, ms.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("B,S1,S2,N1,D,Nidx1,D_idx,topK",
                         SPARSE_GRAD_LARGE_TEST_CONFIGS)
def test_sparse_grad_kl_loss_large_precision(B, S1, S2, N1, D, Nidx1, D_idx, topK, dtype):
    _run_cann_triton_precision(
        B, S1, S2, N1, D, Nidx1, D_idx, topK,
        sparse_pattern="causal_continuous", large_check=True, dtype=dtype,
    )


@pytest.mark.parametrize("sparse_pattern", ["causal_random", "continuous", "repeated"])
def test_sparse_grad_kl_loss_sparse_indices_patterns(sparse_pattern):
    _run_cann_triton_precision(
        B=1, S1=4, S2=2048, N1=32, D=512,
        Nidx1=8, D_idx=128, topK=1024,
        sparse_pattern=sparse_pattern,
    )


@pytest.mark.parametrize("dtype", [ms.float16, ms.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("B,S1,S2,N1,D,Nidx1,D_idx,topK",
                         SPARSE_GRAD_GENERALIZATION_TEST_CONFIGS)
def test_sparse_grad_kl_loss_generalization(B, S1, S2, N1, D, Nidx1, D_idx, topK, dtype):
    _run_triton_numpy_precision(B, S1, S2, N1, D, Nidx1, D_idx, topK, dtype=dtype)


if __name__ == "__main__":
    test_sparse_grad_kl_loss_precision(1, 4, 2048, 32, 512, 8, 128, 1024, ms.float16)
    print("fp16 precision test passed!")
    test_sparse_grad_kl_loss_precision(1, 4, 2048, 32, 512, 8, 128, 1024, ms.bfloat16)
    print("bf16 precision test passed!")
