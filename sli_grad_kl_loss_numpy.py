"""Pure numpy golden reference for sparse_lightning_indexer_grad_kl_loss.

Computes every stage independently and compares against CANN baseline.

Precision model (aligned with CANN kernel):
  - Inputs are quantized to fp16/bf16 to simulate low-precision inputs.
  - All intermediate matmul/softmax/loss stay in fp32 (CANN MM12_OUT_T=float,
    base.h:38), so matmul outputs are NOT quantized back to fp16/bf16.
"""
import numpy as np
import pytest
import mindspore as ms
from sparse_flash_attention_numpy import BF16, _round_bf16

ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})

DROPE = 64


def _quantize(x, dtype):
    """fp32 -> (fp16 | bf16) -> fp32, simulating low-precision inputs.

    Mirrors CANN: cube matmul takes fp16/bf16 inputs but accumulates in fp32.
    bf16 uses round-to-nearest-even (no external bf16 package required).
    """
    if dtype == np.float16:
        return x.astype(np.float16).astype(np.float32)
    if dtype == BF16:
        return _round_bf16(x)
    return x.astype(np.float32)


# ================================================================
# Pure numpy golden reference
# ================================================================
def numpy_reference(q, k, qr, kr, qi, ki, w, si, sm_max, sm_sum, scale, dtype=np.float16):
    """Full numpy implementation, returns dict of every intermediate.

    Aligned with CANN reference: rightDownCausal mask limits effective topK
    to s2RealSize = min(topK, max(S2 - S1 + s1 + 1, 0)) per query position.
    """
    # Simulate low-precision inputs; intermediates stay fp32 (CANN MM12_OUT_T=float)
    q = _quantize(q, dtype)
    k = _quantize(k, dtype)
    qr = _quantize(qr, dtype)
    kr = _quantize(kr, dtype)
    qi = _quantize(qi, dtype)
    ki = _quantize(ki, dtype)
    w = _quantize(w, dtype)
    si = si.astype(np.int32)

    B, S1, N1, D = q.shape
    S2 = k.shape[1]
    Nidx1, D_idx = qi.shape[2], qi.shape[3]
    topK = si.shape[3]
    D_rope = qr.shape[3]

    EPS = 1e-8

    # Per (b,s1) effective topK: rightDownCausal only sees s2RealSize positions
    s2_real = np.zeros((B, S1), dtype=np.int32)
    for b in range(B):
        for s1 in range(S1):
            s2_real[b, s1] = min(topK, max(S2 - S1 + s1 + 1, 0))

    results = {}

    # --- Gather (only s2RealSize elements per query) ---
    key_gathered = np.zeros((B, S1, topK, D), dtype=np.float32)
    ki_gathered = np.zeros((B, S1, topK, D_idx), dtype=np.float32)
    kr_gathered = np.zeros((B, S1, topK, D_rope), dtype=np.float32)
    for b in range(B):
        for s1 in range(S1):
            n = s2_real[b, s1]
            idx = si[b, s1, 0, :n]
            idx = np.clip(idx, 0, S2 - 1)
            key_gathered[b, s1, :n] = k[b, idx, 0, :]
            ki_gathered[b, s1, :n] = ki[b, idx, 0, :]
            kr_gathered[b, s1, :n] = kr[b, idx, 0, :]
    results['key_gathered'] = key_gathered
    results['ki_gathered'] = ki_gathered
    results['kr_gathered'] = kr_gathered

    # --- Stage 1: I = Σ_g W_g * ReLU(qi_g @ ki_gathered^T) ---
    I_scores = np.zeros((B, S1, topK), dtype=np.float32)
    relu_cache = np.zeros((B, S1, Nidx1, topK), dtype=np.float32)
    for b in range(B):
        for s1 in range(S1):
            n = s2_real[b, s1]
            for g in range(Nidx1):
                dot = qi[b, s1, g, :] @ ki_gathered[b, s1, :n].T  # (n,)
                relu = np.maximum(dot, 0.0)
                relu_cache[b, s1, g, :n] = relu
                I_scores[b, s1, :n] += w[b, s1, g] * relu
    results['I_scores'] = I_scores
    results['relu_cache'] = relu_cache

    # --- Stage 2: p from softmaxMax/Sum ---
    p = np.zeros((B, S1, topK), dtype=np.float32)
    for b in range(B):
        for s1 in range(S1):
            n = s2_real[b, s1]
            k_g = key_gathered[b, s1, :n]
            kr_g = kr_gathered[b, s1, :n]
            for h in range(N1):
                scores = (q[b, s1, h] @ k_g.T +
                          qr[b, s1, h] @ kr_g.T) * scale
                sm_max_h = sm_max[b, 0, s1, h]
                sm_sum_h = sm_sum[b, 0, s1, h]
                probs = np.exp(scores - sm_max_h) / (sm_sum_h + EPS)
                p[b, s1, :n] += probs
    p *= (1.0 / N1)
    results['p'] = p

    # --- Stage 3: softmax(I) ---
    softmax_I = np.zeros_like(I_scores)
    for b in range(B):
        for s1 in range(S1):
            n = s2_real[b, s1]
            i_vec = I_scores[b, s1, :n]
            i_max = np.max(i_vec)
            i_exp = np.exp(i_vec - i_max)
            softmax_I[b, s1, :n] = i_exp / (np.sum(i_exp) + EPS)
    results['softmax_I'] = softmax_I

    # --- Stage 4: dI and KL loss ---
    dI = softmax_I - p
    results['dI'] = dI

    kl_loss = 0.0
    for b in range(B):
        for s1 in range(S1):
            n = s2_real[b, s1]
            p_vec = np.maximum(p[b, s1, :n], EPS)
            si_vec = np.maximum(softmax_I[b, s1, :n], EPS)
            kl_loss += np.sum(p_vec * (np.log(p_vec) - np.log(si_vec)))
    results['loss'] = np.array([kl_loss], dtype=np.float32)

    # --- Stage 5: dW, dQueryIndex ---
    dW = np.zeros((B, S1, Nidx1), dtype=np.float32)
    dQueryIndex = np.zeros_like(qi, dtype=np.float32)
    for b in range(B):
        for s1 in range(S1):
            n = s2_real[b, s1]
            for g in range(Nidx1):
                relu_g = relu_cache[b, s1, g, :n]
                relu_mask_g = (relu_g > 0).astype(np.float32)
                dW[b, s1, g] = np.sum(dI[b, s1, :n] * relu_g)
                ds_idx_g = dI[b, s1, :n] * w[b, s1, g] * relu_mask_g
                dQueryIndex[b, s1, g] = ds_idx_g @ ki_gathered[b, s1, :n]
    results['dW'] = dW
    results['dQueryIndex'] = dQueryIndex

    # --- Stage 6: Scatter dKeyIndex ---
    dKeyIndex = np.zeros_like(ki, dtype=np.float32)
    for b in range(B):
        for s1 in range(S1):
            n = s2_real[b, s1]
            for k_idx in range(n):
                target = si[b, s1, 0, k_idx]
                if target < 0 or target >= S2:
                    continue
                for g in range(Nidx1):
                    relu_g = relu_cache[b, s1, g]
                    relu_mask_g = (relu_g[k_idx] > 0).astype(np.float32)
                    dki_contrib = dI[b, s1, k_idx] * w[b, s1, g] * relu_mask_g
                    dKeyIndex[b, target, 0] += dki_contrib * qi[b, s1, g]
    results['dKeyIndex'] = dKeyIndex

    return results


# ================================================================
# Helpers
# ================================================================
def _make_inputs(B, S1, S2, N1, D, Nidx1, D_idx, topK, seed=42):
    rng = np.random.RandomState(seed)
    q = rng.randn(B, S1, N1, D).astype(np.float32)
    k = rng.randn(B, S2, 1, D).astype(np.float32)
    qr = rng.randn(B, S1, N1, DROPE).astype(np.float32)
    kr = rng.randn(B, S2, 1, DROPE).astype(np.float32)
    qi = rng.randn(B, S1, Nidx1, D_idx).astype(np.float32)
    ki = rng.randn(B, S2, 1, D_idx).astype(np.float32)
    w = np.abs(rng.randn(B, S1, Nidx1)).astype(np.float32)
    si = rng.randint(0, S2, (B, S1, 1, topK)).astype(np.int32)
    return q, k, qr, kr, qi, ki, w, si


def _make_causal_sparse_indices(B, S1, S2, topK):
    """rightDownCausal sparse indices: unique keys within each row's causal
    window [0, S2-S1+s1+1), zero-padded. Matches the CANN golden contract and
    the test-file _make_sparse_indices("causal_random") so that numpy and CANN
    see identical sparse_indices (random randint si can violate the causal
    window and trigger CANN nan/overflow at larger N1)."""
    rng = np.random.RandomState(42)
    si = np.zeros((B, S1, 1, topK), dtype=np.int32)
    for b in range(B):
        for s1 in range(S1):
            visible = min(max(S2 - S1 + s1 + 1, 1), S2)
            valid_k = min(topK, visible)
            si[b, s1, 0, :valid_k] = rng.choice(
                visible, size=valid_k, replace=False).astype(np.int32)
    return si


def _compute_sm_stats(q, k, qr, kr, scale, dtype=np.float16):
    """softmaxMax/Sum from the FULL forward FlashAttention: (B, 1, S1, N1).

    These stats come from the forward pass over ALL S2 keys (not just topK).
    The backward operator then uses them to reconstruct per-key probabilities
    for the gathered topK subset.

    Intermediates stay fp32 (CANN forward FA accumulates in fp32).
    """
    B, S1, N1, D = q.shape
    S2 = k.shape[1]
    q_h = _quantize(q, dtype)
    k_h = _quantize(k, dtype)
    qr_h = _quantize(qr, dtype)
    kr_h = _quantize(kr, dtype)
    sm_max = np.full((B, 1, S1, N1), -np.inf, dtype=np.float32)
    sm_sum = np.zeros((B, 1, S1, N1), dtype=np.float32)
    for b in range(B):
        for s1 in range(S1):
            all_k = k_h[b, :, 0, :]
            all_kr = kr_h[b, :, 0, :]
            causal_limit = S2 - S1 + s1 + 1
            for h in range(N1):
                scores = (q_h[b, s1, h] @ all_k.T
                          + qr_h[b, s1, h] @ all_kr.T) * scale
                scores[causal_limit:] = float('-inf')
                s_max = np.max(scores)
                sm_max[b, 0, s1, h] = s_max
                sm_sum[b, 0, s1, h] = np.sum(np.exp(scores - s_max))
    return sm_max, sm_sum


def _to_ms(arr, dtype=None):
    if dtype is None:
        dtype = ms.float32 if arr.dtype == np.float32 else ms.int32
    return ms.Tensor(arr, dtype=dtype)


def _compare(name, ref, test, atol=1e-2, rtol=1e-2):
    ref_f = ref.flatten()
    test_f = test.flatten()
    if ref_f.shape != test_f.shape:
        print(f"  {name}: SHAPE MISMATCH ref={ref.shape} test={test.shape}")
        return False
    close = np.allclose(ref_f, test_f, atol=atol, rtol=rtol)
    max_abs = np.max(np.abs(ref_f - test_f))
    n_mismatch = np.sum(~np.isclose(ref_f, test_f, atol=atol, rtol=rtol))
    pct = 100 * n_mismatch / ref_f.size
    status = "PASS" if close else "FAIL"
    print(f"  {name}: {status}  max_abs_diff={max_abs:.6f}  "
          f"mismatch={n_mismatch}/{ref_f.size} ({pct:.1f}%)")
    if not close:
        # Show first few values
        idx = np.where(~np.isclose(ref_f, test_f, atol=atol, rtol=rtol))[0][:5]
        for i in idx:
            print(f"    [{i}] ref={ref_f[i]:.6f} test={test_f[i]:.6f} "
                  f"diff={abs(ref_f[i]-test_f[i]):.6f}")
    return close


# ================================================================
# Compare CANN vs Numpy (if available)
# ================================================================
# CANN-supported specs for numpy-vs-CANN cross-validation.
# D=512 only, Nidx1 in {8,16,32,64} (CANN doc), topK in {1024,2048}.
# Establishes the numpy ≈ CANN leg of the transitivity proof
# (Triton ≈ numpy ∧ numpy ≈ CANN ⇒ Triton ≈ CANN).
SPARSE_GRAD_CANN_VS_NUMPY_CONFIGS = [
    (1, 4, 2048, 32, 512, 8, 128, 1024),
    (1, 4, 2048, 64, 512, 16, 128, 1024),
    (1, 4, 2048, 128, 512, 32, 128, 1024),
    (1, 4, 4096, 32, 512, 64, 128, 2048),
]

_MS_DTYPE = {np.float16: ms.float16, BF16: ms.bfloat16}
_CANN_VS_NUMPY_TOLS = {np.float16: (1e-3, 1e-3), BF16: (1e-2, 1e-2)}
# dKeyIndex uses scatter-add (atomic-add on hardware vs fixed-order numpy);
# a few collision points diverge by rounding regardless of correctness, so it
# gets a looser atol. fp16 max seen ~0.011, bf16 max seen ~0.065.
_CANN_VS_NUMPY_DKI_TOLS = {np.float16: (1e-2, 1e-2), BF16: (7e-2, 7e-2)}


def _to_ms_dtype(arr, dtype):
    """fp32 numpy -> ms.Tensor of the given low-precision dtype.

    fp16: cast via np.float16. bf16: round-to-nearest-even onto the bf16 grid
    (fp32 repr), then cast to ms.bfloat16 losslessly. Ensures numpy golden
    and CANN see identical input values.
    """
    if dtype == np.float16:
        return ms.Tensor(arr.astype(np.float16), dtype=ms.float16)
    return ms.Tensor(_round_bf16(arr), dtype=ms.bfloat16)


@pytest.mark.parametrize("dtype", [np.float16, BF16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("B,S1,S2,N1,D,Nidx1,D_idx,topK",
                         SPARSE_GRAD_CANN_VS_NUMPY_CONFIGS)
def test_cann_vs_numpy(B, S1, S2, N1, D, Nidx1, D_idx, topK, dtype):
    """Compare CANN baseline against numpy reference (fp16/bf16, multi-shape)."""
    try:
        from sli_grad_kl_loss_cann import (
            SparseLightningIndexerGradKLLoss,
        )
    except ImportError:
        pytest.skip("CANN baseline not available")

    scale = 1.0 / np.sqrt(D)

    q, k, qr, kr, qi, ki, w, si = _make_inputs(
        B, S1, S2, N1, D, Nidx1, D_idx, topK)
    si = _make_causal_sparse_indices(B, S1, S2, topK)
    sm_max, sm_sum = _compute_sm_stats(q, k, qr, kr, scale, dtype)

    ref = numpy_reference(q, k, qr, kr, qi, ki, w, si, sm_max, sm_sum, scale, dtype)

    op = SparseLightningIndexerGradKLLoss()
    d_qi, d_ki, d_w, loss = op(
        _to_ms_dtype(q, dtype),
        _to_ms_dtype(k, dtype),
        _to_ms_dtype(qi, dtype),
        _to_ms_dtype(ki, dtype),
        _to_ms_dtype(w, dtype),
        _to_ms(si, ms.int32),
        _to_ms(sm_max, ms.float32),
        _to_ms(sm_sum, ms.float32),
        query_rope=_to_ms_dtype(qr, dtype),
        key_rope=_to_ms_dtype(kr, dtype),
        scale_value=scale, layout="BSND", sparse_mode=3,
    )

    cann = {
        'dQueryIndex': d_qi.astype(ms.float32).asnumpy(),
        'dKeyIndex': d_ki.astype(ms.float32).asnumpy(),
        'dW': d_w.astype(ms.float32).asnumpy(),
        'loss': loss.astype(ms.float32).asnumpy(),
    }
    strict_atol, strict_rtol = _CANN_VS_NUMPY_TOLS[dtype]
    dki_atol, dki_rtol = _CANN_VS_NUMPY_DKI_TOLS[dtype]
    ok = True
    for name in ('dQueryIndex', 'dW', 'loss'):
        ok = _compare(name, ref[name], cann[name], atol=strict_atol, rtol=strict_rtol) and ok
    ok = _compare('dKeyIndex', ref['dKeyIndex'], cann['dKeyIndex'],
                  atol=dki_atol, rtol=dki_rtol) and ok
    assert ok, f"CANN vs numpy mismatch (dtype={dtype}, shape={(B,S1,S2,N1,D,Nidx1,D_idx,topK)})"


if __name__ == "__main__":
    test_cann_vs_numpy(1, 4, 2048, 32, 512, 8, 128, 1024, np.float16)