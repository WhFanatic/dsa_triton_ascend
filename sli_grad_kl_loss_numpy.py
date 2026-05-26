"""Pure numpy golden reference for sparse_lightning_indexer_grad_kl_loss.

Computes every stage independently and compares against CANN baseline.
"""
import numpy as np
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})

DROPE = 64


# ================================================================
# Pure numpy golden reference
# ================================================================
def numpy_reference(q, k, qr, kr, qi, ki, w, si, sm_max, sm_sum, scale):
    """Full numpy implementation, returns dict of every intermediate.

    Aligned with CANN reference: rightDownCausal mask limits effective topK
    to s2RealSize = min(topK, S2 - S1 + s1 + 1) per query position.
    """
    # Simulate fp16 input precision
    q = q.astype(np.float16).astype(np.float32)
    k = k.astype(np.float16).astype(np.float32)
    qr = qr.astype(np.float16).astype(np.float32)
    kr = kr.astype(np.float16).astype(np.float32)
    qi = qi.astype(np.float16).astype(np.float32)
    ki = ki.astype(np.float16).astype(np.float32)
    w = w.astype(np.float16).astype(np.float32)
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
            s2_real[b, s1] = min(topK, S2 - S1 + s1 + 1)

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
                dot = dot.astype(np.float16).astype(np.float32)
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
                scores = scores.astype(np.float16).astype(np.float32)
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


def _compute_sm_stats(q, k, qr, kr, scale):
    """softmaxMax/Sum from the FULL forward FlashAttention: (B, 1, S1, N1).

    These stats come from the forward pass over ALL S2 keys (not just topK).
    The backward operator then uses them to reconstruct per-key probabilities
    for the gathered topK subset.

    Simulates fp16 matmul precision used by FlashAttention.
    """
    B, S1, N1, D = q.shape
    S2 = k.shape[1]
    q_h = q.astype(np.float16).astype(np.float32)
    k_h = k.astype(np.float16).astype(np.float32)
    qr_h = qr.astype(np.float16).astype(np.float32)
    kr_h = kr.astype(np.float16).astype(np.float32)
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
                scores = scores.astype(np.float16).astype(np.float32)
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
def test_cann_vs_numpy():
    """Compare CANN baseline against numpy reference to verify test setup."""
    print("\n=== Test 6: CANN vs Numpy ===")
    try:
        from sli_grad_kl_loss_cann import (
            SparseLightningIndexerGradKLLoss,
        )
    except ImportError:
        print("  CANN baseline not available, skipping")
        return

    # Must use CANN-compatible specs: N1∈{32,64,128}, Nidx1∈{8,16,32,64},
    # D=512, D_idx=128, K∈{1024,2048,...}
    B, S1, S2, N1, D = 1, 4, 128, 32, 512
    Nidx1, D_idx, topK = 8, 128, 1024
    scale = 1.0 / np.sqrt(D)

    q, k, qr, kr, qi, ki, w, si = _make_inputs(
        B, S1, S2, N1, D, Nidx1, D_idx, topK)
    sm_max, sm_sum = _compute_sm_stats(q, k, qr, kr, scale)

    ref = numpy_reference(q, k, qr, kr, qi, ki, w, si, sm_max, sm_sum, scale)

    op = SparseLightningIndexerGradKLLoss()
    d_qi, d_ki, d_w, loss = op(
        _to_ms(q.astype(np.float16), ms.float16),
        _to_ms(k.astype(np.float16), ms.float16),
        _to_ms(qi.astype(np.float16), ms.float16),
        _to_ms(ki.astype(np.float16), ms.float16),
        _to_ms(w.astype(np.float16), ms.float16),
        _to_ms(si, ms.int32),
        _to_ms(sm_max, ms.float32),
        _to_ms(sm_sum, ms.float32),
        query_rope=_to_ms(qr.astype(np.float16), ms.float16),
        key_rope=_to_ms(kr.astype(np.float16), ms.float16),
        scale_value=scale, layout="BSND", sparse_mode=3,
    )

    cann = {
        'dQueryIndex': d_qi.asnumpy().astype(np.float32),
        'dKeyIndex': d_ki.asnumpy().astype(np.float32),
        'dW': d_w.asnumpy().astype(np.float32),
        'loss': loss.asnumpy().astype(np.float32),
    }

    print("  CANN vs Numpy:")
    _compare("dQueryIndex", ref['dQueryIndex'], cann['dQueryIndex'], atol=0.5)
    _compare("dKeyIndex", ref['dKeyIndex'], cann['dKeyIndex'], atol=0.5)
    _compare("dW", ref['dW'], cann['dW'], atol=0.5)
    _compare("loss", ref['loss'], cann['loss'], atol=0.5)


if __name__ == "__main__":
    test_cann_vs_numpy()