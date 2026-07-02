# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for sparse_flash_attention_grad_triton.

  - test_golden:   triton vs numpy backward golden (algorithm correctness, any shape)
  - test_accuracy: triton vs ms.grad(ops.sparse_flash_attention) (CANN backward, BSND)
  - test_basic:    shape / dtype / finiteness self-checks beyond reference limits
  - test_guards:   unsupported interface params raise ValueError (no NPU needed)

The forward stats (softmax_max/sum) and out are produced by the forward triton op
(test_golden) or by ops.sparse_flash_attention (test_accuracy), so the backward is
tested on consistent forward state.

Run:
    python test_sfa_grad_triton.py                  # quick __main__ debug
    pytest --forked test_sfa_grad_triton.py -v
"""
import math

import pytest
import numpy as np
import mindspore as ms
from mindspore import ops

from sparse_flash_attention_numpy import BF16 as _BF16

ms.set_context(mode=ms.GRAPH_MODE)

D_NOPE = 512
D_ROPE = 64

_NP_DTYPE = {ms.float16: np.float16, ms.bfloat16: _BF16, ms.float32: np.float32}


def _to_np_f32(t):
    return t.astype(ms.float32).asnumpy()


def _make_sparse_indices(B, S1, S2, sparse_count, sparse_block_size, sparse_mode):
    """Front-valid / back=-1 block ids within each row's causal window (CANN contract)."""
    rng = np.random.RandomState(7)
    si = np.full((B, S1, 1, sparse_count), -1, dtype=np.int32)
    act_q, act_k = S1, S2
    for b in range(B):
        for s1 in range(S1):
            if sparse_mode == 0:
                threshold = act_k
            else:
                threshold = act_k - act_q + s1 + 1
            if threshold <= 0:
                continue
            num_blocks = int(np.ceil(threshold / sparse_block_size))
            n = min(sparse_count, num_blocks)
            perm = rng.permutation(num_blocks)[:n]
            si[b, s1, 0, :n] = np.sort(perm).astype(np.int32)
    return ms.Tensor(si, dtype=ms.int32)


def _make_inputs(B, S1, S2, N1, sparse_count, dtype=ms.bfloat16, D=D_NOPE):
    """Random BSND tensors (MQA: N2=1). value passed but ignored (=key)."""
    rng = np.random.RandomState(42)

    def _t(shape):
        return ms.Tensor(rng.randn(*shape).astype(np.float16) * 0.3, dtype=dtype)

    q = _t((B, S1, N1, D))
    k = _t((B, S2, 1, D))
    v = _t((B, S2, 1, D))
    qr = _t((B, S1, N1, D_ROPE))
    kr = _t((B, S2, 1, D_ROPE))
    do = _t((B, S1, N1, D))
    return q, k, v, qr, kr, do


def _allclose(a, b, dtype=ms.bfloat16, scale=1.0):
    rtol = 6e-2 if dtype == ms.bfloat16 else 3e-2
    atol = (5e-2 if dtype == ms.bfloat16 else 2e-2) * scale
    max_diff_hd = 10
    pct_thd = 99.5

    a = np.asarray(a, np.float32).flatten()
    b = np.asarray(b, np.float32).flatten()
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    close = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    fail_mask = ~close
    if fail_mask.any():
        fa, fb = a[fail_mask], b[fail_mask]
        diff = np.abs(fa - fb)
        rtol_only = np.abs(fb) * rtol
        print(f"[diag] failed {fail_mask.sum()}/{fail_mask.size}  "
              f"atol={atol:.4e} rtol={rtol}")
        print(f"[diag]   |diff|  min={diff.min():.6e} max={diff.max():.6e}")
        print(f"[diag]   rtol*|b| min={rtol_only.min():.6e} max={rtol_only.max():.6e}")
        print(f"[diag]   atol-dominated (|b|<{atol/rtol:.4f}): "
              f"{(np.abs(fb) < atol/rtol).sum()}")
    pass_pct = close.sum() / close.size * 100.0
    if pass_pct < pct_thd:
        return False

    diff_abs = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b)) + 1e-10
    rel_err = diff_abs / denom
    if np.any(rel_err[~close] >= max_diff_hd):
        return False
    return True


# ---------------------------------------------------------------------------
# triton vs numpy golden — algorithm correctness, runs on any shape
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("B,S1,S2,N1,sparse_count,sparse_block_size,sparse_mode", [
    (1, 4, 128, 8, 64, 1, 3),       # token-wise, rightDownCausal
    # (1, 4, 128, 8, 64, 1, 0),       # token-wise, full
    (2, 16, 256, 16, 128, 1, 3),    # bigger, multi-batch
    (1, 8, 128, 8, 32, 2, 3),       # block-wise (block_size=2)
    (1, 8, 256, 16, 32, 4, 3),      # block-wise (block_size=4)
    (1, 1, 128, 8, 16, 1, 3),       # S1=1 single query row
    (1, 4, 128, 1, 64, 1, 3),       # N1=1 single head (MQA degenerate, head mask)
    (1, 4, 128, 128, 64, 1, 3),     # N1=128 upper bound (BLOCK_G head tiling)
    (1, 4, 2048, 8, 2048, 1, 3),    # topK=2048, rightDownCausal
    # (1, 4, 2048, 8, 2048, 1, 0),    # topK=2048, full
])
@pytest.mark.parametrize("D", [128, 256, 512])  # CANN fixes 512; golden verifies the rest
@pytest.mark.parametrize("dtype", [ms.float16, ms.bfloat16])
@pytest.mark.parametrize("fwd_source", ["triton", "cann"])
def test_golden(B, S1, S2, N1, sparse_count, sparse_block_size, sparse_mode, D, dtype, fwd_source):
    """Compare triton SFA backward with the numpy golden (D 128/256/512, fp16/bf16).

    fwd_source="triton": triton forward -> triton backward vs golden (end-to-end triton).
    fwd_source="cann":   CANN forward -> triton backward vs golden (isolates backward accuracy).
    """
    if fwd_source == "cann" and D != 512:
        pytest.skip("CANN sparse_flash_attention only supports D=512")

    from sparse_flash_attention_triton import SparseFlashAttentionTriton
    from sparse_flash_attention_grad_triton import SparseFlashAttentionGradTriton
    from sparse_flash_attention_grad_numpy import sparse_flash_attention_grad_golden_bsnd

    np_dtype = _NP_DTYPE[dtype]
    q, k, v, qr, kr, do = _make_inputs(B, S1, S2, N1, sparse_count, dtype, D=D)
    si = _make_sparse_indices(B, S1, S2, sparse_count, sparse_block_size, sparse_mode)
    scale = 1.0 / np.sqrt(D + D_ROPE)

    if fwd_source == "triton":
        fwd = SparseFlashAttentionTriton(
            scale_value=scale, sparse_block_size=sparse_block_size,
            sparse_mode=sparse_mode, return_softmax_lse=True,
        )
        out, smax, ssum = fwd(q, k, v, si, query_rope=qr, key_rope=kr)
    else:
        out, smax, ssum = ops.sparse_flash_attention(
            q, k, v, si, scale,
            query_rope=qr, key_rope=kr,
            layout_query="BSND", layout_kv="BSND",
            sparse_block_size=sparse_block_size, sparse_mode=sparse_mode,
            attention_mode=2, return_softmax_lse=True)

    grad = SparseFlashAttentionGradTriton(
        scale_value=scale, sparse_block_size=sparse_block_size, sparse_mode=sparse_mode,
    )
    dq, dk, dv, dqr, dkr = grad(
        q, k, v, si, do, out, smax, ssum, query_rope=qr, key_rope=kr)

    g_dq, g_dk, g_dv, g_dqr, g_dkr = sparse_flash_attention_grad_golden_bsnd(
        _to_np_f32(q), _to_np_f32(k), _to_np_f32(v), si.asnumpy(),
        _to_np_f32(do), _to_np_f32(out), _to_np_f32(smax), _to_np_f32(ssum),
        _to_np_f32(qr), _to_np_f32(kr),
        scale, [S1] * B, [S2] * B,
        sparse_block_size=sparse_block_size, sparse_mode=sparse_mode, dtype=np_dtype,
    )

    assert _allclose(_to_np_f32(dq), g_dq, dtype), "d_query mismatch vs golden"
    assert _allclose(_to_np_f32(dqr), g_dqr, dtype), "d_query_rope mismatch vs golden"
    assert _allclose(_to_np_f32(dk), g_dk, dtype, scale=math.sqrt(S1)), "d_key mismatch vs golden"
    assert _allclose(_to_np_f32(dv), g_dv, dtype, scale=math.sqrt(S1)), "d_value mismatch vs golden"
    assert _allclose(_to_np_f32(dkr), g_dkr, dtype, scale=math.sqrt(S1)), "d_key_rope mismatch vs golden"


# ---------------------------------------------------------------------------
# triton vs CANN ms.grad(ops.sparse_flash_attention) — NPU backward baseline
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("B,S1,S2,N1,sparse_count,sparse_mode", [
    (1, 4, 128, 16, 64, 3),
    (2, 8, 256, 32, 128, 3),
    # (1, 8, 128, 16, 64, 0),
    (1, 4, 2048, 16, 2048, 3),   # topK=2048 vs CANN
    # (1, 4, 2048, 16, 2048, 0),   # topK=2048, full mode vs CANN
    (1, 512, 4096, 64, 64, 3),   # perf 崩溃 shape vs CANN (大 S1/S2, topK=64)
])
@pytest.mark.parametrize("dtype", [ms.float16, ms.bfloat16])  # bf16 = mindformers compute_dtype
def test_accuracy(B, S1, S2, N1, sparse_count, sparse_mode, dtype):
    """Compare triton SFA backward with CANN's backward via ms.grad.

    ms.grad through ops.sparse_flash_attention lowers to CANN SparseFlashAttentionGrad.
    We build a scalar loss = sum(out * do) so d(loss)/d(input) == the input grad
    contracted with do, matching what our op returns for d_out=do.
    """
    from sparse_flash_attention_grad_triton import SparseFlashAttentionGradTriton

    q, k, v, qr, kr, do = _make_inputs(B, S1, S2, N1, sparse_count, dtype)
    si = _make_sparse_indices(B, S1, S2, sparse_count, 1, sparse_mode)
    scale = 1.0 / np.sqrt(D_NOPE + D_ROPE)

    def _fwd(q_, k_, v_, qr_, kr_):
        out, _, _ = ops.sparse_flash_attention(
            q_, k_, v_, si, scale,
            query_rope=qr_, key_rope=kr_,
            layout_query="BSND", layout_kv="BSND",
            sparse_block_size=1, sparse_mode=sparse_mode,
            attention_mode=2, return_softmax_lse=True)
        return out

    # forward (CANN) for out / softmax stats consumed by the triton backward
    ref_out, ref_max, ref_sum = ops.sparse_flash_attention(
        q, k, v, si, scale, query_rope=qr, key_rope=kr,
        layout_query="BSND", layout_kv="BSND",
        sparse_block_size=1, sparse_mode=sparse_mode,
        attention_mode=2, return_softmax_lse=True)

    # CANN backward: grad of loss=sum(out*do) wrt (q, k, v, qr, kr)
    def _loss(q_, k_, v_, qr_, kr_):
        return (_fwd(q_, k_, v_, qr_, kr_) * do).astype(ms.float32).sum()
    ref_dq, ref_dk, ref_dv, ref_dqr, ref_dkr = ms.grad(
        _loss, grad_position=(0, 1, 2, 3, 4))(q, k, v, qr, kr)

    grad = SparseFlashAttentionGradTriton(scale_value=scale, sparse_mode=sparse_mode)
    dq, dk, dv, dqr, dkr = grad(
        q, k, v, si, do, ref_out, ref_max, ref_sum, query_rope=qr, key_rope=kr)

    # MLA-absorb: value==key, and CANN may attribute the P@dO-path grad to either
    # its d_key or d_value output (forward ignores `value`). Compare the SUM, which
    # is invariant to that split — and equals the merged key grad used in training.
    dkv_merged = _to_np_f32(dk) + _to_np_f32(dv)
    ref_dkv_merged = _to_np_f32(ref_dk) + _to_np_f32(ref_dv)

    assert _allclose(_to_np_f32(dq), _to_np_f32(ref_dq), dtype), "d_query mismatch vs CANN"
    assert _allclose(_to_np_f32(dqr), _to_np_f32(ref_dqr), dtype), "d_query_rope mismatch vs CANN"
    assert _allclose(dkv_merged, ref_dkv_merged, dtype, scale=math.sqrt(S1)), "d_key+d_value mismatch vs CANN"
    assert _allclose(_to_np_f32(dkr), _to_np_f32(ref_dkr), dtype, scale=math.sqrt(S1)), "d_key_rope mismatch vs CANN"


# ---------------------------------------------------------------------------
# functional self-checks — shapes beyond CANN reference constraints
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("B,S1,S2,N1,sparse_count", [
    (1, 128, 1024, 64, 512),
    (2, 64, 512, 32, 256),
    (1, 16, 2048, 64, 2048),   # topK=2048
    (1, 512, 4096, 64, 2048),  # perf shape, topK=2048 (sparse_count > S2, clamp 生效)
])
@pytest.mark.parametrize("D", [128, 256, 512])
@pytest.mark.parametrize("sparse_mode", [3])
@pytest.mark.parametrize("dtype", [ms.float16, ms.bfloat16])
def test_basic(B, S1, S2, N1, sparse_count, D, sparse_mode, dtype):
    """Shape / dtype / finiteness checks (no reference comparison)."""
    from sparse_flash_attention_triton import SparseFlashAttentionTriton
    from sparse_flash_attention_grad_triton import SparseFlashAttentionGradTriton

    q, k, v, qr, kr, do = _make_inputs(B, S1, S2, N1, sparse_count, dtype, D=D)
    si = _make_sparse_indices(B, S1, S2, sparse_count, 1, sparse_mode)
    scale = 1.0 / np.sqrt(D + D_ROPE)

    fwd = SparseFlashAttentionTriton(
        scale_value=scale, sparse_mode=sparse_mode, return_softmax_lse=True)
    out, smax, ssum = fwd(q, k, v, si, query_rope=qr, key_rope=kr)

    grad = SparseFlashAttentionGradTriton(scale_value=scale, sparse_mode=sparse_mode)
    dq, dk, dv, dqr, dkr = grad(
        q, k, v, si, do, out, smax, ssum, query_rope=qr, key_rope=kr)

    assert dq.shape == (B, S1, N1, D) and dq.dtype == dtype, f"dq {dq.shape}/{dq.dtype}"
    assert dqr.shape == (B, S1, N1, D_ROPE) and dqr.dtype == dtype, f"dqr {dqr.shape}"
    assert dk.shape == (B, S2, 1, D) and dk.dtype == dtype, f"dk {dk.shape}/{dk.dtype}"
    assert dv.shape == (B, S2, 1, D) and dv.dtype == dtype, f"dv {dv.shape}"
    assert dkr.shape == (B, S2, 1, D_ROPE) and dkr.dtype == dtype, f"dkr {dkr.shape}"
    for name, t in (("dq", dq), ("dk", dk), ("dv", dv), ("dqr", dqr), ("dkr", dkr)):
        assert np.all(np.isfinite(_to_np_f32(t))), f"{name} has NaN/inf"


# ---------------------------------------------------------------------------
# smoke — fast per-edit regression. Each case is a FULL param tuple (no D/dtype
# cross-product), so the count is exactly what's listed. Reuses test_golden /
# test_accuracy bodies. Run after every edit:  pytest --forked -k smoke
# Run the full matrix only after large structural / logic changes.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("B,S1,S2,N1,sc,bs,mode,D,dtype,fwd_source", [
    (2, 16, 256, 16, 128, 1, 3, 512, ms.float16, "cann"),    # multi-batch, CANN fwd
    (1, 4, 128, 8, 64, 1, 3, 512, ms.bfloat16, "triton"),    # B_S1<BLOCK_S1, triton fwd
    # (1, 4, 128, 8, 64, 1, 0, 256, ms.float16, "triton"),   # sparse_mode 0 (full), D=256
    (1, 1, 128, 8, 16, 1, 3, 128, ms.float16, "triton"),     # S1=1 single row, D=128
    (1, 8, 128, 8, 32, 2, 3, 512, ms.float16, "cann"),       # block-wise (bs=2), CANN fwd
    (1, 4, 2048, 8, 2048, 1, 3, 256, ms.float16, "triton"),  # topK=2048
])
def test_smoke_golden(B, S1, S2, N1, sc, bs, mode, D, dtype, fwd_source):
    """Fast backward golden subset covering the BLOCK_S1 folding risk points."""
    test_golden(B, S1, S2, N1, sc, bs, mode, D, dtype, fwd_source)


@pytest.mark.parametrize("B,S1,S2,N1,sparse_count,sparse_mode,dtype", [
    (2, 8, 256, 32, 128, 3, ms.bfloat16),  # multi-batch + bf16 vs CANN
    # (1, 8, 128, 16, 64, 0, ms.float16),    # full mode vs CANN
    (1, 4, 2048, 16, 2048, 3, ms.float16),  # topK=2048 vs CANN
])
def test_smoke_accuracy(B, S1, S2, N1, sparse_count, sparse_mode, dtype):
    """Fast CANN-baseline subset (backward)."""
    test_accuracy(B, S1, S2, N1, sparse_count, sparse_mode, dtype)


# ---------------------------------------------------------------------------
# negative guards — unsupported interface params must raise, not silently run
# ---------------------------------------------------------------------------
def test_guards():
    """Each unsupported CANN-interface knob raises ValueError at the host gate."""
    from sparse_flash_attention_grad_triton import (
        sparse_flash_attention_grad_triton as gfn, INT64_MAX)

    B, S1, S2, N1, sc, D = 1, 4, 128, 8, 64, D_NOPE
    q, k, v, qr, kr, do = _make_inputs(B, S1, S2, N1, sc, ms.float16, D=D)
    si = _make_sparse_indices(B, S1, S2, sc, 1, 3)
    scale = 1.0 / np.sqrt(D + D_ROPE)
    # dummy forward state (guards fire before any of it is touched)
    out = do
    smax = ms.Tensor(np.zeros((B, 1, S1, N1), np.float32), ms.float32)
    ssum = ms.Tensor(np.ones((B, 1, S1, N1), np.float32), ms.float32)

    def _call(**kw):
        base = dict(scale_value=scale, query_rope=qr, key_rope=kr, sparse_mode=3)
        base.update(kw)
        return gfn(q, k, v, si, do, out, smax, ssum, **base)

    with pytest.raises(ValueError):
        _call(pre_tokens=0)
    with pytest.raises(ValueError):
        _call(next_tokens=0)
    with pytest.raises(ValueError):
        _call(sparse_mode=1)               # only 0 / 3 supported
    with pytest.raises(ValueError):
        _call(deterministic=True)
    with pytest.raises(ValueError):
        _call(sparse_block_size=3)         # not a power of 2
    with pytest.raises(ValueError):
        gfn(q, k, v, si, do, out, smax, ssum,
            scale_value=scale, query_rope=None, key_rope=kr, sparse_mode=3)


if __name__ == "__main__":
    # Diagnostic smoke: run every case to completion (no abort on first mismatch)
    # and print, per gradient, max abs/rel diff + over-tolerance fraction + worst
    # location. Tells "marginal rounding" apart from "structural" errors.
    from sparse_flash_attention_triton import SparseFlashAttentionTriton
    from sparse_flash_attention_grad_triton import SparseFlashAttentionGradTriton
    from sparse_flash_attention_grad_numpy import sparse_flash_attention_grad_golden_bsnd

    def _diag(tri, gold, dtype, scale):
        a = np.asarray(_to_np_f32(tri), np.float32)
        b = np.asarray(gold, np.float32)
        rtol = 6e-2 if dtype == ms.bfloat16 else 3e-2
        atol = (5e-2 if dtype == ms.bfloat16 else 2e-2) * scale
        adiff = np.abs(a - b)
        tol = atol + rtol * np.abs(b)
        over = adiff > tol
        idx = np.unravel_index(np.argmax(adiff), adiff.shape)
        return (bool(np.all(np.isfinite(a))), adiff.max(),
                float(over.mean()), idx, a[idx], b[idx],
                float(np.abs(b).max()))

    def _run(B, S1, S2, N1, sc, bs, mode, D, dtype, fwd_source="triton"):
        q, k, v, qr, kr, do = _make_inputs(B, S1, S2, N1, sc, dtype, D=D)
        si = _make_sparse_indices(B, S1, S2, sc, bs, mode)
        scale = 1.0 / np.sqrt(D + D_ROPE)
        if fwd_source == "triton":
            fwd = SparseFlashAttentionTriton(scale_value=scale, sparse_block_size=bs,
                                             sparse_mode=mode, return_softmax_lse=True)
            out, smax, ssum = fwd(q, k, v, si, query_rope=qr, key_rope=kr)
        else:
            out, smax, ssum = ops.sparse_flash_attention(
                q, k, v, si, scale,
                query_rope=qr, key_rope=kr,
                layout_query="BSND", layout_kv="BSND",
                sparse_block_size=bs, sparse_mode=mode,
                attention_mode=2, return_softmax_lse=True)
        grad = SparseFlashAttentionGradTriton(scale_value=scale,
                                              sparse_block_size=bs, sparse_mode=mode)
        dq, dk, dv, dqr, dkr = grad(q, k, v, si, do, out, smax, ssum,
                                    query_rope=qr, key_rope=kr)
        g = sparse_flash_attention_grad_golden_bsnd(
            _to_np_f32(q), _to_np_f32(k), _to_np_f32(v), si.asnumpy(),
            _to_np_f32(do), _to_np_f32(out), _to_np_f32(smax), _to_np_f32(ssum),
            _to_np_f32(qr), _to_np_f32(kr), scale, [S1] * B, [S2] * B,
            sparse_block_size=bs, sparse_mode=mode, dtype=_NP_DTYPE[dtype])
        print(f"\n=== D={D} mode{mode} bs={bs} {dtype} fwd={fwd_source} (S1={S1} scale_atol={math.sqrt(S1):.1f}) ===")
        for name, t, gld, sc_ in (("dq", dq, g[0], 1), ("dk", dk, g[1], math.sqrt(S1)),
                                  ("dv", dv, g[2], math.sqrt(S1)), ("dqr", dqr, g[3], 1),
                                  ("dkr", dkr, g[4], math.sqrt(S1))):
            fin, mx, frac, idx, av, bv, bmax = _diag(t, gld, dtype, sc_)
            tag = "ok " if frac == 0 else "OVER"
            print(f"  {name:4s} {tag} maxabs={mx:.3e} over={frac:6.2%} "
                  f"|gold|max={bmax:.3e} worst@{idx} tri={av:+.4e} gold={bv:+.4e}"
                  f"{'' if fin else '  [NaN/inf!]'}")

    _run(1, 4, 128, 8, 64, 1, 3, 512, ms.bfloat16, "cann")
    _run(1, 4, 128, 8, 64, 1, 0, 128, ms.bfloat16, "triton")
    _run(1, 8, 128, 8, 32, 2, 3, 256, ms.bfloat16, "triton")
    _run(1, 4, 128, 8, 64, 1, 3, 512, ms.bfloat16, "triton")
    _run(1, 4, 2048, 8, 2048, 1, 3, 256, ms.bfloat16, "triton")
