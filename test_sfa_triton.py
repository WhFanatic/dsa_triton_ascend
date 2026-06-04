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
"""Tests for sparse_flash_attention_triton.

  - test_golden:   triton vs numpy golden (algorithm correctness, any shape)
  - test_accuracy: triton vs ops.sparse_flash_attention (CANN baseline, BSND)
  - test_basic:    shape / dtype / finiteness self-checks beyond reference limits

Run:
    python test_sfa_triton.py                  # quick __main__ debug
    pytest --forked test_sfa_triton.py -v
"""
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops

from sparse_flash_attention_numpy import BF16 as _BF16

ms.set_context(mode=ms.GRAPH_MODE)

D_NOPE = 512
D_ROPE = 64

# ms dtype -> the `dtype` arg the numpy golden rounds to (bf16 uses a pure-numpy
# round-to-nearest-even sentinel, so no external bf16 package is required).
_NP_DTYPE = {ms.float16: np.float16, ms.bfloat16: _BF16, ms.float32: np.float32}


def _to_np_f32(t):
    """Tensor -> fp32 numpy (bf16 asnumpy() is unreliable; cast on-device first)."""
    return t.astype(ms.float32).asnumpy()


def _make_inputs(B, S1, S2, N1, sparse_count, dtype=ms.float16, D=D_NOPE):
    """Random BSND tensors for SFA (MQA: N2=1). value passed but ignored (=key)."""
    rng = np.random.RandomState(42)

    def _t(shape):
        return ms.Tensor(rng.randn(*shape).astype(np.float16), dtype=dtype)

    q = _t((B, S1, N1, D))
    k = _t((B, S2, 1, D))
    v = _t((B, S2, 1, D))
    qr = _t((B, S1, N1, D_ROPE))
    kr = _t((B, S2, 1, D_ROPE))
    si = _make_sparse_indices(B, S1, S2, sparse_count, sparse_block_size=1, sparse_mode=3)
    return q, k, v, qr, kr, si


def _make_sparse_indices(B, S1, S2, sparse_count, sparse_block_size, sparse_mode):
    """Front-valid / back=-1 block ids within each row's causal window.

    Matches the CANN golden contract: valid ids packed at the front, -1 padding
    at the back. block id space = ceil(threshold / sparse_block_size).
    """
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
            # keep the last block (holds the causal boundary) + random earlier ones
            perm = rng.permutation(num_blocks)[:n]
            si[b, s1, 0, :n] = np.sort(perm).astype(np.int32)
    return ms.Tensor(si, dtype=ms.int32)


def _allclose(a, b, dtype=ms.float16):
    # bf16 (8-bit mantissa) needs looser tol than fp16 (10-bit); both accumulate
    # over topK in fp32 but round probs/out to the in/out dtype before bmm2.
    rtol = atol = 4e-2 if dtype == ms.bfloat16 else 2e-2
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    return np.allclose(a, b, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# triton vs numpy golden — algorithm correctness, runs on any shape
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("B,S1,S2,N1,sparse_count,sparse_block_size,sparse_mode", [
    (1, 4, 128, 8, 64, 1, 3),       # token-wise, rightDownCausal
    (1, 4, 128, 8, 64, 1, 0),       # token-wise, full
    (2, 16, 256, 16, 128, 1, 3),    # bigger, multi-batch
    (1, 8, 128, 8, 32, 2, 3),       # block-wise (block_size=2)
    (1, 8, 256, 16, 32, 4, 3),      # block-wise (block_size=4)
    (1, 1, 128, 8, 16, 1, 3),       # S1=1 single query row
])
@pytest.mark.parametrize("D", [128, 256, 512])  # CANN fixes 512; golden verifies the rest
@pytest.mark.parametrize("dtype", [ms.float16, ms.bfloat16])
def test_golden(B, S1, S2, N1, sparse_count, sparse_block_size, sparse_mode, D, dtype):
    """Compare triton SFA with the numpy golden reference (D 128/256/512, fp16/bf16)."""
    from sparse_flash_attention_triton import SparseFlashAttentionTriton
    from sparse_flash_attention_numpy import sparse_flash_attention_golden_bsnd

    np_dtype = _NP_DTYPE[dtype]

    q, k, v, qr, kr, _ = _make_inputs(B, S1, S2, N1, sparse_count, dtype, D=D)
    si = _make_sparse_indices(B, S1, S2, sparse_count, sparse_block_size, sparse_mode)
    scale = 1.0 / np.sqrt(D + D_ROPE)

    cell = SparseFlashAttentionTriton(
        scale_value=scale, sparse_block_size=sparse_block_size,
        sparse_mode=sparse_mode, return_softmax_lse=True,
    )
    out_t, smax_t, ssum_t = cell(q, k, v, si, query_rope=qr, key_rope=kr)

    out_g, smax_g, ssum_g = sparse_flash_attention_golden_bsnd(
        _to_np_f32(q), _to_np_f32(k), _to_np_f32(v), si.asnumpy(),
        _to_np_f32(qr), _to_np_f32(kr),
        scale, [S1] * B, [S2] * B,
        sparse_block_size=sparse_block_size, sparse_mode=sparse_mode,
        return_softmax_lse=True, dtype=np_dtype,
    )

    assert _allclose(_to_np_f32(out_t), out_g, dtype), "attention_out mismatch vs golden"
    assert _allclose(_to_np_f32(smax_t), smax_g, dtype), "softmax_max mismatch vs golden"
    assert _allclose(_to_np_f32(ssum_t), ssum_g, dtype), "softmax_sum mismatch vs golden"


# ---------------------------------------------------------------------------
# triton vs CANN ops.sparse_flash_attention — NPU baseline
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("B,S1,S2,N1,sparse_count,sparse_mode", [
    (1, 4, 128, 16, 64, 3),
    (2, 8, 256, 32, 128, 3),
    (1, 8, 128, 16, 64, 0),
])
@pytest.mark.parametrize("dtype", [ms.float16, ms.bfloat16])  # bf16 = mindformers compute_dtype
def test_accuracy(B, S1, S2, N1, sparse_count, sparse_mode, dtype):
    """Compare triton SFA with ops.sparse_flash_attention (BSND, token-wise)."""
    from sparse_flash_attention_triton import SparseFlashAttentionTriton

    q, k, v, qr, kr, _ = _make_inputs(B, S1, S2, N1, sparse_count, dtype)
    si = _make_sparse_indices(B, S1, S2, sparse_count, 1, sparse_mode)
    scale = 1.0 / np.sqrt(D_NOPE + D_ROPE)

    ref_out, ref_max, ref_sum = ops.sparse_flash_attention(
        q, k, v, si, scale,
        query_rope=qr, key_rope=kr,
        layout_query="BSND", layout_kv="BSND",
        sparse_block_size=1, sparse_mode=sparse_mode,
        attention_mode=2, return_softmax_lse=True,
    )
    cell = SparseFlashAttentionTriton(
        scale_value=scale, sparse_block_size=1,
        sparse_mode=sparse_mode, return_softmax_lse=True,
    )
    tri_out, tri_max, tri_sum = cell(q, k, v, si, query_rope=qr, key_rope=kr)

    assert _allclose(_to_np_f32(tri_out), _to_np_f32(ref_out), dtype), "attention_out mismatch vs CANN"
    assert _allclose(_to_np_f32(tri_max), _to_np_f32(ref_max), dtype), "softmax_max mismatch vs CANN"
    assert _allclose(_to_np_f32(tri_sum), _to_np_f32(ref_sum), dtype), "softmax_sum mismatch vs CANN"


# ---------------------------------------------------------------------------
# functional self-checks — shapes beyond CANN reference constraints
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("B,S1,S2,N1,sparse_count", [
    (1, 128, 1024, 64, 512),
    (2, 64, 512, 32, 256),
])
@pytest.mark.parametrize("D", [128, 256, 512])
@pytest.mark.parametrize("sparse_mode", [0, 3])
@pytest.mark.parametrize("return_lse", [True, False])
@pytest.mark.parametrize("dtype", [ms.float16, ms.bfloat16])
def test_basic(B, S1, S2, N1, sparse_count, D, sparse_mode, return_lse, dtype):
    """Shape / dtype / finiteness checks (no reference comparison)."""
    from sparse_flash_attention_triton import SparseFlashAttentionTriton

    q, k, v, qr, kr, _ = _make_inputs(B, S1, S2, N1, sparse_count, dtype, D=D)
    si = _make_sparse_indices(B, S1, S2, sparse_count, 1, sparse_mode)
    scale = 1.0 / np.sqrt(D + D_ROPE)

    cell = SparseFlashAttentionTriton(
        scale_value=scale, sparse_mode=sparse_mode, return_softmax_lse=return_lse,
    )
    out, smax, ssum = cell(q, k, v, si, query_rope=qr, key_rope=kr)

    assert out.shape == (B, S1, N1, D), f"out shape {out.shape}"
    assert out.dtype == dtype, f"out dtype {out.dtype}"
    assert smax.shape == (B, 1, S1, N1), f"smax shape {smax.shape}"
    assert ssum.shape == (B, 1, S1, N1), f"ssum shape {ssum.shape}"
    assert np.all(np.isfinite(_to_np_f32(out))), "out has NaN/inf"


if __name__ == "__main__":
    test_golden(1, 4, 128, 8, 64, 1, 3, 512, ms.float16)
    print("golden test (D=512, token-wise, mode3, fp16) passed!")
    test_golden(1, 4, 128, 8, 64, 1, 3, 128, ms.float16)
    print("golden test (D=128, token-wise, mode3, fp16) passed!")
    test_golden(1, 8, 128, 8, 32, 2, 3, 256, ms.float16)
    print("golden test (D=256, block-wise bs=2, mode3, fp16) passed!")
    test_golden(1, 4, 128, 8, 64, 1, 3, 512, ms.bfloat16)
    print("golden test (D=512, token-wise, mode3, bf16) passed!")
