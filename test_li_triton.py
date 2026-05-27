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
"""Test lightning_indexer_triton against ops.lightning_indexer.

Run with:
    pytest test_lightning_indexer_triton.py -v
"""
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops

ms.set_context(mode=ms.GRAPH_MODE)


def _make_inputs(B, S1, S2, N1, D, N2=1, dtype=ms.float16):
    """Create random BSND tensors for lightning_indexer."""
    rng = np.random.RandomState(42)
    q = ms.Tensor(rng.randn(B, S1, N1, D).astype(np.float32).astype(np.float16), dtype=dtype)
    k = ms.Tensor(rng.randn(B, S2, N2, D).astype(np.float32).astype(np.float16), dtype=dtype)
    w = ms.Tensor(rng.randn(B, S1, N1).astype(np.float32).astype(np.float16), dtype=dtype)
    return q, k, w


def _allclose_indices_and_values(a_idx, a_val, b_idx, b_val, rtol=1e-3):
    """Check index match and value closeness."""
    assert a_idx.shape == b_idx.shape, f"Shape mismatch: {a_idx.shape} vs {b_idx.shape}"
    idx_match = np.allclose(a_idx.astype(np.float32), b_idx.astype(np.float32))
    val_match = np.allclose(a_val.astype(np.float32), b_val.astype(np.float32), rtol=1e-3, atol=1e-4)
    return idx_match, val_match


@pytest.mark.parametrize("B,S1,S2,N1,D,sparse_count", [
    (1, 4, 128, 8, 128, 32),
    (2, 8, 256, 16, 128, 64),
    (3, 4, 128, 8, 128, 32),
    (2, 1, 128, 8, 128, 32),
    (1, 4, 512, 8, 128, 64),
    (2, 4, 512, 8, 128, 64),
    (2, 8, 256, 8, 128, 16),
])
@pytest.mark.parametrize("sparse_mode", [0, 3])
@pytest.mark.parametrize("return_value", [False, True])
def test_vs_builtin_op(B, S1, S2, N1, D, sparse_count, sparse_mode, return_value):
    """Compare lightning_indexer_triton with ops.lightning_indexer on BSND layout."""
    from lightning_indexer_triton import LightningIndexerTriton

    q, k, w = _make_inputs(B, S1, S2, N1, D)

    ref_idx, ref_val = ops.lightning_indexer(
        q, k, w,
        layout_query="BSND", layout_key="BSND",
        sparse_count=sparse_count, sparse_mode=sparse_mode,
        return_value=return_value,
    )
    cell = LightningIndexerTriton(
        sparse_count=sparse_count, sparse_mode=sparse_mode,
        return_value=return_value,
    )
    tri_idx, tri_val = cell(q, k, w)

    idx_ok, val_ok = _allclose_indices_and_values(
        ref_idx.numpy(), ref_val.numpy(),
        tri_idx.numpy(), tri_val.numpy(),
    )
    assert idx_ok, "Index mismatch between triton and builtin op"
    if return_value:
        assert val_ok, "Value mismatch between triton and builtin op"


@pytest.mark.parametrize("B,S1,S2,N1,N2,D,sparse_count,dtype", [
    (1, 4, 128, 8, 2, 128, 32, ms.float16),
    (2, 8, 256, 16, 4, 128, 64, ms.float16),
    (1, 4, 128, 12, 1, 128, 32, ms.float16),
    (1, 4, 256, 7, 1, 128, 32, ms.float16),
    (1, 4, 128, 8, 1, 64, 32, ms.float16),
    (2, 8, 128, 16, 1, 256, 64, ms.float16),
    (1, 4, 128, 8, 1, 128, 32, ms.float32),
    (2, 8, 128, 16, 2, 96, 64, ms.float32),
])
@pytest.mark.parametrize("sparse_mode", [0, 3])
def test_relaxed_params(B, S1, S2, N1, N2, D, sparse_count, dtype, sparse_mode):
    """Test parameter combinations beyond CANN constraints (no reference comparison)."""
    from lightning_indexer_triton import LightningIndexerTriton

    q, k, w = _make_inputs(B, S1, S2, N1, D, N2=N2, dtype=dtype)

    cell = LightningIndexerTriton(
        sparse_count=sparse_count, sparse_mode=sparse_mode,
        return_value=True,
    )
    tri_idx, tri_val = cell(q, k, w)

    expected_shape = (B, S1, N2, sparse_count)
    assert tri_idx.shape == expected_shape, f"Shape mismatch: {tri_idx.shape} vs {expected_shape}"
    assert tri_val.shape == expected_shape, f"Shape mismatch: {tri_val.shape} vs {expected_shape}"
    assert tri_idx.dtype == ms.int32, f"Index dtype mismatch: {tri_idx.dtype}"
    idx_np = tri_idx.numpy()
    assert np.all((idx_np >= 0) & (idx_np < S2)), "Indices out of range"
    val_np = tri_val.numpy()
    assert np.all(np.isfinite(val_np)), "Values contain NaN or inf"


if __name__== "__main__":
    test_vs_builtin_op(1, 4, 128, 8, 128, 32,0,True)
