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


def _make_inputs(B, S1, S2, N1, N2, D, dtype=ms.float16):
    """Create random BSND tensors for lightning_indexer."""
    rng = np.random.RandomState(42)
    q = ms.Tensor(rng.randn(B, S1, N1, D).astype(np.float32).astype(np.float16), dtype=dtype)
    k = ms.Tensor(rng.randn(B, S2, N2, D).astype(np.float32).astype(np.float16), dtype=dtype)
    w = ms.Tensor(rng.randn(B, S1, N1).astype(np.float32).astype(np.float16), dtype=dtype)
    return q, k, w


def _allclose(a, b):
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    return np.allclose(a.astype(np.float32), b.astype(np.float32), rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("S1, S2, N1, N2, D, sparse_count", [
    # 小 shape, 每行叠多个边界:
    (1, 128, 8, 1, 128, 32),        # S1=1 单 query 行 (causal 下只见 key0)
    (4, 256, 7, 1, 96, 32),         # N1=7 奇数 + D=96 非整除 + S1≠S2
    (8, 128, 16, 2, 64, 64),        # N2>1 分组 + D=64 小 + N1/N2 整除
    (3, 256, 12, 4, 256, 32),       # N2=4 分组 + N1=12 + D=256 大 + S1≠S2
    # 中大 shape: grid/UB/causal/topk:
    (1024, 1024, 64, 1, 128, 512),
    (4096, 4096, 64, 1, 128, 2048),     # 生产 shape
    (4096, 8192, 64, 1, 128, 2048),     # S2>S1 非方阵 (grid 超限回归点)
    (16*1024, 16*1024, 64, 1, 128, 2048),  # 超大, grid 上限压测
])
@pytest.mark.parametrize("B", [1, 2, 3])
@pytest.mark.parametrize("sparse_mode", [0, 3])
@pytest.mark.parametrize("dtype", [ms.float16, ms.float32])
def test_basic(B, S1, S2, N1, N2, D, sparse_count, sparse_mode, dtype, return_value=False):
    """Test parameter combinations beyond CANN constraints (no reference comparison)."""
    from lightning_indexer_triton import LightningIndexerTriton

    q, k, w = _make_inputs(B, S1, S2, N1, N2, D, dtype)

    cell = LightningIndexerTriton(
        sparse_count=sparse_count, sparse_mode=sparse_mode,
        return_value=return_value,
    )
    tri_idx, tri_val = cell(q, k, w)

    expected_shape = (B, S1, N2, sparse_count)
    assert tri_idx.shape == expected_shape, f"Shape mismatch: {tri_idx.shape} vs {expected_shape}"
    assert tri_idx.dtype == ms.int32, f"Index dtype mismatch: {tri_idx.dtype}"
    idx_np = tri_idx.numpy()
    assert np.any((idx_np >= 0) & (idx_np < S2)), ("Indices out of range", idx_np)

    if not return_value:
        return

    assert tri_val.shape == expected_shape, f"Shape mismatch: {tri_val.shape} vs {expected_shape}"
    val_np = tri_val.numpy()
    assert np.any(np.isfinite(val_np)), ("Values contain NaN or inf", val_np)


@pytest.mark.parametrize("S1, S2, N1, N2, D, sparse_count", [
    # N1=64, N2=1, D=128 固定 (参考算子 ops.lightning_indexer 限制)
    (1,        128,       64, 1, 128, 32),     # S1=1, causal 极端边界
    (128,      128,       64, 1, 128, 64),     # 小方阵
    (256,      512,       64, 1, 128, 128),    # 非方阵 S2>S1
    (512,      256,       64, 1, 128, 128),    # 非方阵 S1>S2
    (1024,     1024,      64, 1, 128, 512),    # 中等方阵
    (4096,     4096,      64, 1, 128, 2048),   # 生产 shape, k=S2/2
    (4096,     8192,      64, 1, 128, 2048),   # S2>S1 大非方阵 (grid 超限回归)
    (16*1024,  16*1024,   64, 1, 128, 2048),   # 超大方阵, grid 上限压测
])
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("sparse_mode", [0, 3])
def test_accuracy(B, S1, S2, N1, N2, D, sparse_count, sparse_mode):
    """Compare lightning_indexer_triton with ops.lightning_indexer on BSND layout."""
    from lightning_indexer_triton import LightningIndexerTriton

    q, k, w = _make_inputs(B, S1, S2, N1, N2, D, ms.float16)

    ref_idx, ref_val = ops.lightning_indexer(
        q, k, w,
        layout_query="BSND", layout_key="BSND",
        sparse_count=sparse_count, sparse_mode=sparse_mode,
        return_value=True,
    )
    cell = LightningIndexerTriton(
        sparse_count=sparse_count, sparse_mode=sparse_mode,
        return_value=True,
    )
    tri_idx, tri_val = cell(q, k, w)

    tri_idx_np, tri_val_np = tri_idx.numpy(), tri_val.numpy()
    ref_idx_np, ref_val_np = ref_idx.numpy(), ref_val.numpy()

    idx_valid = tri_idx_np == ref_idx_np
    idx_mismatch = (~idx_valid).sum()
    idx_err_rate = idx_mismatch / tri_idx_np.size

    idx_err = ("Index mismatch between triton and builtin op", tri_idx - ref_idx, tri_idx, ref_idx, tri_val, ref_val)
    val_err = ("Value mismatch between triton and builtin op", tri_val - ref_val, tri_val, ref_val)

    if idx_err_rate < 0.01:
        if idx_mismatch: print(f"Index mismatch: {idx_mismatch} / {tri_idx_np.size}")
        assert _allclose(tri_val_np[idx_valid], ref_val_np[idx_valid]), val_err
    else:
        assert False, idx_err


if __name__== "__main__":
    test_basic(B=1, S1=4096, S2=4096, N1=64, N2=1, D=128, sparse_count=2048, sparse_mode=3, dtype=ms.float16, return_value=True)
    # test_accuracy(B=1, S1=4096, S2=4096, N1=64, N2=1, D=128, sparse_count=2048, sparse_mode=3)
