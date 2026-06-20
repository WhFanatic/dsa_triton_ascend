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

from lightning_indexer_numpy import lightning_indexer_golden_bsnd

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
    # 补充 N1=[32,128], D=[256,512] 覆盖:
    (1024, 1024, 32, 1, 128, 512),
    (1024, 1024, 128, 1, 128, 512),
    (1024, 1024, 64, 1, 256, 512),
    (1024, 1024, 32, 1, 512, 512),
    (1024, 1024, 128, 1, 512, 512),
])
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("sparse_mode", [3])
@pytest.mark.parametrize("dtype", [ms.float16, ms.bfloat16])
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
@pytest.mark.parametrize("sparse_mode", [3])
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

    not_invalid = (tri_idx_np != -1) & (ref_idx_np != -1)
    compare_mask = idx_valid & not_invalid

    if idx_err_rate < 0.01:
        if idx_mismatch: print(f"Index mismatch: {idx_mismatch} / {tri_idx_np.size}")
        if compare_mask.any():
            assert _allclose(tri_val_np[compare_mask], ref_val_np[compare_mask]), val_err
    else:
        assert False, idx_err


@pytest.mark.parametrize("S1, S2, N1, N2, D, sparse_count", [
    (1, 128, 8, 1, 128, 32),
    (4, 256, 7, 1, 96, 32),
    (8, 128, 16, 2, 64, 64),
    (3, 256, 12, 4, 256, 32),
    (128, 128, 64, 1, 128, 64),
    (256, 512, 64, 1, 128, 128),
    (512, 256, 64, 1, 128, 128),
    (1024, 1024, 64, 1, 128, 512),
    (1024, 1024, 32, 1, 128, 512),
    (1024, 1024, 128, 1, 128, 512),
    (1024, 1024, 32, 1, 256, 512),
    (1024, 1024, 32, 1, 512, 512),
    (1024, 1024, 64, 1, 256, 512),
    (1024, 1024, 64, 1, 512, 512),
    (1024, 1024, 128, 1, 256, 512),
    (1024, 1024, 128, 1, 512, 512),
])
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("sparse_mode", [3])
@pytest.mark.parametrize("dtype", [ms.float16, ms.bfloat16])
def test_golden(B, S1, S2, N1, N2, D, sparse_count, sparse_mode, dtype):
    """Compare lightning_indexer_triton with numpy golden reference."""
    from lightning_indexer_triton import LightningIndexerTriton

    q, k, w = _make_inputs(B, S1, S2, N1, N2, D, dtype)

    cell = LightningIndexerTriton(
        sparse_count=sparse_count, sparse_mode=sparse_mode,
        return_value=True,
    )
    tri_idx, tri_val = cell(q, k, w)

    q_np = q.astype(ms.float32).numpy()
    k_np = k.astype(ms.float32).numpy()
    w_np = w.astype(ms.float32).numpy()
    act_q = np.full(B, S1, dtype=np.int32)
    act_k = np.full(B, S2, dtype=np.int32)

    ref_idx, ref_val = lightning_indexer_golden_bsnd(
        q_np, k_np, w_np, act_q, act_k,
        sparse_count=sparse_count, sparse_mode=sparse_mode,
        return_value=True,
        value_dtype="bf16" if dtype == ms.bfloat16 else "fp16",
    )

    tri_idx_np = tri_idx.numpy()
    tri_val_np = tri_val.astype(ms.float32).numpy()

    idx_err = ("Index set mismatch between triton and numpy golden", tri_idx_np, ref_idx)
    val_err = ("Value mismatch between triton and numpy golden", tri_val_np, ref_val)

    # 与 CANN result_compare_method.check_result 同语义比对:
    #  - top-k 同分时 lexsort(golden) 与硬件 topk(triton) 排序/边界成员可互换, 故按
    #    "有效集合 + 选中分值分布"比对, 不做逐位置索引相等;
    #  - 值比对用 CANN 权威判定: 带下限相对误差 (b=max(|a|,|b|, (1/16384)/diff_thd)) +
    #    允许 pct_thd 比例元素超 isclose, 且最大相对误差 < max_diff_hd。bf16 大 shape 下
    #    triton(tl.dot 分块累加) 与 golden(numpy matmul) fp32 累加顺序不同, 个别元素落到
    #    bf16 会差 1 个量化点, 该判定恰好覆盖此类 ULP 噪声 (fp16 因尾数更宽几乎不触发)。
    rtol, atol = 5e-3, 2.5e-5
    diff_thd, max_diff_hd, pct_thd = 0.01, 0.1, 0.05
    rows = B * S1 * N2
    tri_i = tri_idx_np.reshape(rows, sparse_count)
    ref_i = ref_idx.reshape(rows, sparse_count)
    tri_v = tri_val_np.reshape(rows, sparse_count)
    ref_v = ref_val.reshape(rows, sparse_count)

    tv_all, rv_all = [], []
    for r in range(rows):
        tri_m = tri_i[r] != -1
        ref_m = ref_i[r] != -1
        assert tri_m.sum() == ref_m.sum(), idx_err   # 有效 topk 数量一致

        tv = np.sort(tri_v[r][tri_m])
        rv = np.sort(ref_v[r][ref_m])
        tv_all.append(tv)
        rv_all.append(rv)

        # 集合差异只允许发生在 top-k 边界同分: 互斥索引分值需落在边界值容差内
        only_tri = set(tri_i[r][tri_m].tolist()) - set(ref_i[r][ref_m].tolist())
        if only_tri and rv.size:
            bound = rv[0]
            for s2 in only_tri:
                v = tri_v[r][np.where(tri_i[r] == s2)[0][0]]
                assert abs(v - bound) <= rtol * abs(bound) + atol, idx_err

    # 全局分值比对 (CANN 带状相对误差 + 百分比阈值)
    tv = np.concatenate(tv_all) if tv_all else np.zeros(0, np.float32)
    rv = np.concatenate(rv_all) if rv_all else np.zeros(0, np.float32)
    close = np.isclose(tv, rv, rtol=rtol, atol=atol)
    band = np.maximum(np.maximum(np.abs(tv), np.abs(rv)), (1.0 / (1 << 14)) / diff_thd) + 1e-9
    err = np.abs(tv - rv) / band
    fulfill = close.mean() * 100.0 if close.size else 100.0
    max_re = err[~close].max() if (~close).any() else 0.0
    assert fulfill >= (1 - pct_thd) * 100.0 and max_re < max_diff_hd, val_err


@pytest.mark.parametrize("B,S1,S2,N1,N2,D,sparse_count,sparse_mode,dtype", [
    (2, 4, 256, 16, 1, 128, 32, 3, ms.float16),
    (1, 8, 128, 8, 2, 64, 64, 3, ms.bfloat16),
    (3, 128, 128, 64, 1, 128, 64, 3, ms.float16),
])
def test_smoke_golden(B, S1, S2, N1, N2, D, sparse_count, sparse_mode, dtype):
    """Fast golden subset covering GQA and multi-batch scenarios."""
    test_golden(B, S1, S2, N1, N2, D, sparse_count, sparse_mode, dtype)


if __name__== "__main__":
    test_basic(B=1, S1=4096, S2=4096, N1=64, N2=1, D=128, sparse_count=2048, sparse_mode=3, dtype=ms.float16, return_value=True)
    # test_accuracy(B=1, S1=4096, S2=4096, N1=64, N2=1, D=128, sparse_count=2048, sparse_mode=3)
