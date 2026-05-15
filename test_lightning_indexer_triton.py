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
"""Unit tests for triton-ascend lightning_indexer operator.

Run with:
    pytest tests/ut/test_lightning_indexer_triton.py -v

Requires Ascend device with triton-ascend and MindSpore.
"""
import math
import pytest
import numpy as np


def _cpu_golden_lightning_indexer(
    query, key, weights,
    actual_seq_qlen, actual_seq_klen,
    sparse_count, sparse_mode,
):
    """CPU reference implementation matching ops.lightning_indexer semantics.

    Args:
        query: [B, S1, N1, D] float32
        key: [B, S2, 1, D] float32
        weights: [B, S1, N1] float32
        actual_seq_qlen: [B] int32, per-batch query lengths
        actual_seq_klen: [B] float32/int32, per-batch key lengths
        sparse_count: int, top-k
        sparse_mode: 0=default, 3=rightDownCausal

    Returns:
        topk_indices: [B, S1, 1, sparse_count] int32
        topk_values: [B, S1, 1, sparse_count] float32
    """
    B, S1, N1, D = query.shape
    S2 = key.shape[1]

    out_indices = np.full((B, S1, 1, sparse_count), -1, dtype=np.int32)
    out_values = np.full((B, S1, 1, sparse_count), -np.inf, dtype=np.float32)

    for b in range(B):
        act_q = int(actual_seq_qlen[b])
        act_k = int(math.floor(actual_seq_klen[b]))

        if act_q == 0 or act_k == 0:
            continue

        k_b = key[b, :act_k, 0, :].astype(np.float32)
        q_b = query[b, :act_q, :, :].astype(np.float32)
        w_b = weights[b, :act_q, :].astype(np.float32)

        for s1_idx in range(act_q):
            q_s1 = q_b[s1_idx, :, :]
            w_s1 = w_b[s1_idx, :]

            # Q[s1, N1, D] @ K[S2, D]^T -> [N1, S2]
            scores = np.dot(q_s1, k_b.T)
            scores = np.maximum(scores, 0.0)

            # Weight multiply and reduce: [N1, S2] -> [S2]
            reduced = np.sum(scores * w_s1[:, np.newaxis], axis=0)

            if sparse_mode == 3:
                causal_limit = act_k - act_q + s1_idx + 1
                if causal_limit < act_k:
                    reduced[causal_limit:] = -np.inf

            # Stable TopK: sort by (-score, index) for descending score, ascending index for ties
            n = len(reduced)
            sort_keys = list(zip(-reduced, range(n)))
            sort_keys.sort(key=lambda x: (x[0], x[1]))
            sorted_indices = [idx for _, idx in sort_keys]

            k = min(sparse_count, n)
            out_indices[b, s1_idx, 0, :k] = sorted_indices[:k]
            out_values[b, s1_idx, 0, :k] = reduced[sorted_indices[:k]]

    return out_indices, out_values


def _check_topk_equivalence(cpu_indices, triton_indices, cpu_values, triton_values, atol=1e-3):
    """Verify triton output matches CPU golden.

    Checks that for each (b, s1, n2) position, the triton topk results are
    equivalent to the CPU results. Since equal scores may be ordered differently,
    we check that the set of (value, index) pairs matches.
    """
    B, S1, N2, K = cpu_indices.shape
    mismatches = []
    for b in range(B):
        for s1 in range(S1):
            for n2 in range(N2):
                cpu_idx = cpu_indices[b, s1, n2, :]
                cpu_val = cpu_values[b, s1, n2, :]
                tri_idx = triton_indices[b, s1, n2, :]
                tri_val = triton_values[b, s1, n2, :]

                # Filter valid entries (index != -1)
                cpu_valid = cpu_idx >= 0
                tri_valid = tri_idx >= 0

                cpu_pairs = set()
                for i in range(K):
                    if cpu_valid[i]:
                        cpu_pairs.add((int(cpu_idx[i]), float(cpu_val[i])))

                tri_pairs = set()
                for i in range(K):
                    if tri_valid[i]:
                        tri_pairs.add((int(tri_idx[i]), float(tri_val[i])))

                if cpu_pairs != tri_pairs:
                    mismatches.append((b, s1, n2, cpu_pairs - tri_pairs, tri_pairs - cpu_pairs))
    return mismatches


class TestLightningIndexerTriton:
    """Test suite for triton-ascend lightning_indexer."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Skip tests if Ascend device not available."""
        try:
            import mindspore as ms
            ms.set_context(mode=ms.PYNATIVE_MODE)
        except ImportError:
            pytest.skip("MindSpore not available")

    @pytest.mark.parametrize("dtype_str", ["float16", "bfloat16"])
    @pytest.mark.parametrize("B,S1,S2,N1,D", [
        (1, 4, 128, 8, 128),
        (2, 8, 256, 16, 128),
        (1, 16, 512, 32, 128),
    ])
    @pytest.mark.parametrize("sparse_count", [8, 32])
    @pytest.mark.parametrize("sparse_mode", [0, 3])
    @pytest.mark.parametrize("return_value", [True, False])
    def test_bsnd_correctness(self, dtype_str, B, S1, S2, N1, D,
                               sparse_count, sparse_mode, return_value):
        """Test BSND layout correctness against CPU golden."""
        import mindspore as ms
        import numpy as np

        if dtype_str == "float16":
            ms_dtype = ms.float16
            np_dtype = np.float16
        else:
            ms_dtype = ms.bfloat16
            np_dtype = np.float16

        N2 = 1

        rng = np.random.RandomState(42)
        q_np = rng.randn(B, S1, N1, D).astype(np.float32)
        k_np = rng.randn(B, S2, N2, D).astype(np.float32)
        w_np = rng.randn(B, S1, N1).astype(np.float32)

        act_q = np.full((B,), S1, dtype=np.int32)
        act_k = np.full((B,), S2, dtype=np.int32)

        cpu_indices, cpu_values = _cpu_golden_lightning_indexer(
            q_np, k_np, w_np, act_q, act_k,
            sparse_count, sparse_mode,
        )

        try:
            from mindformers.parallel_core.training_graph.ops.lightning_indexer_triton import (
                lightning_indexer_triton,
            )

            q_ms = ms.Tensor(q_np.astype(np_dtype), dtype=ms_dtype)
            k_ms = ms.Tensor(k_np.astype(np_dtype), dtype=ms_dtype)
            w_ms = ms.Tensor(w_np.astype(np_dtype), dtype=ms_dtype)
            act_q_ms = ms.Tensor(act_q, dtype=ms.int32)
            act_k_ms = ms.Tensor(act_k, dtype=ms.int32)

            tri_indices, tri_values = lightning_indexer_triton(
                q_ms, k_ms, w_ms,
                actual_seq_lengths_query=act_q_ms,
                actual_seq_lengths_key=act_k_ms,
                layout_query="BSND",
                layout_key="BSND",
                sparse_count=sparse_count,
                sparse_mode=sparse_mode,
                return_value=return_value,
            )

            tri_indices_np = tri_indices.asnumpy()
            tri_values_np = tri_values.asnumpy() if return_value else cpu_values

            mismatches = _check_topk_equivalence(
                cpu_indices, tri_indices_np,
                cpu_values, tri_values_np.astype(np.float32),
            )
            assert len(mismatches) == 0, f"Mismatches: {mismatches[:5]}"

        except ImportError:
            pytest.skip("triton lightning_indexer not available")

    @pytest.mark.parametrize("B,S1,S2,N1,D", [(2, 8, 200, 16, 128)])
    @pytest.mark.parametrize("sparse_count", [16])
    def test_variable_seq_lens(self, B, S1, S2, N1, D, sparse_count):
        """Test with variable actual_seq_lengths."""
        import mindspore as ms
        import numpy as np
        from mindformers.parallel_core.training_graph.ops.lightning_indexer_triton import (
            lightning_indexer_triton,
        )

        rng = np.random.RandomState(123)
        q_np = rng.randn(B, S1, N1, D).astype(np.float32)
        k_np = rng.randn(B, S2, 1, D).astype(np.float32)
        w_np = rng.randn(B, S1, N1).astype(np.float32)

        act_q = np.array([S1, S1 // 2], dtype=np.int32)
        act_k = np.array([S2, S2 // 2], dtype=np.int32)

        cpu_indices, cpu_values = _cpu_golden_lightning_indexer(
            q_np, k_np, w_np, act_q, act_k,
            sparse_count, 0,
        )

        q_ms = ms.Tensor(q_np.astype(np.float16), dtype=ms.float16)
        k_ms = ms.Tensor(k_np.astype(np.float16), dtype=ms.float16)
        w_ms = ms.Tensor(w_np.astype(np.float16), dtype=ms.float16)
        act_q_ms = ms.Tensor(act_q, dtype=ms.int32)
        act_k_ms = ms.Tensor(act_k, dtype=ms.int32)

        tri_indices, tri_values = lightning_indexer_triton(
            q_ms, k_ms, w_ms,
            actual_seq_lengths_query=act_q_ms,
            actual_seq_lengths_key=act_k_ms,
            layout_query="BSND",
            layout_key="BSND",
            sparse_count=sparse_count,
            return_value=True,
        )

        tri_indices_np = tri_indices.asnumpy()
        tri_values_np = tri_values.asnumpy()

        mismatches = _check_topk_equivalence(
            cpu_indices, tri_indices_np,
            cpu_values, tri_values_np.astype(np.float32),
        )
        assert len(mismatches) == 0, f"Variable seq len mismatches: {mismatches[:5]}"

    def test_n2_validation(self):
        """Test that N2 > 1 raises ValueError."""
        import mindspore as ms
        import numpy as np
        from mindformers.parallel_core.training_graph.ops.lightning_indexer_triton import (
            lightning_indexer_triton,
        )

        q_ms = ms.Tensor(np.random.randn(1, 4, 8, 128).astype(np.float16), dtype=ms.float16)
        k_ms = ms.Tensor(np.random.randn(1, 128, 2, 128).astype(np.float16), dtype=ms.float16)
        w_ms = ms.Tensor(np.random.randn(1, 4, 8).astype(np.float16), dtype=ms.float16)

        with pytest.raises(ValueError, match="N2=1"):
            lightning_indexer_triton(q_ms, k_ms, w_ms)

    def test_interface_compatibility(self):
        """Test that interface matches ops.lightning_indexer parameter names."""
        from mindformers.parallel_core.training_graph.ops.lightning_indexer_triton import (
            lightning_indexer_triton,
        )
        import inspect

        sig = inspect.signature(lightning_indexer_triton)
        params = list(sig.parameters.keys())
        expected = [
            'query', 'key', 'weights',
            'actual_seq_lengths_query', 'actual_seq_lengths_key',
            'block_table', 'layout_query', 'layout_key',
            'sparse_count', 'sparse_mode', 'pre_tokens', 'next_tokens',
            'return_value',
        ]
        for p in expected:
            assert p in params, f"Missing parameter: {p}"
