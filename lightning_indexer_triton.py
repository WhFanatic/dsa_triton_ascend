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
"""Triton-ascend implementation of lightning_indexer operator.

Interface aligned with ops.lightning_indexer.

Supports BSND layout for both query and key. TND layout is supported via
internal BSND conversion (works in PyNative mode; for GRAPH_MODE, caller
should pre-convert to BSND).

PA_BSND layout is not supported.
"""
import triton
import triton.language as tl

import mindspore as ms
from mindspore import ops

INT64_MAX = 9223372036854775807


@triton.jit
def _lightning_indexer_score_kernel(
    q_ptr,
    k_ptr,
    w_ptr,
    score_ptr,
    B,
    S1,
    S2,
    N1,
    N2,
    D,
    act_q_ptr,
    act_k_ptr,
    sparse_mode: tl.constexpr,
    BLOCK_S2: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute reduced scores for lightning_indexer (BSND layout).

    Each program handles one (batch, s1) position:
        score[s2] = sum_g(ReLU(Q[g,:] @ K[s2,:]^T) * W[g])

    Grid: (B * S1,)
    """
    pid = tl.program_id(0)
    b = pid // S1
    s1 = pid % S1

    act_q = tl.load(act_q_ptr + b)
    if s1 >= act_q:
        for s2_start in range(0, S2, BLOCK_S2):
            offs = s2_start + tl.arange(0, BLOCK_S2)
            mask = offs < S2
            tl.store(
                score_ptr + b * S1 * S2 + s1 * S2 + offs,
                tl.full([BLOCK_S2], float('-inf'), dtype=tl.float32),
                mask=mask,
            )
        return

    act_k = tl.load(act_k_ptr + b)

    q_base = (b * S1 + s1) * N1 * D
    w_base = (b * S1 + s1) * N1
    k_base = b * S2 * N2 * D

    for s2_start in range(0, S2, BLOCK_S2):
        s2_offs = s2_start + tl.arange(0, BLOCK_S2)
        s2_valid = s2_offs < S2

        tile_scores = tl.zeros([BLOCK_S2], dtype=tl.float32)

        for d_start in range(0, D, BLOCK_D):
            d_offs_t = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_offs_t < D
            d_size = tl.minimum(BLOCK_D, D - d_start)

            k_offs = (
                k_base
                + s2_offs[:, None] * N2 * D
                + d_offs_t[None, :]
            )
            k_tile = tl.load(
                k_ptr + k_offs,
                mask=s2_valid[:, None] & d_valid[None, :],
                other=0.0,
            )

            for g in range(N1):
                q_offs = q_base + g * D + d_offs_t
                q_g = tl.load(q_ptr + q_offs, mask=d_valid, other=0.0)
                w_g = tl.load(w_ptr + w_base + g)

                q_bc = tl.reshape(q_g, [1, d_size])
                kt_bc = tl.reshape(k_tile, [BLOCK_S2, d_size])
                dot = tl.sum(q_bc * kt_bc, axis=1)

                dot = tl.maximum(dot, 0.0)
                tile_scores += dot * w_g

        if sparse_mode == 3:
            causal_limit = act_k - act_q + s1 + 1
            causal_limit = tl.maximum(causal_limit, 0)
            causal_limit = tl.minimum(causal_limit, S2)
            tile_scores = tl.where(
                s2_offs < causal_limit, tile_scores, float('-inf')
            )

        k_mask = s2_offs < act_k
        tile_scores = tl.where(k_mask, tile_scores, float('-inf'))

        out_offs = b * S1 * S2 + s1 * S2 + s2_offs
        tl.store(score_ptr + out_offs, tile_scores, mask=s2_valid)


def _default_actual_seq_lens(actual_seq_lens, batch_size, seq_len):
    """Build default actual_seq_lengths when None is provided."""
    if actual_seq_lens is not None:
        if isinstance(actual_seq_lens, (list, tuple)):
            return ms.Tensor(list(actual_seq_lens), dtype=ms.int32)
        return actual_seq_lens
    return ms.ops.fill(ms.int32, (batch_size,), seq_len)


def _tnd_cumsum_to_per_batch(cumsum):
    """Convert TND cumulative lengths [B] to per-batch lengths [B]."""
    per_batch = ms.ops.zeros_like(cumsum)
    per_batch[0] = cumsum[0]
    if cumsum.shape[0] > 1:
        per_batch[1:] = cumsum[1:] - cumsum[:-1]
    return per_batch


def _tnd_to_bsnd(tensor, act_seq_per_batch):
    """Convert [T, N, ...] to [B, max_S, N, ...].

    Uses Python-loop based conversion; works in PyNative mode.
    For GRAPH_MODE, caller should pass BSND tensors directly.
    """
    B = act_seq_per_batch.shape[0]
    lengths = [int(act_seq_per_batch[i].asnumpy().item()) for i in range(B)]
    max_seq = max(lengths) if lengths else 0

    if tensor.ndim == 2:
        T, N = tensor.shape
        out = ms.ops.zeros((B, max_seq, N), dtype=tensor.dtype)
        start = 0
        for b_idx in range(B):
            length = lengths[b_idx]
            if length > 0:
                out[b_idx, :length, :] = tensor[start:start + length, :]
                start += length
    elif tensor.ndim == 3:
        T, N, D = tensor.shape
        out = ms.ops.zeros((B, max_seq, N, D), dtype=tensor.dtype)
        start = 0
        for b_idx in range(B):
            length = lengths[b_idx]
            if length > 0:
                out[b_idx, :length, :, :] = tensor[start:start + length, :, :]
                start += length
    else:
        raise ValueError(f"Unexpected ndim: {tensor.ndim}")
    return out


def _bsnd_to_tnd(tensor, act_seq_per_batch):
    """Convert [B, S, N, ...] to [T, N, ...]."""
    B = act_seq_per_batch.shape[0]
    lengths = [int(act_seq_per_batch[i].asnumpy().item()) for i in range(B)]
    total_t = sum(lengths)

    if tensor.ndim == 3:
        N = tensor.shape[2]
        out = ms.ops.zeros((total_t, N), dtype=tensor.dtype)
        start = 0
        for b_idx in range(B):
            length = lengths[b_idx]
            if length > 0:
                out[start:start + length, :] = tensor[b_idx, :length, :]
                start += length
    elif tensor.ndim == 4:
        N = tensor.shape[2]
        K = tensor.shape[3]
        out = ms.ops.zeros((total_t, N, K), dtype=tensor.dtype)
        start = 0
        for b_idx in range(B):
            length = lengths[b_idx]
            if length > 0:
                out[start:start + length, :, :] = tensor[b_idx, :length, :, :]
                start += length
    else:
        raise ValueError(f"Unexpected ndim: {tensor.ndim}")
    return out


def _stable_topk(scores_2d, k):
    """Stable TopK: descending score, ascending index for ties.

    Uses stable sort on the full S2 dimension. For large S2, this can be
    optimized to use topk + partial sort.
    """
    _, s2_len = scores_2d.shape
    k = min(k, s2_len)

    _, sorted_indices = ops.sort(-scores_2d, axis=1, stable=True)
    topk_indices = sorted_indices[:, :k]
    topk_values = ops.gather_d(scores_2d, 1, topk_indices)
    return topk_indices, topk_values


def infer_func(
    query,
    key,
    weights,
    actual_seq_lengths_query,
    actual_seq_lengths_key,
    block_table,
    layout_query,
    layout_key,
    sparse_count,
    sparse_mode,
    pre_tokens,
    next_tokens,
    return_value,
):
    """Infer output shape and dtype for _ms_pyfunc."""
    q_shape = query.shape
    if len(q_shape) == 4:
        B, S1, N1, D = q_shape
    else:
        T1, N1, D = q_shape
        B = 1
        S1 = T1

    k_shape = key.shape
    if len(k_shape) == 4:
        N2 = k_shape[2]
    else:
        N2 = k_shape[1]

    if len(q_shape) == 4:
        out_shape = (B, S1, N2, sparse_count)
    else:
        out_shape = (T1, N2, sparse_count)

    indices = ms.mint.empty(out_shape, dtype=ms.int32)
    values = ms.mint.empty(out_shape, dtype=query.dtype)
    return indices, values


@ms.ops._ms_pyfunc(infer_func=infer_func)
def lightning_indexer_triton(
    query: ms.Tensor,
    key: ms.Tensor,
    weights: ms.Tensor,
    actual_seq_lengths_query=None,
    actual_seq_lengths_key=None,
    block_table=None,
    layout_query="BSND",
    layout_key="BSND",
    sparse_count=2048,
    sparse_mode=0,
    pre_tokens=INT64_MAX,
    next_tokens=INT64_MAX,
    return_value=False,
):
    """Triton-ascend lightning_indexer, aligned with ops.lightning_indexer.

    Args:
        query: [B,S1,N1,D] BSND or [T1,N1,D] TND, fp16/bf16
        key: [B,S2,N2,D] BSND or [T2,N2,D] TND, fp16/bf16
        weights: [B,S1,N1] BSND or [T1,N1] TND, fp16/bf16/fp32
        actual_seq_lengths_query: [B] int32 or None (defaults to full seq)
        actual_seq_lengths_key: [B] int32 or None (defaults to full seq)
        block_table: unsupported, must be None
        layout_query: "BSND" or "TND"
        layout_key: "BSND" or "TND"
        sparse_count: top-k count
        sparse_mode: 0=default, 3=rightDownCausal
        pre_tokens: ignored in triton path
        next_tokens: ignored in triton path
        return_value: if True, return (indices, values); else values is dummy

    Returns:
        (sparseIndicesOut, sparseValuesOut)
    """
    if block_table is not None:
        raise ValueError("PA_BSND / block_table not supported in triton lightning_indexer")

    is_tnd = (layout_query == "TND")

    if is_tnd:
        act_q_cumsum = actual_seq_lengths_query
        act_k_cumsum = actual_seq_lengths_key
        act_q_pb = _tnd_cumsum_to_per_batch(act_q_cumsum)
        act_k_pb = _tnd_cumsum_to_per_batch(act_k_cumsum)
        q_bsnd = _tnd_to_bsnd(query, act_q_pb)
        w_bsnd = _tnd_to_bsnd(weights, act_q_pb)
        if layout_key == "TND":
            k_bsnd = _tnd_to_bsnd(key, act_k_pb)
        else:
            k_bsnd = key
        act_q = act_q_pb
        act_k = act_k_pb
    else:
        q_bsnd = query
        k_bsnd = key
        w_bsnd = weights
        B = q_bsnd.shape[0]
        act_q = _default_actual_seq_lens(actual_seq_lengths_query, B, q_bsnd.shape[1])
        act_k = _default_actual_seq_lens(actual_seq_lengths_key, B, k_bsnd.shape[1])

    B = q_bsnd.shape[0]
    S1 = q_bsnd.shape[1]
    N1 = q_bsnd.shape[2]
    D = q_bsnd.shape[3]
    S2 = k_bsnd.shape[1]
    N2 = k_bsnd.shape[2]

    if N2 != 1:
        raise ValueError(f"lightning_indexer_triton requires N2=1 (k_head_num=1), got N2={N2}")

    q_flat = q_bsnd.reshape(B * S1, N1, D).contiguous()
    k_flat = k_bsnd.reshape(B * S2, N2, D).contiguous()
    w_flat = w_bsnd.reshape(B * S1, N1).contiguous()

    scores_flat = ms.mint.empty((B * S1, S2), dtype=ms.float32)

    BLOCK_S2 = 256
    BLOCK_D = 64
    grid = (B * S1,)

    _lightning_indexer_score_kernel[grid](
        q_flat,
        k_flat,
        w_flat,
        scores_flat,
        B,
        S1,
        S2,
        N1,
        N2,
        D,
        act_q,
        act_k,
        sparse_mode=sparse_mode,
        BLOCK_S2=BLOCK_S2,
        BLOCK_D=BLOCK_D,
    )

    topk_indices_flat, topk_values_flat = _stable_topk(scores_flat, sparse_count)

    topk_indices = topk_indices_flat.reshape(B, S1, N2, sparse_count)
    if return_value:
        topk_values = topk_values_flat.to(dtype=query.dtype).reshape(B, S1, N2, sparse_count)
    else:
        topk_values = ms.ops.zeros((B, S1, N2, sparse_count), dtype=query.dtype)

    if is_tnd:
        topk_indices = _bsnd_to_tnd(topk_indices, act_q_pb)
        topk_values = _bsnd_to_tnd(topk_values, act_q_pb)

    return topk_indices, topk_values
