"""Triton-ascend implementation of lightning_indexer operator.

Interface aligned with ops.lightning_indexer.

Supports BSND layout for both query and key. TND layout is supported via
internal BSND conversion (works in PyNative mode; for GRAPH_MODE, caller
should pre-convert to BSND).

PA_BSND layout is not supported.
"""
import triton
import triton.language as tl
import triton.backends.ascend.runtime

import mindspore as ms
from mindspore import ops, mint
from typing import Tuple

INT64_MAX = 9223372036854775807


# # TODO: autotune 暂不支持 mindspore
# @triton.autotune(
#     configs=[], # Ascend backend 自动生成候选配置
#     key=["S2", "D", "G"],
# )
@triton.jit
def _lightning_indexer_score_kernel(
    q_ptr, k_ptr, w_ptr, score_ptr, # Input/output tensors
    B, S1, S2, N1, N2, D, G,        # B: batch size, S1: query sequence length, S2: key sequence length, N1: query group size, N2: key group size, D: head dimension
    act_q_ptr, act_k_ptr,           # valid query and key sequence length
    sparse_mode: tl.constexpr,
    BLOCK_S2: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute reduced scores for lightning_indexer (BSND layout).

    Grid: (B * S1 * N2,)
    Each program handles one (batch, s1, n2) position:
        score[b, s1, n2, s2] = sum_{g in group}(ReLU(Q[b,s1,g,:] @ K[b,s2,n2,:]^T) * W[b,s1,g])
    where group = [n2 * G, (n2+1) * G), G = N1 // N2.
    """
    pid = tl.program_id(0)
    n2 = pid % N2
    tmp = pid // N2
    s1 = tmp % S1
    b = tmp // S1
    score_row_base = (b * S1 * N2 + s1 * N2 + n2) * S2

    act_q = tl.load(act_q_ptr + b) # 当前 sample 的有效 query 序列长度

    # Mask out invalid query positions
    if s1 >= act_q:
        for s2_start in range(0, S2, BLOCK_S2):
            offs = s2_start + tl.arange(0, BLOCK_S2)
            tl.store(
                score_ptr + score_row_base + offs,
                tl.full([BLOCK_S2], float('-inf'), dtype=tl.float32),
                mask=offs < S2,
            )
        return

    act_k = tl.load(act_k_ptr + b) # 当前 sample 的有效 key 序列长度

    # Base offsets: q[B, S1, N1, D], k[B, S2, N2, D], w[B, S1, N1]
    q_base = ((b * S1 + s1) * N1 + n2 * G) * D   # start of this n2's query group
    w_base = (b * S1 + s1) * N1 + n2 * G
    k_base = (b * S2 * N2 + n2) * D               # key head n2, stride N2*D between s2 positions

    # 对长度为 S2 的 key 序列分块处理
    for s2_start in range(0, S2, BLOCK_S2):
        s2_offs = s2_start + tl.arange(0, BLOCK_S2)
        s2_valid = s2_offs < S2

        tile_scores = tl.zeros([BLOCK_S2], dtype=tl.float32)

        for g in range(G):
            w_g = tl.load(w_ptr + w_base + g) # 当前 query head 的标量权重

            full_dot = tl.zeros([BLOCK_S2], dtype=tl.float32)
            for d_start in range(0, D, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < D

                # k layout: [B, S2, N2, D] → offset = b*S2*N2*D + s2*N2*D + n2*D + d
                k_offs = k_base + s2_offs[:, None] * (N2 * D) + d_offs[None, :]
                k_tile = tl.load(
                    k_ptr + k_offs,
                    mask=s2_valid[:, None] & d_valid[None, :],
                    other=0.0,
                )

                q_offs = q_base + g * D + d_offs
                q_g = tl.load(q_ptr + q_offs, mask=d_valid, other=0.0)

                q_bc = tl.reshape(q_g, [BLOCK_D, 1])
                kt_bc = tl.reshape(k_tile, [BLOCK_S2, BLOCK_D])
                full_dot += tl.reshape(tl.dot(kt_bc, q_bc), [BLOCK_S2])

            full_dot = tl.maximum(full_dot, 0.0)
            tile_scores += full_dot * w_g

        # Causal mask (sparse_mode == 3: rightDownCausal)
        if sparse_mode == 3:
            causal_limit = tl.minimum(tl.maximum(act_k - act_q + s1 + 1, 0), S2)
            tile_scores = tl.where(s2_offs < causal_limit, tile_scores, float('-inf'))

        tile_scores = tl.where(s2_offs < act_k, tile_scores, float('-inf'))

        tl.store(score_ptr + score_row_base + s2_offs, tile_scores, mask=s2_valid)


def _default_actual_seq_lens(actual_seq_lens, batch_size, seq_len):
    return ms.ops.fill(ms.int32, (batch_size,), seq_len) if actual_seq_lens is None else \
           ms.Tensor(list(actual_seq_lens), dtype=ms.int32) if isinstance(actual_seq_lens, (list, tuple)) else \
           actual_seq_lens


def _tnd_cumsum_to_per_batch(cumsum):
    return cumsum - ops.pad(cumsum[:-1], (1, 0))


def _tnd_to_bsnd(tensor, act_seq_per_batch):
    """Convert [T, N, ...] to [B, max_S, N, ...].

    Uses Python-loop based conversion; works in PyNative mode.
    For GRAPH_MODE, caller should pass BSND tensors directly.
    """
    assert ms.get_context('mode') == ms.PYNATIVE_MODE, "Only PyNative mode is supported."
    B = act_seq_per_batch.shape[0]
    lengths = [int(act_seq_per_batch[i].asnumpy().item()) for i in range(B)]
    max_seq = max(lengths) if lengths else 0

    assert tensor.ndim in (2, 3), f"Unexpected ndim: {tensor.ndim}"

    out = ms.ops.zeros((B, max_seq, *tensor.shape[1:]), dtype=tensor.dtype)
    start = 0
    for b_idx in range(B):
        length = lengths[b_idx]
        if length > 0:
            out[b_idx, :length] = tensor[start:start + length]
            start += length

    return out


def _bsnd_to_tnd(tensor, act_seq_per_batch):
    """Convert [B, S, N, ...] to [T, N, ...]."""
    assert ms.get_context('mode') == ms.PYNATIVE_MODE, "Only PyNative mode is supported."
    B = act_seq_per_batch.shape[0]
    lengths = [int(act_seq_per_batch[i].asnumpy().item()) for i in range(B)]
    total_t = sum(lengths)

    assert tensor.ndim in (3, 4), f"Unexpected ndim: {tensor.ndim}"

    out = ms.ops.zeros((total_t, *tensor.shape[2:]), dtype=tensor.dtype)
    start = 0
    for b_idx in range(B):
        length = lengths[b_idx]
        if length > 0:
            out[start:start + length] = tensor[b_idx, :length]
            start += length

    return out


def _stable_topk(scores_2d, k):
    """Stable TopK: descending score, ascending index for ties.

    Uses stable sort on the full S2 dimension. For large S2, this can be
    optimized to use topk + partial sort.
    """
    _, s2_len = scores_2d.shape
    k = min(k, s2_len)

    _, sorted_indices = mint.sort(-scores_2d, dim=1, stable=True)
    topk_indices = sorted_indices[:, :k].to(ms.int32)
    topk_values = ops.gather_d(scores_2d, 1, topk_indices)
    return topk_indices, topk_values


def _infer_core(
    q_bsnd: ms.Tensor,
    k_bsnd: ms.Tensor,
    w_bsnd: ms.Tensor,
    act_q: ms.Tensor,
    act_k: ms.Tensor,
    sparse_count: int,
    sparse_mode: int,
    return_value: bool,
) -> Tuple[ms.Tensor, ms.Tensor]:
    """Infer output shape and dtype for _ms_pyfunc."""
    out_shape = (*q_bsnd.shape[:-2], k_bsnd.shape[-2], sparse_count)
    indices = ms.mint.empty(out_shape, dtype=ms.int32)
    values = ms.mint.empty(out_shape, dtype=q_bsnd.dtype)
    return indices, values


@ms.ops._ms_pyfunc(infer_func=_infer_core)
def _lightning_indexer_core(
    q_bsnd: ms.Tensor,
    k_bsnd: ms.Tensor,
    w_bsnd: ms.Tensor,
    act_q: ms.Tensor,
    act_k: ms.Tensor,
    sparse_count: int,
    sparse_mode: int,
    return_value: bool,
) -> Tuple[ms.Tensor, ms.Tensor]:

    B, S1, N1, D = q_bsnd.shape
    _, S2, N2, _ = k_bsnd.shape

    if N1 % N2 != 0:
        raise ValueError(f"N1({N1}) must be divisible by N2({N2})")

    G = N1 // N2  # GQA group size: number of query heads per key head

    q_flat = q_bsnd.contiguous()
    k_flat = k_bsnd.contiguous()
    w_flat = w_bsnd.contiguous()

    # scores: [B * S1 * N2, S2] — each (b, s1, n2) produces one row
    scores_flat = ms.mint.empty((B * S1 * N2, S2), dtype=ms.float32)

    _lightning_indexer_score_kernel[(B * S1 * N2,)](
        q_flat, k_flat, w_flat, scores_flat,
        B, S1, S2, N1, N2, D, G,
        act_q, act_k,
        sparse_mode=sparse_mode,
        BLOCK_S2=128, # TODO: autotune 情况下无需传入
        BLOCK_D=64,   # TODO: autotune 情况下无需传入
    )

    topk_indices_flat, topk_values_flat = _stable_topk(scores_flat, sparse_count)

    topk_indices = topk_indices_flat.reshape(B, S1, N2, sparse_count)
    if return_value:
        topk_values = topk_values_flat.to(dtype=q_bsnd.dtype).reshape(B, S1, N2, sparse_count)
    else:
        topk_values = ms.ops.zeros((B, S1, N2, sparse_count), dtype=q_bsnd.dtype)

    return topk_indices, topk_values


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
        act_q_pb = _tnd_cumsum_to_per_batch(actual_seq_lengths_query)
        act_k_pb = _tnd_cumsum_to_per_batch(actual_seq_lengths_key)
        q_bsnd = _tnd_to_bsnd(query, act_q_pb)
        w_bsnd = _tnd_to_bsnd(weights, act_q_pb)
        k_bsnd = _tnd_to_bsnd(key, act_k_pb) if layout_key == "TND" else key
        act_q = act_q_pb
        act_k = act_k_pb
    else:
        q_bsnd = query
        k_bsnd = key
        w_bsnd = weights
        B = q_bsnd.shape[0]
        act_q = _default_actual_seq_lens(actual_seq_lengths_query, B, q_bsnd.shape[1])
        act_k = _default_actual_seq_lens(actual_seq_lengths_key, B, k_bsnd.shape[1])

    topk_indices, topk_values = _lightning_indexer_core(
        q_bsnd, k_bsnd, w_bsnd, act_q, act_k,
        sparse_count, sparse_mode, return_value,
    )

    if is_tnd:
        topk_indices = _bsnd_to_tnd(topk_indices, act_q_pb)
        topk_values = _bsnd_to_tnd(topk_values, act_q_pb)

    return topk_indices, topk_values
