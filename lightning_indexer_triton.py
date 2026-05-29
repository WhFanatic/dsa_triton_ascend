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

INT64_MAX = 9223372036854775807


def _patch_triton_ascend_mindspore_dtype_bytes():
    try:
        from triton.backends.ascend.runtime import utils as ascend_utils
        from triton.backends.ascend.runtime import autotuner as ascend_autotuner
    except ImportError:
        return

    origin_func = getattr(ascend_utils, "get_byte_per_numel", None)
    if origin_func is None:
        return
    if getattr(origin_func, "_mindspore_dtype_patched", False):
        if hasattr(ascend_autotuner, "get_byte_per_numel"):
            ascend_autotuner.get_byte_per_numel = origin_func
        return

    dtype_bytes = {}

    def add_dtype(dtype_name, byte_size):
        dtype = getattr(ms, dtype_name, None)
        if dtype is not None:
            dtype_bytes[dtype] = byte_size

    for dtype_name in ("int8", "uint8", "bool_"):
        add_dtype(dtype_name, 1)
    for dtype_name in ("float16", "bfloat16", "int16", "uint16"):
        add_dtype(dtype_name, 2)
    for dtype_name in ("float32", "int32", "uint32"):
        add_dtype(dtype_name, 4)
    for dtype_name in ("float64", "int64", "uint64"):
        add_dtype(dtype_name, 8)

    def patched_get_byte_per_numel(dtype):
        try:
            if dtype in dtype_bytes:
                return dtype_bytes[dtype]
        except TypeError:
            pass
        return origin_func(dtype)

    patched_get_byte_per_numel._mindspore_dtype_patched = True
    ascend_utils.get_byte_per_numel = patched_get_byte_per_numel
    if hasattr(ascend_autotuner, "get_byte_per_numel"):
        ascend_autotuner.get_byte_per_numel = patched_get_byte_per_numel


_patch_triton_ascend_mindspore_dtype_bytes()


@triton.autotune(
    configs=[
        # BLOCK_G = 16:Cube M 维最小可用值,适合 G 较大或寄存器压力高
        triton.Config({"BLOCK_S2": 128,  "BLOCK_D": 64,  "BLOCK_G": 16}),
        triton.Config({"BLOCK_S2": 128,  "BLOCK_D": 128, "BLOCK_G": 16}),
        triton.Config({"BLOCK_S2": 256,  "BLOCK_D": 64,  "BLOCK_G": 16}),
        triton.Config({"BLOCK_S2": 256,  "BLOCK_D": 128, "BLOCK_G": 16}),
        triton.Config({"BLOCK_S2": 512,  "BLOCK_D": 64,  "BLOCK_G": 16}),

        # BLOCK_G = 32:中等 G,平衡 K 复用与 tile 大小
        triton.Config({"BLOCK_S2": 128,  "BLOCK_D": 64,  "BLOCK_G": 32}),
        triton.Config({"BLOCK_S2": 128,  "BLOCK_D": 128, "BLOCK_G": 32}),
        triton.Config({"BLOCK_S2": 256,  "BLOCK_D": 64,  "BLOCK_G": 32}),
        triton.Config({"BLOCK_S2": 256,  "BLOCK_D": 128, "BLOCK_G": 32}),
        triton.Config({"BLOCK_S2": 512,  "BLOCK_D": 64,  "BLOCK_G": 32}),

        # BLOCK_G = 64:G 较小时直接整块吃完;G 较大时单次 dot 形状最饱满
        triton.Config({"BLOCK_S2": 128,  "BLOCK_D": 64,  "BLOCK_G": 64}),
        triton.Config({"BLOCK_S2": 128,  "BLOCK_D": 128, "BLOCK_G": 64}),
        triton.Config({"BLOCK_S2": 256,  "BLOCK_D": 64,  "BLOCK_G": 64}),
        triton.Config({"BLOCK_S2": 256,  "BLOCK_D": 128, "BLOCK_G": 64}),
        triton.Config({"BLOCK_S2": 512,  "BLOCK_D": 64,  "BLOCK_G": 64}),
    ],
    key=["S2", "D", "G", "sparse_mode"],
)
@triton.jit
def _lightning_indexer_score_kernel(
    q_ptr, k_ptr, w_ptr, score_ptr, # Input/output tensors
    B, S1, S2, N1, N2, D, G,        # B: batch size, S1: query sequence length, S2: key sequence length, N1: query group size, N2: key group size, D: head dimension
    act_q_ptr, act_k_ptr,           # valid query and key sequence length
    sparse_mode: tl.constexpr,
    BLOCK_S2: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    """Compute reduced scores for lightning_indexer (BSND layout).

    Grid: (B * S1 * N2, cdiv(S2, BLOCK_S2))
    每个 program 处理一个 (b, s1, n2) 位置的一个 S2 tile。
        score[b, s1, n2, s2] = sum_{g in group}(ReLU(Q[b,s1,g,:] @ K[b,s2,n2,:]^T) * W[b,s1,g])
    where group = [n2 * G, (n2+1) * G), G = N1 // N2.
    """
    pid_bsn = tl.program_id(0)
    pid_s2  = tl.program_id(1)

    n2  = pid_bsn % N2
    tmp = pid_bsn // N2
    s1  = tmp % S1
    b   = tmp // S1

    s2_offs  = pid_s2 * BLOCK_S2 + tl.arange(0, BLOCK_S2)
    s2_valid = s2_offs < S2

    score_row_base = (b * S1 * N2 + s1 * N2 + n2) * S2

    act_q = tl.load(act_q_ptr + b) # 当前 sample 的有效 query 序列长度
    act_k = tl.load(act_k_ptr + b) # 当前 sample 的有效 key 序列长度

    # Causal limit
    if sparse_mode == 3:
        causal_limit = tl.minimum(tl.maximum(act_k - act_q + s1 + 1, 0), S2)
        visible_limit = tl.minimum(act_k, causal_limit)
    else:
        causal_limit = S2
        visible_limit = act_k

    # 早退: invalid query positions or 本 S2 tile 完全在可见范围之外
    # 注: triton-ascend kernel 中只能有一个早退出口, 否则会有 bug
    if s1 >= act_q or pid_s2 * BLOCK_S2 >= visible_limit:
        tl.store(
            score_ptr + score_row_base + s2_offs,
            tl.full([BLOCK_S2], float('-inf'), dtype=tl.float32),
            mask=s2_valid,
        )
        return

    # Base offsets: q[B, S1, N1, D], k[B, S2, N2, D], w[B, S1, N1]
    k_row_stride = N2 * D
    k_base = b * S2 * k_row_stride + n2 * D
    q_base = ((b * S1 + s1) * N1 + n2 * G) * D
    w_base = (b * S1 + s1) * N1 + n2 * G

    tile_scores = tl.zeros([BLOCK_S2], dtype=tl.float32)

    # G 维分块, 外层循环; K tile 在 D 内循环 load, 被 G 内循环复用
    for g_start in range(0, G, BLOCK_G):
        g_rel   = g_start + tl.arange(0, BLOCK_G)
        g_valid = g_rel < G
        w_g = tl.load(w_ptr + w_base + g_rel, mask=g_valid, other=0.0).to(tl.float32) # W: [BLOCK_G]

        # 累加 [BLOCK_G, BLOCK_S2] 的 QK^T(沿 D 维)
        acc = tl.zeros([BLOCK_G, BLOCK_S2], dtype=tl.float32)

        for d_start in range(0, D, BLOCK_D):
            d_offs  = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_offs < D

            # Q tile: [BLOCK_G, BLOCK_D]
            q_offs = q_base + g_rel[:, None] * D + d_offs[None, :]
            q_tile = tl.load(
                q_ptr + q_offs,
                mask=g_valid[:, None] & d_valid[None, :],
                other=0.0,
            )

            # K tile: [BLOCK_S2, BLOCK_D]
            # 注:此处每个 (g_start, d_start) 都会 load 一次 K,
            # 当 BLOCK_G == G 时只走 1 次 g 循环,K 不会重复 load;
            # 当 BLOCK_G < G 时,K 沿 g 维度重复 load 是必要代价,
            # 由 autotune 在重 load 和 tile shape 之间权衡。
            k_offs = k_base + s2_offs[:, None] * k_row_stride + d_offs[None, :]
            k_tile = tl.load(
                k_ptr + k_offs,
                mask=s2_valid[:, None] & d_valid[None, :],
                other=0.0,
            )

            # Cube MMA: [BLOCK_G, BLOCK_D] x [BLOCK_D, BLOCK_S2]
            acc += tl.dot(q_tile, tl.trans(k_tile))

        # ReLU + W 加权
        acc = tl.maximum(acc, 0.0)
        acc = tl.where(g_valid[:, None], acc, 0.0)
        tile_scores += tl.sum(acc * w_g[:, None], axis=0)

    # Causal mask (sparse_mode == 3: rightDownCausal)
    if sparse_mode == 3:
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


def _stable_topk(scores_2d, k, stable=False):
    _, s2_len = scores_2d.shape
    k = min(k, s2_len)

    if stable:
        _, sorted_indices = mint.sort(-scores_2d, dim=1, stable=True)
        topk_indices = sorted_indices[:, :k].to(ms.int32)
        topk_values = ops.gather_d(scores_2d, 1, topk_indices)
    else:
        # 直接使用 mint.topk, 性能更好, 但同分情况排序结果与 CANN 参考实现有差异
        topk_values, topk_indices = mint.topk(scores_2d, k, dim=1, largest=True, sorted=True)
        topk_indices = topk_indices.to(ms.int32)

    # -inf positions are invalid -> index -1, aligned with builtin op
    invalid = topk_values == float('-inf')
    topk_indices = mint.where(invalid, -1, topk_indices)

    return topk_indices, topk_values


def _infer_score_launch(
    q_flat: ms.Tensor,
    k_flat: ms.Tensor,
    w_flat: ms.Tensor,
    scores_flat: ms.Tensor,
    act_q: ms.Tensor,
    act_k: ms.Tensor,
    B: int,
    S1: int,
    S2: int,
    N1: int,
    N2: int,
    D: int,
    G: int,
    sparse_mode: int,
) -> ms.Tensor:
    return ms.mint.empty_like(scores_flat)


@ms.ops._ms_pyfunc(infer_func=_infer_score_launch)
def _lightning_indexer_score_core(
    q_flat: ms.Tensor,
    k_flat: ms.Tensor,
    w_flat: ms.Tensor,
    scores_flat: ms.Tensor,
    act_q: ms.Tensor,
    act_k: ms.Tensor,
    B: int,
    S1: int,
    S2: int,
    N1: int,
    N2: int,
    D: int,
    G: int,
    sparse_mode: int,
) -> ms.Tensor:
    # autotune 选 BLOCK_S2 后由 triton 自己根据 meta 算 cdiv
    def grid_fn(meta):
        return (B * S1 * N2, triton.cdiv(S2, meta["BLOCK_S2"]))

    _lightning_indexer_score_kernel[grid_fn](
        q_flat, k_flat, w_flat, scores_flat,
        B, S1, S2, N1, N2, D, G,
        act_q, act_k,
        sparse_mode=sparse_mode,
    )

    return scores_flat


class LightningIndexerTriton(ms.nn.Cell):
    """nn.Cell wrapper for lightning_indexer_triton.

    Args:
        sparse_count: top-k count
        sparse_mode: 3=default, rightDownCausal
        layout_query: "BSND" or "TND"
        layout_key: "BSND" or "TND"
        return_value: if True, return (indices, values); else values is dummy
        pre_tokens: ignored in triton path
        next_tokens: ignored in triton path
    """

    def __init__(
        self,
        sparse_count=2048,
        sparse_mode=3,
        layout_query="BSND",
        layout_key="BSND",
        return_value=False,
        pre_tokens=INT64_MAX,
        next_tokens=INT64_MAX,
    ):
        super().__init__()
        self.sparse_count = sparse_count
        self.sparse_mode = sparse_mode
        self.layout_query = layout_query
        self.layout_key = layout_key
        self.return_value = return_value
        self.pre_tokens = pre_tokens
        self.next_tokens = next_tokens

    def construct(
        self,
        query,
        key,
        weights,
        actual_seq_lengths_query=None,
        actual_seq_lengths_key=None,
        block_table=None,
    ):
        return lightning_indexer_triton(
            query, key, weights,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            layout_query=self.layout_query,
            layout_key=self.layout_key,
            sparse_count=self.sparse_count,
            sparse_mode=self.sparse_mode,
            pre_tokens=self.pre_tokens,
            next_tokens=self.next_tokens,
            return_value=self.return_value,
        )


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
    sparse_mode=3,
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
        sparse_mode: 3=default, rightDownCausal
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
        act_q = _tnd_cumsum_to_per_batch(actual_seq_lengths_query)
        act_k = _tnd_cumsum_to_per_batch(actual_seq_lengths_key)
        q_bsnd = _tnd_to_bsnd(query, act_q)
        w_bsnd = _tnd_to_bsnd(weights, act_q)
        k_bsnd = _tnd_to_bsnd(key, act_k) if layout_key == "TND" else key
    else:
        q_bsnd = query
        k_bsnd = key
        w_bsnd = weights
        B = q_bsnd.shape[0]
        act_q = _default_actual_seq_lens(actual_seq_lengths_query, B, q_bsnd.shape[1])
        act_k = _default_actual_seq_lens(actual_seq_lengths_key,   B, k_bsnd.shape[1])

    B, S1, N1, D = q_bsnd.shape
    _, S2, N2, _ = k_bsnd.shape

    if N1 % N2 != 0:
        raise ValueError(f"N1({N1}) must be divisible by N2({N2})")

    G = N1 // N2

    q_flat = q_bsnd.contiguous()
    k_flat = k_bsnd.contiguous()
    w_flat = w_bsnd.contiguous()

    scores_flat = ms.mint.empty((B * S1 * N2, S2), dtype=ms.float32)

    scores_flat = _lightning_indexer_score_core(
        q_flat, k_flat, w_flat, scores_flat,
        act_q.to('Ascend'), act_k.to('Ascend'),
        B, S1, S2, N1, N2, D, G,
        sparse_mode,
    )

    topk_indices_flat, topk_values_flat = _stable_topk(scores_flat, sparse_count)
    topk_indices = topk_indices_flat.reshape(B, S1, N2, sparse_count)
    if return_value:
        topk_values = topk_values_flat.to(dtype=q_bsnd.dtype).reshape(B, S1, N2, sparse_count)
    else:
        topk_values = ms.ops.zeros((B, S1, N2, sparse_count), dtype=q_bsnd.dtype)

    if is_tnd:
        topk_indices = _bsnd_to_tnd(topk_indices, act_q)
        topk_values = _bsnd_to_tnd(topk_values, act_q)

    return topk_indices, topk_values
