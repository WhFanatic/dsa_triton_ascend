"""Triton-ascend implementation of SparseLightningIndexerGradKLLoss operator.

Replaces the MindSpore Custom op wrapper that calls
aclnnSparseLightningIndexerGradKLLoss.

Supports BSND layout with sparse_mode=3 (rightDownCausal).
"""
import triton
import triton.language as tl
import mindspore as ms
from mindspore import ops


# Helper: blocked dot product  vec[D] · mat[BLOCK_K, D] → [BLOCK_K]
@triton.jit
def _block_dot_1xD_vs_KxD(
    vec_ptr,
    vec_base,
    mat_ptr,
    mat_base,
    mat_stride_k,
    D_total,
    k_arange,
    topk_mask,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for d_start in range(0, D_total, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_valid = d_offs < D_total

        v = tl.load(vec_ptr + vec_base + d_offs,
                     mask=d_valid, other=0.0).to(tl.float32)
        m = tl.load(mat_ptr + mat_base + k_arange[:, None] * mat_stride_k + d_offs[None, :],
                     mask=topk_mask[:, None] & d_valid[None, :],
                     other=0.0).to(tl.float32)

        acc += tl.sum(v[None, :] * m, axis=1)
    return acc


# Gather kernel  (with batch offset fix)
@triton.jit
def _gather_kv_kernel(
    src_ptr,
    indices_ptr,
    dst_ptr,
    S1,
    S2,
    topK,
    D,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Gather key/keyIndex tokens at sparseIndices positions.

    Grid: (B * S1,)
    src: [B, S2, D] flat   dst: [B*S1, topK, D]
    """
    pid = tl.program_id(0)
    b = pid // S1
    batch_src_offset = b * S2 * D

    for k_start in range(0, topK, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < topK

        idx = tl.load(indices_ptr + pid * topK + k_offs, mask=k_mask, other=0)
        idx = tl.maximum(tl.minimum(idx, S2 - 1), 0)

        for d_start in range(0, D, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D

            src_offs = batch_src_offset + idx[:, None] * D + d_offs[None, :]
            vals = tl.load(src_ptr + src_offs,
                           mask=k_mask[:, None] & d_mask[None, :], other=0.0)

            dst_offs = pid * topK * D + k_offs[:, None] * D + d_offs[None, :]
            tl.store(dst_ptr + dst_offs, vals,
                     mask=k_mask[:, None] & d_mask[None, :])


# Main fused kernel
@triton.jit
def _indexer_grad_kl_loss_kernel(
    query_ptr,
    key_gathered_ptr,
    query_index_ptr,
    key_index_gathered_ptr,
    weights_ptr,
    di_ptr,
    d_query_index_ptr,
    d_weights_ptr,
    loss_ptr,
    s_idx_buf_ptr,
    B,
    S1,
    N1,
    Nidx1,
    D,
    D_idx,
    topK,
    scale_value,
    act_q_ptr,
    act_k_ptr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused kernel: scores → distributions → dW / dQueryIndex / KL loss / dI.

    Grid: (B * S1,)
    """
    pid = tl.program_id(0)
    b = pid // S1
    s1 = pid % S1

    act_q = tl.load(act_q_ptr + b)
    if s1 >= act_q:
        return

    act_k = tl.load(act_k_ptr + b)
    causal_limit = act_k - act_q + s1 + 1
    causal_limit_f = causal_limit.to(tl.float32)
    num_k = tl.minimum(topK, act_k)

    k_arange = tl.arange(0, BLOCK_K)
    topk_mask = k_arange < topK

    qi_base = pid * Nidx1 * D_idx
    ki_g_base = pid * topK * D_idx
    w_base = pid * Nidx1
    q_base = pid * N1 * D
    kg_base = pid * topK * D

    # ============================================================
    # Stage 1: I = Σ_g  W_g · ReLU(queryIndex_g @ keyIndex_gathered^T)
    #          Cache relu_g into s_idx_buf for Stage 5.
    # ============================================================
    i_acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    for g in range(Nidx1):
        w_g = tl.load(weights_ptr + w_base + g).to(tl.float32)

        full_dot = _block_dot_1xD_vs_KxD(
            query_index_ptr, qi_base + g * D_idx,
            key_index_gathered_ptr, ki_g_base, D_idx,
            D_idx, k_arange, topk_mask,
            BLOCK_K, BLOCK_D,
        )

        s_idx_g = tl.where(topk_mask, full_dot, 0.0)
        relu_g = tl.maximum(s_idx_g, 0.0)
        i_acc += relu_g * w_g

        # Cache relu_g for Stage 5
        buf_offs = pid * Nidx1 * BLOCK_K + g * BLOCK_K + k_arange
        tl.store(s_idx_buf_ptr + buf_offs, relu_g, mask=topk_mask)

    # ============================================================
    # Stage 2: p from main attention softmax (average over heads, L1 norm)
    # ============================================================
    p_sum = tl.zeros([BLOCK_K], dtype=tl.float32)

    for g in range(N1):
        full_dot = _block_dot_1xD_vs_KxD(
            query_ptr, q_base + g * D,
            key_gathered_ptr, kg_base, D,
            D, k_arange, topk_mask,
            BLOCK_K, BLOCK_D,
        )

        s_main_g = full_dot * scale_value.to(tl.float32)
        s_main_g = tl.where(k_arange < causal_limit_f, s_main_g, float('-inf'))
        s_main_g = tl.where(k_arange < num_k, s_main_g, float('-inf'))
        s_main_g = tl.where(topk_mask, s_main_g, float('-inf'))

        s_max_g = tl.max(s_main_g, axis=0)
        s_exp_g = tl.exp(s_main_g - s_max_g)
        s_sum_g = tl.sum(s_exp_g, axis=0)
        p_sum += tl.where(topk_mask, s_exp_g / (s_sum_g + 1e-12), 0.0)

    n1_recip = 1.0 / N1.to(tl.float32)
    p = p_sum * n1_recip
    p = p / (tl.sum(tl.abs(p), axis=0) + 1e-12)

    # ============================================================
    # Stage 3: softmax(I) and dI = softmax(I) - p
    # ============================================================
    i_masked = tl.where(topk_mask, i_acc, float('-inf'))
    i_max = tl.max(i_masked, axis=0)
    i_exp = tl.exp(i_masked - i_max)
    i_sum_val = tl.sum(i_exp, axis=0)
    softmax_i = tl.where(topk_mask, i_exp / (i_sum_val + 1e-12), 0.0)

    di = softmax_i - p
    tl.store(di_ptr + pid * topK + k_arange, di.to(tl.float32), mask=topk_mask)

    # ============================================================
    # Stage 4: KL loss  (atomic accumulation)
    # ============================================================
    safe_ratio = p / (softmax_i + 1e-12)
    kl_elem = p * tl.log(safe_ratio + 1e-12)
    kl_elem = tl.where(topk_mask, kl_elem, 0.0)
    tl.atomic_add(loss_ptr, tl.sum(kl_elem, axis=0))

    # ============================================================
    # Stage 5: dW, dQueryIndex  (reuse cached relu_g from Stage 1)
    # ============================================================
    for g in range(Nidx1):
        w_g = tl.load(weights_ptr + w_base + g).to(tl.float32)

        # Load cached relu_g
        buf_offs = pid * Nidx1 * BLOCK_K + g * BLOCK_K + k_arange
        relu_g = tl.load(s_idx_buf_ptr + buf_offs, mask=topk_mask, other=0.0)
        relu_mask_g = (relu_g > 0.0).to(tl.float32)

        # dW_g = Σ_k  dI_k · ReLU(S_idx_gk)
        dw_g = tl.sum(di * relu_g, axis=0)
        tl.store(d_weights_ptr + pid * Nidx1 + g,
                 dw_g.to(d_weights_ptr.dtype.element_ty))

        # dS_idx_gk = dI_k · W_g · mask_gk
        ds_idx_g = di * w_g * relu_mask_g

        # dQueryIndex_gd = Σ_k  dS_idx_gk · keyIndex_gathered_kd
        for d_start in range(0, D_idx, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_offs < D_idx

            ki_offs = ki_g_base + k_arange[:, None] * D_idx + d_offs[None, :]
            ki_tile = tl.load(key_index_gathered_ptr + ki_offs,
                              mask=topk_mask[:, None] & d_valid[None, :],
                              other=0.0).to(tl.float32)

            dqi_val = tl.sum(ds_idx_g[:, None] * ki_tile, axis=0)

            dqi_offs = pid * Nidx1 * D_idx + g * D_idx + d_offs
            tl.store(d_query_index_ptr + dqi_offs,
                     dqi_val.to(d_query_index_ptr.dtype.element_ty), mask=d_valid)


# Scatter-add dKeyIndex kernel
@triton.jit
def _scatter_dkey_index_kernel(
    query_index_ptr,
    key_index_gathered_ptr,
    weights_ptr,
    di_ptr,
    sparse_indices_ptr,
    d_key_index_ptr,
    B,
    S1,
    S2,
    Nidx1,
    D_idx,
    topK,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Scatter-add dKeyIndex.  Grid: (B * S1 * topK,)"""
    pid = tl.program_id(0)
    bs1_idx = pid // topK
    k_idx = pid % topK
    b = bs1_idx // S1

    qi_base = bs1_idx * Nidx1 * D_idx
    ki_g_base = bs1_idx * topK * D_idx
    w_base = bs1_idx * Nidx1

    di_k = tl.load(di_ptr + bs1_idx * topK + k_idx).to(tl.float32)

    target_k = tl.load(sparse_indices_ptr + bs1_idx * topK + k_idx)
    target_k = tl.maximum(tl.minimum(target_k, S2 - 1), 0)

    for g in range(Nidx1):
        w_g = tl.load(weights_ptr + w_base + g).to(tl.float32)

        # Recompute scalar dot  queryIndex[g,:] · keyIndex_gathered[k_idx,:]
        s_idx_gk = tl.zeros([1], dtype=tl.float32)
        for d_start in range(0, D_idx, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_offs < D_idx

            qi_g = tl.load(query_index_ptr + qi_base + g * D_idx + d_offs,
                           mask=d_valid, other=0.0).to(tl.float32)
            ki_gk = tl.load(key_index_gathered_ptr + ki_g_base + k_idx * D_idx + d_offs,
                            mask=d_valid, other=0.0).to(tl.float32)
            s_idx_gk += tl.sum(qi_g * ki_gk)

        relu_mask = (s_idx_gk > 0.0).to(tl.float32)
        dki_contrib = di_k * w_g * relu_mask

        for d_start in range(0, D_idx, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_offs < D_idx

            qi_g = tl.load(query_index_ptr + qi_base + g * D_idx + d_offs,
                           mask=d_valid, other=0.0).to(tl.float32)
            dki_vals = dki_contrib * qi_g
            dki_offs = b * S2 * D_idx + target_k * D_idx + d_offs

            tl.atomic_add(d_key_index_ptr + dki_offs,
                          dki_vals.to(d_key_index_ptr.dtype.element_ty),
                          mask=d_valid)


# Python helpers
def _default_actual_seq(actual_seq, batch_size, seq_len):
    if actual_seq is None:
        return ops.fill(ms.int32, (batch_size,), seq_len)
    if isinstance(actual_seq, (list, tuple)):
        return ms.Tensor(list(actual_seq), dtype=ms.int32)
    return actual_seq


# Core: shape-inference function for _ms_pyfunc
def _infer_core(query, key, query_index, key_index, weights,
                sparse_indices, actual_seq_qlen, actual_seq_klen,
                scale_value):
    """Return output placeholders with correct shapes / dtypes."""
    return (
        ms.mint.empty_like(query_index),      # d_query_index
        ms.mint.empty_like(key_index),         # d_key_index
        ms.mint.empty_like(weights),           # d_weights
        ms.mint.empty((1,), dtype=ms.float32), # loss
    )
 
 
# Core: Triton kernel orchestration (all params are concrete Tensors)
@ms.ops._ms_pyfunc(infer_func=_infer_core)
def _sparse_lightning_indexer_grad_kl_loss_core(
    query: ms.Tensor,
    key: ms.Tensor,
    query_index: ms.Tensor,
    key_index: ms.Tensor,
    weights: ms.Tensor,
    sparse_indices: ms.Tensor,
    actual_seq_qlen: ms.Tensor,
    actual_seq_klen: ms.Tensor,
    scale_value: float,
) -> tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
    """Core implementation — all arguments are resolved Tensors or plain scalars.
 
    Shapes (BSND, Nidx2=N2=1 already validated by caller):
        query:          [B, S1, N1, D]
        key:            [B, S2, 1,  D]
        query_index:    [B, S1, Nidx1, D_idx]
        key_index:      [B, S2, 1,     D_idx]
        weights:        [B, S1, Nidx1]
        sparse_indices: [B, S1, 1, topK]
        actual_seq_qlen:[B]
        actual_seq_klen:[B]
    """
    B, S1, N1, D = query.shape
    S2 = key.shape[1]
    Nidx1, D_idx = query_index.shape[2], query_index.shape[3]
    topK = sparse_indices.shape[3]
    BLOCK_K, BLOCK_D = 128, 64
 
    # Flatten (squeeze N2=Nidx2=1)
    q_flat = query.reshape(B * S1, N1, D).contiguous()
    k_flat = key.reshape(B * S2, D).contiguous()
    qi_flat = query_index.reshape(B * S1, Nidx1, D_idx).contiguous()
    ki_flat = key_index.reshape(B * S2, D_idx).contiguous()
    w_flat = weights.reshape(B * S1, Nidx1).contiguous()
    sparse_flat = sparse_indices.reshape(B * S1, topK).contiguous()
 
    # --- Allocate intermediates & outputs ---
    di = ms.mint.empty((B * S1, topK), dtype=ms.float32)
    s_idx_buf = ms.mint.empty((B * S1, Nidx1, BLOCK_K), dtype=ms.float32)
 
    d_query_index = ms.mint.zeros((B * S1, Nidx1, D_idx), dtype=query_index.dtype)
    d_key_index = ms.mint.zeros((B * S2, D_idx), dtype=key_index.dtype)
    d_weights = ms.mint.zeros((B * S1, Nidx1), dtype=weights.dtype)
    loss = ms.mint.zeros((1,), dtype=ms.float32)
 
    key_gathered = ms.mint.empty((B * S1, topK, D), dtype=key.dtype)
    key_index_gathered = ms.mint.empty((B * S1, topK, D_idx), dtype=key_index.dtype)
 
    grid_bs1 = (B * S1,)
 
    # Gather keys & key_indices at sparse positions
    _gather_kv_kernel[grid_bs1](
        k_flat, sparse_flat, key_gathered,
        S1, S2, topK, D,
        BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
    )
    _gather_kv_kernel[grid_bs1](
        ki_flat, sparse_flat, key_index_gathered,
        S1, S2, topK, D_idx,
        BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
    )
 
    # Main fused kernel
    _indexer_grad_kl_loss_kernel[grid_bs1](
        q_flat, key_gathered,
        qi_flat, key_index_gathered,
        w_flat,
        di,
        d_query_index, d_weights, loss,
        s_idx_buf,
        B, S1, N1, Nidx1, D, D_idx, topK,
        scale_value,
        actual_seq_qlen, actual_seq_klen,
        BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
    )
 
    # Scatter-add dKeyIndex
    _scatter_dkey_index_kernel[(B * S1 * topK,)](
        qi_flat, key_index_gathered,
        w_flat, di, sparse_flat,
        d_key_index,
        B, S1, S2, Nidx1, D_idx, topK,
        BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
    )
 
    # Reshape outputs back to original layout
    d_query_index = d_query_index.reshape(query_index.shape)
    d_key_index = d_key_index.reshape(key_index.shape)
    d_weights = d_weights.reshape(weights.shape)
 
    return d_query_index, d_key_index, d_weights, loss
 
 
# Public API: interface-compatible with the CANN op
def sparse_lightning_indexer_grad_kl_loss_triton(
    query: ms.Tensor,
    key: ms.Tensor,
    query_index: ms.Tensor,
    key_index: ms.Tensor,
    weights: ms.Tensor,
    sparse_indices: ms.Tensor,
    softmax_max: ms.Tensor,
    softmax_sum: ms.Tensor,
    query_rope: ms.Tensor = None,
    key_rope: ms.Tensor = None,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    scale_value: float = 1.0,
    layout: str = "BSND",
    sparse_mode: int = 3,
    pre_tokens: int = 9223372036854775807,
    next_tokens: int = 9223372036854775807,
    deterministic: bool = False,
):
    """Triton-ascend SparseLightningIndexerGradKLLoss.
 
    Drop-in replacement for the CANN aclnnSparseLightningIndexerGradKLLoss op.
    This wrapper validates constraints, resolves optional / polymorphic
    arguments, and delegates to the strictly-typed core decorated with
    ``@ms.ops._ms_pyfunc``.
 
    Args:
        query:          [B, S1, N1, D], fp16/bf16
        key:            [B, S2, N2, D], fp16/bf16   (N2 must be 1)
        query_index:    [B, S1, Nidx1, D_idx], fp16/bf16
        key_index:      [B, S2, Nidx2, D_idx], fp16/bf16  (Nidx2 must be 1)
        weights:        [B, S1, Nidx1], fp16/bf16/fp32
        sparse_indices: [B, S1, Nidx2, topK], int32  (Nidx2 must be 1)
        softmax_max:    unused in triton path
        softmax_sum:    unused in triton path
        query_rope:     unused in triton path
        key_rope:       unused in triton path
        actual_seq_qlen: [B] int32, list/tuple, or None (defaults to S1)
        actual_seq_klen: [B] int32, list/tuple, or None (defaults to S2)
        scale_value:    scaling factor for attention scores
        layout:         "BSND" only
        sparse_mode:    3 (rightDownCausal) only
        pre_tokens:     unused in triton path
        next_tokens:    unused in triton path
        deterministic:  unused in triton path
 
    Returns:
        (d_query_index, d_key_index, d_weights, loss)
    """
    # Validate constraints
    if layout != "BSND":
        raise ValueError("Only BSND layout is supported in triton path")
    if sparse_mode != 3:
        raise ValueError("Only sparse_mode=3 (rightDownCausal) is supported")
    assert sparse_indices.shape[2] == 1, "Nidx2 must be 1 (MQA constraint)"
 
    # Resolve optional / polymorphic arguments
    B, S1 = query.shape[0], query.shape[1]
    S2 = key.shape[1]
    act_q = _default_actual_seq(actual_seq_qlen, B, S1)
    act_k = _default_actual_seq(actual_seq_klen, B, S2)
 
    # Delegate to strictly-typed core
    return _sparse_lightning_indexer_grad_kl_loss_core(
        query, key, query_index, key_index, weights,
        sparse_indices, act_q, act_k, scale_value,
    )
