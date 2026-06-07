"""Triton-ascend SparseLightningIndexerGradKLLoss.

BSND layout, sparse_mode=3 (rightDownCausal).
Reuses softmaxMax/softmaxSum from forward pass for numerical consistency with CANN.

Stages (per (b,s1) position):
  1. I[k] = sum_g W[g] * ReLU(qi[g] @ ki[idx[k]]^T)
  2. p[k] = (1/N1) sum_h softmax(score_h)[k]  (teacher)
  3-4. softmax(I) -> KL(p || softmax(I)) loss -> dI
  5. dW, dQueryIndex, dKeyIndex from chain rule
"""
import triton
import triton.language as tl
import mindspore as ms
from mindspore import ops


# vec[D] @ mat[K, D]^T -> [K]
@triton.jit
def _block_dot_1xD_vs_KxD(
    vec_ptr, vec_base,
    mat_ptr, mat_base, mat_stride_k,
    D_total, k_arange, topk_mask,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
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


# gather kernel
@triton.jit
def _gather_kv_kernel(
    src_ptr, indices_ptr, dst_ptr,
    act_q_ptr, act_k_ptr,
    S1, S2, topK, D,
    VALID_K: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Gather sparse KV at positions specified by indices.

    Grid: (B * S1,)   src: [B, S2, D]   dst: [B*S1, topK, D]
    For each (b,s1), only gathers valid causal rows.
    """
    pid = tl.program_id(0)
    b = pid // S1
    s1 = pid % S1
    batch_src_offset = b * S2 * D
    act_q = tl.load(act_q_ptr + b)
    act_k = tl.load(act_k_ptr + b)
    s2_real = tl.minimum(topK, tl.maximum(act_k - act_q + s1 + 1, 0))
    for k_start in range(0, VALID_K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < s2_real
        idx = tl.load(indices_ptr + pid * topK + k_offs, mask=k_mask, other=0)
        idx = tl.maximum(tl.minimum(idx, S2 - 1), 0)
        for d_start in range(0, D, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D
            mask_2d = k_mask[:, None] & d_mask[None, :]
            src_offs = batch_src_offset + idx[:, None] * D + d_offs[None, :]
            vals = tl.load(src_ptr + src_offs, mask=mask_2d, other=0.0)
            dst_offs = pid * topK * D + k_offs[:, None] * D + d_offs[None, :]
            tl.store(dst_ptr + dst_offs, vals, mask=mask_2d)


@triton.jit
def _indexer_grad_kl_loss_kernel(
    query_ptr, key_gathered_ptr,
    query_rope_ptr, key_rope_gathered_ptr,
    query_index_ptr, key_index_gathered_ptr,
    weights_ptr,
    softmax_max_ptr, softmax_sum_ptr,              # from forward pass
    di_ptr,
    d_query_index_ptr, d_weights_ptr, loss_ptr,
    s_idx_buf_ptr, buf_i_ptr, buf_p_ptr,
    B, S1, N1, Nidx1, D, D_rope, D_idx, topK,
    scale_value,
    act_q_ptr, act_k_ptr,
    VALID_K: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Fused gradient + KL loss. Grid: (B*S1,), one (b,s1) per program.

    dKeyIndex computed by separate scatter kernel.
    s_idx_buf: ReLU(dot) per (g,k), reused in stage 5 & scatter kernel.
    buf_i:     I[k] weighted index-level scores.
    buf_p:     p[k] averaged teacher distribution.
    di:        softmax(I) - p, gradient signal back to I.
    """
    pid = tl.program_id(0)
    b = pid // S1
    s1 = pid % S1

    act_q = tl.load(act_q_ptr + b)
    if s1 >= act_q:
        return

    act_k = tl.load(act_k_ptr + b)
    # s2_real: number of valid key positions within causal window (s1 + right context)
    s2_real = tl.minimum(topK, tl.maximum(act_k - act_q + s1 + 1, 0))

    qi_base = pid * Nidx1 * D_idx
    ki_g_base = pid * topK * D_idx
    w_base = pid * Nidx1
    q_base = pid * N1 * D
    kg_base = pid * topK * D
    qr_base = pid * N1 * D_rope
    kr_g_base = pid * topK * D_rope
    sm_base = pid * N1
    local_k = tl.arange(0, BLOCK_K)

    # Stage 1: I[k] = sum_g W_g * ReLU(qi_g @ ki_gathered[k]^T)
    for k_start in range(0, VALID_K, BLOCK_K):
        k_offs = k_start + local_k
        k_mask = k_offs < s2_real
        i_tile = tl.zeros([BLOCK_K], dtype=tl.float32)
        for g in range(Nidx1):
            w_g = tl.load(weights_ptr + w_base + g).to(tl.float32)
            dot = _block_dot_1xD_vs_KxD(
                query_index_ptr, qi_base + g * D_idx,
                key_index_gathered_ptr, ki_g_base + k_start * D_idx, D_idx,
                D_idx, local_k, k_mask, BLOCK_K, BLOCK_D,
            )
            dot = tl.where(k_mask, dot, 0.0)
            relu = tl.maximum(dot, 0.0)
            i_tile += w_g * relu
            tl.store(s_idx_buf_ptr + pid * Nidx1 * topK + g * topK + k_offs,
                     relu, mask=k_mask)
        tl.store(buf_i_ptr + pid * topK + k_offs, i_tile, mask=k_mask)

    # Stage 2: p[k] = (1/N1) sum_h softmax(score_h)[k], using forward softmaxMax/Sum.
    for k_start in range(0, VALID_K, BLOCK_K):
        k_offs = k_start + local_k
        k_mask = k_offs < s2_real
        tl.store(buf_p_ptr + pid * topK + k_offs,
                 tl.zeros([BLOCK_K], dtype=tl.float32), mask=k_mask)

    for h in range(N1):
        sm_max_h = tl.load(softmax_max_ptr + sm_base + h).to(tl.float32)
        sm_sum_h = tl.load(softmax_sum_ptr + sm_base + h).to(tl.float32)

        for k_start in range(0, VALID_K, BLOCK_K):
            k_offs = k_start + local_k
            k_mask = k_offs < s2_real

            dot = _block_dot_1xD_vs_KxD(
                query_ptr, q_base + h * D,
                key_gathered_ptr, kg_base + k_start * D, D,
                D, local_k, k_mask, BLOCK_K, BLOCK_D,
            )
            dot += _block_dot_1xD_vs_KxD(
                query_rope_ptr, qr_base + h * D_rope,
                key_rope_gathered_ptr, kr_g_base + k_start * D_rope, D_rope,
                D_rope, local_k, k_mask, BLOCK_K, BLOCK_D,
            )
            scores = dot * scale_value
            probs = tl.exp(scores - sm_max_h) / (sm_sum_h + 1e-8)
            probs = tl.where(k_mask, probs, 0.0)

            old_p = tl.load(buf_p_ptr + pid * topK + k_offs,
                            mask=k_mask, other=0.0)
            tl.store(buf_p_ptr + pid * topK + k_offs,
                     old_p + probs, mask=k_mask)

    for k_start in range(0, VALID_K, BLOCK_K):
        k_offs = k_start + local_k
        k_mask = k_offs < s2_real
        p_tile = tl.load(buf_p_ptr + pid * topK + k_offs,
                         mask=k_mask, other=0.0) * (1.0 / N1)
        tl.store(buf_p_ptr + pid * topK + k_offs, p_tile, mask=k_mask)

    # Stage 3+4: softmax(I) -> KL(p || softmax(I)) loss, dI = softmax(I) - p
    i_max = tl.full([1], float('-inf'), dtype=tl.float32)
    for k_start in range(0, VALID_K, BLOCK_K):
        k_offs = k_start + local_k
        k_mask = k_offs < s2_real
        i_tile = tl.load(buf_i_ptr + pid * topK + k_offs,
                         mask=k_mask, other=float('-inf'))
        i_max = tl.maximum(i_max, tl.max(i_tile, axis=0))

    i_sum = tl.zeros([1], dtype=tl.float32)
    for k_start in range(0, VALID_K, BLOCK_K):
        k_offs = k_start + local_k
        k_mask = k_offs < s2_real
        i_tile = tl.load(buf_i_ptr + pid * topK + k_offs,
                         mask=k_mask, other=float('-inf'))
        exp_i = tl.exp(i_tile - i_max)
        exp_i = tl.where(k_mask, exp_i, 0.0)
        i_sum += tl.sum(exp_i, axis=0)

    kl_total = tl.zeros([1], dtype=tl.float32)
    for k_start in range(0, VALID_K, BLOCK_K):
        k_offs = k_start + local_k
        k_mask = k_offs < s2_real
        i_tile = tl.load(buf_i_ptr + pid * topK + k_offs,
                         mask=k_mask, other=float('-inf'))
        softmax_i = tl.exp(i_tile - i_max) / (i_sum + 1e-8)
        softmax_i = tl.where(k_mask, softmax_i, 0.0)
        p_tile = tl.load(buf_p_ptr + pid * topK + k_offs,
                         mask=k_mask, other=0.0)
        di_tile = softmax_i - p_tile
        tl.store(di_ptr + pid * topK + k_offs, di_tile, mask=k_mask)
        p_clamped = tl.maximum(p_tile, 1e-8)
        si_clamped = tl.maximum(softmax_i, 1e-8)
        kl = p_clamped * (tl.log(p_clamped) - tl.log(si_clamped))
        kl = tl.where(k_mask, kl, 0.0)
        kl_total += tl.sum(kl, axis=0)

    tl.atomic_add(loss_ptr, tl.sum(kl_total))

    # Stage 5: dW, dQueryIndex from chain rule
    for g in range(Nidx1):
        w_g = tl.load(weights_ptr + w_base + g).to(tl.float32)
        dw_g = tl.zeros([1], dtype=tl.float32)
        for k_start in range(0, VALID_K, BLOCK_K):
            k_offs = k_start + local_k
            k_mask = k_offs < s2_real
            relu_tile = tl.load(
                s_idx_buf_ptr + pid * Nidx1 * topK + g * topK + k_offs,
                mask=k_mask, other=0.0)
            di_tile = tl.load(di_ptr + pid * topK + k_offs,
                              mask=k_mask, other=0.0)
            relu_mask = (relu_tile > 0.0).to(tl.float32)
            dw_g += tl.sum(di_tile * relu_tile, axis=0)
            ds_idx = di_tile * w_g * relu_mask
            for d_start in range(0, D_idx, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < D_idx
                ki_tile = tl.load(
                    key_index_gathered_ptr + ki_g_base
                    + k_offs[:, None] * D_idx + d_offs[None, :],
                    mask=k_mask[:, None] & d_valid[None, :],
                    other=0.0).to(tl.float32)
                dqi_contrib = tl.sum(ds_idx[:, None] * ki_tile, axis=0)
                dqi_offs = pid * Nidx1 * D_idx + g * D_idx + d_offs
                old_dqi = tl.load(d_query_index_ptr + dqi_offs,
                                  mask=d_valid, other=0.0).to(tl.float32)
                tl.store(d_query_index_ptr + dqi_offs,
                         (old_dqi + dqi_contrib).to(
                             d_query_index_ptr.dtype.element_ty),
                         mask=d_valid)
        tl.store(d_weights_ptr + pid * Nidx1 + g,
                 tl.sum(dw_g).to(d_weights_ptr.dtype.element_ty))


# scatter-add dKeyIndex kernel
@triton.jit
def _scatter_dkey_index_kernel(
    query_index_ptr,
    weights_ptr, di_ptr, sparse_indices_ptr,
    s_idx_buf_ptr,
    d_key_index_ptr,
    act_q_ptr, act_k_ptr,
    B, S1, S2, Nidx1, D_idx, topK,
    valid_k,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Scatter-add dKeyIndex. Grid: (B*S1, cdiv(valid_k, BLOCK_K)).

    dki[b, target_k] += dI[k] * w[g] * 1_{relu>0} * qi[g]
    """
    bs1_idx = tl.program_id(0)
    k_block = tl.program_id(1)
    b = bs1_idx // S1
    s1 = bs1_idx % S1

    act_q = tl.load(act_q_ptr + b)
    act_k = tl.load(act_k_ptr + b)
    s2_real = tl.minimum(topK, tl.maximum(act_k - act_q + s1 + 1, 0))
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < tl.minimum(s2_real, valid_k)

    qi_base = bs1_idx * Nidx1 * D_idx
    w_base = bs1_idx * Nidx1
    di_k = tl.load(di_ptr + bs1_idx * topK + k_offsets,
                   mask=k_mask, other=0.0).to(tl.float32)
    target_k = tl.load(sparse_indices_ptr + bs1_idx * topK + k_offsets,
                       mask=k_mask, other=0)
    target_k = tl.maximum(tl.minimum(target_k, S2 - 1), 0)
    for g in range(Nidx1):
        w_g = tl.load(weights_ptr + w_base + g).to(tl.float32)
        relu_gk = tl.load(s_idx_buf_ptr + bs1_idx * Nidx1 * topK
                          + g * topK + k_offsets,
                          mask=k_mask, other=0.0).to(tl.float32)
        relu_mask = (relu_gk > 0.0).to(tl.float32)
        dki_contrib = di_k * w_g * relu_mask
        for d_start in range(0, D_idx, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_offs < D_idx
            qi_g = tl.load(query_index_ptr + qi_base + g * D_idx + d_offs,
                           mask=d_valid, other=0.0).to(tl.float32)
            dki_vals = dki_contrib[:, None] * qi_g[None, :]
            dki_offs = b * S2 * D_idx + target_k[:, None] * D_idx + d_offs[None, :]
            tl.atomic_add(d_key_index_ptr + dki_offs,
                          dki_vals.to(d_key_index_ptr.dtype.element_ty),
                          mask=k_mask[:, None] & d_valid[None, :])


# helpers
def _default_actual_seq(actual_seq, seq_len, ref_tensor):
    device_zeros = ops.cast(
        ops.zeros_like(ref_tensor[:, 0, 0, 0]), ms.int32)
    if actual_seq is None:
        return device_zeros + seq_len
    if isinstance(actual_seq, (list, tuple)):
        return device_zeros + ms.Tensor(actual_seq, dtype=ms.int32)
    return actual_seq + device_zeros


def _infer_core(query, key, query_rope, key_rope,
                query_index, key_index, weights,
                sparse_indices, softmax_max, softmax_sum,
                actual_seq_qlen, actual_seq_klen,
                scale_value):
    return (
        ms.mint.empty_like(query_index),
        ms.mint.empty_like(key_index),
        ms.mint.empty_like(weights),
        ms.mint.empty((1,), dtype=ms.float32),
    )


# orchestration
@ms.ops._ms_pyfunc(infer_func=_infer_core)
def _sparse_lightning_indexer_grad_kl_loss_core(
    query: ms.Tensor,
    key: ms.Tensor,
    query_rope: ms.Tensor,
    key_rope: ms.Tensor,
    query_index: ms.Tensor,
    key_index: ms.Tensor,
    weights: ms.Tensor,
    sparse_indices: ms.Tensor,
    softmax_max: ms.Tensor,
    softmax_sum: ms.Tensor,
    actual_seq_qlen: ms.Tensor,
    actual_seq_klen: ms.Tensor,
    scale_value: float,
) -> tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
    B, S1, N1, D = query.shape
    S2 = key.shape[1]
    D_rope = query_rope.shape[3]
    Nidx1, D_idx = query_index.shape[2], query_index.shape[3]
    topK = sparse_indices.shape[3]
    valid_k = min(topK, S2)

    BLOCK_K = 128
    BLOCK_D = 64
    BLOCK_K_SCATTER = 16

    # Flatten N2=1, Nidx2=1 dimensions (MQA: single KV head)
    q_flat = query.reshape(B * S1, N1, D).contiguous()
    k_flat = key.reshape(B * S2, D).contiguous()
    qr_flat = query_rope.reshape(B * S1, N1, D_rope).contiguous()
    kr_flat = key_rope.reshape(B * S2, D_rope).contiguous()
    qi_flat = query_index.reshape(B * S1, Nidx1, D_idx).contiguous()
    ki_flat = key_index.reshape(B * S2, D_idx).contiguous()
    w_flat = weights.reshape(B * S1, Nidx1).contiguous()
    sparse_flat = sparse_indices.reshape(B * S1, topK).contiguous()

    # softmaxMax/Sum: (B, N2=1, S1, N1) -> (B*S1, N1)
    sm_max_flat = softmax_max.reshape(B * S1, N1).contiguous()
    sm_sum_flat = softmax_sum.reshape(B * S1, N1).contiguous()

    # Intermediates
    di = ms.mint.empty((B * S1, topK), dtype=ms.float32)
    s_idx_buf = ms.mint.empty((B * S1, Nidx1, topK), dtype=ms.float32)
    buf_i = ms.mint.empty((B * S1, topK), dtype=ms.float32)
    buf_p = ms.mint.empty((B * S1, topK), dtype=ms.float32)

    # Outputs
    d_query_index = ms.mint.zeros((B * S1, Nidx1, D_idx), dtype=query_index.dtype)
    d_key_index = ms.mint.zeros((B * S2, D_idx), dtype=key_index.dtype)
    d_weights = ms.mint.zeros((B * S1, Nidx1), dtype=weights.dtype)
    loss = ms.mint.zeros((1,), dtype=ms.float32)

    # Gathered
    key_gathered = ms.mint.empty((B * S1, topK, D), dtype=key.dtype)
    key_index_gathered = ms.mint.empty((B * S1, topK, D_idx), dtype=key_index.dtype)
    key_rope_gathered = ms.mint.empty((B * S1, topK, D_rope), dtype=key_rope.dtype)

    # gather key / keyIndex / keyRope at sparse positions
    grid_bs1 = (B * S1,)
    _gather_kv_kernel[grid_bs1](
        k_flat, sparse_flat, key_gathered,
        actual_seq_qlen, actual_seq_klen,
        S1, S2, topK, D,
        VALID_K=valid_k, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D)
    _gather_kv_kernel[grid_bs1](
        ki_flat, sparse_flat, key_index_gathered,
        actual_seq_qlen, actual_seq_klen,
        S1, S2, topK, D_idx,
        VALID_K=valid_k, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D)
    _gather_kv_kernel[grid_bs1](
        kr_flat, sparse_flat, key_rope_gathered,
        actual_seq_qlen, actual_seq_klen,
        S1, S2, topK, D_rope,
        VALID_K=valid_k, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D)

    _indexer_grad_kl_loss_kernel[grid_bs1](
        q_flat, key_gathered,
        qr_flat, key_rope_gathered,
        qi_flat, key_index_gathered,
        w_flat,
        sm_max_flat, sm_sum_flat,
        di,
        d_query_index, d_weights, loss,
        s_idx_buf, buf_i, buf_p,
        B, S1, N1, Nidx1, D, D_rope, D_idx, topK,
        scale_value,
        actual_seq_qlen, actual_seq_klen,
        VALID_K=valid_k, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
    )

    _scatter_dkey_index_kernel[(B * S1, triton.cdiv(valid_k, BLOCK_K_SCATTER))](
        qi_flat,
        w_flat, di, sparse_flat,
        s_idx_buf,
        d_key_index,
        actual_seq_qlen, actual_seq_klen,
        B, S1, S2, Nidx1, D_idx, topK,
        valid_k,
        BLOCK_K=BLOCK_K_SCATTER, BLOCK_D=BLOCK_D,
    )

    d_query_index = d_query_index.reshape(query_index.shape)
    d_key_index = d_key_index.reshape(key_index.shape)
    d_weights = d_weights.reshape(weights.shape)
    return d_query_index, d_key_index, d_weights, loss


class SparseLightningIndexerGradKLLossTriton(ms.nn.Cell):
    """nn.Cell wrapper for sparse_lightning_indexer_grad_kl_loss_triton.

    Args:
        scale_value: scaling factor for attention scores
        layout: only "BSND" is supported
        sparse_mode: only 3 (rightDownCausal) is supported
        pre_tokens: ignored in triton path
        next_tokens: ignored in triton path
        deterministic: ignored in triton path
    """

    def __init__(
        self,
        scale_value=1.0,
        layout="BSND",
        sparse_mode=3,
        pre_tokens=9223372036854775807,
        next_tokens=9223372036854775807,
        deterministic=False,
    ):
        super().__init__()
        self.scale_value = scale_value
        self.layout = layout
        self.sparse_mode = sparse_mode
        self.pre_tokens = pre_tokens
        self.next_tokens = next_tokens
        self.deterministic = deterministic

    def construct(
        self,
        query, key,
        query_index, key_index,
        weights, sparse_indices,
        softmax_max, softmax_sum,
        query_rope=None, key_rope=None,
        actual_seq_qlen=None, actual_seq_klen=None,
    ):
        return sparse_lightning_indexer_grad_kl_loss_triton(
            query, key,
            query_index, key_index,
            weights, sparse_indices,
            softmax_max, softmax_sum,
            query_rope=query_rope, key_rope=key_rope,
            actual_seq_qlen=actual_seq_qlen, actual_seq_klen=actual_seq_klen,
            scale_value=self.scale_value,
            layout=self.layout, sparse_mode=self.sparse_mode,
            pre_tokens=self.pre_tokens, next_tokens=self.next_tokens,
            deterministic=self.deterministic,
        )


# public API
def sparse_lightning_indexer_grad_kl_loss_triton(
    query: ms.Tensor, key: ms.Tensor,
    query_index: ms.Tensor, key_index: ms.Tensor,
    weights: ms.Tensor, sparse_indices: ms.Tensor,
    softmax_max: ms.Tensor, softmax_sum: ms.Tensor,
    query_rope: ms.Tensor = None, key_rope: ms.Tensor = None,
    actual_seq_qlen=None, actual_seq_klen=None,
    scale_value: float = 1.0,
    layout: str = "BSND", sparse_mode: int = 3,
    pre_tokens: int = 9223372036854775807,
    next_tokens: int = 9223372036854775807,
    deterministic: bool = False,
):
    """Drop-in replacement for aclnnSparseLightningIndexerGradKLLoss."""
    if layout != "BSND":
        raise ValueError("Only BSND layout is supported in triton path")
    if sparse_mode != 3:
        raise ValueError("Only sparse_mode=3 (rightDownCausal) is supported")
    assert sparse_indices.shape[2] == 1, "Nidx2 must be 1 (MQA constraint)"

    B, S1, N1 = query.shape[0], query.shape[1], query.shape[2]
    S2, N2 = key.shape[1], key.shape[2]

    if query_rope is None:
        query_rope = ms.mint.zeros((B, S1, N1, 1), dtype=query.dtype)
    if key_rope is None:
        key_rope = ms.mint.zeros((B, S2, N2, 1), dtype=key.dtype)

    act_q = _default_actual_seq(actual_seq_qlen, S1, query)
    act_k = _default_actual_seq(actual_seq_klen, S2, key)

    return _sparse_lightning_indexer_grad_kl_loss_core(
        query, key, query_rope, key_rope,
        query_index, key_index, weights,
        sparse_indices, softmax_max, softmax_sum,
        act_q, act_k, scale_value,
    )
