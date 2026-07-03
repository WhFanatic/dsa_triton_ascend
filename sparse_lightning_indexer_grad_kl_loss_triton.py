"""Triton-ascend SparseLightningIndexerGradKLLoss.

BSND layout, sparse_mode=3 (rightDownCausal).
Reuses softmaxMax/softmaxSum from forward pass for numerical consistency with CANN.

Stages (per (b,s1) position):
  1. I[k] = sum_g W[g] * ReLU(qi[g] @ ki[idx[k]]^T)
  2. p[k] = (1/N1) sum_h softmax(score_h)[k]  (teacher)
  3-4. softmax(I) -> KL(p || softmax(I)) loss -> dI
  5. dW, dQueryIndex, dKeyIndex from chain rule
"""
import os

import triton
import triton.language as tl
import mindspore as ms
from mindspore import ops, runtime


# Per-chunk memory cap (MB) for intermediate tensors allocated inside one launch
# of the fused grad kernel.  Device HBM is ~60 GB, but a single overly large
# launch causes memory-pool allocation/free overhead to dominate wall-clock time.
_SLI_CHUNK_BUDGET_MB = float(os.getenv("SLI_CHUNK_BUDGET_MB", "350"))
_SLI_MIN_CHUNK = int(os.getenv("SLI_MIN_CHUNK", "64"))
_SLISYNC = os.getenv("SLI_SYNC", "0") == "1"
_DEBUG_DUMP = {}


def _compute_s1_chunk(B, S1, N1, Nidx1, D, D_rope, D_idx, topK,
                      budget_mb=_SLI_CHUNK_BUDGET_MB,
                      min_chunk=_SLI_MIN_CHUNK):
    """Pick an S1 chunk size so one launch's intermediates fit in the cap."""
    budget_bytes = int(budget_mb * (1 << 20))
    if budget_bytes <= 0:
        return S1
    # Approximate bytes created per (b, s1) in _sparse_lightning_indexer_grad_kl_loss_core.
    per_s1 = (
        N1 * (D + D_rope) * 2        # q_all
        + N1 * 4 * 2                 # sm_max_flat + sm_sum_flat
        + Nidx1 * D_idx * 2          # qi_flat
        + Nidx1 * 2                  # w_flat
        + topK * 4                   # sparse_flat
        + topK * D_idx * 2           # key_index_gathered
        + topK * 4 * 3               # di, buf_p, buf_i
        + Nidx1 * topK * 2           # s_idx_buf
        + topK * D_idx * 4           # dki_workspace
        + Nidx1 * D_idx * 2          # d_query_index
        + Nidx1 * 2                  # d_weights
        + 4                          # loss_parts
    ) * B
    if per_s1 <= 0:
        return S1
    chunk = budget_bytes // per_s1
    chunk = max(min_chunk, chunk)
    return min(chunk, S1)


@triton.jit
def _sli_grad_fused_kernel(
    query_all_ptr, key_all_ptr,
    key_index_ptr,
    query_index_ptr,
    weights_ptr,
    key_index_gathered_ptr,
    sparse_indices_ptr,
    softmax_max_ptr, softmax_sum_ptr,
    buf_p_ptr, buf_i_ptr,
    di_ptr, loss_ptr, s_idx_buf_ptr,
    d_query_index_ptr,
    d_weights_ptr,
    dki_workspace_ptr,
    B, S1, S2, N1, Nidx1, D, D_rope, D_idx, topK,
    scale_value, S1_OFFSET,
    act_q_ptr, act_k_ptr,
    VALID_K: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_D_IDX: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_K_A: tl.constexpr,
    BLOCK_K_B: tl.constexpr,
    BLOCK_D_G: tl.constexpr,
    BLOCK_G_G: tl.constexpr,
):
    """Single fused kernel: K1 (teacher+indexer+KL+dI) + K2 (dW+dQI) + K3 (dKI workspace).

    Grid: (B*S1,). One program per (b,s1), running all stages back-to-back in a
    single launch. Saves one kernel launch + sync vs the previous 2-kernel split,
    keeps di / buf_p / buf_i / s_idx_buf / key_index_gathered L2-hot across stages.
    """
    pid = tl.program_id(0)
    b = pid // S1
    s1 = pid % S1
    s1_global = s1 + S1_OFFSET

    act_q = tl.load(act_q_ptr + b)
    if s1_global >= act_q:
        return

    act_k = tl.load(act_k_ptr + b)
    s2_real = tl.minimum(topK, tl.maximum(act_k - act_q + s1_global + 1, 0))
    s2_bound = tl.minimum(s2_real, VALID_K)

    d_all = D + D_rope
    q_base = pid * N1 * d_all
    k_batch_base = b * S2 * d_all
    sm_base = pid * N1
    inv_n1 = 1.0 / N1
    qi_base = pid * Nidx1 * D_idx
    ki_g_base = pid * topK * D_idx
    w_base = pid * Nidx1
    ki_src_base = b * S2 * D_idx
    sidx_base = pid * Nidx1 * topK
    buf_p_base = pid * topK
    buf_i_base = pid * topK

    local_k = tl.arange(0, BLOCK_K)
    h_local = tl.arange(0, BLOCK_H)
    d_local = tl.arange(0, BLOCK_D)
    d_idx_local = tl.arange(0, BLOCK_D_IDX)
    g_local = tl.arange(0, BLOCK_G)

    # Stage T+I merged: per K-tile gather ki, compute teacher p[k] and indexer I[k].
    # ki gather has a D_idx outer loop to support D_idx > BLOCK_D_IDX.
    # idx_scores cumulated across D_idx blocks before relu; teacher
    # (query_all/key_all) is D_idx-independent and stays as-is.
    for k_start in range(0, VALID_K, BLOCK_K):
        if k_start < s2_real:
            k_offs = k_start + local_k
            k_mask_real = k_offs < s2_real

            idx = tl.load(sparse_indices_ptr + pid * topK + k_offs,
                          mask=k_mask_real, other=0)
            idx = tl.maximum(tl.minimum(idx, S2 - 1), 0)

            # Ki gather across all D_idx blocks (stored to HBM for Pass A/B).
            for d_idx_start in range(0, D_idx, BLOCK_D_IDX):
                d_idx_loc = d_idx_start + d_idx_local
                d_idx_mask = d_idx_loc < D_idx
                mask_2d = k_mask_real[:, None] & d_idx_mask[None, :]
                ki_vals = tl.load(
                    key_index_ptr + ki_src_base + idx[:, None] * D_idx + d_idx_loc[None, :],
                    mask=mask_2d, other=0.0)
                tl.store(
                    key_index_gathered_ptr + ki_g_base
                    + k_offs[:, None] * D_idx + d_idx_loc[None, :],
                    ki_vals, mask=mask_2d)

            # Teacher: compute p[k] (only when k < s2_bound; for production
            # shape s2_bound == s2_real so this branch is always taken).
            if k_start < s2_bound:
                k_mask_bound = k_offs < s2_bound
                p_tile_acc = tl.zeros([BLOCK_K], dtype=tl.float32)
                for h_start in range(0, N1, BLOCK_H):
                    h_offs = h_start + h_local
                    h_mask = h_offs < N1
                    sm_max = tl.load(softmax_max_ptr + sm_base + h_offs,
                                     mask=h_mask, other=0.0).to(tl.float32)
                    sm_sum = tl.load(softmax_sum_ptr + sm_base + h_offs,
                                     mask=h_mask, other=1.0).to(tl.float32)
                    inv_sum = 1.0 / (sm_sum + 1e-8)

                    scores = tl.zeros([BLOCK_H, BLOCK_K], dtype=tl.float32)
                    for d_start in range(0, d_all, BLOCK_D):
                        d_offs = d_start + d_local
                        d_valid = d_offs < d_all
                        q_tile = tl.load(
                            query_all_ptr + q_base
                            + h_offs[:, None] * d_all + d_offs[None, :],
                            mask=h_mask[:, None] & d_valid[None, :],
                            other=0.0)
                        k_tile = tl.load(
                            key_all_ptr + k_batch_base
                            + idx[:, None] * d_all + d_offs[None, :],
                            mask=k_mask_bound[:, None] & d_valid[None, :],
                            other=0.0)
                        scores += tl.dot(q_tile, tl.trans(k_tile))

                    probs = tl.exp(scores * scale_value - sm_max[:, None]) * inv_sum[:, None]
                    probs = tl.where(h_mask[:, None] & k_mask_bound[None, :], probs, 0.0)
                    p_tile_acc += tl.sum(probs, axis=0)

                p_tile = p_tile_acc * inv_n1
                tl.store(buf_p_ptr + buf_p_base + k_offs, p_tile, mask=k_mask_bound)

            # Indexer: compute I[k] = Σ_g W[g]·ReLU(Σ_d qi_d·ki_d^T).
            # idx_scores cumulated across D_idx blocks, then relu applied.
            i_tile = tl.zeros([BLOCK_K], dtype=tl.float32)
            for g_start in range(0, Nidx1, BLOCK_G):
                g_offs = g_start + g_local
                g_mask = g_offs < Nidx1

                idx_scores_full = tl.zeros([BLOCK_G, BLOCK_K], dtype=tl.float32)
                for d_idx_start in range(0, D_idx, BLOCK_D_IDX):
                    d_idx_loc = d_idx_start + d_idx_local
                    d_idx_mask = d_idx_loc < D_idx
                    qi_tile = tl.load(
                        query_index_ptr + qi_base
                        + g_offs[:, None] * D_idx + d_idx_loc[None, :],
                        mask=g_mask[:, None] & d_idx_mask[None, :], other=0.0)
                    ki_vals = tl.load(
                        key_index_gathered_ptr + ki_g_base
                        + k_offs[:, None] * D_idx + d_idx_loc[None, :],
                        mask=k_mask_real[:, None] & d_idx_mask[None, :], other=0.0)
                    idx_scores_full += tl.dot(qi_tile, tl.trans(ki_vals))

                relu = tl.maximum(idx_scores_full, 0.0)
                relu = tl.where(g_mask[:, None] & k_mask_real[None, :], relu, 0.0)
                w_g = tl.load(weights_ptr + w_base + g_offs,
                              mask=g_mask, other=0.0).to(tl.float32)
                i_tile += tl.sum(relu * w_g[:, None], axis=0)
                tl.store(
                    s_idx_buf_ptr + sidx_base
                    + g_offs[:, None] * topK + k_offs[None, :],
                    relu.to(s_idx_buf_ptr.dtype.element_ty),
                    mask=g_mask[:, None] & k_mask_real[None, :])
            tl.store(buf_i_ptr + buf_i_base + k_offs, i_tile, mask=k_mask_real)

    # Stage Final: load i_full / p_full (L2-hot from above), softmax + KL + dI.
    valid_k_offs = tl.arange(0, VALID_K)
    valid_k_mask = valid_k_offs < s2_real
    i_full = tl.load(buf_i_ptr + buf_i_base + valid_k_offs,
                     mask=valid_k_mask, other=float('-inf'))
    i_max = tl.max(i_full, axis=0)
    exp_i_full = tl.where(valid_k_mask, tl.exp(i_full - i_max), 0.0)
    i_sum = tl.sum(exp_i_full, axis=0)
    inv_i_sum = 1.0 / (i_sum + 1e-8)
    log_i_sum = tl.log(i_sum + 1e-8)
    softmax_i_full = exp_i_full * inv_i_sum
    p_full = tl.load(buf_p_ptr + buf_p_base + valid_k_offs,
                     mask=valid_k_mask, other=0.0)
    di_full = tl.where(valid_k_mask, softmax_i_full - p_full, 0.0)
    tl.store(di_ptr + pid * topK + valid_k_offs, di_full, mask=valid_k_mask)
    p_clamped = tl.maximum(p_full, 1e-8)
    log_softmax_i = i_full - i_max - log_i_sum
    log_si_clamped = tl.maximum(log_softmax_i, -18.420680743952367)
    kl_full = tl.where(valid_k_mask,
                       p_clamped * (tl.log(p_clamped) - log_si_clamped),
                       0.0)
    tl.store(loss_ptr + pid, tl.sum(kl_full, axis=0))

    # Pass A: g outer -> dW (D-independent) then dQI (D outer, K inner).
    # di / s_idx_buf / ki_gathered are L2-hot from Stages T/I/Final above.
    local_k_a = tl.arange(0, BLOCK_K_A)
    local_k_b = tl.arange(0, BLOCK_K_B)
    g_local_g = tl.arange(0, BLOCK_G_G)
    d_local_g = tl.arange(0, BLOCK_D_G)

    for g_start in range(0, Nidx1, BLOCK_G_G):
        g_offs = g_start + g_local_g
        g_mask = g_offs < Nidx1
        w_g = tl.load(weights_ptr + w_base + g_offs,
                      mask=g_mask, other=0.0).to(tl.float32)

        dw_acc = tl.zeros([BLOCK_G_G], dtype=tl.float32)
        for k_start in range(0, VALID_K, BLOCK_K_A):
            k_offs = k_start + local_k_a
            k_mask = k_offs < s2_bound
            relu_tile = tl.load(
                s_idx_buf_ptr + sidx_base
                + g_offs[:, None] * topK + k_offs[None, :],
                mask=g_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
            di_tile = tl.load(di_ptr + pid * topK + k_offs, mask=k_mask, other=0.0)
            dw_acc += tl.sum(di_tile[None, :] * relu_tile, axis=1)
        tl.store(d_weights_ptr + w_base + g_offs,
                 dw_acc.to(d_weights_ptr.dtype.element_ty),
                 mask=g_mask)

        for d_start in range(0, D_idx, BLOCK_D_G):
            d_offs_g = d_start + d_local_g
            d_valid_g = d_offs_g < D_idx
            dqi_acc = tl.zeros([BLOCK_G_G, BLOCK_D_G], dtype=tl.float32)
            for k_start in range(0, VALID_K, BLOCK_K_A):
                k_offs = k_start + local_k_a
                k_mask = k_offs < s2_bound

                relu_tile = tl.load(
                    s_idx_buf_ptr + sidx_base
                    + g_offs[:, None] * topK + k_offs[None, :],
                    mask=g_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
                di_tile = tl.load(di_ptr + pid * topK + k_offs, mask=k_mask, other=0.0)
                relu_mask = (relu_tile > 0.0).to(tl.float32)
                ds_idx = di_tile[None, :] * w_g[:, None] * relu_mask

                ki_tile = tl.load(
                    key_index_gathered_ptr + ki_g_base
                    + k_offs[:, None] * D_idx + d_offs_g[None, :],
                    mask=k_mask[:, None] & d_valid_g[None, :], other=0.0)

                dqi_acc += tl.dot(ds_idx.to(ki_tile.dtype), ki_tile)

            dqi_offs = qi_base + g_offs[:, None] * D_idx + d_offs_g[None, :]
            tl.store(d_query_index_ptr + dqi_offs,
                     dqi_acc.to(d_query_index_ptr.dtype.element_ty),
                     mask=g_mask[:, None] & d_valid_g[None, :])

    # Pass B: K outer -> D outer, g inner -> dKI workspace.
    # di / s_idx_buf reads land L2-hot from Pass A above.
    for k_start in range(0, VALID_K, BLOCK_K_B):
        k_offs = k_start + local_k_b
        k_mask = k_offs < s2_bound
        di_k = tl.load(di_ptr + pid * topK + k_offs, mask=k_mask, other=0.0).to(tl.float32)

        for d_start in range(0, D_idx, BLOCK_D_G):
            d_offs_g = d_start + d_local_g
            d_valid_g = d_offs_g < D_idx
            dki_acc = tl.zeros([BLOCK_K_B, BLOCK_D_G], dtype=tl.float32)

            for g_start in range(0, Nidx1, BLOCK_G_G):
                g_offs = g_start + g_local_g
                g_mask = g_offs < Nidx1

                # Issue qi_tile load first so MTE2 overlaps with vector compute below.
                qi_tile = tl.load(
                    query_index_ptr + qi_base
                    + g_offs[:, None] * D_idx + d_offs_g[None, :],
                    mask=g_mask[:, None] & d_valid_g[None, :], other=0.0)
                w_g = tl.load(weights_ptr + w_base + g_offs,
                              mask=g_mask, other=0.0).to(tl.float32)

                relu_gk = tl.load(
                    s_idx_buf_ptr + sidx_base
                    + g_offs[:, None] * topK + k_offs[None, :],
                    mask=g_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
                relu_mask = (relu_gk > 0.0).to(tl.float32)

                ds_idx_gk = di_k[None, :] * w_g[:, None] * relu_mask

                qi_tile = tl.load(
                    query_index_ptr + qi_base
                    + g_offs[:, None] * D_idx + d_offs_g[None, :],
                    mask=g_mask[:, None] & d_valid_g[None, :], other=0.0)

                ds_idx_kg = tl.trans(ds_idx_gk).to(qi_tile.dtype)
                dki_acc += tl.dot(ds_idx_kg, qi_tile)

            dki_offs = pid * topK * D_idx + k_offs[:, None] * D_idx + d_offs_g[None, :]
            tl.store(dki_workspace_ptr + dki_offs, dki_acc,
                     mask=k_mask[:, None] & d_valid_g[None, :])


@triton.jit
def _sli_grad_scatter_b_kernel(
    dki_workspace_ptr,
    sparse_indices_ptr,
    d_key_index_ptr,
    act_q_ptr, act_k_ptr,
    B, S1, S2, D_idx, topK, valid_k,
    S1_OFFSET,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Scatter dKI workspace into d_key_index via atomic_add.

    Grid: (B*S1, cdiv(valid_k, BLOCK_K), cdiv(D_idx, BLOCK_D)).
    """
    pid = tl.program_id(0)
    k_block = tl.program_id(1)
    d_block = tl.program_id(2)
    b = pid // S1
    s1 = pid % S1
    s1_global = s1 + S1_OFFSET

    act_q = tl.load(act_q_ptr + b)
    if s1_global >= act_q:
        return

    act_k = tl.load(act_k_ptr + b)
    s2_real = tl.minimum(topK, tl.maximum(act_k - act_q + s1_global + 1, 0))
    s2_bound = tl.minimum(s2_real, valid_k)

    k_start = k_block * BLOCK_K
    if k_start >= s2_bound:
        return

    k_offs = k_start + tl.arange(0, BLOCK_K)
    k_mask = k_offs < s2_bound
    target_k = tl.load(sparse_indices_ptr + pid * topK + k_offs,
                       mask=k_mask, other=0)
    target_k = tl.maximum(tl.minimum(target_k, S2 - 1), 0)

    d_start = d_block * BLOCK_D
    d_offs = d_start + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D_idx

    mask_2d = k_mask[:, None] & d_mask[None, :]
    wksp_offs = pid * topK * D_idx + k_offs[:, None] * D_idx + d_offs[None, :]
    dki_tile = tl.load(dki_workspace_ptr + wksp_offs, mask=mask_2d, other=0.0)

    dst_offs = b * S2 * D_idx + target_k[:, None] * D_idx + d_offs[None, :]
    tl.atomic_add(d_key_index_ptr + dst_offs, dki_tile, mask=mask_2d)


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
                scale_value, s1_offset):
    return (
        ms.mint.empty_like(query_index),
        ms.mint.empty(key_index.shape, dtype=ms.float32),
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
    s1_offset: int,
) -> tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
    B, S1, N1, D = query.shape
    S2 = key.shape[1]
    D_rope = query_rope.shape[3]
    Nidx1, D_idx = query_index.shape[2], query_index.shape[3]
    topK = sparse_indices.shape[3]
    valid_k = min(topK, S2)

    BLOCK_K_GATHER = 256
    BLOCK_K_MAIN = 128
    BLOCK_K_TEACHER = 64
    BLOCK_H_MAIN = 32
    BLOCK_H_TEACHER = 64
    BLOCK_G_MAIN = 64
    BLOCK_D_GATHER = 128
    BLOCK_D_MAIN = 64
    BLOCK_D_TEACHER = 128
    BLOCK_K_QUERY_WEIGHT = 64
    BLOCK_G_QUERY_WEIGHT = 64
    BLOCK_D_QUERY_WEIGHT = 128
    BLOCK_K_SCATTER = 64
    BLOCK_D_SCATTER = 128
    BLOCK_K_SCATTER_KERNEL = 256

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

    di = ms.mint.zeros((B * S1, topK), dtype=ms.float32)
    s_idx_buf = ms.mint.zeros((B * S1, Nidx1, topK), dtype=ms.float32)
    buf_i = ms.mint.zeros((B * S1, topK), dtype=ms.float32)
    buf_p = ms.mint.zeros((B * S1, topK), dtype=ms.float32)

    # Workspace for Pass B: per-(b,s1,k) dKI before cross-program reduction.
    # Only valid (b,s1,k) entries are stored and later read; use empty to
    # avoid paying for zero-fill of this large scratch buffer.
    dki_workspace = ms.mint.empty((B * S1, topK, D_idx), dtype=ms.float32)

    # Outputs
    d_query_index = ms.mint.zeros((B * S1, Nidx1, D_idx), dtype=query_index.dtype)
    d_key_index_acc = ms.mint.zeros((B * S2, D_idx), dtype=ms.float32)
    d_weights = ms.mint.zeros((B * S1, Nidx1), dtype=weights.dtype)
    loss_parts = ms.mint.zeros((B * S1,), dtype=ms.float32)

    # Gathered (the fused teacher kernel below writes key_index_gathered as a
    # side-effect, eliminating the standalone _gather_kv_kernel)
    key_index_gathered = ms.mint.zeros((B * S1, topK, D_idx), dtype=key_index.dtype)

    # Concatenate query+query_rope and key+key_rope along the feature dimension
    # so Stage T computes the full attention score with a single dot loop.
    q_all = ms.mint.cat([q_flat, qr_flat], dim=-1).contiguous()
    k_all = ms.mint.cat([k_flat, kr_flat], dim=-1).contiguous()

    grid_bs1 = (B * S1,)
    if _SLISYNC:
        runtime.synchronize()

    _sli_grad_fused_kernel[grid_bs1](
        q_all, k_all,
        ki_flat,
        qi_flat,
        w_flat,
        key_index_gathered,
        sparse_flat,
        sm_max_flat, sm_sum_flat,
        buf_p, buf_i,
        di, loss_parts, s_idx_buf,
        d_query_index,
        d_weights,
        dki_workspace,
        B, S1, S2, N1, Nidx1, D, D_rope, D_idx, topK,
        scale_value, s1_offset,
        actual_seq_qlen, actual_seq_klen,
        VALID_K=valid_k, BLOCK_K=BLOCK_K_TEACHER, BLOCK_D=BLOCK_D_TEACHER,
        BLOCK_D_IDX=BLOCK_D_GATHER,
        BLOCK_H=BLOCK_H_TEACHER,
        BLOCK_G=BLOCK_G_MAIN,
        BLOCK_K_A=BLOCK_K_QUERY_WEIGHT,
        BLOCK_K_B=BLOCK_K_SCATTER,
        BLOCK_D_G=BLOCK_D_QUERY_WEIGHT,
        BLOCK_G_G=BLOCK_G_QUERY_WEIGHT,
    )

    _sli_grad_scatter_b_kernel[
        (B * S1, triton.cdiv(valid_k, BLOCK_K_SCATTER_KERNEL),
         triton.cdiv(D_idx, BLOCK_D_SCATTER))
    ](
        dki_workspace,
        sparse_flat,
        d_key_index_acc,
        actual_seq_qlen, actual_seq_klen,
        B, S1, S2, D_idx, topK, valid_k,
        s1_offset,
        BLOCK_K=BLOCK_K_SCATTER_KERNEL, BLOCK_D=BLOCK_D_SCATTER,
    )

    if os.getenv("SLI_DUMP"):
        _DEBUG_DUMP.setdefault("act", []).append(
            (actual_seq_qlen.asnumpy().copy(),
             actual_seq_klen.asnumpy().copy()))
        _DEBUG_DUMP.setdefault("chunks", []).append(
            (s_idx_buf, buf_i, buf_p, di, loss_parts, key_index_gathered))
    d_query_index = d_query_index.reshape(query_index.shape)
    d_key_index_acc = d_key_index_acc.reshape(key_index.shape)
    d_weights = d_weights.reshape(weights.shape)
    loss = ops.sum(loss_parts).reshape((1,))
    return d_query_index, d_key_index_acc, d_weights, loss


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
    D = query.shape[3]
    Nidx1 = query_index.shape[2]
    D_idx = query_index.shape[3]
    topK = sparse_indices.shape[3]

    if query_rope is None:
        query_rope = ms.mint.zeros((B, S1, N1, 1), dtype=query.dtype)
    if key_rope is None:
        key_rope = ms.mint.zeros((B, S2, N2, 1), dtype=key.dtype)
    D_rope = query_rope.shape[3]

    act_q = _default_actual_seq(actual_seq_qlen, S1, query)
    act_k = _default_actual_seq(actual_seq_klen, S2, key)

    s1_chunk = _compute_s1_chunk(B, S1, N1, Nidx1, D, D_rope, D_idx, topK)

    if S1 <= s1_chunk:
        d_query_index, d_key_index_acc, d_weights, loss = (
            _sparse_lightning_indexer_grad_kl_loss_core(
                query, key, query_rope, key_rope,
                query_index, key_index, weights,
                sparse_indices, softmax_max, softmax_sum,
                act_q, act_k, scale_value, 0,
            )
        )
        return (
            d_query_index,
            ops.cast(d_key_index_acc, key_index.dtype),
            d_weights,
            loss,
        )

    d_query_chunks = []
    d_weight_chunks = []
    d_key_index_acc = None
    loss_total = None

    for start in range(0, S1, s1_chunk):
        end = min(start + s1_chunk, S1)
        d_query_chunk, d_key_chunk, d_weight_chunk, loss_chunk = (
            _sparse_lightning_indexer_grad_kl_loss_core(
                query[:, start:end, :, :],
                key,
                query_rope[:, start:end, :, :],
                key_rope,
                query_index[:, start:end, :, :],
                key_index,
                weights[:, start:end, :],
                sparse_indices[:, start:end, :, :],
                softmax_max[:, :, start:end, :],
                softmax_sum[:, :, start:end, :],
                act_q, act_k, scale_value, start,
            )
        )
        d_query_chunks.append(d_query_chunk)
        d_weight_chunks.append(d_weight_chunk)
        d_key_index_acc = (
            d_key_chunk if d_key_index_acc is None
            else d_key_index_acc + d_key_chunk
        )
        loss_total = loss_chunk if loss_total is None else loss_total + loss_chunk

    d_query_index = ops.concat(tuple(d_query_chunks), axis=1)
    d_weights = ops.concat(tuple(d_weight_chunks), axis=1)
    return (
        d_query_index,
        ops.cast(d_key_index_acc, key_index.dtype),
        d_weights,
        loss_total,
    )