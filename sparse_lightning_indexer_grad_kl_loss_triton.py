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


SPARSE_GRAD_S1_CHUNK = 512
_SLISYNC = os.getenv("SLI_SYNC", "0") == "1"
_DEBUG_DUMP = {}


# gather kernel
@triton.jit
def _gather_kv_kernel(
    src_idx_ptr, indices_ptr, dst_idx_ptr,
    act_q_ptr, act_k_ptr,
    S1, S2, topK, S1_OFFSET,
    D_IDX: tl.constexpr,
    VALID_K: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Gather key_index at sparse positions (K1, post K1+K2 fusion).

    Grid: (B * S1, cdiv(valid_k, BLOCK_K), cdiv(D_IDX, BLOCK_D)).
    Only key_index is materialized — K3/K4 still read key_index_gathered
    from HBM. K2 (_teacher_distribution_kernel) inlines key/key_rope
    gather to avoid the original 9GB HBM round-trip.
    """
    pid = tl.program_id(0)
    k_block = tl.program_id(1)
    d_block = tl.program_id(2)
    b = pid // S1
    s1 = pid % S1
    s1_global = s1 + S1_OFFSET
    act_q = tl.load(act_q_ptr + b)
    act_k = tl.load(act_k_ptr + b)
    s2_real = tl.minimum(topK, tl.maximum(act_k - act_q + s1_global + 1, 0))
    s2_real = tl.where(s1_global < act_q, s2_real, 0)
    s2_bound = tl.minimum(s2_real, VALID_K)
    if k_block * BLOCK_K >= s2_bound:
        return

    k_offs = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offs < s2_bound
    idx = tl.load(indices_ptr + pid * topK + k_offs, mask=k_mask, other=0)
    idx = tl.maximum(tl.minimum(idx, S2 - 1), 0)

    d_offs = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D_IDX
    mask_2d = k_mask[:, None] & d_mask[None, :]
    src_offs = b * S2 * D_IDX + idx[:, None] * D_IDX + d_offs[None, :]
    vals = tl.load(src_idx_ptr + src_offs, mask=mask_2d, other=0.0)
    dst_offs = pid * topK * D_IDX + k_offs[:, None] * D_IDX + d_offs[None, :]
    tl.store(dst_idx_ptr + dst_offs, vals, mask=mask_2d)


@triton.jit
def _teacher_distribution_kernel(
    query_ptr, key_ptr,
    query_rope_ptr, key_rope_ptr,
    key_index_ptr,
    key_index_gathered_ptr,
    sparse_indices_ptr,
    softmax_max_ptr, softmax_sum_ptr,
    buf_p_ptr,
    B, S1, S2, N1, D, D_rope, D_idx, topK,
    scale_value, S1_OFFSET,
    act_q_ptr, act_k_ptr,
    VALID_K: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_D_IDX: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused gather + teacher p[k] averaged over heads + ki gather
    (K1+K2 fusion, formerly two separate kernels).

    Grid: (B*S1, cdiv(valid_k, BLOCK_K)).
    Each program inline-gathers k/kr at sparse positions for its K-tile
    (single shared idx load per tile) and computes scores/softmax without
    writing key_gathered / key_rope_gathered to HBM. Then it gathers
    key_index at the same sparse positions into key_index_gathered (this
    replaces the standalone _gather_kv_kernel).
    """
    pid = tl.program_id(0)
    k_block = tl.program_id(1)
    b = pid // S1
    s1 = pid % S1
    s1_global = s1 + S1_OFFSET

    act_q = tl.load(act_q_ptr + b)
    if s1_global >= act_q:
        return

    act_k = tl.load(act_k_ptr + b)
    s2_real = tl.minimum(topK, tl.maximum(act_k - act_q + s1_global + 1, 0))
    s2_bound = tl.minimum(s2_real, VALID_K)
    k_start = k_block * BLOCK_K
    if k_start >= s2_bound:
        return

    local_k = tl.arange(0, BLOCK_K)
    h_local = tl.arange(0, BLOCK_H)
    d_local = tl.arange(0, BLOCK_D)
    k_offs = k_start + local_k
    k_mask = k_offs < s2_bound

    # Inline gather: load sparse_indices once per K-tile, shared by
    # both D and D_rope inner loops; ~1KB UB (int32 x BLOCK_K=128).
    idx = tl.load(sparse_indices_ptr + pid * topK + k_offs,
                  mask=k_mask, other=0)
    idx = tl.maximum(tl.minimum(idx, S2 - 1), 0)

    q_base = pid * N1 * D
    k_batch_base = b * S2 * D
    qr_base = pid * N1 * D_rope
    kr_batch_base = b * S2 * D_rope
    sm_base = pid * N1
    inv_n1 = 1.0 / N1

    # Fast path when BLOCK_H >= N1 (typical: N1=64, BLOCK_H=64).
    # Eliminates outer h-loop overhead and lets the cube + softmax stages
    # overlap better.
    if BLOCK_H >= N1:
        h_offs = h_local
        h_mask = h_offs < N1
        scores = tl.zeros([BLOCK_H, BLOCK_K], dtype=tl.float32)

        for d_start in range(0, D, BLOCK_D):
            d_offs = d_start + d_local
            d_valid = d_offs < D
            q_tile = tl.load(
                query_ptr + q_base + h_offs[:, None] * D + d_offs[None, :],
                mask=h_mask[:, None] & d_valid[None, :],
                other=0.0)
            k_tile = tl.load(
                key_ptr + k_batch_base
                + idx[:, None] * D + d_offs[None, :],
                mask=k_mask[:, None] & d_valid[None, :],
                other=0.0)
            scores += tl.dot(q_tile, tl.trans(k_tile))

        for d_start in range(0, D_rope, BLOCK_D):
            d_offs = d_start + d_local
            d_valid = d_offs < D_rope
            qr_tile = tl.load(
                query_rope_ptr + qr_base
                + h_offs[:, None] * D_rope + d_offs[None, :],
                mask=h_mask[:, None] & d_valid[None, :],
                other=0.0)
            kr_tile = tl.load(
                key_rope_ptr + kr_batch_base
                + idx[:, None] * D_rope + d_offs[None, :],
                mask=k_mask[:, None] & d_valid[None, :],
                other=0.0)
            scores += tl.dot(qr_tile, tl.trans(kr_tile))

        sm_max = tl.load(softmax_max_ptr + sm_base + h_offs,
                         mask=h_mask, other=0.0).to(tl.float32)
        sm_sum = tl.load(softmax_sum_ptr + sm_base + h_offs,
                         mask=h_mask, other=1.0).to(tl.float32)
        inv_sum = 1.0 / (sm_sum + 1e-8)
        # combined scale: probs = exp(scores * scale - sm_max) * inv_sum
        probs = tl.exp(scores * scale_value - sm_max[:, None]) * inv_sum[:, None]
        probs = tl.where(h_mask[:, None] & k_mask[None, :], probs, 0.0)
        p_acc = tl.sum(probs, axis=0)
    else:
        p_acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        for h_start in range(0, N1, BLOCK_H):
            h_offs = h_start + h_local
            h_mask = h_offs < N1
            scores = tl.zeros([BLOCK_H, BLOCK_K], dtype=tl.float32)

            for d_start in range(0, D, BLOCK_D):
                d_offs = d_start + d_local
                d_valid = d_offs < D
                q_tile = tl.load(
                    query_ptr + q_base + h_offs[:, None] * D + d_offs[None, :],
                    mask=h_mask[:, None] & d_valid[None, :],
                    other=0.0)
                k_tile = tl.load(
                    key_ptr + k_batch_base
                    + idx[:, None] * D + d_offs[None, :],
                    mask=k_mask[:, None] & d_valid[None, :],
                    other=0.0)
                scores += tl.dot(q_tile, tl.trans(k_tile))

            for d_start in range(0, D_rope, BLOCK_D):
                d_offs = d_start + d_local
                d_valid = d_offs < D_rope
                qr_tile = tl.load(
                    query_rope_ptr + qr_base
                    + h_offs[:, None] * D_rope + d_offs[None, :],
                    mask=h_mask[:, None] & d_valid[None, :],
                    other=0.0)
                kr_tile = tl.load(
                    key_rope_ptr + kr_batch_base
                    + idx[:, None] * D_rope + d_offs[None, :],
                    mask=k_mask[:, None] & d_valid[None, :],
                    other=0.0)
                scores += tl.dot(qr_tile, tl.trans(kr_tile))

            sm_max = tl.load(softmax_max_ptr + sm_base + h_offs,
                             mask=h_mask, other=0.0).to(tl.float32)
            sm_sum = tl.load(softmax_sum_ptr + sm_base + h_offs,
                             mask=h_mask, other=1.0).to(tl.float32)
            probs = tl.exp(scores * scale_value - sm_max[:, None]) / (
                sm_sum[:, None] + 1e-8)
            probs = tl.where(h_mask[:, None] & k_mask[None, :], probs, 0.0)
            p_acc += tl.sum(probs, axis=0)

    tl.store(buf_p_ptr + pid * topK + k_offs,
             p_acc * inv_n1, mask=k_mask)

    # Inline gather of key_index (replaces standalone _gather_kv_kernel).
    # Same k-block, reuse idx already loaded. Tile D_idx by BLOCK_D_IDX.
    ki_src_base = b * S2 * D_idx
    ki_dst_base = pid * topK * D_idx
    d_idx_local = tl.arange(0, BLOCK_D_IDX)
    for d_start in range(0, D_idx, BLOCK_D_IDX):
        d_offs = d_start + d_idx_local
        d_valid = d_offs < D_idx
        mask_2d = k_mask[:, None] & d_valid[None, :]
        ki_vals = tl.load(
            key_index_ptr + ki_src_base + idx[:, None] * D_idx + d_offs[None, :],
            mask=mask_2d, other=0.0)
        tl.store(
            key_index_gathered_ptr + ki_dst_base
            + k_offs[:, None] * D_idx + d_offs[None, :],
            ki_vals, mask=mask_2d)


@triton.jit
def _indexer_grad_kl_loss_kernel(
    query_index_ptr, key_index_gathered_ptr,
    weights_ptr,
    di_ptr,
    loss_ptr,
    s_idx_buf_ptr, buf_i_ptr, buf_p_ptr,
    S1, Nidx1, D_idx, topK,
    S1_OFFSET,
    act_q_ptr, act_k_ptr,
    VALID_K: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    """Indexer score, teacher KL, and dI. Grid: (B*S1,).

    Stage 1: I[k]=sum_g W_g*ReLU(qi_g @ ki_gathered[k]^T), saves s_idx_buf.
    Stage 3+4: softmax(I) -> KL(p || softmax(I)), dI = softmax(I) - p.
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

    qi_base = pid * Nidx1 * D_idx
    ki_g_base = pid * topK * D_idx
    w_base = pid * Nidx1
    local_k = tl.arange(0, BLOCK_K)
    g_local = tl.arange(0, BLOCK_G)
    d_idx_local = tl.arange(0, BLOCK_D)

    # Stage 1: I[k] = sum_g W_g * ReLU(qi_g @ ki_gathered[k]^T)
    for k_start in range(0, VALID_K, BLOCK_K):
        if k_start < s2_real:
            k_offs = k_start + local_k
            k_mask = k_offs < s2_real
            i_tile = tl.zeros([BLOCK_K], dtype=tl.float32)
            for g_start in range(0, Nidx1, BLOCK_G):
                g_offs = g_start + g_local
                g_mask = g_offs < Nidx1
                idx_scores = tl.zeros([BLOCK_G, BLOCK_K], dtype=tl.float32)

                for d_start in range(0, D_idx, BLOCK_D):
                    d_offs = d_start + d_idx_local
                    d_valid = d_offs < D_idx
                    qi_tile = tl.load(
                        query_index_ptr + qi_base
                        + g_offs[:, None] * D_idx + d_offs[None, :],
                        mask=g_mask[:, None] & d_valid[None, :],
                        other=0.0)
                    ki_tile = tl.load(
                        key_index_gathered_ptr + ki_g_base
                        + k_offs[:, None] * D_idx + d_offs[None, :],
                        mask=k_mask[:, None] & d_valid[None, :],
                        other=0.0)
                    idx_scores += tl.dot(qi_tile, tl.trans(ki_tile))

                relu = tl.maximum(idx_scores, 0.0)
                relu = tl.where(g_mask[:, None] & k_mask[None, :], relu, 0.0)
                w_g = tl.load(weights_ptr + w_base + g_offs,
                              mask=g_mask, other=0.0).to(tl.float32)
                i_tile += tl.sum(relu * w_g[:, None], axis=0)
                tl.store(
                    s_idx_buf_ptr + pid * Nidx1 * topK
                    + g_offs[:, None] * topK + k_offs[None, :],
                    relu.to(s_idx_buf_ptr.dtype.element_ty),
                    mask=g_mask[:, None] & k_mask[None, :])
            tl.store(buf_i_ptr + pid * topK + k_offs, i_tile, mask=k_mask)

    # Stage 3+4: softmax(I) -> KL(p || softmax(I)) loss, dI = softmax(I) - p
    # Load full I[0:VALID_K] into UB once (fp32, ~8KB at VALID_K=2048) and
    # share across max/sum/softmax stages to avoid re-reading buf_i 3x.
    valid_k_offs = tl.arange(0, VALID_K)
    valid_k_mask = valid_k_offs < s2_real
    i_full = tl.load(buf_i_ptr + pid * topK + valid_k_offs,
                     mask=valid_k_mask, other=float('-inf'))
    i_max = tl.max(i_full, axis=0)
    exp_i_full = tl.where(valid_k_mask, tl.exp(i_full - i_max), 0.0)
    i_sum = tl.sum(exp_i_full, axis=0)
    inv_i_sum = 1.0 / (i_sum + 1e-8)
    log_i_sum = tl.log(i_sum + 1e-8)
    softmax_i_full = exp_i_full * inv_i_sum
    p_full = tl.load(buf_p_ptr + pid * topK + valid_k_offs,
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


@triton.jit
def _query_index_weight_grad_kernel(
    query_index_ptr,
    key_index_gathered_ptr,
    weights_ptr,
    di_ptr,
    s_idx_buf_ptr,
    d_query_index_ptr,
    d_weights_ptr,
    act_q_ptr, act_k_ptr,
    S1, Nidx1, D_idx, topK,
    S1_OFFSET,
    VALID_K: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    """Stage 5: dW and dQueryIndex from chain rule.

    Grid: (B*S1, cdiv(Nidx1, BLOCK_G)). Each program produces dW and
    dQueryIndex for a whole [BLOCK_G, D_idx] tile. The K reduction is done
    via tl.dot([BLOCK_G, BLOCK_K] x [BLOCK_K, BLOCK_D]) so the cube unit
    carries the FMAs and BLOCK_G works as a vector dimension, not a scalar
    static_range. BLOCK_D should cover D_idx in one shot to avoid splitting.
    """
    pid = tl.program_id(0)
    g_block = tl.program_id(1)
    b = pid // S1
    s1 = pid % S1
    s1_global = s1 + S1_OFFSET

    act_q = tl.load(act_q_ptr + b)
    if s1_global >= act_q:
        return

    act_k = tl.load(act_k_ptr + b)
    s2_real = tl.minimum(topK, tl.maximum(act_k - act_q + s1_global + 1, 0))
    s2_bound = tl.minimum(s2_real, VALID_K)
    if s2_bound <= 0:
        return

    qi_base = pid * Nidx1 * D_idx
    ki_g_base = pid * topK * D_idx
    w_base = pid * Nidx1
    sidx_base = pid * Nidx1 * topK
    local_k = tl.arange(0, BLOCK_K)
    g_local = tl.arange(0, BLOCK_G)
    g_offs = g_block * BLOCK_G + g_local
    g_mask = g_offs < Nidx1
    d_offs = tl.arange(0, BLOCK_D)
    d_valid = d_offs < D_idx

    w_g = tl.load(weights_ptr + w_base + g_offs,
                  mask=g_mask, other=0.0).to(tl.float32)

    dw_acc = tl.zeros([BLOCK_G], dtype=tl.float32)
    dqi_acc = tl.zeros([BLOCK_G, BLOCK_D], dtype=tl.float32)

    for k_start in range(0, VALID_K, BLOCK_K):
        k_offs = k_start + local_k
        k_mask = k_offs < s2_bound

        # relu_tile: [BLOCK_G, BLOCK_K]. Stored in input dtype (fp16/bf16) to
        # halve HBM bandwidth; promote to fp32 here for accurate dW reduction.
        relu_tile = tl.load(
            s_idx_buf_ptr + sidx_base
            + g_offs[:, None] * topK + k_offs[None, :],
            mask=g_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        # di_tile: [BLOCK_K], shared across all g in this block
        di_tile = tl.load(di_ptr + pid * topK + k_offs,
                          mask=k_mask, other=0.0)

        # dW_g += sum_k(di_k * relu_gk)
        dw_acc += tl.sum(di_tile[None, :] * relu_tile, axis=1)

        # ds_idx: [BLOCK_G, BLOCK_K]
        relu_mask = (relu_tile > 0.0).to(tl.float32)
        ds_idx = di_tile[None, :] * w_g[:, None] * relu_mask

        # ki_tile: [BLOCK_K, BLOCK_D], shared across all g
        # Keep ki_tile in fp16 to halve UB pressure on tight backends
        # (e.g. CANN 9.0 + Ascend910_9382 multi-buffer pass). tl.dot
        # accepts fp16 x fp16 -> fp32 accumulator.
        ki_tile = tl.load(
            key_index_gathered_ptr + ki_g_base
            + k_offs[:, None] * D_idx + d_offs[None, :],
            mask=k_mask[:, None] & d_valid[None, :],
            other=0.0)

        # dqi_acc += [BLOCK_G, BLOCK_K] @ [BLOCK_K, BLOCK_D]
        dqi_acc += tl.dot(ds_idx.to(ki_tile.dtype), ki_tile)

    # Store dQueryIndex tile [BLOCK_G, BLOCK_D]
    dqi_offs = (qi_base
                + g_offs[:, None] * D_idx + d_offs[None, :])
    tl.store(d_query_index_ptr + dqi_offs,
             dqi_acc.to(d_query_index_ptr.dtype.element_ty),
             mask=g_mask[:, None] & d_valid[None, :])

    # Store dW tile [BLOCK_G]
    tl.store(d_weights_ptr + w_base + g_offs,
             dw_acc.to(d_weights_ptr.dtype.element_ty),
             mask=g_mask)


# scatter-add dKeyIndex kernel
@triton.jit
def _scatter_dkey_index_kernel(
    query_index_ptr,
    weights_ptr, di_ptr, sparse_indices_ptr,
    s_idx_buf_ptr,
    d_key_index_ptr,
    act_q_ptr, act_k_ptr,
    B, S1, S2, Nidx1, D_idx, topK,
    valid_k, S1_OFFSET,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    """Scatter-add dKeyIndex. Grid: (B*S1, cdiv(valid_k, BLOCK_K), D-blocks).

    dki[b, target_k] += dI[k] * w[g] * 1_{relu>0} * qi[g]

    Vectorized over g: instead of a scalar for-loop over Nidx1, we tile g into
    BLOCK_G chunks and fuse the (di * w * mask) construction with a
    [BLOCK_K, BLOCK_G] @ [BLOCK_G, BLOCK_D] tl.dot. This moves the FMAs to
    the cube unit (matching the dQueryIndex kernel's strategy) and replaces
    the serial broadcast-add with a single cube call per g-tile.
    """
    bs1_idx = tl.program_id(0)
    k_block = tl.program_id(1)
    d_block = tl.program_id(2)
    b = bs1_idx // S1
    s1 = bs1_idx % S1
    s1_global = s1 + S1_OFFSET

    act_q = tl.load(act_q_ptr + b)
    act_k = tl.load(act_k_ptr + b)
    if s1_global >= act_q:
        return

    s2_real = tl.minimum(topK, tl.maximum(act_k - act_q + s1_global + 1, 0))
    s2_bound = tl.minimum(s2_real, valid_k)
    if k_block * BLOCK_K >= s2_bound:
        return

    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < s2_bound

    qi_base = bs1_idx * Nidx1 * D_idx
    w_base = bs1_idx * Nidx1
    sidx_base = bs1_idx * Nidx1 * topK
    di_k = tl.load(di_ptr + bs1_idx * topK + k_offsets,
                   mask=k_mask, other=0.0).to(tl.float32)
    target_k = tl.load(sparse_indices_ptr + bs1_idx * topK + k_offsets,
                       mask=k_mask, other=0)
    target_k = tl.maximum(tl.minimum(target_k, S2 - 1), 0)

    d_offs = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    d_valid = d_offs < D_idx
    g_local = tl.arange(0, BLOCK_G)

    dki_acc = tl.zeros([BLOCK_K, BLOCK_D], dtype=tl.float32)

    for g_start in range(0, Nidx1, BLOCK_G):
        g_offs = g_start + g_local
        g_mask = g_offs < Nidx1

        # w_g: [BLOCK_G]
        w_g = tl.load(weights_ptr + w_base + g_offs,
                      mask=g_mask, other=0.0).to(tl.float32)

        # relu_gk: [BLOCK_G, BLOCK_K]
        relu_gk = tl.load(
            s_idx_buf_ptr + sidx_base
            + g_offs[:, None] * topK + k_offsets[None, :],
            mask=g_mask[:, None] & k_mask[None, :], other=0.0)
        relu_mask = (relu_gk > 0.0).to(tl.float32)

        # ds_idx[g, k] = di_k * w_g * mask -> [BLOCK_G, BLOCK_K]
        ds_idx_gk = di_k[None, :] * w_g[:, None] * relu_mask

        # qi tile: [BLOCK_G, BLOCK_D]
        # Cast to fp16 to halve UB pressure (matches dQueryIndex path,
        # tl.dot accepts fp16 x fp16 -> fp32 accumulator).
        qi_tile = tl.load(
            query_index_ptr + qi_base
            + g_offs[:, None] * D_idx + d_offs[None, :],
            mask=g_mask[:, None] & d_valid[None, :], other=0.0)

        # dki_acc[k, d] += sum_g ds_idx[g,k] * qi[g,d]
        #              = trans(ds_idx)[k,g] @ qi[g,d]
        ds_idx_kg = tl.trans(ds_idx_gk).to(qi_tile.dtype)
        dki_acc += tl.dot(ds_idx_kg, qi_tile)

    dki_offs = b * S2 * D_idx + target_k[:, None] * D_idx + d_offs[None, :]
    tl.atomic_add(d_key_index_ptr + dki_offs,
                  dki_acc,
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
    BLOCK_K_TEACHER = 128
    BLOCK_H_MAIN = 32
    BLOCK_H_TEACHER = 64
    BLOCK_G_MAIN = 16
    BLOCK_D_GATHER = 128
    BLOCK_D_MAIN = 64
    BLOCK_D_TEACHER = 128
    BLOCK_K_QUERY_WEIGHT = 128
    BLOCK_G_QUERY_WEIGHT = 32
    BLOCK_D_QUERY_WEIGHT = 128
    BLOCK_K_SCATTER = 256
    BLOCK_D_SCATTER = 128
    BLOCK_G_SCATTER = 32

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

    # Intermediates. s_idx_buf is stored in the source dtype (typically fp16)
    # to halve HBM bandwidth — query_weight & scatter only need its sign
    # (for relu_mask) and a single fp16-precise value (for dW). Quality of
    # relu output is bounded by the fp16 qi/ki inputs that produced it.
    di = ms.mint.zeros((B * S1, topK), dtype=ms.float32)
    s_idx_buf = ms.mint.zeros((B * S1, Nidx1, topK), dtype=query_index.dtype)
    buf_i = ms.mint.zeros((B * S1, topK), dtype=ms.float32)
    buf_p = ms.mint.zeros((B * S1, topK), dtype=ms.float32)

    # Outputs
    d_query_index = ms.mint.zeros((B * S1, Nidx1, D_idx), dtype=query_index.dtype)
    d_key_index_acc = ms.mint.zeros((B * S2, D_idx), dtype=ms.float32)
    d_weights = ms.mint.zeros((B * S1, Nidx1), dtype=weights.dtype)
    loss_parts = ms.mint.zeros((B * S1,), dtype=ms.float32)

    # Gathered (the fused teacher kernel below writes key_index_gathered as a
    # side-effect, eliminating the standalone _gather_kv_kernel)
    key_index_gathered = ms.mint.zeros((B * S1, topK, D_idx), dtype=key_index.dtype)

    grid_bs1 = (B * S1,)
    if _SLISYNC:
        runtime.synchronize()

    _teacher_distribution_kernel[
        (B * S1, triton.cdiv(valid_k, BLOCK_K_TEACHER))
    ](
        q_flat, k_flat,
        qr_flat, kr_flat,
        ki_flat,
        key_index_gathered,
        sparse_flat,
        sm_max_flat, sm_sum_flat,
        buf_p,
        B, S1, S2, N1, D, D_rope, D_idx, topK,
        scale_value, s1_offset,
        actual_seq_qlen, actual_seq_klen,
        VALID_K=valid_k, BLOCK_K=BLOCK_K_TEACHER, BLOCK_D=BLOCK_D_TEACHER,
        BLOCK_D_IDX=BLOCK_D_GATHER,
        BLOCK_H=BLOCK_H_TEACHER,
    )

    if _SLISYNC:
        runtime.synchronize()

    _indexer_grad_kl_loss_kernel[grid_bs1](
        qi_flat, key_index_gathered,
        w_flat,
        di,
        loss_parts,
        s_idx_buf, buf_i, buf_p,
        S1, Nidx1, D_idx, topK,
        s1_offset,
        actual_seq_qlen, actual_seq_klen,
        VALID_K=valid_k, BLOCK_K=BLOCK_K_MAIN, BLOCK_D=BLOCK_D_MAIN,
        BLOCK_G=BLOCK_G_MAIN,
    )

    if _SLISYNC:
        runtime.synchronize()

    _query_index_weight_grad_kernel[
        (B * S1, triton.cdiv(Nidx1, BLOCK_G_QUERY_WEIGHT))
    ](
        qi_flat,
        key_index_gathered,
        w_flat,
        di,
        s_idx_buf,
        d_query_index,
        d_weights,
        actual_seq_qlen, actual_seq_klen,
        S1, Nidx1, D_idx, topK,
        s1_offset,
        VALID_K=valid_k, BLOCK_K=BLOCK_K_QUERY_WEIGHT,
        BLOCK_D=BLOCK_D_QUERY_WEIGHT, BLOCK_G=BLOCK_G_QUERY_WEIGHT,
    )

    _scatter_dkey_index_kernel[
        (B * S1, triton.cdiv(valid_k, BLOCK_K_SCATTER),
         triton.cdiv(D_idx, BLOCK_D_SCATTER))
    ](
        qi_flat,
        w_flat, di, sparse_flat,
        s_idx_buf,
        d_key_index_acc,
        actual_seq_qlen, actual_seq_klen,
        B, S1, S2, Nidx1, D_idx, topK,
        valid_k, s1_offset,
        BLOCK_K=BLOCK_K_SCATTER, BLOCK_D=BLOCK_D_SCATTER,
        BLOCK_G=BLOCK_G_SCATTER,
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

    if query_rope is None:
        query_rope = ms.mint.zeros((B, S1, N1, 1), dtype=query.dtype)
    if key_rope is None:
        key_rope = ms.mint.zeros((B, S2, N2, 1), dtype=key.dtype)

    act_q = _default_actual_seq(actual_seq_qlen, S1, query)
    act_k = _default_actual_seq(actual_seq_klen, S2, key)

    if S1 <= SPARSE_GRAD_S1_CHUNK:
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

    for start in range(0, S1, SPARSE_GRAD_S1_CHUNK):
        end = min(start + SPARSE_GRAD_S1_CHUNK, S1)
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
