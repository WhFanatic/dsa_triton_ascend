"""Correctness-first Triton-Ascend dense LightningIndexer KL loss backward.

Only BSND layout, sparse_mode=3, and Nidx2=1 are supported.
The operator returns gradients for queryIndex, keyIndex, weights, and loss.
"""
import triton
import triton.language as tl
import mindspore as ms
from mindspore import ops


SUPPORTED_D = (128, 256, 512)
SUPPORTED_NIDX1 = (32, 64, 128)
INT64_MAX = 9223372036854775807
MAX_TRITON_ASCEND_COREDIM = 65535


def _bs1_chunk_for_core_dim(total_bs1, programs_per_bs1):
    programs_per_bs1 = max(1, programs_per_bs1)
    return max(1, min(total_bs1, MAX_TRITON_ASCEND_COREDIM // programs_per_bs1))


def _check_in(name, value, supported):
    if value not in supported:
        raise ValueError(f"{name} must be one of {supported}, got {value}")


def _default_actual_seq(actual_seq, seq_len, ref_tensor):
    device_zeros = ops.cast(ops.zeros_like(ref_tensor[:, 0, 0, 0]), ms.int64)
    if actual_seq is None:
        return device_zeros + seq_len
    if isinstance(actual_seq, (list, tuple)):
        return device_zeros + ms.Tensor(actual_seq, dtype=ms.int64)
    return actual_seq + device_zeros


@triton.jit
def _dense_indexer_dot_g_tile(
    query_index_ptr, key_index_ptr,
    B, S1, S2, Nidx1, D_idx,
    bs1_idx, b, g, s2_offsets, k_mask,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    qi_base = bs1_idx * Nidx1 * D_idx + g * D_idx
    ki_base = b * S2 * D_idx
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for d_start in range(0, D_idx, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D_idx
        qi = tl.load(query_index_ptr + qi_base + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        ki = tl.load(
            key_index_ptr + ki_base + s2_offsets[:, None] * D_idx + d_offsets[None, :],
            mask=k_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(ki * qi[None, :], axis=1)
    return acc


@triton.jit
def _dense_indexer_i_tile(
    query_index_ptr, key_index_ptr, weights_ptr,
    B, S1, S2, Nidx1, D_idx,
    bs1_idx, b, s2_offsets, k_mask,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    w_base = bs1_idx * Nidx1
    i_tile = tl.zeros([BLOCK_K], dtype=tl.float32)
    for g in range(Nidx1):
        dot = _dense_indexer_dot_g_tile(
            query_index_ptr, key_index_ptr,
            B, S1, S2, Nidx1, D_idx,
            bs1_idx, b, g, s2_offsets, k_mask,
            BLOCK_K, BLOCK_D,
        )
        relu = tl.maximum(dot, 0.0)
        relu = tl.where(k_mask, relu, 0.0)
        w_g = tl.load(weights_ptr + w_base + g).to(tl.float32)
        i_tile += w_g * relu
    return i_tile


@triton.jit
def _dense_teacher_p_tile(
    query_ptr, key_ptr, query_rope_ptr, key_rope_ptr,
    softmax_max_ptr, softmax_sum_ptr,
    B, S1, S2, N1, N2, G, D, D_rope,
    bs1_idx, b, s1, s2_offsets, k_mask,
    scale_value,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    HAS_ROPE: tl.constexpr,
):
    p_tile = tl.zeros([BLOCK_K], dtype=tl.float32)
    q_base = bs1_idx * N1 * D
    k_base = b * S2 * N2 * D
    qr_base = bs1_idx * N1 * D_rope
    kr_base = b * S2 * N2 * D_rope
    sm_base = bs1_idx * N1
    for h in range(N1):
        kv_h = h // G
        score = tl.zeros([BLOCK_K], dtype=tl.float32)
        for d_start in range(0, D, BLOCK_D):
            d_offsets = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offsets < D
            q = tl.load(query_ptr + q_base + h * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
            k = tl.load(
                key_ptr + k_base + s2_offsets[:, None] * N2 * D + kv_h * D + d_offsets[None, :],
                mask=k_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            score += tl.sum(k * q[None, :], axis=1)
        if HAS_ROPE:
            for dr_start in range(0, D_rope, BLOCK_D):
                dr_offsets = dr_start + tl.arange(0, BLOCK_D)
                dr_mask = dr_offsets < D_rope
                qr = tl.load(
                    query_rope_ptr + qr_base + h * D_rope + dr_offsets,
                    mask=dr_mask,
                    other=0.0,
                ).to(tl.float32)
                kr = tl.load(
                    key_rope_ptr + kr_base + s2_offsets[:, None] * N2 * D_rope + kv_h * D_rope + dr_offsets[None, :],
                    mask=k_mask[:, None] & dr_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                score += tl.sum(kr * qr[None, :], axis=1)
        score = score * scale_value
        sm_max = tl.load(softmax_max_ptr + sm_base + h).to(tl.float32)
        sm_sum = tl.load(softmax_sum_ptr + sm_base + h).to(tl.float32)
        prob = tl.exp(score - sm_max) / (sm_sum + 1.0e-8)
        p_tile += tl.where(k_mask, prob, 0.0)
    return p_tile * (1.0 / N1)


@triton.jit
def _dense_indexer_stats_kernel(
    query_index_ptr, key_index_ptr, weights_ptr,
    max_index_ptr, sum_index_ptr,
    act_q_ptr, act_k_ptr,
    B, S1, S2, Nidx1, D_idx,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    bs1_idx = tl.program_id(0)
    b = bs1_idx // S1
    s1 = bs1_idx % S1
    act_q = tl.load(act_q_ptr + b)
    act_k = tl.load(act_k_ptr + b)
    visible = tl.minimum(tl.maximum(act_k - act_q + s1 + 1, 0), S2)
    valid_q = s1 < act_q

    local_k = tl.arange(0, BLOCK_K)
    i_max = tl.full([1], float("-inf"), dtype=tl.float32)
    i_sum = tl.zeros([1], dtype=tl.float32)
    for k_start in range(0, S2, BLOCK_K):
        s2_offsets = k_start + local_k
        k_mask = valid_q & (s2_offsets < visible)
        i_tile = _dense_indexer_i_tile(
            query_index_ptr, key_index_ptr, weights_ptr,
            B, S1, S2, Nidx1, D_idx,
            bs1_idx, b, s2_offsets, k_mask,
            BLOCK_K, BLOCK_D,
        )
        i_masked = tl.where(k_mask, i_tile, float("-inf"))
        m_new = tl.maximum(i_max, tl.max(i_masked, axis=0))
        m_new_safe = tl.where(m_new > float("-inf"), m_new, 0.0)
        alpha = tl.where(i_max > float("-inf"), tl.exp(i_max - m_new_safe), 1.0)
        exp_i = tl.where(k_mask, tl.exp(i_tile - m_new_safe), 0.0)
        i_sum = i_sum * alpha + tl.sum(exp_i, axis=0)
        i_max = m_new

    stat_lane = tl.arange(0, 1)
    tl.store(max_index_ptr + bs1_idx + stat_lane, i_max, mask=stat_lane == 0)
    tl.store(sum_index_ptr + bs1_idx + stat_lane, i_sum, mask=stat_lane == 0)


@triton.jit
def _dense_main_grad_kernel(
    query_index_ptr, key_index_ptr, weights_ptr,
    di_ptr,
    d_query_index_ptr, d_weights_ptr,
    act_q_ptr, act_k_ptr,
    B, S1, S2, N1, N2, G, Nidx1, D, D_idx, D_rope,
    scale_value, BS1_OFFSET,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    HAS_ROPE: tl.constexpr,
):
    bs1_idx = BS1_OFFSET + tl.program_id(0)
    g = tl.program_id(1)
    d_block = tl.program_id(2)
    b = bs1_idx // S1
    s1 = bs1_idx % S1

    act_q = tl.load(act_q_ptr + b)
    act_k = tl.load(act_k_ptr + b)
    visible = tl.minimum(tl.maximum(act_k - act_q + s1 + 1, 0), S2)
    valid_q = s1 < act_q

    d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D_idx
    local_k = tl.arange(0, BLOCK_K)
    dqi = tl.zeros([BLOCK_D], dtype=tl.float32)
    dw = tl.zeros([1], dtype=tl.float32)
    w_g = tl.load(weights_ptr + bs1_idx * Nidx1 + g).to(tl.float32)
    ki_base = b * S2 * D_idx

    for k_start in range(0, S2, BLOCK_K):
        s2_offsets = k_start + local_k
        k_mask = valid_q & (s2_offsets < visible)
        di = tl.load(di_ptr + bs1_idx * S2 + s2_offsets, mask=k_mask, other=0.0).to(tl.float32)

        dot_g = _dense_indexer_dot_g_tile(
            query_index_ptr, key_index_ptr,
            B, S1, S2, Nidx1, D_idx,
            bs1_idx, b, g, s2_offsets, k_mask,
            BLOCK_K, BLOCK_D,
        )
        relu_g = tl.maximum(dot_g, 0.0)
        relu_mask = (dot_g > 0.0).to(tl.float32)
        dw += tl.sum(di * relu_g, axis=0)

        ki = tl.load(
            key_index_ptr + ki_base + s2_offsets[:, None] * D_idx + d_offsets[None, :],
            mask=k_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        dqi += tl.sum((di * w_g * relu_mask)[:, None] * ki, axis=0)

    dqi_offs = bs1_idx * Nidx1 * D_idx + g * D_idx + d_offsets
    tl.store(d_query_index_ptr + dqi_offs, dqi.to(d_query_index_ptr.dtype.element_ty), mask=d_mask)
    tl.store(d_weights_ptr + bs1_idx * Nidx1 + g, tl.sum(dw).to(d_weights_ptr.dtype.element_ty), mask=d_block == 0)


@triton.jit
def _dense_loss_kernel(
    query_ptr, key_ptr, query_rope_ptr, key_rope_ptr,
    query_index_ptr, key_index_ptr, weights_ptr,
    softmax_max_ptr, softmax_sum_ptr,
    max_index_ptr, sum_index_ptr,
    loss_ptr, di_ptr,
    act_q_ptr, act_k_ptr,
    B, S1, S2, N1, N2, G, Nidx1, D, D_idx, D_rope,
    scale_value,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    HAS_ROPE: tl.constexpr,
):
    bs1_idx = tl.program_id(0)
    b = bs1_idx // S1
    s1 = bs1_idx % S1

    act_q = tl.load(act_q_ptr + b)
    act_k = tl.load(act_k_ptr + b)
    visible = tl.minimum(tl.maximum(act_k - act_q + s1 + 1, 0), S2)
    valid_q = s1 < act_q

    i_max = tl.load(max_index_ptr + bs1_idx).to(tl.float32)
    i_sum = tl.load(sum_index_ptr + bs1_idx).to(tl.float32)
    has_valid = valid_q & (i_sum > 0.0)
    i_sum_safe = tl.maximum(i_sum, 1.0e-8)

    local_k = tl.arange(0, BLOCK_K)
    local_loss = tl.zeros([1], dtype=tl.float32)
    for k_start in range(0, S2, BLOCK_K):
        s2_offsets = k_start + local_k
        k_mask = valid_q & (s2_offsets < visible)
        i_tile = _dense_indexer_i_tile(
            query_index_ptr, key_index_ptr, weights_ptr,
            B, S1, S2, Nidx1, D_idx,
            bs1_idx, b, s2_offsets, k_mask,
            BLOCK_K, BLOCK_D,
        )
        student = tl.exp(i_tile - i_max) / i_sum_safe
        student = tl.where(k_mask & has_valid, student, 0.0)
        teacher = _dense_teacher_p_tile(
            query_ptr, key_ptr, query_rope_ptr, key_rope_ptr,
            softmax_max_ptr, softmax_sum_ptr,
            B, S1, S2, N1, N2, G, D, D_rope,
            bs1_idx, b, s1, s2_offsets, k_mask,
            scale_value,
            BLOCK_K, BLOCK_D, HAS_ROPE,
        )
        teacher = tl.where(k_mask & has_valid, teacher, 0.0)

        di = student - teacher
        di_offs = bs1_idx * S2 + s2_offsets
        tl.store(di_ptr + di_offs, di.to(di_ptr.dtype.element_ty), mask=k_mask)

        p_clamped = tl.maximum(teacher, 1.0e-8)
        q_clamped = tl.maximum(student, 1.0e-8)
        kl = p_clamped * (tl.log(p_clamped) - tl.log(q_clamped))
        kl = tl.where(k_mask & has_valid, kl, 0.0)
        local_loss += tl.sum(kl, axis=0)

    loss_lane = tl.arange(0, 1)
    tl.atomic_add(loss_ptr + loss_lane, local_loss, mask=loss_lane == 0)


@triton.jit
def _dense_dkey_index_kernel(
    query_index_ptr, key_index_ptr, weights_ptr,
    di_ptr,
    d_key_index_ptr,
    act_q_ptr, act_k_ptr,
    B, S1, S2, N1, N2, G, Nidx1, D, D_idx, D_rope,
    scale_value, BS1_OFFSET,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    HAS_ROPE: tl.constexpr,
):
    bs1_idx = BS1_OFFSET + tl.program_id(0)
    k_block = tl.program_id(1)
    d_block = tl.program_id(2)
    b = bs1_idx // S1
    s1 = bs1_idx % S1

    act_q = tl.load(act_q_ptr + b)
    act_k = tl.load(act_k_ptr + b)
    visible = tl.minimum(tl.maximum(act_k - act_q + s1 + 1, 0), S2)
    valid_q = s1 < act_q

    local_k = tl.arange(0, BLOCK_K)
    s2_offsets = k_block * BLOCK_K + local_k
    k_mask = valid_q & (s2_offsets < visible)

    di = tl.load(di_ptr + bs1_idx * S2 + s2_offsets, mask=k_mask, other=0.0).to(tl.float32)

    d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D_idx
    qi_base = bs1_idx * Nidx1 * D_idx
    w_base = bs1_idx * Nidx1

    dki_acc = tl.zeros([BLOCK_K, BLOCK_D], dtype=tl.float32)
    for g in range(Nidx1):
        dot_g = _dense_indexer_dot_g_tile(
            query_index_ptr, key_index_ptr,
            B, S1, S2, Nidx1, D_idx,
            bs1_idx, b, g, s2_offsets, k_mask,
            BLOCK_K, BLOCK_D,
        )
        relu_mask = (dot_g > 0.0).to(tl.float32)
        w_g = tl.load(weights_ptr + w_base + g).to(tl.float32)
        coeff = di * w_g * relu_mask
        qi = tl.load(
            query_index_ptr + qi_base + g * D_idx + d_offsets,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        dki_acc += coeff[:, None] * qi[None, :]

    dki_offsets = b * S2 * D_idx + s2_offsets[:, None] * D_idx + d_offsets[None, :]
    tl.atomic_add(d_key_index_ptr + dki_offsets, dki_acc, mask=k_mask[:, None] & d_mask[None, :])


def _infer_dense_index_stats(query_index, key_index, weights,
                             actual_seq_q, actual_seq_k):
    B, S1 = query_index.shape[0], query_index.shape[1]
    return (
        ms.mint.empty((B, S1), dtype=ms.float32),
        ms.mint.empty((B, S1), dtype=ms.float32),
    )


@ms.ops._ms_pyfunc(infer_func=_infer_dense_index_stats)
def _dense_index_stats_core(
    query_index: ms.Tensor,
    key_index: ms.Tensor,
    weights: ms.Tensor,
    actual_seq_q: ms.Tensor,
    actual_seq_k: ms.Tensor,
) -> tuple[ms.Tensor, ms.Tensor]:
    B, S1, Nidx1, D_idx = query_index.shape
    S2 = key_index.shape[1]

    qi_flat = query_index.reshape(B * S1, Nidx1, D_idx).contiguous()
    ki_flat = key_index.reshape(B * S2, D_idx).contiguous()
    w_flat = weights.reshape(B * S1, Nidx1).contiguous()

    max_index = ms.mint.empty((B * S1,), dtype=ms.float32)
    sum_index = ms.mint.empty((B * S1,), dtype=ms.float32)
    _dense_indexer_stats_kernel[(B * S1,)](
        qi_flat, ki_flat, w_flat,
        max_index, sum_index,
        actual_seq_q, actual_seq_k,
        B, S1, S2, Nidx1, D_idx,
        BLOCK_K=64, BLOCK_D=64,
    )
    return max_index.reshape((B, S1)), sum_index.reshape((B, S1))


def _infer_dense_backward(query, key, query_rope, key_rope,
                          query_index, key_index, weights,
                          softmax_max, softmax_sum,
                          actual_seq_q, actual_seq_k, scale_value, d_rope):
    return (
        ms.mint.empty_like(query_index),
        ms.mint.empty_like(key_index),
        ms.mint.empty_like(weights),
        ms.mint.empty((1,), dtype=ms.float32),
    )


@ms.ops._ms_pyfunc(infer_func=_infer_dense_backward)
def _dense_loss_backward_core(
    query: ms.Tensor,
    key: ms.Tensor,
    query_rope: ms.Tensor,
    key_rope: ms.Tensor,
    query_index: ms.Tensor,
    key_index: ms.Tensor,
    weights: ms.Tensor,
    softmax_max: ms.Tensor,
    softmax_sum: ms.Tensor,
    actual_seq_q: ms.Tensor,
    actual_seq_k: ms.Tensor,
    scale_value: float,
    d_rope: int,
) -> tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
    B, S1, N1, D = query.shape
    S2, N2 = key.shape[1], key.shape[2]
    G = N1 // N2
    Nidx1, D_idx = query_index.shape[2], query_index.shape[3]

    q_flat = query.reshape(B * S1, N1, D).contiguous()
    k_flat = key.reshape(B * S2, N2, D).contiguous()
    if d_rope > 0:
        qr_flat = query_rope.reshape(B * S1, N1, d_rope).contiguous()
        kr_flat = key_rope.reshape(B * S2, N2, d_rope).contiguous()
    else:
        qr_flat = q_flat
        kr_flat = k_flat
    qi_flat = query_index.reshape(B * S1, Nidx1, D_idx).contiguous()
    ki_flat = key_index.reshape(B * S2, D_idx).contiguous()
    w_flat = weights.reshape(B * S1, Nidx1).contiguous()
    sm_max_flat = softmax_max.transpose(0, 2, 1, 3).reshape(B * S1, N1).contiguous()
    sm_sum_flat = softmax_sum.transpose(0, 2, 1, 3).reshape(B * S1, N1).contiguous()

    max_index = ms.mint.empty((B * S1,), dtype=ms.float32)
    sum_index = ms.mint.empty((B * S1,), dtype=ms.float32)
    d_query_index = ms.mint.zeros((B * S1, Nidx1, D_idx), dtype=query_index.dtype)
    d_key_index_acc = ms.mint.zeros((B * S2, D_idx), dtype=ms.float32)
    d_weights = ms.mint.zeros((B * S1, Nidx1), dtype=weights.dtype)
    loss = ms.mint.zeros((1,), dtype=ms.float32)
    di_ws = ms.mint.empty((B * S1, S2), dtype=ms.float32)

    BLOCK_K = 64
    BLOCK_D = 64
    _dense_indexer_stats_kernel[(B * S1,)](
        qi_flat, ki_flat, w_flat,
        max_index, sum_index,
        actual_seq_q, actual_seq_k,
        B, S1, S2, Nidx1, D_idx,
        BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
    )

    _dense_loss_kernel[(B * S1,)](
        q_flat, k_flat, qr_flat, kr_flat,
        qi_flat, ki_flat, w_flat,
        sm_max_flat, sm_sum_flat,
        max_index, sum_index,
        loss, di_ws,
        actual_seq_q, actual_seq_k,
        B, S1, S2, N1, N2, G, Nidx1, D, D_idx, d_rope,
        scale_value,
        BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D, HAS_ROPE=(d_rope > 0),
    )

    total_bs1 = B * S1
    num_d_blocks = (D_idx + BLOCK_D - 1) // BLOCK_D
    main_bs1_chunk = _bs1_chunk_for_core_dim(total_bs1, Nidx1 * num_d_blocks)
    for bs1_start in range(0, total_bs1, main_bs1_chunk):
        bs1_chunk = min(main_bs1_chunk, total_bs1 - bs1_start)
        _dense_main_grad_kernel[(bs1_chunk, Nidx1, num_d_blocks)](
            qi_flat, ki_flat, w_flat,
            di_ws,
            d_query_index, d_weights,
            actual_seq_q, actual_seq_k,
            B, S1, S2, N1, N2, G, Nidx1, D, D_idx, d_rope,
            scale_value, bs1_start,
            BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D, HAS_ROPE=(d_rope > 0),
        )

    num_k_blocks = (S2 + BLOCK_K - 1) // BLOCK_K
    dkey_bs1_chunk = _bs1_chunk_for_core_dim(total_bs1, num_k_blocks * num_d_blocks)
    for bs1_start in range(0, total_bs1, dkey_bs1_chunk):
        bs1_chunk = min(dkey_bs1_chunk, total_bs1 - bs1_start)
        _dense_dkey_index_kernel[(bs1_chunk, num_k_blocks, num_d_blocks)](
            qi_flat, ki_flat, w_flat,
            di_ws,
            d_key_index_acc,
            actual_seq_q, actual_seq_k,
            B, S1, S2, N1, N2, G, Nidx1, D, D_idx, d_rope,
            scale_value, bs1_start,
            BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D, HAS_ROPE=(d_rope > 0),
        )

    return (
        d_query_index.reshape(query_index.shape),
        ops.cast(d_key_index_acc.reshape(key_index.shape), key_index.dtype),
        d_weights.reshape(weights.shape),
        loss,
    )


def _infer_dense_backward_with_index(query, key, query_index, key_index, weights,
                                     softmax_max, softmax_sum,
                                     softmax_max_index, softmax_sum_index,
                                     query_rope, key_rope,
                                     actual_seq_q, actual_seq_k, scale_value, d_rope):
    return (
        ms.mint.empty_like(query_index),
        ms.mint.empty_like(key_index),
        ms.mint.empty_like(weights),
        ms.mint.empty((1,), dtype=ms.float32),
    )


@ms.ops._ms_pyfunc(infer_func=_infer_dense_backward_with_index)
def _dense_loss_backward_with_index_core(
    query: ms.Tensor,
    key: ms.Tensor,
    query_index: ms.Tensor,
    key_index: ms.Tensor,
    weights: ms.Tensor,
    softmax_max: ms.Tensor,
    softmax_sum: ms.Tensor,
    softmax_max_index: ms.Tensor,
    softmax_sum_index: ms.Tensor,
    query_rope: ms.Tensor,
    key_rope: ms.Tensor,
    actual_seq_q: ms.Tensor,
    actual_seq_k: ms.Tensor,
    scale_value: float,
    d_rope: int,
) -> tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
    B, S1, N1, D = query.shape
    S2, N2 = key.shape[1], key.shape[2]
    G = N1 // N2
    Nidx1, D_idx = query_index.shape[2], query_index.shape[3]

    q_flat = query.reshape(B * S1, N1, D).contiguous()
    k_flat = key.reshape(B * S2, N2, D).contiguous()
    if d_rope > 0:
        qr_flat = query_rope.reshape(B * S1, N1, d_rope).contiguous()
        kr_flat = key_rope.reshape(B * S2, N2, d_rope).contiguous()
    else:
        qr_flat = q_flat
        kr_flat = k_flat
    qi_flat = query_index.reshape(B * S1, Nidx1, D_idx).contiguous()
    ki_flat = key_index.reshape(B * S2, D_idx).contiguous()
    w_flat = weights.reshape(B * S1, Nidx1).contiguous()
    sm_max_flat = softmax_max.transpose(0, 2, 1, 3).reshape(B * S1, N1).contiguous()
    sm_sum_flat = softmax_sum.transpose(0, 2, 1, 3).reshape(B * S1, N1).contiguous()
    max_index = softmax_max_index.reshape(B * S1).contiguous()
    sum_index = softmax_sum_index.reshape(B * S1).contiguous()

    d_query_index = ms.mint.zeros((B * S1, Nidx1, D_idx), dtype=query_index.dtype)
    d_key_index_acc = ms.mint.zeros((B * S2, D_idx), dtype=ms.float32)
    d_weights = ms.mint.zeros((B * S1, Nidx1), dtype=weights.dtype)
    loss = ms.mint.zeros((1,), dtype=ms.float32)
    di_ws = ms.mint.empty((B * S1, S2), dtype=ms.float32)

    BLOCK_K = 64
    BLOCK_D = 64
    _dense_loss_kernel[(B * S1,)](
        q_flat, k_flat, qr_flat, kr_flat,
        qi_flat, ki_flat, w_flat,
        sm_max_flat, sm_sum_flat,
        max_index, sum_index,
        loss, di_ws,
        actual_seq_q, actual_seq_k,
        B, S1, S2, N1, N2, G, Nidx1, D, D_idx, d_rope,
        scale_value,
        BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D, HAS_ROPE=(d_rope > 0),
    )

    total_bs1 = B * S1
    num_d_blocks = (D_idx + BLOCK_D - 1) // BLOCK_D
    main_bs1_chunk = _bs1_chunk_for_core_dim(total_bs1, Nidx1 * num_d_blocks)
    for bs1_start in range(0, total_bs1, main_bs1_chunk):
        bs1_chunk = min(main_bs1_chunk, total_bs1 - bs1_start)
        _dense_main_grad_kernel[(bs1_chunk, Nidx1, num_d_blocks)](
            qi_flat, ki_flat, w_flat,
            di_ws,
            d_query_index, d_weights,
            actual_seq_q, actual_seq_k,
            B, S1, S2, N1, N2, G, Nidx1, D, D_idx, d_rope,
            scale_value, bs1_start,
            BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D, HAS_ROPE=(d_rope > 0),
        )

    num_k_blocks = (S2 + BLOCK_K - 1) // BLOCK_K
    dkey_bs1_chunk = _bs1_chunk_for_core_dim(total_bs1, num_k_blocks * num_d_blocks)
    for bs1_start in range(0, total_bs1, dkey_bs1_chunk):
        bs1_chunk = min(dkey_bs1_chunk, total_bs1 - bs1_start)
        _dense_dkey_index_kernel[(bs1_chunk, num_k_blocks, num_d_blocks)](
            qi_flat, ki_flat, w_flat,
            di_ws,
            d_key_index_acc,
            actual_seq_q, actual_seq_k,
            B, S1, S2, N1, N2, G, Nidx1, D, D_idx, d_rope,
            scale_value, bs1_start,
            BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D, HAS_ROPE=(d_rope > 0),
        )

    return (
        d_query_index.reshape(query_index.shape),
        ops.cast(d_key_index_acc.reshape(key_index.shape), key_index.dtype),
        d_weights.reshape(weights.shape),
        loss,
    )


def _validate_dense_index_inputs(query_index, key_index, weights,
                                 layout, sparse_mode,
                                 pre_tokens=INT64_MAX, next_tokens=INT64_MAX):
    if layout != "BSND":
        raise ValueError("Only BSND layout is supported")
    if sparse_mode != 3:
        raise ValueError("Only sparse_mode=3 (rightDownCausal) is supported")
    if pre_tokens != INT64_MAX or next_tokens != INT64_MAX:
        raise ValueError("pre_tokens/next_tokens only support the official default value")
    if query_index.ndim != 4 or key_index.ndim != 4:
        raise ValueError("query_index and key_index must be 4D BSND tensors")
    if weights.ndim != 3:
        raise ValueError("weights must be [B, S1, Nidx1]")

    B, S1, Nidx1, D_idx = query_index.shape
    if key_index.shape[0] != B or key_index.shape[2] != 1:
        raise ValueError("key_index must be [B, S2, 1, D_idx]")
    if weights.shape != (B, S1, Nidx1):
        raise ValueError("weights must be [B, S1, Nidx1]")
    if key_index.shape[3] != D_idx:
        raise ValueError("query_index and key_index must share D_idx")

    _check_in("Nidx1", Nidx1, SUPPORTED_NIDX1)
    _check_in("D_idx", D_idx, SUPPORTED_D)


def _validate_dense_dtypes(query, key, query_index, key_index, weights,
                           softmax_max, softmax_sum):
    data_dtype = query.dtype
    qk_dtypes = [ms.float16]
    if hasattr(ms, "bfloat16"):
        qk_dtypes.append(ms.bfloat16)
    if data_dtype not in qk_dtypes:
        raise TypeError(f"query dtype must be float16 or bfloat16, got {data_dtype}")
    if key.dtype != data_dtype or query_index.dtype != data_dtype or key_index.dtype != data_dtype:
        raise TypeError("query, key, query_index, and key_index must share dtype")
    if weights.dtype != data_dtype and weights.dtype != ms.float32:
        raise TypeError("weights dtype must match query dtype or be float32")
    if softmax_max.dtype != ms.float32 or softmax_sum.dtype != ms.float32:
        raise TypeError("softmax_max and softmax_sum must be float32")


def _validate_rope_inputs(query, key, query_rope, key_rope):
    if (query_rope is None) != (key_rope is None):
        raise ValueError("query_rope and key_rope must be both provided or both omitted")
    if query_rope is None:
        return 0
    if query_rope.ndim != 4 or key_rope.ndim != 4:
        raise ValueError("query_rope and key_rope must be 4D BSND tensors")
    if query_rope.shape[:3] != query.shape[:3]:
        raise ValueError("query_rope must share [B, S1, N1] with query")
    if key_rope.shape[:3] != key.shape[:3]:
        raise ValueError("key_rope must share [B, S2, N2] with key")
    if query_rope.shape[3] != key_rope.shape[3]:
        raise ValueError("query_rope and key_rope must share Drope")
    if query_rope.shape[3] != 64:
        raise ValueError("Only Drope=64 is supported for query_rope/key_rope")
    if query_rope.dtype != query.dtype or key_rope.dtype != key.dtype:
        raise TypeError("query_rope/key_rope dtype must match query/key dtype")
    return query_rope.shape[3]


def _validate_dense_inputs(query, key, query_index, key_index, weights,
                           softmax_max, softmax_sum, layout, sparse_mode,
                           query_rope=None, key_rope=None,
                           pre_tokens=INT64_MAX, next_tokens=INT64_MAX):
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must be 4D BSND tensors")
    _validate_dense_index_inputs(
        query_index, key_index, weights,
        layout, sparse_mode,
        pre_tokens=pre_tokens, next_tokens=next_tokens,
    )
    _validate_dense_dtypes(query, key, query_index, key_index, weights, softmax_max, softmax_sum)

    B, S1, N1, D = query.shape
    if key.shape[0] != B or key.shape[3] != D:
        raise ValueError("key must be [B, S2, N2, D_attn]")
    N2 = key.shape[2]
    if N2 <= 0 or N1 % N2 != 0:
        raise ValueError("query N1 must be divisible by key N2")
    if query_index.shape[0] != B or query_index.shape[1] != S1:
        raise ValueError("query_index must share B and S1 with query")
    if key_index.shape[1] != key.shape[1]:
        raise ValueError("key_index S2 must match key S2")
    softmax_shape = (B, N2, S1, N1 // N2)
    if softmax_max.shape != softmax_shape or softmax_sum.shape != softmax_shape:
        raise ValueError(f"softmax_max and softmax_sum must be [B, N2, S1, G], got {softmax_max.shape}")

    _check_in("D_attn", D, SUPPORTED_D)
    return _validate_rope_inputs(query, key, query_rope, key_rope)


def _reshape_dense_index_stat(stat, B, S1, name):
    if stat.shape == (B, S1) or stat.shape == (B * S1,):
        return stat.reshape(B * S1).contiguous()
    if stat.shape == (B, 1, S1):
        return stat.reshape(B * S1).contiguous()
    if stat.shape == (B, 1, S1, 1):
        return stat.reshape(B * S1).contiguous()
    raise ValueError(f"{name} must have shape [B,S1], [B*S1], [B,1,S1], or [B,1,S1,1], got {stat.shape}")


def dense_loss_backward_triton(
    query: ms.Tensor,
    key: ms.Tensor,
    query_index: ms.Tensor,
    key_index: ms.Tensor,
    weights: ms.Tensor,
    softmax_max: ms.Tensor,
    softmax_sum: ms.Tensor,
    softmax_max_index=None,
    softmax_sum_index=None,
    scale_value: float = 1.0,
    query_rope=None,
    key_rope=None,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout: str = "BSND",
    sparse_mode: int = 3,
    pre_tokens: int = INT64_MAX,
    next_tokens: int = INT64_MAX,
):
    """Dense LightningIndexer KL loss backward.

    The public signature is aligned with the official dense
    lightning-indexer grad KL loss interface. If softmax_max_index and
    softmax_sum_index are omitted, the triton wrapper computes them internally
    for backward compatibility with the first correctness test path.

    Returns:
        dQueryIndex, dKeyIndex, dWeights, loss
    """
    d_rope = _validate_dense_inputs(
        query, key, query_index, key_index, weights,
        softmax_max, softmax_sum, layout, sparse_mode,
        query_rope=query_rope, key_rope=key_rope,
        pre_tokens=pre_tokens, next_tokens=next_tokens,
    )
    query_rope_arg = query if query_rope is None else query_rope
    key_rope_arg = key if key_rope is None else key_rope
    act_q = _default_actual_seq(actual_seq_qlen, query.shape[1], query)
    act_k = _default_actual_seq(actual_seq_klen, key.shape[1], key)
    if (softmax_max_index is None) != (softmax_sum_index is None):
        raise ValueError("softmax_max_index and softmax_sum_index must be both provided or both omitted")
    if softmax_max_index is not None:
        if softmax_max_index.dtype != ms.float32 or softmax_sum_index.dtype != ms.float32:
            raise TypeError("softmax_max_index and softmax_sum_index must be float32")
        B, S1 = query.shape[0], query.shape[1]
        max_index = _reshape_dense_index_stat(softmax_max_index, B, S1, "softmax_max_index")
        sum_index = _reshape_dense_index_stat(softmax_sum_index, B, S1, "softmax_sum_index")
        return _dense_loss_backward_with_index_core(
            query.contiguous(), key.contiguous(),
            query_index.contiguous(), key_index.contiguous(), weights.contiguous(),
            softmax_max.contiguous(), softmax_sum.contiguous(),
            max_index, sum_index,
            query_rope_arg.contiguous(), key_rope_arg.contiguous(),
            act_q, act_k, scale_value, d_rope,
        )
    return _dense_loss_backward_core(
        query.contiguous(), key.contiguous(),
        query_rope_arg.contiguous(), key_rope_arg.contiguous(),
        query_index.contiguous(), key_index.contiguous(), weights.contiguous(),
        softmax_max.contiguous(), softmax_sum.contiguous(),
        act_q, act_k, scale_value, d_rope,
    )


def npu_dense_lightning_indexer_softmax_lse_triton(
    query_index: ms.Tensor,
    key_index: ms.Tensor,
    weights: ms.Tensor,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout: str = "BSND",
    sparse_mode: int = 3,
    pre_tokens: int = INT64_MAX,
    next_tokens: int = INT64_MAX,
):
    """Official-style dense lightning-indexer softmax LSE helper.

    Returns:
        softmax_max_index, softmax_sum_index with shape [B, 1, S1]
    """
    _validate_dense_index_inputs(
        query_index, key_index, weights,
        layout, sparse_mode,
        pre_tokens=pre_tokens, next_tokens=next_tokens,
    )
    act_q = _default_actual_seq(actual_seq_qlen, query_index.shape[1], query_index)
    act_k = _default_actual_seq(actual_seq_klen, key_index.shape[1], key_index)
    max_index, sum_index = _dense_index_stats_core(
        query_index.contiguous(), key_index.contiguous(), weights.contiguous(),
        act_q, act_k,
    )
    B, S1 = query_index.shape[0], query_index.shape[1]
    return max_index.reshape((B, 1, S1)), sum_index.reshape((B, 1, S1))


def dense_lightning_indexer_softmax_lse_triton(
    query_index: ms.Tensor,
    key_index: ms.Tensor,
    weights: ms.Tensor,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout: str = "BSND",
    sparse_mode: int = 3,
    pre_tokens: int = INT64_MAX,
    next_tokens: int = INT64_MAX,
):
    return npu_dense_lightning_indexer_softmax_lse_triton(
        query_index, key_index, weights,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_klen=actual_seq_klen,
        layout=layout, sparse_mode=sparse_mode,
        pre_tokens=pre_tokens, next_tokens=next_tokens,
    )


def dense_lightning_indexer_grad_kl_loss_triton(
    query: ms.Tensor,
    key: ms.Tensor,
    query_index: ms.Tensor,
    key_index: ms.Tensor,
    weights: ms.Tensor,
    softmax_max: ms.Tensor,
    softmax_sum: ms.Tensor,
    softmax_max_index: ms.Tensor,
    softmax_sum_index: ms.Tensor,
    scale_value: float,
    query_rope=None,
    key_rope=None,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout: str = "BSND",
    sparse_mode: int = 3,
    pre_tokens: int = INT64_MAX,
    next_tokens: int = INT64_MAX,
):
    """Official-style dense lightning-indexer grad KL loss wrapper."""
    return dense_loss_backward_triton(
        query, key, query_index, key_index, weights,
        softmax_max, softmax_sum,
        softmax_max_index, softmax_sum_index,
        scale_value,
        query_rope=query_rope, key_rope=key_rope,
        actual_seq_qlen=actual_seq_qlen, actual_seq_klen=actual_seq_klen,
        layout=layout, sparse_mode=sparse_mode,
        pre_tokens=pre_tokens, next_tokens=next_tokens,
    )


def npu_dense_lightning_indexer_grad_kl_loss_triton(
    query: ms.Tensor,
    key: ms.Tensor,
    query_index: ms.Tensor,
    key_index: ms.Tensor,
    weights: ms.Tensor,
    softmax_max: ms.Tensor,
    softmax_sum: ms.Tensor,
    softmax_max_index: ms.Tensor,
    softmax_sum_index: ms.Tensor,
    scale_value: float,
    query_rope=None,
    key_rope=None,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout: str = "BSND",
    sparse_mode: int = 3,
    pre_tokens: int = INT64_MAX,
    next_tokens: int = INT64_MAX,
):
    return dense_lightning_indexer_grad_kl_loss_triton(
        query, key, query_index, key_index, weights,
        softmax_max, softmax_sum,
        softmax_max_index, softmax_sum_index,
        scale_value,
        query_rope=query_rope, key_rope=key_rope,
        actual_seq_qlen=actual_seq_qlen, actual_seq_klen=actual_seq_klen,
        layout=layout, sparse_mode=sparse_mode,
        pre_tokens=pre_tokens, next_tokens=next_tokens,
    )


class DenseLightningIndexerSoftmaxLseTriton(ms.nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, query_index, key_index, weight,
                  actual_seq_qlen=None, actual_seq_klen=None,
                  layout="BSND", sparse_mode=3,
                  pre_tokens=INT64_MAX, next_tokens=INT64_MAX):
        return npu_dense_lightning_indexer_softmax_lse_triton(
            query_index, key_index, weight,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen,
            layout=layout,
            sparse_mode=sparse_mode,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
        )


class DenseLossBackwardTriton(ms.nn.Cell):
    def __init__(self, scale_value=1.0, layout="BSND", sparse_mode=3,
                 pre_tokens=INT64_MAX, next_tokens=INT64_MAX):
        super().__init__()
        self.scale_value = scale_value
        self.layout = layout
        self.sparse_mode = sparse_mode
        self.pre_tokens = pre_tokens
        self.next_tokens = next_tokens

    def construct(self, query, key, query_index, key_index, weights,
                  softmax_max, softmax_sum,
                  softmax_max_index=None, softmax_sum_index=None,
                  scale_value=None,
                  query_rope=None, key_rope=None,
                  actual_seq_qlen=None, actual_seq_klen=None):
        effective_scale = self.scale_value if scale_value is None else scale_value
        return dense_loss_backward_triton(
            query, key, query_index, key_index, weights,
            softmax_max, softmax_sum,
            softmax_max_index, softmax_sum_index,
            effective_scale,
            query_rope=query_rope, key_rope=key_rope,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen,
            layout=self.layout,
            sparse_mode=self.sparse_mode,
            pre_tokens=self.pre_tokens,
            next_tokens=self.next_tokens,
        )


class DenseLightningIndexerGradKLLossTriton(ms.nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, query, key, query_index, key_index, weights,
                  softmax_max, softmax_sum,
                  softmax_max_index, softmax_sum_index,
                  scale_value=1.0,
                  query_rope=None, key_rope=None,
                  actual_seq_qlen=None, actual_seq_klen=None,
                  layout="BSND", sparse_mode=3,
                  pre_tokens=INT64_MAX, next_tokens=INT64_MAX):
        return dense_loss_backward_triton(
            query, key, query_index, key_index, weights,
            softmax_max, softmax_sum,
            softmax_max_index, softmax_sum_index,
            scale_value,
            query_rope=query_rope, key_rope=key_rope,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen,
            layout=layout,
            sparse_mode=sparse_mode,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
        )
