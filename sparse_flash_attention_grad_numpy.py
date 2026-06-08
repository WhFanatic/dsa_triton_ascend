"""NumPy golden reference for sparse_flash_attention_grad (BSND, MLA-absorb).

Host-runnable ground truth for the backward pass, paired with the forward golden
in sparse_flash_attention_numpy.py. Standard flash-attention backward: the
forward softmax stats (softmax_max / softmax_sum) are reused to reconstruct the
attention probabilities P, so no online-softmax recompute is needed.

Per (b, s1), N2=1 / MQA, g = N1 query heads share one gathered KV:
  q_full[h] = concat(q_nope[b,s1,h], q_rope[b,s1,h])      -> [D+Dr]
  k_full[p] = concat(k_nope[b,p,0],  k_rope[b,p,0])        -> [D+Dr]
  score[h,p] = (q_nope·k_nope + q_rope·k_rope) * scale
  P[h,p]     = exp(score[h,p] - softmax_max[h]) / softmax_sum[h]
  dP[h,p]    = dO[h] · v[p]            (v[p] = k_nope[p]; MLA-absorb value=key[:D])
  delta[h]   = dO[h] · O[h]           (== sum_p P[h,p]·dP[h,p])
  dS[h,p]    = P[h,p] · (dP[h,p] - delta[h]) · scale

Gradients (dq/dq_rope are per-(b,s1); dk/dk_rope/dv scatter-add across s1 rows
that select the same KV token):
  d_query[h]      = sum_p dS[h,p] · k_nope[p]
  d_query_rope[h] = sum_p dS[h,p] · k_rope[p]
  d_key[u]       += sum_h dS[h,p->u] · q_nope[h]
  d_key_rope[u]  += sum_h dS[h,p->u] · q_rope[h]
  d_value[u]     += sum_h P[h,p->u]  · dO[h]      (no scale; P@dO path)

d_key (QK path) and d_value (P@V path) are returned SEPARATELY, matching CANN's
5-output contract: key and value are independent inputs even though value=key in
MLA-absorb; a caller that shares the tensor sums the two gradients itself.
"""
import numpy as np

from sparse_flash_attention_numpy import BF16, _round_bf16, _gather_positions


def sparse_flash_attention_grad_golden_bsnd(
    q_nope, k_nope, value, sparse_indices,
    d_out, out, softmax_max, softmax_sum,
    q_rope, k_rope,
    scale_value,
    act_q, act_k,
    sparse_block_size=1,
    sparse_mode=3,
    dtype=np.float16,
):
    """Reference SFA backward on BSND inputs (all numpy, fp32 accumulation).

    Args:
        q_nope: [B, S1, N1, D]            (D in 128/256/512)
        k_nope: [B, S2, 1, D]
        value:  [B, S2, 1, D]             IGNORED (value = k_nope in MLA-absorb)
        sparse_indices: [B, S1, 1, sparse_count] int32 (block ids, -1 = invalid)
        d_out:  [B, S1, N1, D]            gradient of attention output
        out:    [B, S1, N1, D]            forward attention output
        softmax_max: [B, 1, S1, N1] fp32  forward online-softmax max
        softmax_sum: [B, 1, S1, N1] fp32  forward online-softmax sum
        q_rope: [B, S1, N1, Dr]           (Dr = 64)
        k_rope: [B, S2, 1, Dr]
        scale_value: softmax scale (1/sqrt(d_k))
        act_q, act_k: [B] int, per-batch valid query/key lengths
        sparse_block_size: 1 (token-wise) or 2^n in [1,128] (block-wise)
        sparse_mode: 0 (full) or 3 (rightDownCausal)
        dtype: np.float16 / BF16 / np.float32 — outputs rounded to this dtype

    Returns:
        d_query:      [B, S1, N1, D]
        d_key:        [B, S2, 1, D]
        d_value:      [B, S2, 1, D]
        d_query_rope: [B, S1, N1, Dr]
        d_key_rope:   [B, S2, 1, Dr]
    """
    B, S1, N1, D = q_nope.shape
    S2 = k_nope.shape[1]
    Dr = q_rope.shape[-1]

    qn = q_nope.astype(np.float32)
    kn = k_nope.astype(np.float32)
    qr = q_rope.astype(np.float32)
    kr = k_rope.astype(np.float32)
    do = d_out.astype(np.float32)
    o = out.astype(np.float32)
    smax = softmax_max.astype(np.float32)
    ssum = softmax_sum.astype(np.float32)

    d_query = np.zeros((B, S1, N1, D), dtype=np.float32)
    d_query_rope = np.zeros((B, S1, N1, Dr), dtype=np.float32)
    d_key = np.zeros((B, S2, 1, D), dtype=np.float32)
    d_key_rope = np.zeros((B, S2, 1, Dr), dtype=np.float32)
    d_value = np.zeros((B, S2, 1, D), dtype=np.float32)

    def _round(x):
        if dtype == np.float16:
            return x.astype(np.float16).astype(np.float32)
        if dtype == BF16:
            return _round_bf16(x)
        return x

    for b in range(B):
        aq = int(act_q[b])
        ak = int(act_k[b])
        k_all = kn[b, :, 0, :]          # [S2, D]
        kr_all = kr[b, :, 0, :]         # [S2, Dr]
        v_all = k_all                   # MLA-absorb: value = key[..,:D]
        for s1 in range(aq):
            # rightDownCausal: leading rows (query longer than key) see nothing.
            if sparse_mode != 0 and s1 < aq - ak:
                continue
            if sparse_mode == 0:
                threshold = ak
            else:
                threshold = ak - aq + s1 + 1
            if threshold <= 0:
                continue

            pos = _gather_positions(sparse_indices[b, s1, 0, :],
                                    sparse_block_size, threshold, ak)
            if len(pos) == 0:
                continue
            pos = np.asarray(pos, dtype=np.int64)

            k_sp = np.concatenate([k_all[pos], kr_all[pos]], axis=-1)   # [P, D+Dr]
            kn_sp = k_all[pos]                                          # [P, D]
            krp_sp = kr_all[pos]                                       # [P, Dr]
            v_sp = v_all[pos]                                          # [P, D]

            for h in range(N1):
                q_full = np.concatenate([qn[b, s1, h], qr[b, s1, h]], axis=-1)  # [D+Dr]
                scores = (k_sp @ q_full) * scale_value                          # [P]
                # reuse forward stats to rebuild P (no online-softmax recompute)
                P = np.exp(scores - smax[b, 0, s1, h]) / ssum[b, 0, s1, h]      # [P]
                dPv = v_sp @ do[b, s1, h]                                       # [P] dO·v
                delta = float(do[b, s1, h] @ o[b, s1, h])                       # rowsum(dO*O)
                dS = P * (dPv - delta) * scale_value                            # [P]

                # dq/dq_rope: per-(b,s1,h)
                d_query[b, s1, h] += dS @ kn_sp                                 # [D]
                d_query_rope[b, s1, h] += dS @ krp_sp                           # [Dr]

                # dk/dk_rope/dv: scatter-add over the gathered positions
                np.add.at(d_key[b, :, 0, :], pos, np.outer(dS, qn[b, s1, h]))
                np.add.at(d_key_rope[b, :, 0, :], pos, np.outer(dS, qr[b, s1, h]))
                np.add.at(d_value[b, :, 0, :], pos, np.outer(P, do[b, s1, h]))

    return (_round(d_query), _round(d_key), _round(d_value),
            _round(d_query_rope), _round(d_key_rope))
