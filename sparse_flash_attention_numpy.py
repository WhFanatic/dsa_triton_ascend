"""NumPy golden reference for sparse_flash_attention (BSND, MLA-absorb).

Host-runnable ground truth, ported from the CANN golden
ops-transformer/attention/sparse_flash_attention/tests/pytest/
sparse_flash_attention_golden.py (`_t_increattention_bnsd` + `gatherKV`).

Semantics (per (b, s1), N2=1 / MQA, g = N1 query heads share one gathered KV):
  q_full[h] = concat(q_nope[b,s1,h], q_rope[b,s1,h])      -> [D+Dr]
  k_full[p] = concat(k_nope[b,p,0],  k_rope[b,p,0])        -> [D+Dr]
  score[h,p] = (q_nope·k_nope + q_rope·k_rope) * scale
  out[h]     = softmax_p(score[h,:]) @ k_nope[b, p, 0, :]   -> [D]
value is taken as key[..,:D] (K and V share the compressed latent c_kv); the
passed `value` tensor is IGNORED, matching CANN attention_mode=2.

sparse_mode: 0 = full (threshold = act_k); 3 = rightDownCausal
(threshold = act_k - act_q + s1 + 1; rows with s1 < act_q - act_k output 0).
"""
import math
import numpy as np

# bf16 sentinel for the `dtype` arg (numpy has no native bfloat16). Rounding is
# done with a pure-numpy round-to-nearest-even so the golden stays self-contained.
BF16 = "bfloat16"


def _round_bf16(x):
    """fp32 -> bf16 -> fp32 via round-to-nearest-even (truncate low 16 mantissa
    bits). Inputs are finite attention probs/outputs, so NaN/inf edge handling
    is unnecessary."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    bias = ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((u + bias) & np.uint32(0xFFFF0000)).view(np.float32)


INT64_MAX = 9223372036854775807


def _gather_positions(sparse_idx_row, sparse_block_size, threshold, act_k):
    """Expand sparse block ids into token positions; mirrors CANN gatherKV.

    front-valid / back=-1 contract: a -1 entry terminates the scan (break).
    """
    sparse_blockcount = sparse_idx_row.shape[0]
    valid_count = min(sparse_blockcount, math.ceil(threshold / sparse_block_size))
    positions = []
    for i in range(valid_count):
        sid = int(sparse_idx_row[i])
        if sid == -1:
            break
        begin = sid * sparse_block_size
        end = begin + sparse_block_size
        if end > act_k:
            end = act_k
        if begin >= threshold:
            continue
        if end <= threshold:
            positions.extend(range(begin, end))
        else:
            positions.extend(range(begin, threshold))
    return positions


def sparse_flash_attention_golden_bsnd(
    q_nope, k_nope, value, sparse_indices,
    q_rope, k_rope,
    scale_value,
    act_q, act_k,
    sparse_block_size=1,
    sparse_mode=3,
    return_softmax_lse=True,
    dtype=np.float16,
):
    """Reference SFA forward on BSND inputs (all numpy, fp32 accumulation).

    Args:
        q_nope: [B, S1, N1, D]            (D in 128/256/512)
        k_nope: [B, S2, 1, D]
        value:  [B, S2, 1, D]             IGNORED (value = k_nope in MLA-absorb)
        sparse_indices: [B, S1, 1, sparse_count] int32 (block ids, -1 = invalid)
        q_rope: [B, S1, N1, Dr]           (Dr = 64)
        k_rope: [B, S2, 1, Dr]
        scale_value: softmax scale (1/sqrt(d_k))
        act_q, act_k: [B] int, per-batch valid query/key lengths
        sparse_block_size: 1 (token-wise) or 2^n in [1,128] (block-wise)
        sparse_mode: 0 (full) or 3 (rightDownCausal)
        return_softmax_lse: also return softmax_max / softmax_sum
        dtype: np.float16 / BF16 ("bfloat16") / np.float32 — probs and output
               rounded to this dtype before bmm2 (matches CANN golden)

    Returns:
        out:         [B, S1, N1, D] (same dtype-rounded as inputs)
        softmax_max: [B, 1, S1, N1] fp32
        softmax_sum: [B, 1, S1, N1] fp32
    """
    B, S1, N1, D = q_nope.shape
    S2 = k_nope.shape[1]
    Dr = q_rope.shape[-1]

    qn = q_nope.astype(np.float32)
    kn = k_nope.astype(np.float32)
    qr = q_rope.astype(np.float32)
    kr = k_rope.astype(np.float32)

    out = np.zeros((B, S1, N1, D), dtype=np.float32)
    softmax_max = np.zeros((B, 1, S1, N1), dtype=np.float32)
    softmax_sum = np.zeros((B, 1, S1, N1), dtype=np.float32)

    # softmax probs cast to in/out dtype before bmm2 (matches CANN golden,
    # which rounds fp16 AND bf16 probs before the P@V matmul).
    def _round(x):
        if dtype == np.float16:
            return x.astype(np.float16).astype(np.float32)
        if dtype == BF16:
            return _round_bf16(x)
        return x  # fp32: keep full precision

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
            v_sp = v_all[pos]                                           # [P, D]

            for h in range(N1):
                q_full = np.concatenate([qn[b, s1, h], qr[b, s1, h]], axis=-1)  # [D+Dr]
                scores = (k_sp @ q_full) * scale_value                          # [P]
                x_max = scores.max()
                exp = np.exp(scores - x_max)
                x_sum = exp.sum()
                probs = _round(exp / x_sum)
                out[b, s1, h, :] = probs @ v_sp
                if return_softmax_lse:
                    softmax_max[b, 0, s1, h] = x_max
                    softmax_sum[b, 0, s1, h] = x_sum

    # round final output to the in/out dtype (returned as fp32 for comparison)
    if dtype == np.float16:
        out = out.astype(np.float16).astype(np.float32)
    elif dtype == BF16:
        out = _round_bf16(out)
    return out, softmax_max, softmax_sum
