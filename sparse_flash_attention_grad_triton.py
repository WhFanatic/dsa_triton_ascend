"""Triton-ascend implementation of sparse_flash_attention_grad (SFA backward).

Interface aligned with CANN aclnnSparseFlashAttentionGrad (op_host/
sparse_flash_attention_grad_def.cpp): 12 inputs / 7 attrs / 5 outputs, MLA-absorb
(attention_mode=2 semantics), MQA (N2=1). Pairs with the forward
sparse_flash_attention_triton.py; reuses the forward softmax_max / softmax_sum to
rebuild the attention probabilities P (no online-softmax recompute).

Math (per (b,s1), N2=1 / MQA, g = N1 heads share one gathered KV):
  score[h,p] = (q_nope·k_nope + q_rope·k_rope) * scale
  P[h,p]     = exp(score[h,p] - softmax_max[h]) / softmax_sum[h]
  delta[h]   = sum_d dO[h,d]·O[h,d]                       (rowsum(dO*O))
  dS[h,p]    = P[h,p] · (dO[h]·k_nope[p] - delta[h]) · scale
  d_query[h]      = sum_p dS[h,p] · k_nope[p]
  d_query_rope[h] = sum_p dS[h,p] · k_rope[p]
  d_key[u]       += sum_h dS[h,p->u] · q_nope[h]          (scatter-add over s1)
  d_key_rope[u]  += sum_h dS[h,p->u] · q_rope[h]          (scatter-add over s1)
  d_value[u]     += sum_h P[h,p->u]  · dO[h]              (scatter-add, P@dO path)

d_key (QK path) and d_value (P@dO path) are returned SEPARATELY, matching CANN's
5-output contract (key/value are independent inputs even though value=key in
MLA-absorb). A caller that shares the tensor sums the two itself.

Layout: BSND (TND/PA normalized to BSND on host, PyNative-only — same as forward).
sparse_mode 0 (full) / 3 (rightDownCausal). sparse_block_size 1 or 2^n in [1,128]
(block-wise pre-expanded to token indices on host). pre/next_tokens only default.
deterministic: only False (the scatter-add path is order-nondeterministic by
nature; True would require a serialized reduction not implemented here).

--------------------------------------------------------------------------------
mindformers integration (P.Morph bprop) — see CLAUDE.md 接入点 table
--------------------------------------------------------------------------------
The forward call site mindformers .../transformer/dsa/dsa_attention.py wraps the
forward in `P.Morph(self._sparse_flash_attention_forward, self.sfa_infer_shape,
...)` WITHOUT a bprop, relying on autodiff through the inner
ops.sparse_flash_attention (which lowers to CANN SparseFlashAttentionGrad).
After swapping the forward to the triton `_ms_pyfunc` op, that autodiff chain is
broken, so the Morph MUST be given an explicit bprop that calls this op. Paste:

    # in DSAttention.__init__, replacing the P.Morph(...) construction:
    self.sparse_flash_attention = P.Morph(
        self._sparse_flash_attention_forward,
        self.sfa_infer_shape,
        lambda *args: (args[0], mstype.float32, mstype.float32),
    ).add_prim_attr("self_define_shard", True)
    self.sparse_flash_attention.bprop = self._sfa_bprop   # attach backward

    # forward signature is _sparse_flash_attention_forward(self, q, k, v,
    #   topk_indices, query_rope, key_rope, actual_seq_qlen, actual_seq_kvlen);
    # Morph bprop receives (*inputs, out, dout) where out/dout are 3-tuples
    # (attention_out, softmax_max, softmax_sum). Only d(attention_out) feeds back.
    def _sfa_bprop(self, q, k, v, topk_indices, query_rope, key_rope,
                   actual_seq_qlen, actual_seq_kvlen, out, dout):
        attention_out, softmax_max, softmax_sum = out
        d_attention_out = dout[0]
        dq, dk, dv, dqr, dkr = sparse_flash_attention_grad_triton(
            q, k, v, topk_indices, d_attention_out, attention_out,
            softmax_max, softmax_sum, self.softmax_scale,
            query_rope=query_rope, key_rope=key_rope,
            actual_seq_lengths_query=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_kvlen,
            layout=self.input_layout, sparse_mode=<forward sparse_mode>)
        # k and v share the compressed latent c_kv -> the caller's split feeds
        # both back into the same key tensor; sum dk+dv for the merged k grad.
        d_qkv_rope_pad = self.zeros_like(query_rope)  # if rope grads are unused
        return (dq, dk + dv, self.zeros_like(v), dqr, dkr,
                None if actual_seq_qlen is None else self.zeros_like(actual_seq_qlen),
                None if actual_seq_kvlen is None else self.zeros_like(actual_seq_kvlen))
    # NB: align the returned tuple arity/order with the Morph forward inputs;
    # q_rope/k_rope grads (dqr/dkr) route to the upstream concat that built q/k.
--------------------------------------------------------------------------------
"""
import triton
import triton.language as tl
import triton.backends.ascend.runtime

import mindspore as ms
from mindspore import ops

# host-side layout helpers shared with the forward (same normalization rules)
from sparse_flash_attention_triton import (
    _default_actual_seq_lens, _tnd_cumsum_to_per_batch,
    _tnd_to_bsnd, _bsnd_to_tnd, _expand_block_indices,
)

INT64_MAX = 9223372036854775807

_VALID_D = (128, 256, 512)
_D_ROPE = 64
_VALID_N1 = (1, 2, 4, 8, 16, 32, 64, 128)


def _next_pow2(x):
    # Ascend 要求 kernel grid 每维都是 2 的幂, 否则分核映射出错 -> aicore trap。
    return 1 << (x - 1).bit_length() if x > 1 else 1


def _patch_triton_ascend_mindspore_dtype_bytes():
    """修补 triton-ascend autotuner 的 dtype 字节数查询 (见 sparse_flash_attention_triton)。"""
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
    for byte_size, names in (
        (1, ("int8", "uint8", "bool_")),
        (2, ("float16", "bfloat16", "int16", "uint16")),
        (4, ("float32", "int32", "uint32")),
        (8, ("float64", "int64", "uint64")),
    ):
        for name in names:
            dt = getattr(ms, name, None)
            if dt is not None:
                dtype_bytes[dt] = byte_size

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

def _prune_configs(configs, named_args, **kwargs):
    """autotune config 过滤 (UB 上限 + grid pow2 + grid 总数上限)。

    SFA backward kernel: 无 BLOCK_DV 维度，D 和 D_ROPE 通过 BLOCK_D 分块。
    UB 占用项：
      - q_nope/q_rope/do_tile/o_tile: 4 × BLOCK_G × max(D, D_ROPE) × 2 (fp16/bf16)
      - k_full/kr_full: 2 × BLOCK_K × max(D, D_ROPE) × 2
      - scores/P/dS: 3 × BLOCK_G × BLOCK_K × 4 (fp32)
      - acc_dq/acc_dqr: 2 × BLOCK_G × max(D, D_ROPE) × 4 (fp32)
      - q_sub/do_sub (内层 D 循环): 2 × BLOCK_G × BLOCK_D × 2
      - dk_contrib/dv_contrib (内层 D 循环): 2 × BLOCK_K × BLOCK_D × 4
      - qr_sub/dkr_contrib (D 循环结束后): BLOCK_G × D_ROPE × 2 + BLOCK_K × D_ROPE × 4
      - 转置缓冲: BLOCK_K × D_MAX × 2 (仅 k_full)
    """
    _UB_LIMIT_BYTES = 192 * 1024
    _GRID_LIMIT = 131072

    def _get(name):
        if name in named_args:
            return named_args[name]
        return kwargs.get(name, None)

    N1 = _get("N1")
    BS1 = _get("B_S1")
    D = _get("D")
    D_ROPE = _get("D_ROPE")
    # 取 D 和 D_ROPE 的最大值用于统一估算
    D_MAX = max(D, D_ROPE) if None not in (D, D_ROPE) else (D or D_ROPE or 512)

    def _estimate_ub_bytes(block_g, block_k, block_d):
        if None in (block_g, block_k, block_d):
            return 0

        # 按 kernel 实际执行阶段分阶段估算峰值 UB；不同阶段使用的缓冲不叠加。
        # Phase 1: 加载 resident tiles + 计算 scores/P/dS/acc_dq/acc_dqr（D-loop 之前）。
        # Phase 2: D-tile 循环内部（q_sub/do_sub/dk_contrib/dv_contrib 与 P/dS 转置缓冲共存）。
        # Phase 3: D-loop 结束后 scatter dkr（qr_sub/dkr_contrib）。

        # Phase 1
        q_nope = block_g * D_MAX * 2          # fp16/bf16
        q_rope = block_g * D_ROPE * 2 if D_ROPE else 0
        do_tile = block_g * D_MAX * 2
        o_tile = block_g * D_MAX * 2
        k_full = block_k * D_MAX * 2          # gather 结果
        kr_full = block_k * D_ROPE * 2 if D_ROPE else 0

        scores = block_g * block_k * 4        # fp32
        P = block_g * block_k * 4
        dS = block_g * block_k * 4
        dS_bf16 = block_g * block_k * 2       # cast before dot

        acc_dq = block_g * D_MAX * 4          # fp32 accumulator
        acc_dqr = block_g * D_ROPE * 4 if D_ROPE else 0

        trans_k = block_k * D_MAX * 2         # tl.trans(k_full)

        phase1 = (
            q_nope + q_rope + do_tile + o_tile + k_full + kr_full
            + scores + P + dS + dS_bf16
            + acc_dq + acc_dqr + trans_k
        )

        # Phase 2: P/dS/dS_bf16 仍存活，并生成 P_bf16/P_t_cast/dS_t_cast
        P_bf16 = block_g * block_k * 2
        P_t_cast = block_g * block_k * 2
        dS_t_cast = block_g * block_k * 2

        q_sub = block_g * block_d * 2
        do_sub = block_g * block_d * 2
        dk_contrib = block_k * block_d * 4    # fp32
        dv_contrib = block_k * block_d * 4

        phase2 = (
            scores + P + dS + dS_bf16 + P_bf16 + P_t_cast + dS_t_cast
            + acc_dq + acc_dqr
            + q_sub + do_sub + dk_contrib + dv_contrib
        )

        # Phase 3: dkr scatter，qr_sub 复用 q_rope 数据但 worst-case 重新 load
        qr_sub = block_g * D_ROPE * 2 if D_ROPE else 0
        dkr_contrib = block_k * D_ROPE * 4 if D_ROPE else 0

        phase3 = (
            scores + P + dS + dS_bf16 + P_bf16 + P_t_cast + dS_t_cast
            + acc_dq + acc_dqr
            + qr_sub + dkr_contrib
        )

        total = max(phase1, phase2, phase3)

        # 10% 余量给编译器临时缓冲；分阶段估算本身已比全加保守
        return int(total * 1.1)

    topK = _get("topK")

    kept = []
    for c in configs:
        bg = c.kwargs.get("BLOCK_G")
        bk = c.kwargs.get("BLOCK_K")
        bd = c.kwargs.get("BLOCK_D")

        # BLOCK_D > D 浪费：内层 D-loop 只有 1 个 tile，多分配的 BLOCK_D 不会被使用，
        # 但 q_sub/do_sub/dk_contrib/dv_contrib 仍按 BLOCK_D 分配 UB 与 dot 算力。
        if D and bd > D:
            continue

        # BLOCK_K > pow2(topK) 没意义：多出的 lane 全是 tok==-1 mask 空转，
        # 不会带来更多并行度，只浪费 UB 和 dot 算力。
        if topK and bk > _next_pow2(topK):
            continue

        if _estimate_ub_bytes(bg, bk, bd) > _UB_LIMIT_BYTES:
            continue

        # Grid 限制：不基于 BLOCK_G > N1 剪枝（kernel 内部有 g_valid 掩码处理）
        if None not in (BS1, N1) and bg:
            grid0 = _next_pow2(BS1)
            grid1 = _next_pow2((N1 + bg - 1) // bg)
            if grid0 * grid1 > _GRID_LIMIT:
                continue
        kept.append(c)

    if not kept:
        print('Warning: all autotune params pruned')
        kept = [min(configs, key=lambda c: _estimate_ub_bytes(
            c.kwargs.get("BLOCK_G"), c.kwargs.get("BLOCK_K"),
            c.kwargs.get("BLOCK_D")))]
    return kept

def _grad_configs():
    # base 已去重；mb/uf 锁到 (True, True) — 这两个旋钮的最优值在已落地的 autotune
    # 历史里高度一致 (multibuffer=True 几乎总赢，unit_flag=True 同步原语更省)，全展开
    # 会让 autotune 多跑 4 倍但常常选回同一组合。需要重新 sweep 时改这一行即可。
    base = [
        {"BLOCK_G": 8,  "BLOCK_K": 16,  "BLOCK_D": 128},
        {"BLOCK_G": 8,  "BLOCK_K": 16,  "BLOCK_D": 256},
        {"BLOCK_G": 8,  "BLOCK_K": 16,  "BLOCK_D": 512},
        {"BLOCK_G": 16, "BLOCK_K": 16,  "BLOCK_D": 128},
        {"BLOCK_G": 16, "BLOCK_K": 16,  "BLOCK_D": 256},
        {"BLOCK_G": 16, "BLOCK_K": 16,  "BLOCK_D": 512},
        {"BLOCK_G": 4,  "BLOCK_K": 16,  "BLOCK_D": 128},
        {"BLOCK_G": 4,  "BLOCK_K": 16,  "BLOCK_D": 256},
        {"BLOCK_G": 4,  "BLOCK_K": 16,  "BLOCK_D": 512},
        {"BLOCK_G": 16, "BLOCK_K": 32,  "BLOCK_D": 128},
        {"BLOCK_G": 16, "BLOCK_K": 32,  "BLOCK_D": 256},
        {"BLOCK_G": 16, "BLOCK_K": 64,  "BLOCK_D": 128},
        {"BLOCK_G": 16, "BLOCK_K": 128, "BLOCK_D": 128},
        {"BLOCK_G": 16, "BLOCK_K": 256, "BLOCK_D": 128},
        {"BLOCK_G": 16, "BLOCK_K": 256, "BLOCK_D": 512},
        {"BLOCK_G": 16, "BLOCK_K": 512, "BLOCK_D": 128},
        {"BLOCK_G": 16, "BLOCK_K": 512, "BLOCK_D": 256},
        {"BLOCK_G": 16, "BLOCK_K": 512, "BLOCK_D": 512},
        {"BLOCK_G": 32, "BLOCK_K": 256, "BLOCK_D": 128},
        {"BLOCK_G": 32, "BLOCK_K": 512, "BLOCK_D": 256},
        {"BLOCK_G": 8,  "BLOCK_K": 32,  "BLOCK_D": 256},
        {"BLOCK_G": 8,  "BLOCK_K": 64,  "BLOCK_D": 256},
        {"BLOCK_G": 8,  "BLOCK_K": 256, "BLOCK_D": 256},
        {"BLOCK_G": 8,  "BLOCK_K": 256, "BLOCK_D": 512},
        {"BLOCK_G": 8,  "BLOCK_K": 512, "BLOCK_D": 256},
        {"BLOCK_G": 8,  "BLOCK_K": 512, "BLOCK_D": 512},
        {"BLOCK_G": 4,  "BLOCK_K": 256, "BLOCK_D": 128},
        {"BLOCK_G": 4,  "BLOCK_K": 512, "BLOCK_D": 256},
    ]
    return [triton.Config({**c, "multibuffer": True, "unit_flag": True}) for c in base]


@triton.autotune(
    configs=_grad_configs(),
    key=["B_S1", "N1", "D", "D_ROPE"],
    reset_to_zero=["dk_ptr", "dkr_ptr", "dv_ptr"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _sfa_grad_kernel(
    q_ptr, qr_ptr,                       # query[B,S1,N1,D], query_rope[B,S1,N1,Dr]
    k_ptr, kr_ptr, v_ptr,                # key/key_rope/value, all [B,S2,1,*] (v aliases k)
    sparse_ptr,                          # token indices [B,S1,1,topK] int32 (block pre-expanded)
    do_ptr, o_ptr,                       # d_out[B,S1,N1,D], out[B,S1,N1,D]
    sm_max_ptr, sm_sum_ptr,              # forward softmax stats, flat (b*S1+s1)*N1 + g
    dq_ptr, dqr_ptr,                     # outputs: d_query, d_query_rope
    dk_ptr, dkr_ptr, dv_ptr,             # outputs (fp32 workspace): d_key, d_key_rope, d_value
    act_q_ptr, act_k_ptr,
    B_S1, S1, S2, N1, topK,
    D: tl.constexpr, D_ROPE: tl.constexpr,
    scale_value,
    sparse_mode: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """SFA backward over sparsely gathered KV (BSND, MQA / N2=1), single pass.

    Grid: (_next_pow2(B*S1), _next_pow2(cdiv(N1, BLOCK_G))), both pow2-padded.
    Each program owns one (b,s1) and BLOCK_G query heads.

    Per topK block: rebuild scores (q·k) -> P (from saved softmax stats) -> dPv
    (dO·v, v=k_nope) -> dS = P·(dPv - delta)·scale. Accumulate dq/dqr resident
    (no cross-program contention — each head row owned by one program); scatter-add
    dk/dkr (dS·q) and dv (P·dO) into fp32 workspaces (many s1 rows hit one KV token).

    No early-return (triton-ascend drops stores after early-return): inactive rows
    are folded into tok_valid so dS/P become 0 and all contributions vanish.
    """
    pid_bs1 = tl.program_id(0)
    pid_g = tl.program_id(1)

    bs1_in_range = pid_bs1 < B_S1
    pid_bs1 = tl.where(bs1_in_range, pid_bs1, 0)

    b = pid_bs1 // S1
    s1 = pid_bs1 % S1

    g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    g_valid = g_offs < N1

    act_q = tl.load(act_q_ptr + b)
    act_k = tl.load(act_k_ptr + b)

    if sparse_mode == 0:
        threshold = act_k
    else:
        threshold = act_k - act_q + s1 + 1
    row_active = bs1_in_range & (s1 < act_q) & (threshold > 0)

    d_offs_full = tl.arange(0, D)
    dr_offs_full = tl.arange(0, D_ROPE)

    q_base = (b * S1 + s1) * N1 * D
    qr_base = (b * S1 + s1) * N1 * D_ROPE
    o_base = (b * S1 + s1) * N1 * D
    k_base = b * S2 * D
    kr_base = b * S2 * D_ROPE
    v_base = b * S2 * D
    sp_base = (b * S1 + s1) * topK
    sm_base = (b * S1 + s1) * N1

    # resident per-head tiles (loaded once)
    q_nope = tl.load(q_ptr + q_base + g_offs[:, None] * D + d_offs_full[None, :],
                     mask=g_valid[:, None], other=0.0)
    q_rope = tl.load(qr_ptr + qr_base + g_offs[:, None] * D_ROPE + dr_offs_full[None, :],
                     mask=g_valid[:, None], other=0.0)
    do_tile = tl.load(do_ptr + o_base + g_offs[:, None] * D + d_offs_full[None, :],
                      mask=g_valid[:, None], other=0.0)
    o_tile = tl.load(o_ptr + o_base + g_offs[:, None] * D + d_offs_full[None, :],
                     mask=g_valid[:, None], other=0.0)

    sm_max = tl.load(sm_max_ptr + sm_base + g_offs, mask=g_valid, other=0.0)
    sm_sum = tl.load(sm_sum_ptr + sm_base + g_offs, mask=g_valid, other=1.0)

    # delta[g] = rowsum(dO * O)  (== sum_p P·dPv)
    delta = tl.sum(do_tile.to(tl.float32) * o_tile.to(tl.float32), axis=1)

    acc_dq = tl.zeros([BLOCK_G, D], dtype=tl.float32)
    acc_dqr = tl.zeros([BLOCK_G, D_ROPE], dtype=tl.float32)

    for blk_start in range(0, topK, BLOCK_K):
        blk_offs = blk_start + tl.arange(0, BLOCK_K)
        blk_in_count = blk_offs < topK
        tok = tl.load(sparse_ptr + sp_base + blk_offs, mask=blk_in_count, other=-1)
        tok_valid = blk_in_count & (tok != -1) & (tok < threshold) & (tok < act_k) & row_active

        k_full = tl.load(
            k_ptr + k_base + tok[:, None] * D + d_offs_full[None, :],
            mask=tok_valid[:, None], other=0.0)
        kr_full = tl.load(
            kr_ptr + kr_base + tok[:, None] * D_ROPE + dr_offs_full[None, :],
            mask=tok_valid[:, None], other=0.0)

        # scores[g,k] = (q_nope·k_nope + q_rope·k_rope)·scale
        scores = tl.dot(q_nope, tl.trans(k_full)).to(tl.float32)
        scores += tl.dot(q_rope, tl.trans(kr_full)).to(tl.float32)
        scores = scores * scale_value

        # P from saved stats; invalid tokens naturally become 0 via -inf scores
        scores = tl.where(tok_valid[None, :], scores, float('-inf'))
        P = tl.exp(scores - sm_max[:, None]) / sm_sum[:, None]

        # dPv[g,k] = dO·v, v = k_nope (MLA-absorb)
        dPv = tl.dot(do_tile, tl.trans(k_full)).to(tl.float32)
        dS = P * (dPv - delta[:, None]) * scale_value
        dS_bf16 = dS.to(k_full.dtype)

        # dq/dqr accumulate (owned, no atomics)
        acc_dq += tl.dot(dS_bf16, k_full).to(tl.float32)
        acc_dqr += tl.dot(dS_bf16, kr_full).to(tl.float32)

        # scatter dk/dv over D-tiles
        # pre-cast transposed tiles once per k-block to avoid repeated conversions
        P_bf16 = P.to(k_full.dtype)
        P_t_cast = tl.trans(P_bf16)
        dS_t_cast = tl.trans(dS_bf16)
        for d_start in range(0, D, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_offs < D
            q_sub = tl.load(
                q_ptr + q_base + g_offs[:, None] * D + d_offs[None, :],
                mask=g_valid[:, None] & d_valid[None, :], other=0.0)
            do_sub = tl.load(
                do_ptr + o_base + g_offs[:, None] * D + d_offs[None, :],
                mask=g_valid[:, None] & d_valid[None, :], other=0.0)
            dk_contrib = tl.dot(dS_t_cast, q_sub).to(tl.float32)
            dv_contrib = tl.dot(P_t_cast, do_sub).to(tl.float32)
            dk_offs = v_base + tok[:, None] * D + d_offs[None, :]
            tl.atomic_add(dk_ptr + dk_offs, dk_contrib,
                          mask=tok_valid[:, None] & d_valid[None, :])
            tl.atomic_add(dv_ptr + dk_offs, dv_contrib,
                          mask=tok_valid[:, None] & d_valid[None, :])

        # scatter dkr
        qr_sub = tl.load(
            qr_ptr + qr_base + g_offs[:, None] * D_ROPE + dr_offs_full[None, :],
            mask=g_valid[:, None], other=0.0)
        dkr_contrib = tl.dot(dS_t_cast, qr_sub).to(tl.float32)
        dkr_offs = kr_base + tok[:, None] * D_ROPE + dr_offs_full[None, :]
        tl.atomic_add(dkr_ptr + dkr_offs, dkr_contrib, mask=tok_valid[:, None])

    # store dq/dqr (per (b,s1,head-group), no contention)
    tl.store(dq_ptr + q_base + g_offs[:, None] * D + d_offs_full[None, :],
             acc_dq.to(dq_ptr.dtype.element_ty),
             mask=g_valid[:, None] & row_active)
    tl.store(dqr_ptr + qr_base + g_offs[:, None] * D_ROPE + dr_offs_full[None, :],
             acc_dqr.to(dqr_ptr.dtype.element_ty),
             mask=g_valid[:, None] & row_active)


# ---------------------------------------------------------------------------
# _ms_pyfunc core. Type annotations are required by _ms_pyfunc shape/dtype
# inference and must match between infer_func and core.
# ---------------------------------------------------------------------------
def _infer_sfa_grad(
    q_flat: ms.Tensor, qr_flat: ms.Tensor,
    k_flat: ms.Tensor, kr_flat: ms.Tensor, v_flat: ms.Tensor,
    sparse_flat: ms.Tensor,
    do_flat: ms.Tensor, o_flat: ms.Tensor,
    sm_max_flat: ms.Tensor, sm_sum_flat: ms.Tensor,
    dq_buf: ms.Tensor, dqr_buf: ms.Tensor,
    dk_buf: ms.Tensor, dkr_buf: ms.Tensor, dv_buf: ms.Tensor,
    act_q: ms.Tensor, act_k: ms.Tensor,
    B_S1: int, S1: int, S2: int, N1: int, topK: int,
    D: int, D_ROPE: int,
    scale_value: float,
    sparse_mode: int,
) -> tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
    return (ms.mint.empty_like(dq_buf), ms.mint.empty_like(dqr_buf),
            ms.mint.empty_like(dk_buf), ms.mint.empty_like(dkr_buf),
            ms.mint.empty_like(dv_buf))

import os
import pickle
from datetime import datetime

def _save_sfa_grad_inputs(
    q_flat, qr_flat, k_flat, kr_flat, v_flat,
    sparse_flat,
    do_flat, o_flat,
    sm_max_flat, sm_sum_flat,
    dq_buf, dqr_buf, dk_buf, dkr_buf, dv_buf,
    act_q, act_k,
    B_S1, S1, S2, N1, topK,
    D, D_ROPE, scale_value, sparse_mode,
    save_dir="/tmp/sfa_inputs"
):
    """保存 _sfa_grad_kernel 的所有输入到本地文件"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_path = os.path.join(save_dir, f"sfa_grad_inputs_{timestamp}.pkl")
    
    import torch
    import numpy as np
    import mindspore as ms
    
    def ms_tensor_to_torch(ms_t):
        """将 ms.Tensor 转为 torch.Tensor，兼容 bfloat16"""
        if not hasattr(ms_t, "dtype"):
            return ms_t  # 不是 ms.Tensor
        
        ms_dtype = ms_t.dtype
        shape = ms_t.shape
        
        # bfloat16 特殊处理：通过 uint16 字节视图中转
        if ms_dtype == ms.bfloat16:
            uint16_tensor = ms_t.view(ms.uint16)
            np_uint16 = uint16_tensor.asnumpy()
            torch_uint16 = torch.from_numpy(np_uint16)
            return torch_uint16.view(torch.bfloat16).reshape(shape)
        
        # 其他类型正常转换
        try:
            np_arr = ms_t.asnumpy()
            dtype_map = {
                ms.float16: torch.float16,
                ms.float32: torch.float32,
                ms.float64: torch.float64,
                ms.int8: torch.int8,
                ms.int16: torch.int16,
                ms.int32: torch.int32,
                ms.int64: torch.int64,
                ms.uint8: torch.uint8,
                ms.bool_: torch.bool,
            }
            torch_dtype = dtype_map.get(ms_dtype, torch.float32)
            return torch.from_numpy(np_arr).to(torch_dtype)
        except Exception as e:
            # 兜底：float32 中转
            np_f32 = ms_t.astype(ms.float32).asnumpy()
            target_dtype = dtype_map.get(ms_dtype, torch.float32)
            return torch.from_numpy(np_f32).to(target_dtype)

    inputs = {
        "q_flat":       ms_tensor_to_torch(q_flat),
        "qr_flat":      ms_tensor_to_torch(qr_flat),
        "k_flat":       ms_tensor_to_torch(k_flat),
        "kr_flat":      ms_tensor_to_torch(kr_flat),
        "v_flat":       ms_tensor_to_torch(v_flat),
        "sparse_flat":  ms_tensor_to_torch(sparse_flat),
        "do_flat":      ms_tensor_to_torch(do_flat),
        "o_flat":       ms_tensor_to_torch(o_flat),
        "sm_max_flat":  ms_tensor_to_torch(sm_max_flat),
        "sm_sum_flat":  ms_tensor_to_torch(sm_sum_flat),
        "dq_buf":       ms_tensor_to_torch(dq_buf),
        "dqr_buf":      ms_tensor_to_torch(dqr_buf),
        "dk_buf":       ms_tensor_to_torch(dk_buf),
        "dkr_buf":      ms_tensor_to_torch(dkr_buf),
        "dv_buf":       ms_tensor_to_torch(dv_buf),
        "act_q":        ms_tensor_to_torch(act_q),
        "act_k":        ms_tensor_to_torch(act_k),
        "B_S1": B_S1, "S1": S1, "S2": S2, "N1": N1, "topK": topK,
        "D": D, "D_ROPE": D_ROPE, "scale_value": scale_value,
        "sparse_mode": sparse_mode,
    }
    
    with open(save_path, "wb") as f:
        pickle.dump(inputs, f)
    
    print(f"[SFA] Grad 输入已保存到: {save_path}")
    return save_path

@ms.ops._ms_pyfunc(infer_func=_infer_sfa_grad)
def _sfa_grad_core(
    q_flat: ms.Tensor, qr_flat: ms.Tensor,
    k_flat: ms.Tensor, kr_flat: ms.Tensor, v_flat: ms.Tensor,
    sparse_flat: ms.Tensor,
    do_flat: ms.Tensor, o_flat: ms.Tensor,
    sm_max_flat: ms.Tensor, sm_sum_flat: ms.Tensor,
    dq_buf: ms.Tensor, dqr_buf: ms.Tensor,
    dk_buf: ms.Tensor, dkr_buf: ms.Tensor, dv_buf: ms.Tensor,
    act_q: ms.Tensor, act_k: ms.Tensor,
    B_S1: int, S1: int, S2: int, N1: int, topK: int,
    D: int, D_ROPE: int,
    scale_value: float,
    sparse_mode: int,
) -> tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
    # Fixed block config + single launch (NO autotune): autotune re-runs the
    # kernel many times to benchmark, which double-counts the atomic_add scatter
    # into dk/dkr/dv. See _select_block_config.
    grid = lambda meta: (
        _next_pow2(B_S1), 
        _next_pow2(triton.cdiv(N1, meta['BLOCK_G']))
    )
    # _save_sfa_grad_inputs(
    #     q_flat, qr_flat, k_flat, kr_flat, v_flat,
    #     sparse_flat,
    #     do_flat, o_flat,
    #     sm_max_flat, sm_sum_flat,
    #     dq_buf, dqr_buf, dk_buf, dkr_buf, dv_buf,
    #     act_q, act_k,
    #     B_S1, S1, S2, N1, topK,
    #     D, D_ROPE, scale_value, sparse_mode,
    #     save_dir="/home/z00841464/SFA/data/sfa_grad_inputs"
    # )

    _sfa_grad_kernel[grid](
        q_flat, qr_flat,
        k_flat, kr_flat, v_flat,
        sparse_flat,
        do_flat, o_flat,
        sm_max_flat, sm_sum_flat,
        dq_buf, dqr_buf,
        dk_buf, dkr_buf, dv_buf,
        act_q, act_k,
        B_S1, S1, S2, N1, topK,
        D, D_ROPE,
        scale_value,
        sparse_mode=sparse_mode
    )
    return dq_buf, dqr_buf, dk_buf, dkr_buf, dv_buf


# ---------------------------------------------------------------------------
# public API — aligned with CANN aclnnSparseFlashAttentionGrad signature
# ---------------------------------------------------------------------------
def sparse_flash_attention_grad_triton(
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    sparse_indices: ms.Tensor,
    d_out: ms.Tensor,
    out: ms.Tensor,
    softmax_max: ms.Tensor,
    softmax_sum: ms.Tensor,
    scale_value: float = 1.0,
    query_rope=None,
    key_rope=None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    sparse_block_size: int = 1,
    layout: str = "BSND",
    sparse_mode: int = 3,
    pre_tokens: int = INT64_MAX,
    next_tokens: int = INT64_MAX,
    deterministic: bool = False,
):
    """SparseFlashAttentionGrad (MLA-absorb, MQA). Backward of SFA forward.

    Args:
        query: [B,S1,N1,D] BSND or [T1,N1,D] TND, fp16/bf16 (D in 128/256/512)
        key:   [B,S2,1,D] BSND / [T2,1,D] TND
        value: same layout/shape as key; IGNORED in MLA-absorb (value=key[..,:D]),
               but its gradient d_value IS returned separately (CANN parity).
        sparse_indices: [B,S1,1,sparse_count] (or TND) int32, block ids, -1 invalid
        d_out: [B,S1,N1,D] gradient of the attention output
        out:   [B,S1,N1,D] forward attention output
        softmax_max/sum: [B,1,S1,N1] BSND / [1,T1,N1] TND fp32 — forward stats
        scale_value: softmax scale (1/sqrt(d_k))
        query_rope/key_rope: [.,N,Dr] (Dr=64), required (no empty rope)
        actual_seq_lengths_query/kv: [B] int32 / list / None (cumulative for TND)
        sparse_block_size: 1 (token-wise) or 2^n in [1,128] (block-wise)
        layout: "BSND" / "TND"
        sparse_mode: 0 (full) / 3 (rightDownCausal)
        pre_tokens/next_tokens: only default INT64_MAX supported
        deterministic: only False (scatter-add is order-nondeterministic)

    Returns:
        (d_query, d_key, d_value, d_query_rope, d_key_rope), each matching the
        shape/dtype of its corresponding forward input. d_key (QK path) and
        d_value (P@dO path) are separate; a caller sharing key==value sums them.
    """
    if pre_tokens != INT64_MAX or next_tokens != INT64_MAX:
        raise ValueError("pre_tokens/next_tokens only support default INT64_MAX")
    if sparse_mode not in (0, 3):
        raise ValueError("Only sparse_mode 0 (full) / 3 (rightDownCausal) supported")
    if query_rope is None or key_rope is None:
        raise ValueError("query_rope and key_rope are required (no empty rope)")
    if deterministic:
        raise ValueError("deterministic=True is not supported (scatter-add path)")
    if sparse_block_size < 1 or (sparse_block_size & (sparse_block_size - 1)) != 0 \
            or sparse_block_size > 128:
        raise ValueError("sparse_block_size must be a power of 2 in [1,128]")

    is_tnd = (layout == "TND")

    # Pre-init cross-branch vars (GRAPH_MODE parser checks definition across all
    # branches before dead-branch elimination — see forward + memory note).
    act_q_pb = None
    act_k_pb = None

    # ---- normalize TND -> dense BSND (PyNative only; GRAPH_MODE caller pre-converts) ----
    if is_tnd:
        act_q_pb = _tnd_cumsum_to_per_batch(actual_seq_lengths_query)
        act_k_pb = _tnd_cumsum_to_per_batch(actual_seq_lengths_kv)
        q_bsnd = _tnd_to_bsnd(query, act_q_pb)
        qr_bsnd = _tnd_to_bsnd(query_rope, act_q_pb)
        si_bsnd = _tnd_to_bsnd(sparse_indices, act_q_pb)
        do_bsnd = _tnd_to_bsnd(d_out, act_q_pb)
        o_bsnd = _tnd_to_bsnd(out, act_q_pb)
        k_bsnd = _tnd_to_bsnd(key, act_k_pb)
        kr_bsnd = _tnd_to_bsnd(key_rope, act_k_pb)
        # softmax (1,T1,N1) -> (B,1,S1,N1): drop N2 axis, tnd-unpack S1, restore N2
        sm_max_bsnd = _tnd_to_bsnd(softmax_max.transpose(1, 0, 2), act_q_pb).transpose(0, 2, 1, 3)
        sm_sum_bsnd = _tnd_to_bsnd(softmax_sum.transpose(1, 0, 2), act_q_pb).transpose(0, 2, 1, 3)
    else:
        q_bsnd, qr_bsnd, si_bsnd = query, query_rope, sparse_indices
        do_bsnd, o_bsnd = d_out, out
        k_bsnd, kr_bsnd = key, key_rope
        sm_max_bsnd, sm_sum_bsnd = softmax_max, softmax_sum

    B, S1, N1, D = q_bsnd.shape
    S2 = k_bsnd.shape[1]
    D_ROPE = qr_bsnd.shape[-1]

    if D not in _VALID_D:
        raise ValueError(f"D must be one of {_VALID_D}, got {D}")
    if D_ROPE != _D_ROPE:
        raise ValueError(f"rope dim must be {_D_ROPE}, got {D_ROPE}")
    if k_bsnd.shape[2] != 1:
        raise ValueError("Only N2=1 (MQA) is supported")
    if N1 not in _VALID_N1:
        raise ValueError(f"N1 must be one of {_VALID_N1}, got {N1}")

    act_q = _default_actual_seq_lens(
        None if is_tnd else actual_seq_lengths_query, B, S1)
    act_k = _default_actual_seq_lens(
        None if is_tnd else actual_seq_lengths_kv, B, S2)
    if is_tnd:
        act_q = act_q_pb.to(ms.int32)
        act_k = act_k_pb.to(ms.int32)

    # block-wise -> token-wise indices (kernel is purely token-wise)
    si_tok = _expand_block_indices(si_bsnd, sparse_block_size)
    topK = si_tok.shape[-1]

    # flatten (N2=1 / MQA). MLA-absorb: value=key[..,:D] -> kernel v_ptr aliases
    # k_flat (passed `value` ignored); d_value still computed into its own buffer.
    q_flat = q_bsnd.contiguous()
    qr_flat = qr_bsnd.contiguous()
    do_flat = do_bsnd.contiguous()
    o_flat = o_bsnd.contiguous()
    k_flat = k_bsnd.reshape(B * S2, D).contiguous()
    kr_flat = kr_bsnd.reshape(B * S2, D_ROPE).contiguous()
    v_flat = k_flat
    sparse_flat = si_tok.reshape(B * S1, topK).to(ms.int32).contiguous()
    sm_max_flat = sm_max_bsnd.reshape(B * S1 * N1).astype(ms.float32).contiguous()
    sm_sum_flat = sm_sum_bsnd.reshape(B * S1 * N1).astype(ms.float32).contiguous()

    # Output buffers must live on-device (mint.zeros defaults to CPU -> triton
    # rejects the pointer). dk/dkr/dv accumulate via atomic_add, so fp32 workspace.
    dq_buf = ms.mint.zeros((B, S1, N1, D), dtype=q_bsnd.dtype).to('Ascend')
    dqr_buf = ms.mint.zeros((B, S1, N1, D_ROPE), dtype=qr_bsnd.dtype).to('Ascend')
    dk_buf = ms.mint.zeros((B * S2, D), dtype=ms.float32).to('Ascend')
    dkr_buf = ms.mint.zeros((B * S2, D_ROPE), dtype=ms.float32).to('Ascend')
    dv_buf = ms.mint.zeros((B * S2, D), dtype=ms.float32).to('Ascend')

    dq_buf, dqr_buf, dk_buf, dkr_buf, dv_buf = _sfa_grad_core(
        q_flat, qr_flat,
        k_flat, kr_flat, v_flat,
        sparse_flat,
        do_flat, o_flat,
        sm_max_flat, sm_sum_flat,
        dq_buf, dqr_buf,
        dk_buf, dkr_buf, dv_buf,
        act_q.to('Ascend'), act_k.to('Ascend'),
        B * S1, S1, S2, N1, topK,
        D, D_ROPE,
        float(scale_value),
        sparse_mode,
    )

    # fp32 workspace -> key/value dtype, reshape to BSND [B,S2,1,*]
    d_key = dk_buf.reshape(B, S2, 1, D).astype(key.dtype)
    d_key_rope = dkr_buf.reshape(B, S2, 1, D_ROPE).astype(key_rope.dtype)
    d_value = dv_buf.reshape(B, S2, 1, D).astype(value.dtype)
    d_query, d_query_rope = dq_buf, dqr_buf

    if is_tnd:
        d_query = _bsnd_to_tnd(d_query, act_q_pb)
        d_query_rope = _bsnd_to_tnd(d_query_rope, act_q_pb)
        d_key = _bsnd_to_tnd(d_key, act_k_pb)
        d_key_rope = _bsnd_to_tnd(d_key_rope, act_k_pb)
        d_value = _bsnd_to_tnd(d_value, act_k_pb)

    return d_query, d_key, d_value, d_query_rope, d_key_rope


class SparseFlashAttentionGradTriton(ms.nn.Cell):
    """nn.Cell wrapper around sparse_flash_attention_grad_triton; see that function."""

    def __init__(
        self,
        scale_value=1.0,
        sparse_block_size=1,
        layout="BSND",
        sparse_mode=3,
        pre_tokens=INT64_MAX,
        next_tokens=INT64_MAX,
        deterministic=False,
    ):
        super().__init__()
        self.scale_value = scale_value
        self.sparse_block_size = sparse_block_size
        self.layout = layout
        self.sparse_mode = sparse_mode
        self.pre_tokens = pre_tokens
        self.next_tokens = next_tokens
        self.deterministic = deterministic

    def construct(
        self,
        query, key, value, sparse_indices,
        d_out, out, softmax_max, softmax_sum,
        query_rope=None, key_rope=None,
        actual_seq_lengths_query=None, actual_seq_lengths_kv=None,
    ):
        return sparse_flash_attention_grad_triton(
            query, key, value, sparse_indices,
            d_out, out, softmax_max, softmax_sum,
            scale_value=self.scale_value,
            query_rope=query_rope, key_rope=key_rope,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            sparse_block_size=self.sparse_block_size,
            layout=self.layout,
            sparse_mode=self.sparse_mode,
            pre_tokens=self.pre_tokens,
            next_tokens=self.next_tokens,
            deterministic=self.deterministic,
        )