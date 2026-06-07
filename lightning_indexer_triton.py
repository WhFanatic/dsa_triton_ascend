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


def _next_pow2(x):
    # Ascend 要求 kernel grid 每维都是 2 的幂, 否则分核映射出错 -> aicore trap。
    # padding 多出的 program 在 kernel 里靠 bsn_in_range 掩码空转。
    return 1 << (x - 1).bit_length() if x > 1 else 1


def _patch_triton_ascend_mindspore_dtype_bytes():
    """修补 triton-ascend autotuner 的 dtype 字节数查询。

    autotuner 估 UB 占用时会调 get_byte_per_numel(dtype), 但它只认 torch dtype,
    传 MindSpore dtype 会抛错把整个 autotune 带挂; 这里补一张 ms dtype -> 字节的映射兜住。
    """
    try:
        from triton.backends.ascend.runtime import utils as ascend_utils
        from triton.backends.ascend.runtime import autotuner as ascend_autotuner
    except ImportError:
        return

    origin_func = getattr(ascend_utils, "get_byte_per_numel", None)
    if origin_func is None:
        return
    # 已打过补丁: 让 autotuner 模块也指向同一函数 (re-import 时它可能仍握着旧引用)。
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
            # 个别 dtype 对象不可哈希, in 会抛 TypeError; 落回原实现。
            pass
        return origin_func(dtype)

    patched_get_byte_per_numel._mindspore_dtype_patched = True
    ascend_utils.get_byte_per_numel = patched_get_byte_per_numel
    if hasattr(ascend_autotuner, "get_byte_per_numel"):
        ascend_autotuner.get_byte_per_numel = patched_get_byte_per_numel


_patch_triton_ascend_mindspore_dtype_bytes()


def _prune_configs(configs, named_args, **kwargs):
    """autotune config 过滤

    - UB 容量上限(910B 单核 ~192KB)
    - grid program 总数上限(实测 131072 可跑, 262144 静默失败)
    - grid0/grid1 必须是 2 的幂, 否则 runtime 分核映射出错 -> aicore trap
    """
    _UB_LIMIT_BYTES = 192 * 1024
    _GRID_LIMIT = 65535  # Ascend coreDim 硬件上限

    def _estimate_ub_bytes(block_s2, block_d, block_g):
        """粗估单 tile 主要 buffer 的 UB 占用 (bytes)。

        1.25 系数近似 double-buffering + tl.trans 临时空间, 由实测反推:
        (BLOCK_S2=256,BLOCK_D=128,BLOCK_G=64) 实测要 ~262KB。
        """
        if block_s2 is None or block_d is None or block_g is None:
            return 0
        acc        = block_g * block_s2 * 4   # acc[BLOCK_G, BLOCK_S2] fp32
        k_tile     = block_s2 * block_d * 2   # k_tile[BLOCK_S2, BLOCK_D] fp16
        q_tile     = block_g  * block_d * 2   # q_tile[BLOCK_G, BLOCK_D] fp16
        trans_tmp  = block_s2 * block_d * 2   # tl.trans 中间空间
        tile_score = block_s2 * 4             # tile_scores[BLOCK_S2] fp32
        total = acc + k_tile + q_tile + trans_tmp + tile_score
        return int(total * 1.25)

    def _get(name):
        if name in named_args:
            return named_args[name]
        return kwargs.get(name, None)

    B  = _get("B");  S1 = _get("S1"); N2 = _get("N2")
    S2 = _get("S2"); D  = _get("D");  G  = _get("G")

    kept = []
    for c in configs:
        bs1 = c.kwargs.get("BLOCK_S1")
        bs2 = c.kwargs.get("BLOCK_S2")
        bd  = c.kwargs.get("BLOCK_D")
        bg  = c.kwargs.get("BLOCK_G")

        if _estimate_ub_bytes(bs2, bd, bg) > _UB_LIMIT_BYTES:
            continue

        if None not in (S2, D, G):
            if bs2 > S2 or bd > D or bg > G:
                continue

        if None not in (B, S1, N2, S2) and bs1 and bs2:
            bsn = B * S1 * N2
            grid0 = _next_pow2((bsn + bs1 - 1) // bs1)
            grid1 = _next_pow2((S2 + bs2 - 1) // bs2)
            if grid0 * grid1 > _GRID_LIMIT:
                continue

        kept.append(c)

    if not kept:
        # 兜底: 取 UB 占用最小的一个, 避免 autotune 无 config 可用
        print('Warning: all autotune params pruned')
        kept = [min(
            configs,
            key=lambda c: _estimate_ub_bytes(
                c.kwargs.get("BLOCK_S2"),
                c.kwargs.get("BLOCK_D"),
                c.kwargs.get("BLOCK_G"),
            ),
        )]
    return kept


@triton.autotune(
    configs=[
        # BLOCK_S1=8/4: 小~中 shape, 并行度高; 每档给 G=16 与 G=64 两种 reduce 宽度。
        triton.Config({"BLOCK_S1": 8, "BLOCK_S2": 128, "BLOCK_D": 128, "BLOCK_G": 16}),  # S1=128 选中
        triton.Config({"BLOCK_S1": 8, "BLOCK_S2": 128, "BLOCK_D": 128, "BLOCK_G": 64}),  # S1=1024/2048 选中
        triton.Config({"BLOCK_S1": 4, "BLOCK_S2": 128, "BLOCK_D": 128, "BLOCK_G": 16}),
        triton.Config({"BLOCK_S1": 4, "BLOCK_S2": 128, "BLOCK_D": 128, "BLOCK_G": 64}),

        # BLOCK_S1=1/2: 并行度最高, 但只在 B*S1 很小时存活 (大 shape 下 grid0 超 coreDim 被剪)。
        triton.Config({"BLOCK_S1": 1, "BLOCK_S2": 128, "BLOCK_D": 128, "BLOCK_G": 16}),  # S1=4096 选中
        triton.Config({"BLOCK_S1": 1, "BLOCK_S2": 128, "BLOCK_D": 128, "BLOCK_G": 64}),
        triton.Config({"BLOCK_S1": 2, "BLOCK_S2": 128, "BLOCK_D": 128, "BLOCK_G": 16}),

        # 大 BLOCK_S1: 大 shape 用它把 grid0 压回限内, 同时调大 BLOCK_S2 摊薄 grid1、调小 BLOCK_D 守 UB。
        triton.Config({"BLOCK_S1": 16, "BLOCK_S2": 256, "BLOCK_D": 128, "BLOCK_G": 16}),  # S1=16384 选中
        triton.Config({"BLOCK_S1": 16, "BLOCK_S2": 128, "BLOCK_D": 128, "BLOCK_G": 16}),
        triton.Config({"BLOCK_S1": 32, "BLOCK_S2": 256, "BLOCK_D": 128, "BLOCK_G": 16}),
        triton.Config({"BLOCK_S1": 128, "BLOCK_S2": 256,  "BLOCK_D": 64, "BLOCK_G": 16}),
        triton.Config({"BLOCK_S1": 256, "BLOCK_S2": 512,  "BLOCK_D": 32, "BLOCK_G": 16}),
        triton.Config({"BLOCK_S1": 512, "BLOCK_S2": 1024, "BLOCK_D": 16, "BLOCK_G": 16}),
    ],
    key=["B", "S1", "S2", "N1", "N2", "D"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _lightning_indexer_score_kernel(
    q_ptr, k_ptr, w_ptr, score_ptr, # Input/output tensors
    B, S1, S2, N1, N2, D, G,        # B: batch size, S1: query sequence length, S2: key sequence length, N1: query group size, N2: key group size, D: head dimension
    act_q_ptr, act_k_ptr,           # valid query and key sequence length
    sparse_mode: tl.constexpr,
    BLOCK_S1: tl.constexpr,
    BLOCK_S2: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    """Compute reduced scores for lightning_indexer (BSND layout).

    Grid: (cdiv(B*S1*N2, BLOCK_S1), cdiv(S2, BLOCK_S2)), 两维都 pow2-padded。
    把 (b,s1,n2) 摊平成 bsn 维再按 BLOCK_S1 分块, 是为了让 grid0 不随 S1 线性膨胀撑破
    coreDim 上限。每个 program 算 BLOCK_S1 个 bsn 位置的同一个 S2 tile:
        score[b,s1,n2,s2] = sum_{g in group}(ReLU(Q[b,s1,g,:] @ K[b,s2,n2,:]^T) * W[b,s1,g])
    其中 group = [n2*G, (n2+1)*G), G = N1 // N2。
    """
    pid_bsn_blk = tl.program_id(0)
    pid_s2      = tl.program_id(1)

    s2_offs  = pid_s2 * BLOCK_S2 + tl.arange(0, BLOCK_S2)
    s2_valid = s2_offs < S2

    k_row_stride = N2 * D
    bsn_limit = B * S1 * N2
    bsn_base = pid_bsn_blk * BLOCK_S1

    # 串行扫本 block 的 BLOCK_S1 个 bsn 位置。踩坑: triton-ascend 一个 kernel 里多个
    # early-return + store 会丢 store, 所以全程不 return, 无效位置也走完、只置空 store mask。
    for i in range(BLOCK_S1):
        bsn = bsn_base + i
        bsn_in_range = bsn < bsn_limit
        # pow2-padding 会多出越界 program; 钳到 0 只为地址不 trap, 不写脏数据靠末尾 store_mask。
        bsn = tl.where(bsn_in_range, bsn, 0)

        n2  = bsn % N2
        tmp = bsn // N2
        s1  = tmp % S1
        b   = tmp // S1

        score_row_base = bsn * S2

        act_q = tl.load(act_q_ptr + b) # 当前 sample 的有效 query 序列长度
        act_k = tl.load(act_k_ptr + b) # 当前 sample 的有效 key 序列长度

        # rightDownCausal: query s1 可见 key 上界 = act_k-act_q+s1+1 (右下对齐), 再和 act_k/S2 取交。
        if sparse_mode == 3:
            causal_limit  = tl.minimum(tl.maximum(act_k - act_q + s1 + 1, 0), S2)
            visible_limit = tl.minimum(act_k, causal_limit)
        else:
            causal_limit  = S2
            visible_limit = act_k

        # bsn 合法 + query 行有效 + 本 S2 tile 在可见范围内, 三者缺一就整段跳过、直接输出 -inf。
        need_compute = bsn_in_range & (s1 < act_q) & (pid_s2 * BLOCK_S2 < visible_limit)

        if need_compute:
            # offset 基址 (内存布局 q[B,S1,N1,D] / k[B,S2,N2,D] / w[B,S1,N1])。
            k_base = b * S2 * k_row_stride + n2 * D
            q_base = ((b * S1 + s1) * N1 + n2 * G) * D
            w_base = (b * S1 + s1) * N1 + n2 * G

            tile_scores = tl.zeros([BLOCK_S2], dtype=tl.float32)

            # G 外 / D 内分块。k_tile 与 g 无关却每个 g-block 重 load 一遍 (G=64/BLOCK_G=16 时 4 次),
            # K 带宽吃紧时可提到 g 循环外复用。
            for g_start in range(0, G, BLOCK_G):
                g_rel   = g_start + tl.arange(0, BLOCK_G)
                g_valid = g_rel < G
                w_g = tl.load(w_ptr + w_base + g_rel, mask=g_valid, other=0.0).to(tl.float32)

                acc = tl.zeros([BLOCK_G, BLOCK_S2], dtype=tl.float32)

                for d_start in range(0, D, BLOCK_D):
                    d_offs  = d_start + tl.arange(0, BLOCK_D)
                    d_valid = d_offs < D

                    q_offs = q_base + g_rel[:, None] * D + d_offs[None, :]
                    q_tile = tl.load(
                        q_ptr + q_offs,
                        mask=g_valid[:, None] & d_valid[None, :],
                        other=0.0,
                    )

                    k_offs = k_base + s2_offs[:, None] * k_row_stride + d_offs[None, :]
                    k_tile = tl.load(
                        k_ptr + k_offs,
                        mask=s2_valid[:, None] & d_valid[None, :],
                        other=0.0,
                    )

                    acc += tl.dot(q_tile, tl.trans(k_tile))

                # indexer 打分 = 各 head ReLU(Q·K) 按 W 加权求和; padding 的 g 行先清零。
                acc = tl.maximum(acc, 0.0)
                acc = tl.where(g_valid[:, None], acc, 0.0)
                tile_scores += tl.sum(acc * w_g[:, None], axis=0)

            # 超出 causal / act_k 的不可见位置置 -inf, topk 时会被映射成 index -1。
            if sparse_mode == 3:
                tile_scores = tl.where(s2_offs < causal_limit, tile_scores, float('-inf'))
            tile_scores = tl.where(s2_offs < act_k, tile_scores, float('-inf'))
        else:
            tile_scores = tl.full([BLOCK_S2], float('-inf'), dtype=tl.float32)

        # s2_valid 卡 S2 尾块, bsn_in_range 卡 padding 越界行; 任一为假都不写。
        store_mask = s2_valid & bsn_in_range
        tl.store(score_ptr + score_row_base + s2_offs, tile_scores, mask=store_mask)


def _default_actual_seq_lens(actual_seq_lens, batch_size, seq_len):
    # None -> 默认整段; list/tuple -> 转 int32 张量; 否则原样透传。
    return ms.ops.fill(ms.int32, (batch_size,), seq_len) if actual_seq_lens is None else \
           ms.Tensor(list(actual_seq_lens), dtype=ms.int32) if isinstance(actual_seq_lens, (list, tuple)) else \
           actual_seq_lens


def _tnd_cumsum_to_per_batch(cumsum):
    # TND 的 actual_seq_lengths 是累积前缀和; 错位相减还原成每 batch 实际长度。
    return cumsum - ops.pad(cumsum[:-1], (1, 0))


def _tnd_to_bsnd(tensor, act_seq_per_batch):
    """[T, N, ...] -> [B, max_S, N, ...], 按各 batch 实际长度填充、其余补零。

    python 循环做变长切片, 只能 PyNative (GRAPH_MODE 无法处理数据依赖的切片);
    GRAPH_MODE 调用方请直接传 BSND。
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
    """[B, S, N, ...] -> [T, N, ...], _tnd_to_bsnd 的逆操作 (同样 PyNative-only)。"""
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
        # stable=True: 用 sort 保证同分时稳定顺序, 与 CANN 参考对齐 (比 topk 慢)。
        _, sorted_indices = mint.sort(-scores_2d, dim=1, stable=True)
        topk_indices = sorted_indices[:, :k].to(ms.int32)
        topk_values = ops.gather_d(scores_2d, 1, topk_indices)
    else:
        # mint.topk 更快; 但同分时排序结果与 CANN 不保证一致, 对数值无影响。
        topk_values, topk_indices = mint.topk(scores_2d, k, dim=1, largest=True, sorted=True)
        topk_indices = topk_indices.to(ms.int32)

    # -inf 占位的不可见位置, index 统一置 -1, 与 ops 算子语义一致。
    invalid = topk_values == float('-inf')
    topk_indices = mint.where(invalid, -1, topk_indices)

    return topk_indices, topk_values


# _ms_pyfunc 的 shape/dtype 推导: 输出与 scores_flat 同形同类型。
# 参数类型注解是 _ms_pyfunc 推导依赖的, 勿删 (须与下面 core 对齐)。
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


# 参数类型注解同样被 _ms_pyfunc 依赖, 勿删。
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
    # grid 两维都 _next_pow2 向上取整 (Ascend 非 2 的幂 grid 会 trap); 越界 program 由
    # kernel 内 bsn_in_range 空转。padding 须与 _prune_configs 判定一致。
    def grid_fn(meta): return (
        _next_pow2(triton.cdiv(B * S1 * N2, meta["BLOCK_S1"])),
        _next_pow2(triton.cdiv(S2, meta["BLOCK_S2"])),
    )

    _lightning_indexer_score_kernel[grid_fn](
        q_flat, k_flat, w_flat, scores_flat,
        B, S1, S2, N1, N2, D, G,
        act_q, act_k,
        sparse_mode=sparse_mode,
    )

    return scores_flat


class LightningIndexerTriton(ms.nn.Cell):
    """nn.Cell wrapper around lightning_indexer_triton; see that function for arg semantics."""

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

    # TND 入口: 累积长度还原成 per-batch, 转 BSND 喂 kernel; 出口再转回 TND。
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

    G = N1 // N2   # GQA: 每个 key head 对应的 query head 组宽

    q_flat = q_bsnd.contiguous()
    k_flat = k_bsnd.contiguous()
    w_flat = w_bsnd.contiguous()

    # 预填 -inf: kernel 只写可见位置, 未触及的保持 -inf -> topk 视为无效 (index -1)。
    scores_flat = ms.mint.full((B * S1 * N2, S2), float('-inf'), dtype=ms.float32)

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
