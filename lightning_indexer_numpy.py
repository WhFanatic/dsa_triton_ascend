"""NumPy golden reference for lightning_indexer (BSND layout).

Host-runnable ground truth, ported from the CANN golden
ops-transformer/attention/lightning_indexer/tests/pytest/
lightning_indexer_golden.py (`GeneralizedLI`).

Algorithm (per (b, s1, n2)):
  score[n2,s2] = sum_{g in [n2*G,(n2+1)*G)} (ReLU(Q[b,s1,g,:] @ K[b,s2,n2,:]^T) * W[b,s1,g])
  Apply causal mask (sparse_mode=3): positions >= causal_limit or >= act_k -> -inf
  Cast scores to bf16, then stable sort descending -> topK indices
  -inf positions get index = -1

GQA: G = N1 // N2. Each key head n2 reduces over ONLY its own query-head group
g in [n2*G, (n2+1)*G) (per-group reduce), matching the op formula (group size g)
and the triton kernel. N2=1 degenerates to a single global reduce over all N1.
"""
import numpy as np


def _round_bf16(x):
    """fp32 -> bf16 -> fp32 via round-to-nearest-even."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    bias = ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((u + bias) & np.uint32(0xFFFF0000)).view(np.float32)


def _quantize(x, value_dtype):
    """量化分值到目标低精度再回 fp32, 与 triton 返回 dtype 对齐。

    triton 路径分值在 fp32 累加后 cast 到输出 dtype (= 输入 dtype); golden 在
    排序/返回前做同样量化, fp16/bf16 各自逐值可比。-inf 经 fp16 cast 保持 -inf。
    """
    if value_dtype == "fp16":
        return x.astype(np.float16).astype(np.float32)
    return _round_bf16(x)


def lightning_indexer_golden_bsnd(
    q, k, w, act_q, act_k,
    sparse_count=2048, sparse_mode=3,
    return_value=False,
    value_dtype="bf16",
):
    """Reference lightning_indexer on BSND inputs (pure numpy, fp32 accumulation).

    Args:
        q: [B, S1, N1, D] fp32 numpy array
        k: [B, S2, N2, D] fp32 numpy array
        w: [B, S1, N1] fp32 numpy array
        act_q: [B] int array, actual query sequence lengths per batch
        act_k: [B] int array, actual key sequence lengths per batch
        sparse_count: top-k count
        sparse_mode: 0=full, 3=rightDownCausal
        return_value: if True, return (indices, values); else values is dummy
        value_dtype: "bf16" or "fp16", score quantization matching triton output dtype

    Returns:
        (topk_indices, topk_values)
        topk_indices: [B, S1, N2, sparse_count] int32
        topk_values: [B, S1, N2, sparse_count] (same dtype as q if return_value, else zeros)
    """
    B, S1, N1, D = q.shape
    _, S2, N2, _ = k.shape
    G = N1 // N2

    topk_indices = np.full((B, S1, N2, sparse_count), -1, dtype=np.int32)
    topk_values = np.zeros((B, S1, N2, sparse_count), dtype=np.float32)

    for b in range(B):
        cur_act_q = int(act_q[b])
        cur_act_k = int(act_k[b])

        for s1 in range(cur_act_q):
            for n2 in range(N2):
                scores = np.zeros(S2, dtype=np.float32)

                # per-group reduce: 只累加属于本 key head n2 的 query head 组 [n2*G,(n2+1)*G)
                for g in range(n2 * G, (n2 + 1) * G):
                    q_vec = q[b, s1, g, :]
                    k_vec = k[b, :, n2, :]
                    dots = k_vec @ q_vec
                    relu_dots = np.maximum(dots, 0.0)
                    scores += relu_dots * w[b, s1, g]

                if sparse_mode == 3:
                    causal_limit = min(max(cur_act_k - cur_act_q + s1 + 1, 0), S2)
                    scores[causal_limit:] = -np.inf
                    scores[cur_act_k:] = -np.inf
                else:
                    scores[cur_act_k:] = -np.inf

                scores_bf16 = _quantize(scores, value_dtype)

                indices = np.arange(S2, dtype=np.int32)
                sort_keys = np.stack([-scores_bf16, indices.astype(np.float32)], axis=1)
                sorted_order = np.lexsort((sort_keys[:, 1], sort_keys[:, 0]))
                sorted_indices = indices[sorted_order]

                actual_selected = min(cur_act_k, sparse_count)
                for i in range(actual_selected):
                    idx = sorted_indices[i]
                    if scores_bf16[idx] == -np.inf:
                        continue
                    topk_indices[b, s1, n2, i] = idx
                    topk_values[b, s1, n2, i] = scores_bf16[idx]

    if return_value:
        return topk_indices, topk_values
    else:
        dummy_values = np.zeros_like(topk_values)
        return topk_indices, dummy_values
