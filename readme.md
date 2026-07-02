# Triton-Ascend implementation of DSA

用 triton-ascend 重写的 DSA（DeepSeek Sparse Attention）算子，接口分别与 `ops.lightning_indexer`、`aclnnSparseLightningIndexerGradKLLoss`、`ops.sparse_flash_attention` 、'ops.sparse_flash_attention_grad'和dense_lightning_indexer_grad_kl_loss对齐，通过 `ops._ms_pyfunc()` 接入 mindspore 静态图。

## 环境依赖

- CANN 9.0.0
- mindspore 2.9.0
- triton-ascend 3.2.1及以上
- pytest-forked (全量测试需要 pytest --forked ...)

## 文件

### Triton 算子

| 算子 | 实现 | 测试 | 性能测试 |
|------|------|------|----------|
| lightning_indexer | `lightning_indexer_triton.py` | `test_li_triton.py` | `perf_li_triton.py` |
| sparse_lightning_indexer_grad_kl_loss | `sparse_lightning_indexer_grad_kl_loss_triton.py` | `test_sli_grad_kl_loss_triton.py` | `perf_sli_grad_kl_loss_triton.py` |
| sparse_flash_attention (前向) | `sparse_flash_attention_triton.py` | `test_sfa_triton.py` | `perf_sfa_triton.py` |
| sparse_flash_attention (反向) | `sparse_flash_attention_grad_triton.py` | `test_sfa_grad_triton.py` | `perf_sfa_grad_triton.py` |
| dense_loss_backward | `dense_loss_backward_triton.py` | `test_dense_loss_backward_triton.py` | `perf_dense_loss_backward_triton.py` |

## lightning_indexer_triton

### 接口

```python
lightning_indexer_triton(
    query,                          # [B,S1,N1,D] 或 [T1,N1,D], fp16/bf16
    key,                            # [B,S2,N2,D] 或 [T2,N2,D], fp16/bf16
    weights,                        # [B,S1,N1] 或 [T1,N1], fp16/bf16/fp32
    actual_seq_lengths_query=None,  # [B] int32, list/tuple, 或 None (默认全序列)
    actual_seq_lengths_key=None,    # [B] int32, list/tuple, 或 None (默认全序列)
    block_table=None,               # 暂不支持 (PA_BSND)
    layout_query="BSND",            # "BSND" 或 "TND"
    layout_key="BSND",              # "BSND" 或 "TND"
    sparse_count=2048,              # top-k
    sparse_mode=0,                  # 0=default, 3=rightDownCausal
    pre_tokens=INT64_MAX,           # 暂未使用
    next_tokens=INT64_MAX,          # 暂未使用
    return_value=False,             # True 返回 (indices, values), False 返回 (indices, zeros)
) -> (sparseIndicesOut, sparseValuesOut)
```

### 算法

对每个 (batch, s1, n2) 位置:

1. 对每个 query head g in group (G = N1 // N2)，Q[g,:] @ K[s2,n2,:]^T → full_dot[s2]
2. ReLU(full_dot) * W[g] → 累加到 tile_scores
3. sparse_mode=3 时应用 rightDownCausal mask: s2 < act_k - act_q + s1 + 1
4. act_k mask: s2 >= act_k → -inf
5. Stable TopK (descending score, ascending index for ties)

### triton kernel

`_lightning_indexer_score_kernel`:
- Grid: `(B * S1 * N2,)`, 每个 program 处理一个 (batch, s1, n2)
- `BLOCK_S2 = 128`, `BLOCK_D = 64`
- 对 s2 分块，每块内遍历 G 个 group，每个 group 内遍历 D 分块累加完整点积后 ReLU
- 内部 fp32 精度计算

TopK 用 `mint.sort(stable=True)` 保证等分时按下标升序。

### TND 布局转换

`_tnd_to_bsnd` / `_bsnd_to_tnd`:
- `_tnd_cumsum_to_per_batch`: 将 TND 累积长度 tensor 转为 per-batch 实际长度
- `_tnd_to_bsnd`: 将 [T, N, ...] 转为 [B, max_S, N, ...]，基于 Python 循环分 batch 拷贝
- `_bsnd_to_tnd`: 将 [B, S, N, ...] 转回 [T, N, ...]
- 仅在 PyNative 模式下工作

## sparse_lightning_indexer_grad_kl_loss_triton

### 接口

```python
sparse_lightning_indexer_grad_kl_loss_triton(
    query,                   # [B,S1,N1,D], fp16/bf16
    key,                     # [B,S2,1,D], fp16/bf16
    query_index,             # [B,S1,Nidx1,D_idx], fp16/bf16
    key_index,               # [B,S2,1,D_idx], fp16/bf16
    weights,                 # [B,S1,Nidx1], fp16/bf16/fp32
    sparse_indices,          # [B,S1,1,topK], int32
    softmax_max,             # [B,1,S1,N1], fp32, from forward FlashAttention
    softmax_sum,             # [B,1,S1,N1], fp32, from forward FlashAttention
    query_rope=None,         # [B,S1,N1,DRope], fp16/bf16
    key_rope=None,           # [B,S2,1,DRope], fp16/bf16
    scale_value=1.0,         # softmax scale
    layout="BSND",           # 仅支持 "BSND"
    sparse_mode=3,           # 仅支持 3 (rightDownCausal)
    ...
) -> (dQueryIndex, dKeyIndex, dWeights, loss)
```

### 算法

对每个 (batch, s1) 位置:

1. I[k] = Σ_g W_g · ReLU(qi_g @ ki_gathered[k]^T) — index-level scores
2. p[k] = (1/N1) Σ_h softmax(score_h)[k] — teacher distribution, 复用 forward softmaxMax/Sum
3. softmax(I) → KL(p || softmax(I)) loss → dI = softmax(I) - p
4. 链式法则求 dW, dQueryIndex, dKeyIndex

### triton kernels

- `_gather_kv_kernel`: 按 sparse_indices 从 key/key_index/key_rope 中 gather，Grid: `(B * S1,)`
- `_indexer_grad_kl_loss_kernel`: 主融合 kernel，完成 stage 1-5，Grid: `(B * S1,)`
- `_scatter_dkey_index_kernel`: scatter-add 计算 dKeyIndex，Grid: `(B * S1 * topK,)`
- `BLOCK_K = 128`, `BLOCK_D = 64`

## sparse_flash_attention_triton

### 接口

```python
sparse_flash_attention_triton(
    query,                          # [B,S1,N1,D] BSND 或 [T1,N1,D] TND, fp16/bf16 (D∈{128,256,512})
    key,                            # [B,S2,1,D] BSND / [T2,1,D] TND / [block_num,block_size,1,D] PA_BSND
    value,                          # 与 key 同布局同形; MLA-absorb 下忽略(value=key[:D]), 仅为对齐 CANN 接口保留
    sparse_indices,                 # [B,S1,1,sparse_count] int32, block id, -1 无效
    scale_value=1.0,                # softmax 缩放
    block_table=None,               # [B,max_blocks] int32, PA_BSND 必传
    actual_seq_lengths_query=None,  # [B] int32/list/None (TND 为累积前缀和)
    actual_seq_lengths_kv=None,
    query_rope=None,                # [.,N,Dr] (Dr=64), 必传
    key_rope=None,                  # 必传
    sparse_block_size=1,            # 1(token-wise) 或 [1,128] 2 的幂(block-wise)
    layout_query="BSND",            # BSND / TND
    layout_kv="BSND",               # BSND / TND / PA_BSND
    sparse_mode=0,                  # 0(全计算) / 3(rightDownCausal)
    pre_tokens=INT64_MAX,           # 仅默认值
    next_tokens=INT64_MAX,          # 仅默认值
    attention_mode=2,               # 仅 2(MLA-absorb)
    return_softmax_lse=False,       # True 返回 softmax_max/sum
    block_size=0,                   # PA_BSND 的 block token 数
) -> (attention_out, softmax_max, softmax_sum)
```

### 算法

对每个 (batch, s1) 位置（N2=1, N1 个 query head 共享同一份 gather KV）:

1. q_full = concat(query[:D], query_rope[:64]); k_full = concat(key[:D], key_rope[:64])
2. 按 sparse_indices gather KV; sparse_mode=3 时 threshold = act_k-act_q+s1+1, mode=0 时 = act_k
3. score[h,p] = (q_nope·k_nope + q_rope·k_rope) * scale, 无效位 -inf
4. 在线 softmax(flow over topK) → out = softmax · value, 同时出 softmax_max/sum
5. MLA-absorb: value = key[:D]（K/V 共享压缩 latent c_kv）, 传入的 value 张量被忽略

### triton kernel

`_sfa_kernel`:
- Grid: `(_next_pow2(B*S1), _next_pow2(cdiv(N1, BLOCK_G)))`, 两维 pow2-padded
- 每 program 处理一个 (b,s1) 的 BLOCK_G 个 query head, inline gather KV 行
- online-softmax 流式累加, acc[BLOCK_G,D] fp32 常驻; value=key[:D]（v_ptr 复用 k_flat）全宽 [BLOCK_K,D] 载入做 PV（D 越小 UB 越宽松, autotune 可选更大 BLOCK_K）
- block-wise(sparse_block_size>1) 在 host 侧展开成逐 token 索引, kernel 始终 token-wise
- 输出 softmax_max/sum 布局: BSND→(B,1,S1,N1), TND→(1,T1,N1)

### 布局转换

TND / PA_BSND 在 host 侧归一到 dense BSND 再进 kernel（与 lightning_indexer 一致, PyNative-only）:
- `_tnd_to_bsnd` / `_bsnd_to_tnd`: TND 累积长度 ↔ BSND
- `_pa_to_bsnd`: 按 block_table 把分页 cache 反分页成 dense BSND
- `_expand_block_indices`: block id → token id（纯静态算子, GRAPH_MODE 亦可）

## sparse_flash_attention_grad_triton

### 接口

```python
sparse_flash_attention_grad_triton(
    query,                          # [B,S1,N1,D] BSND 或 [T1,N1,D] TND, fp16/bf16
    key,                            # [B,S2,1,D] BSND / [T2,1,D] TND
    value,                          # 与 key 同布局同形; MLA-absorb 下忽略, 但 d_value 仍独立返回
    sparse_indices,                 # [B,S1,1,sparse_count] int32, block id, -1 无效
    d_out,                          # [B,S1,N1,D] 梯度输出
    out,                            # [B,S1,N1,D] 前向输出
    softmax_max,                    # [B,1,S1,N1] fp32, 前向 softmax_max
    softmax_sum,                    # [B,1,S1,N1] fp32, 前向 softmax_sum
    scale_value=1.0,                # softmax 缩放
    query_rope=None,                # [.,N,Dr] (Dr=64), 必传
    key_rope=None,                  # 必传
    actual_seq_lengths_query=None,  # [B] int32/list/None
    actual_seq_lengths_kv=None,
    sparse_block_size=1,            # 1(token-wise) 或 [1,128] 2 的幂
    layout="BSND",                  # BSND / TND
    sparse_mode=3,                  # 0 / 3
    pre_tokens=INT64_MAX,           # 仅默认值
    next_tokens=INT64_MAX,          # 仅默认值
    deterministic=False,            # 仅 False (scatter-add 路径天然非确定性)
) -> (d_query, d_key, d_value, d_query_rope, d_key_rope)
```

### 算法

对每个 (batch, s1) 位置（N2=1, MQA）:

1. Host 侧 concat nope|rope: q_cat = [q, q_rope], k_cat = [k, k_rope], D_TOT = D + D_ROPE
2. 复用前向 softmax_max/sum 重建 P = exp(score - max) / sum
3. delta = rowsum(dO * O), host 预计算
4. dS = P · (dO·k - delta) · scale
5. dq = dS^T · k, dk = dS · q, dv = P^T · dO

### triton kernel

`_sfa_grad_kernel`:
- Grid: `(_next_pow2(B*S1),)`, 每 program 处理一个 (b,s1) 的所有 query head
- **Stage A**: 计算 dq/dqr, 同时将 dS/P 写入 GM workspace (bf16)
  - 每 head chunk (BLOCK_G heads) 内循环 topK blocks
  - scores = q_cat · k_cat^T · scale, P = exp(scores - max) / sum
  - dPv = dO_pad · k_cat^T, dS = P · (dPv - delta) · scale
  - dq_acc += dS · k_cat, 写 dS/P 到 workspace
- **Stage B1**: 计算 dk/dkr (读 workspace dS, 无需重新 gather k_cat)
  - dk_acc = dS^T · q_cat, atomic_add 到全局 dkcat
- **Stage B2**: 计算 dv (读 workspace P, 无需重新 gather)
  - dv_acc = P^T · dO, atomic_add 到全局 dv
- `BLOCK_G = 8~16`, `BLOCK_K_A = 32`, `BLOCK_K_B = 32`, `BLOCK_D = 128`
- Host 侧 split nope|rope 列回 dq/dqr, dk/dkr

## dense_loss_backward_triton

### 接口

```python
dense_loss_backward_triton(
    query,                   # [B,S1,N1,D], fp16/bf16
    key,                     # [B,S2,N2,D], fp16/bf16
    query_index,             # [B,S1,Nidx1,D_idx], fp16/bf16
    key_index,               # [B,S2,1,D_idx], fp16/bf16
    weights,                 # [B,S1,Nidx1], fp16/bf16/fp32
    softmax_max,             # [B,N2,S1,G] fp32, from forward FlashAttention
    softmax_sum,             # [B,N2,S1,G] fp32, from forward FlashAttention
    softmax_max_index=None,  # [B,1,S1] fp32, 可选 (不提供则内部计算)
    softmax_sum_index=None,  # [B,1,S1] fp32, 可选
    scale_value=1.0,         # softmax scale
    query_rope=None,         # [B,S1,N1,DRope], fp16/bf16
    key_rope=None,           # [B,S2,N2,DRope], fp16/bf16
    actual_seq_qlen=None,    # [B] int32/list/None
    actual_seq_klen=None,    # [B] int32/list/None
    layout="BSND",           # 仅支持 "BSND"
    sparse_mode=3,           # 仅支持 3 (rightDownCausal)
    pre_tokens=INT64_MAX,    # 仅默认值
    next_tokens=INT64_MAX,   # 仅默认值
) -> (dQueryIndex, dKeyIndex, dWeights, loss)
```

### 辅助接口

```python
dense_lightning_indexer_softmax_lse_triton(
    query_index,             # [B,S1,Nidx1,D_idx]
    key_index,               # [B,S2,1,D_idx]
    weights,                 # [B,S1,Nidx1]
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout="BSND",
    sparse_mode=3,
) -> (softmax_max_index, softmax_sum_index)  # [B,1,S1] fp32
```

### 算法

对每个 (batch, s1) 位置:

1. **Index stats**: I[k] = Σ_g W_g · ReLU(qi_g @ ki[k]^T), stable softmax → max_index, sum_index
2. **Teacher distribution**: p[k] = (1/N1) Σ_h softmax(score_h)[k], 复用 forward softmax_max/sum
3. **KL loss**: student = softmax(I), loss = Σ p·log(p/student)
4. **dI = student - p**, 链式法则求 dQueryIndex, dKeyIndex, dWeights

### triton kernels

- `_dense_indexer_stats_kernel`: 计算 index-level softmax stats (max/sum), Grid: `(B*S1,)`
- `_dense_loss_kernel`: 融合 loss + dI 计算, Grid: `(B*S1,)`
- `_dense_main_grad_kernel`: 计算 dQueryIndex + dWeights, Grid: `(B*S1,)` (分块避免 core dim 溢出)
- `_dense_dkey_index_kernel`: 计算 dKeyIndex (scatter-add over s1), Grid: `(B, num_k_blocks, num_di_blocks)`
- `BLOCK_K = 64`, `BLOCK_D = 64`, `BLOCK_G` 自适应 Nidx1×D_idx

## 限制

- lightning_indexer: N1 % N2 == 0（测试覆盖 N2=1）
- sparse_lightning_indexer_grad_kl_loss: N2=1, Nidx2=1（MQA 约束），仅 sparse_mode=3
- sparse_flash_attention: attention_mode=2(MLA-absorb), N2=1(MQA), D∈{128,256,512}, Dr=64, rope 必传; sparse_mode 0/3; sparse_block_size 1 或 [1,128] 2 的幂; pre/next_tokens 仅默认值
- sparse_flash_attention_grad: attention_mode=2(MLA-absorb), N2=1(MQA), D∈{128,256,512}, Dr=64, rope 必传; sparse_mode 0/3; deterministic 仅 False; pre/next_tokens 仅默认值
- dense_loss_backward: 仅 BSND 布局, sparse_mode=3, Nidx2=1, D∈{128,256,512}, D_idx∈{128,256,512}, Nidx1∈{32,64,128}
- 不支持 PA_BSND / block_table（lightning_indexer 与 grad_kl_loss）
- TND 在 PyNative 下通过布局转换支持，静态图需调用方预转 BSND
- pre_tokens / next_tokens 参数暂未使用

## 测试
测试前执行环境信息：
export ASCEND_RT_VISIBLE_DEVICES=0
export TRITON_END=mindspore
export TRITON_BACKEND=mindspore
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TRITON_CACHE_DIR=./my_triton_cache

### 精度测试命令

```bash
# LightningIndexer
pytest --forked test_li_triton.py -v

# SparseLightningIndexerGradKLLoss
pytest --forked test_sli_grad_kl_loss_triton.py -v

# SparseFlashAttention 前向
pytest --forked test_sfa_triton.py -v

# SparseFlashAttention 反向
pytest --forked test_sfa_grad_triton.py -v

# DenseLossBackward
pytest test_dense_loss_backward_triton.py -v
```

### 性能能命令：(计时 + speedup（triton vs CANN）)

```bash
# LightningIndexer
TRITON_PRINT_AUTOTUNING=1 python perf_li_triton.py

# SparseLightningIndexerGradKLLoss
bash script/profile_sparse.sh timing

# SparseFlashAttention 前向
TRITON_PRINT_AUTOTUNING=1 python perf_sfa_triton.py

# SparseFlashAttention 反向
TRITON_PRINT_AUTOTUNING=1 python perf_sfa_grad_triton.py

# DenseLossBackward
TRITON_PRINT_AUTOTUNING=1 python perf_dense_loss_backward_triton.py
```

## 参考

- [triton-ascend 通过 `ops._ms_pyfunc()` 接入 mindspore 静态图](https://gitcode.com/Ascend/triton-ascend/issues/283)
- [CANN lightning_indexer](https://gitcode.com/cann/ops-transformer/tree/master/attention/lightning_indexer)
- [CANN sparse_lightning_indexer_grad_kl_loss](https://gitcode.com/cann/ops-transformer/tree/master/attention/sparse_lightning_indexer_grad_kl_loss)
- [mindspore.ops.lightning_indexer](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.lightning_indexer.html)
