# Triton-Ascend implementation of DSA

用 triton-ascend 重写的 DSA（DeepSeek Sparse Attention）算子，接口分别与 `ops.lightning_indexer` 和 `aclnnSparseLightningIndexerGradKLLoss` 对齐，通过 `ops._ms_pyfunc()` 接入 mindspore 静态图。

## 环境依赖

- CANN 9.0.0
- mindspore 2.9.0
- triton-ascend 3.2.1

## 文件

| 文件 | 说明 |
|------|------|
| `lightning_indexer_triton.py` | triton-ascend lightning_indexer 算子实现 |
| `sparse_lightning_indexer_grad_kl_loss_triton.py` | triton-ascend sparse_lightning_indexer_grad_kl_loss 算子实现 |
| `test_li_triton.py` | lightning_indexer 单算子测试 |
| `test_sli_grad_kl_loss_triton.py` | sparse_lightning_indexer_grad_kl_loss 单算子测试 |

## 泛化性配置

| 算子 | 参数 | 支持范围 |
|------|------|----------|
| lightning_indexer | N1（dsa_indexer_n_heads） | 32, 64, 128 |
| | D（dsa_indexer_head_dim） | 128, 256, 512 |
| sparse_lightning_indexer_grad_kl_loss | Nidx1（dsa_indexer_n_heads） | 32, 64, 128 |
| | D_idx（dsa_indexer_head_dim） | 128, 256, 512 |
| | D（query/key head_dim） | 128, 256, 512 |
| sparse_flash_attention | D（head_dim） | 128, 256, 512 |

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

## 限制

- lightning_indexer: N1 % N2 == 0（测试覆盖 N2=1）
- sparse_lightning_indexer_grad_kl_loss: N2=1, Nidx2=1（MQA 约束），仅 sparse_mode=3
- 不支持 PA_BSND / block_table
- TND 在 PyNative 下通过布局转换支持，静态图需调用方预转 BSND
- pre_tokens / next_tokens 参数暂未使用

## 测试

```bash
pytest test_li_triton.py -v
pytest test_sli_grad_kl_loss_triton.py -v
```

- `test_li_triton.py`: 参数化覆盖 BSND 布局、fp16、sparse_mode 0/3、return_value True/False，对比 `ops.lightning_indexer`
- `test_sli_grad_kl_loss_triton.py`: 对比 CANN `SparseLightningIndexerGradKLLoss` 验证 dQueryIndex、dKeyIndex、dWeights、loss 一致性

## 接入路径

在 mindformers 中替换原有 CANN 算子调用:

- `ops.lightning_indexer` → `lightning_indexer_triton` @ `dsa_indexer.py`
- `ops.Custom("aclnnSparseLightningIndexerGradKLLoss", ...)` → `sparse_lightning_indexer_grad_kl_loss_triton` @ `sparse_lightning_indexer_grad_kl_loss.py`

## 参考

- [triton-ascend 通过 `ops._ms_pyfunc()` 接入 mindspore 静态图](https://gitcode.com/Ascend/triton-ascend/issues/283)
- [CANN lightning_indexer](https://gitcode.com/cann/ops-transformer/tree/master/attention/lightning_indexer)
- [CANN sparse_lightning_indexer_grad_kl_loss](https://gitcode.com/cann/ops-transformer/tree/master/attention/sparse_lightning_indexer_grad_kl_loss)
- [mindspore.ops.lightning_indexer](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.lightning_indexer.html)
