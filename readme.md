# LightningIndexer Triton-Ascend

基于 `references/lightning_indexer` 的 Ascend C 实现，用 triton-ascend 重写的 lightning_indexer 算子，接口与 `ops.lightning_indexer` 对齐。

## 文件

| 文件 | 说明 |
|------|------|
| `lightning_indexer_triton.py` | triton-ascend 算子实现 |
| `test_lightning_indexer_triton.py` | 单算子测试 |
| `prompts.md` | 原始调用路径与 triton-ascend 接入示例 |

## 接口

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

## 算法

对每个 (batch, s1) 位置:

1. 对每个 head g，Q[g,:] @ K[s2,:]^T → full_dot[s2]
2. ReLU(full_dot) * W[g] → 累加到 tile_scores
3. sparse_mode=3 时应用 rightDownCausal mask: s2 < act_k - act_q + s1 + 1
4. act_k mask: s2 >= act_k → -inf
5. Stable TopK (descending score, ascending index for ties)

## triton kernel

`_lightning_indexer_score_kernel`:
- Grid: `(B * S1,)`, 每个 program 处理一个 (batch, s1)
- `BLOCK_S2 = 256`, `BLOCK_D = 64`
- 对 s2 分块，每块内遍历 N1 个 group，每个 group 内遍历 D 分块累加完整点积后 ReLU
- 内部 fp32 精度计算

TopK 用 `ops.sort(stable=True)` 保证等分时按下标升序。

## TND 布局转换

`_tnd_to_bsnd` / `_bsnd_to_tnd`:
- `_tnd_cumsum_to_per_batch`: 将 TND 累积长度 tensor 转为 per-batch 实际长度
- `_tnd_to_bsnd`: 将 [T, N, ...] 转为 [B, max_S, N, ...]，基于 Python 循环分 batch 拷贝
- `_bsnd_to_tnd`: 将 [B, S, N, ...] 转回 [T, N, ...]
- 仅在 PyNative 模式下工作

## 限制

- N2 (k_head_num) 必须为 1
- 不支持 PA_BSND / block_table
- TND 在 PyNative 下通过布局转换支持，静态图需调用方预转 BSND
- pre_tokens / next_tokens 参数暂未使用

## 测试

```bash
pytest test_lightning_indexer_triton.py -v
```

参数化覆盖: BSND 布局、fp16、sparse_mode 0/3、return_value True/False。
对比 `ops.lightning_indexer` 验证 indices 和 values 一致性。

## 参考

- `references/lightning_indexer/` - Ascend C 原始实现
- `prompts.md` - 算子调用路径与 triton-ascend 接入模式
