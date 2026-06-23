# SparseLightningIndexerGradKLLoss 需求分析文档

## 一、需求概述

采用 triton-ascend 实现 `SparseLightningIndexerGradKLLoss`，入参和出参不变，支持以下参数范围：

| 参数 | 对应张量 | shape 维度 | 需求值范围 |
|---|---|---|---|
| query/key 的 head_dim | `query` | `[B, S1, N1, D]` | D ∈ {128, 256, 512} |
| | `key` | `[B, S2, 1, D]` | D ∈ {128, 256, 512} |
| 索引头数 | `query_index` | `[B, S1, Nidx1, D_idx]` | Nidx1 ∈ {32, 64, 128} |
| 索引头维度 | `query_index` | `[B, S1, Nidx1, D_idx]` | D_idx ∈ {128, 256, 512} |
| | `key_index` | `[B, S2, 1, D_idx]` | D_idx ∈ {128, 256, 512} |
| | `weights` | `[B, S1, Nidx1]` | Nidx1 ∈ {32, 64, 128} |

CANN 参考实现：[ops-transformer/sparse_lightning_indexer_grad_kl_loss](https://gitcode.com/cann/opstransformer/tree/master/attention/sparse_lightning_indexer_grad_kl_loss)

模型入口：`mindformers/parallel_core/training_graph/ops/sparse_lightning_indexer_grad_kl_loss.py`（调用 `aclnnSparseLightningIndexerGradKLLoss`）

## 二、参数对应关系

### 2.1 入参全量

| 序号 | 参数名 | shape（BSND） | dtype | 说明 |
|---|---|---|---|---|
| 1 | query | `[B, S1, N1, D]` | fp16/bf16 | 主 attention 的 Q |
| 2 | key | `[B, S2, 1, D]` | fp16/bf16 | 主 attention 的 K（MQA：N2=1） |
| 3 | query_index | `[B, S1, Nidx1, D_idx]` | fp16/bf16 | 闪电索引器的 Q |
| 4 | key_index | `[B, S2, 1, D_idx]` | fp16/bf16 | 闪电索引器的 K（MQA：Nidx2=1） |
| 5 | weights | `[B, S1, Nidx1]` | fp16/bf16/fp32 | 索引器权重 |
| 6 | sparse_indices | `[B, S1, 1, topK]` | int32 | 前向 lightning_indexer 选出的 topK 索引 |
| 7 | softmax_max | `[B, 1, S1, N1]` | fp32 | 主 attention 前向的 softmax 最大值 |
| 8 | softmax_sum | `[B, 1, S1, N1]` | fp32 | 主 attention 前向的 softmax 累加和 |
| 9 | query_rope | `[B, S1, N1, 64]` | fp16/bf16 | query 的 RoPE 部分 |
| 10 | key_rope | `[B, S2, 1, 64]` | fp16/bf16 | key 的 RoPE 部分 |

标量属性：`scale_value (float)`、`layout ("BSND"/"TND")`、`sparse_mode (3)`、`actual_seq_qlen/klen (可选)`

### 2.2 出参

| 序号 | 参数名 | shape | dtype |
|---|---|---|---|
| 1 | d_query_index | 同 query_index | 同 query_index |
| 2 | d_key_index | 同 key_index | 同 key_index |
| 3 | d_weights | 同 weights | 同 weights |
| 4 | loss | `[1]` | fp32 |

## 三、需求参数约束 vs CANN vs Triton 对比

| 维度 | 需求 | CANN README | Triton 实现 |
|---|---|---|---|
| N1（query 头数） | 未明确要求 | 32, 64, 128 | 32, 64, 128 |
| **Nidx1（索引头数）** | **32, 64, 128** | 8, 16, 32, 64 | 32, 64, 128 |
| **D（query/key 头维度）** | **128, 256, 512** | 512 | 128, 256, 512 |
| **D_idx（索引头维度）** | **128, 256, 512** | 128（典型值） | 128, 256, 512 |
| N2（key 头数） | — | 1 | 1 |
| Nidx2（索引 key 头数） | — | 1 | 1 |
| Drope（RoPE 维度） | — | 64 | 64 |
| topK（稀疏数量） | — | 1024~8192（步长 1024） | 任意 |
| B（batch） | — | 1~256 | ✓ |
| S1（query 序列长度） | — | 1~8K | ✓ |
| S2（key 序列长度） | — | 1~128K | ✓ |

## 四、CANN 满足度判定

| 需求参数 | CANN 是否支持 | 判定 |
|---|---|---|
| Nidx1 = 32 | ✅ | 满足 |
| Nidx1 = 64 | ✅ | 满足 |
| **Nidx1 = 128** | ❌ CANN 上限为 64 | **不满足** |
| **D = 128** | ❌ CANN 仅支持 512 | **不满足** |
| **D = 256** | ❌ CANN 仅支持 512 | **不满足** |
| D = 512 | ✅ | 满足 |
| D_idx = 128 | ✅ | 满足 |
| **D_idx = 256** | ❌ CANN 仅支持 128 | **不满足** |
| **D_idx = 512** | ❌ CANN 仅支持 128 | **不满足** |

**结论：8 项需求参数中 4 项超出 CANN 规格，CANN 不能完全满足诉求。**

## 五、Triton 实现现状

当前 triton-ascend 实现（`sparse_lightning_indexer_grad_kl_loss_triton.py`）已支持全部需求参数范围，可作为 CANN 的超集替代。

### 5.1 功能支持度

| 特性 | 状态 |
|---|---|
| Nidx1 = 32, 64, 128 | ✅ 已支持 |
| D = 128, 256, 512 | ✅ 已支持 |
| D_idx = 128, 256, 512 | ✅ 已支持 |
| fp16 / bf16 | ✅ 已支持 |
| sparse_mode = 3（rightDownCausal） | ✅ 已支持 |
| BSND layout | ✅ 已支持 |
| TND layout | ❌ 未支持（CANN 支持） |

### 5.2 性能现状（生产 shape：B=1, S1/S2=4096, N1=64, D=512, Nidx1=64, D_idx=128, topK=2048）

| 指标 | CANN | Triton | 比值 |
|---|---|---|---|
| 单次调用耗时 | 30.2 ms | 3719 ms | 0.008x（123 倍慢） |
| Kernel 数量 | 1 个融合 kernel | 5 个独立 kernel | — |
| 峰值 HBM 占用 | 低 | ~18 GB（gather 中间张量） | — |

### 5.3 已知问题

1. **性能差距巨大**：比 CANN 慢 123 倍，目标 0.5 倍（即不超过 CANN 的 2 倍）
2. **显存瓶颈**：3 个 gather kernel 创建约 11 GB 中间张量，在 A3（CANN 9.0.0）上导致 OOM
3. **无 autotune**：所有 BLOCK 参数硬编码，无法针对不同 shape 找到最优 tile 配置
4. **dKeyIndex 精度偏差**：scatter atomic_add 导致 0.024% 元素与 CANN 不一致

## 六、差距分析与优先级

| 序号 | 差距项 | 说明 | 优先级 |
|---|---|---|---|
| 1 | 性能（123x CANN） | 需大幅优化，达到 CANN 的 0.5 倍 | P0 |
| 2 | A3 内存 OOM | gather 中间张量需分 tile 处理 | P0 |
| 3 | 无 autotune | 添加 autotune + UB prune 机制 | P1 |
| 4 | scatter 精度 | 用 tiled reduction 替代 atomic_add | P2 |
| 5 | TND layout | 补齐 CANN 对齐的 layout 支持 | P3 |

## 七、参考文件

| 文件 | 说明 |
|---|---|
| `ops-transformer/attention/sparse_lightning_indexer_grad_kl_loss/README.md` | CANN 算子规格文档 |
| `dsa_triton_ascend_v2/sparse_lightning_indexer_grad_kl_loss_triton.py` | Triton 实现 |
| `dsa_triton_ascend_v2/sli_grad_kl_loss_cann.py` | CANN 调用封装 |
| `dsa_triton_ascend_v2/test_sli_grad_kl_loss_triton.py` | 测试用例 |
| `dsa_triton_ascend_v2/readme.md` | 项目整体说明 |
