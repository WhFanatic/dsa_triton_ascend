# Dense算子（DenseLossBackward）Triton实现优化总结

---

## 一、整体优化效果概述

### 1.1 优化目标与结果

在 Ascend910_9382 + CANN 9.0 平台上，Dense 算子的 Triton 实现经历了从基线到最终 **0.3× CANN**（与 CANN 持平）的迭代优化。

Dense 算子计算模式接近 SLI 的简化版——无 sparse gather、全 dense K 维循环，优化重心在**计算路径升级**（scalar→cube）和**中间结果复用**。

### 1.2 优化里程碑

| 阶段 | 关键推动力 | 效果 |
|------|-----------|------|
| 初始基线 | 功能验证 | 基准线 |
| 基线优化 | Cell 重构 | 精度通过 |
| di workspace | 消除 dquery/dkey 中 teacher/student 重算 | **781ms** |
| dkey grid 重构 | G-iteration 内化，4D→3D grid | **414ms**（1.88×）|
| LSE online stats | 两趟→单趟 chunked max+sum | LSE 69ms |
| Scalar→Cube 路径 | tl.sum→tl.dot，BLOCK_G tile | **CANN 持平** |
| S1=4096 兼容 | dkey grid 重构，S1 内部循环 | 全 shape 达标 |
| Causal skip | `k_start < visible` 提前跳过 | **0.3× CANN** |

### 1.3 核心架构图

```
┌─────────────────────────────────────────────────────────┐
│ Dense Pipeline (三个 kernel, 两个 _ms_pyfunc)            │
│                                                         │
│ [K1] _dense_indexer_stats_kernel                        │
│   Online chunked max+sum for student softmax            │
│   Grid: (B*S1,) → outputs: max_index, sum_index         │
│                                                         │
│ [K2] _dense_loss_kernel                                │
│   Teacher dist. + student softmax + KL loss + di        │
│   Grid: (B*S1,) → outputs: loss, di_ws [B*S1, S2]      │
│   di = student - teacher → workspace for K3/K4 reuse    │
│                                                         │
│ [K3] _dense_main_grad_kernel                            │
│   dQueryIndex + dWeights (reads di from workspace)      │
│   Grid: (bs1_chunk,) with BLOCK_G tile                  │
│   qi[BG,D] @ ki[K,D] → dot[BG,K] → dw/dqi cube acc     │
│                                                         │
│ [K4] _dense_dkey_index_kernel                           │
│   dKeyIndex (accumulates over all s1)                   │
│   Grid: (B, num_k_blocks, num_di_blocks)                │
│   s1-loop internal: dS^T @ qi → dki_acc                 │
└─────────────────────────────────────────────────────────┘
```

---

## 二、优化点子章节

### 2.1 di Workspace：消除 Teacher/Student 重算（commit `26950ad`，→781ms）

- **原始设计**：三个下游 kernel（`_dense_main_grad_kernel` 做 dquery/dweight、`_dense_dkey_index_kernel` 做 dkey）各自独立计算 `student = softmax(i_tile)` 和 `teacher = p_tile`，然后求 `di = student - teacher`。每个 K-tile 内：
  - `_dense_indexer_i_tile`：G 循环 q@k + ReLU + W 加权求和
  - `_dense_teacher_p_tile`：N1 循环 Q@K + QR@KR + softmax
  - 两项计算被**三次重复执行**（loss kernel 一次 + dquery kernel 一次 + dkey kernel 一次）

- **优化**：`_dense_loss_kernel` 在计算 loss 的同时将 `di` 物化写入 workspace `di_ws[B*S1, S2]`（fp32），`_dense_main_grad_kernel` 和 `_dense_dkey_index_kernel` 改为直接从 workspace `tl.load(di)`，**完全消除** teacher/student 重算

- **收益**：
  - `_dense_main_grad_kernel` 签名从接收 `query_ptr, key_ptr, query_rope_ptr, key_rope_ptr, softmax_max_ptr, softmax_sum_ptr, max_index_ptr, sum_index_ptr` 简化为仅需 `di_ptr`
  - `_dense_dkey_index_kernel` 同理
  - 端到端 → 781ms

### 2.2 dkey Grid 重构：G-iteration 内化 + 降维（commit `1eec4dd`，781ms→414ms）

- **原始设计**：`_dense_dkey_index_kernel` 使用 4D grid：`(bs1_chunk, Nidx1, kd_chunk)`，其中 kd_chunk 是 K 块和 D 块的平铺组合。每个 program 处理一个 group `g`，需要构造复杂的 `kd_chunk` 拆分逻辑（`num_k_blocks * num_d_blocks` 平铺后按 Ascend coreDim 限值分片）

- **优化**：
  - Grid 降为 3D：`(bs1_chunk, num_k_blocks, num_d_blocks)`，去掉 Nidx1 维度
  - G 维度循环（`for g in range(Nidx1)`）内化到 kernel 内部
  - 每个 program 串行遍历所有 G 组，内部累加 `dki_acc`，最后做**单次** `tl.atomic_add`
  - 消除 host 侧 kd_chunk 复杂拆分逻辑

- **收益**：781ms→414ms（1.88×）。核心原因是减少了 kernel launch 次数（Nidx1 倍）和 grid 管理开销

### 2.3 Scalar→Cube 路径升级（commit `c963ddb`，CANN 持平）

这是 Dense 算子**最大的一次架构变更**，将三处核心计算从 scalar (vector) 路径升级为 cube (matrix) 路径。

**2.3.1 QI·KI dot：tl.sum(D 维度) → tl.dot（矩阵乘法）**

- **原始**：`_dense_indexer_dot_g_tile` 对每个 group `g` 做 scalar D 维度循环：
  ```python
  for g in range(Nidx1):
      for d_start in range(0, D_idx, BLOCK_D):
          qi = tl.load(...)  # [BLOCK_D]
          ki = tl.load(...)  # [BLOCK_K, BLOCK_D]
          acc += tl.sum(ki * qi[None, :], axis=1)  # vector mul-add
  ```
- **优化**：`_dense_indexer_qi_cube` 加载 `qi[NIDX1, D_idx]` 全量 + `_dense_indexer_ki_cube` 加载 `ki[BLOCK_K, D_idx]` + 单次 `tl.dot(qi, ki^T)`：
  ```python
  qi = tl.load(...)  # [NIDX1, D_idx]
  ki = tl.load(...)  # [BLOCK_K, D_idx]
  dot = tl.dot(qi, tl.trans(ki))  # cube: [NIDX1, BLOCK_K] in one shot
  i_tile = tl.sum(tl.maximum(dot, 0.0) * w[:, None], axis=0)  # ReLU + weighted sum
  ```
- **收益**：G×D 维度的 Nidx1×cdiv(D,BLOCK_D) 次 vector 乘加合并为 1 次 cube dot，cube 利用率大幅提升

**2.3.2 dQueryIndex：vector 累加 → tl.dot(cube) + L0C 驻留**

- **原始**：`_dense_main_grad_kernel` 每个 group 独立程序，dqi 用 `tl.sum(di[:,None]*ki, axis=0)` vector 累加
- **优化**：
  - Grid 从 `(bs1_chunk, Nidx1, num_d_blocks)` 降为 `(bs1_chunk,)`
  - BLOCK_G tile 一次处理多个 group：`qi_g[BLOCK_G, D_IDX]`
  - dqi 累加：`tl.dot(coeff.to(ki.dtype), ki, acc=dqi)`，cube L0C 驻留
  - dw 累加：`tl.sum(relu_g * di[None, :], axis=1)`，一次覆盖 BLOCK_G 组
- **配置选择**：`_grad_block_gd(Nidx1, D_idx)` 根据 `Nidx1 * D_idx <= 8192` 决定 BLOCK_G = Nidx1（全量一次）或分块

**2.3.3 dKeyIndex：vector 累加 → tl.dot(cube)**

- **原始**：`coeff[:, None] * qi[None, :]` vector 外积 → `tl.atomic_add`，每 (s1, g) 一次
- **优化**：`tl.dot(tl.trans(coeff), qi_g_sub, acc=dki_blk)`，跨所有 G 组累积后单次 store

### 2.4 LSE Online Stats：两趟→单趟 chunked（commit `f74fe0d`，LSE 69ms）

- **原始设计**：Pass 1 遍历全部 S2 找 max（`i_max`），Pass 2 用 safe max 计算 softmax sum。两个循环遍历完整 S2
- **优化**：单趟 chunked online max+sum——与 SFA forward 的 online-softmax 同款技术：
  ```python
  for k_start in range(0, S2, BLOCK_K):
      i_masked = tl.where(k_mask, i_tile, float("-inf"))
      m_new = tl.maximum(i_max, tl.max(i_masked, axis=0))
      alpha = tl.where(i_max > -inf, tl.exp(i_max - m_new), 1.0)
      exp_i = tl.where(k_mask, tl.exp(i_tile - m_new), 0.0)
      i_sum = i_sum * alpha + tl.sum(exp_i, axis=0)
      i_max = m_new
  ```
- **效果**：消除 `valid_count` 计数和 `has_valid` 分支，单趟完成

### 2.5 Causal Skip：提前跳出不可见 K-block（commit `5c9868a`，最终 0.3×）

- **思路**：Dense 模式下 causal 窗口固定，`visible = min(max(act_k - act_q + s1 + 1, 0), S2)` 是严格上界。`_dense_loss_kernel` 中 K 循环遍历 `range(0, S2, BLOCK_K)`，但大量 K-block（尤其是 s1 较小的行）完全不参与 causal 窗口
- **优化**：`if k_start < visible` 提前跳过完全不可见的 K-block。对于 S1=4096、causal 窗口从 1 到 4096 变化：
  - s1=0：visible=1，只算 1 个 K-block（原来算 64 个）
  - s1=2048：visible=2048，算 32 个（原来 64 个）
  - s1=4096：visible=4096，全算
  - 平均省 ~50% 的 K-block 循环
- **注意**：这是 Dense 模式的独有优化——Sparse 模式中 sparse_indices 已预选 topK，窗口天然受限，不需要此优化

### 2.6 dkey Grid 二次重构：S1=4096 兼容（commit `8761340`）

- **问题**：S1=512 时 dkey kernel grid `(bs1_chunk, num_k_blocks, num_d_blocks)` 中 `bs1_chunk` 较小（`_bs1_chunk_for_core_dim` 限值），但 S1=4096 时 grid 维度爆炸。原设计每个 program 一个 (b,s1)，S1 在 grid 维度
- **优化**：dkey kernel grid 从 `(bs1_chunk, num_k_blocks, num_d_blocks)` 改为 `(B, num_k_blocks, num_d_blocks)`：
  - S1 循环从 grid 维度移入 kernel 内部：`for s1 in range(S1)`
  - 每个 program 负责完整的 `(b, k_block, di_block)`，内部遍历所有 s1
  - `dki_blk` 跨 S1 累积，最后单次 `tl.store`（非 atomic_add——每个 program 写独立位置）
  - 消除 bs1_chunk 拆分管理逻辑
- **收益**：S1=512 和 S1=4096 统一处理，性能无损。grid 大小从 `O(B*S1)` 降为 `O(B)`，对 S1=4096 尤其重要

### 2.7 代码模块化（共用 Cube 算子）

- 将 `_dense_indexer_qi_cube`、`_dense_indexer_ki_cube`、`_dense_indexer_dot_cube` 抽为独立 `@triton.jit` 函数
- 三个 kernel（stats/loss/main_grad）共享同一套 cube 原语
- `_dense_teacher_p_tile` 独立封装 teacher 分布计算（含 H 循环、score、softmax）
- `_grad_block_gd(Nidx1, D_idx)` 自动选择 BLOCK_G/BLOCK_DI：
  - 小 shape（`Nidx1 * D_idx ≤ 8192`）：BLOCK_G = Nidx1 全覆盖
  - 大 shape：BLOCK_G = max(16, 8192/D_idx)，BLOCK_DI = min(D_idx, 128)

### 2.8 双路径 API 设计

Dense 算子提供两套 LSE 入口以满足不同调用场景：

- **内部计算**（`_dense_loss_backward_core`）：kernel 内部运行 `_dense_indexer_stats_kernel` 计算 LSE
- **外部传入**（`_dense_loss_backward_with_index_core`）：接受外部预计算的 `softmax_max_index`/`softmax_sum_index`，跳过 stats kernel
- 对外暴露 `npu_dense_lightning_indexer_softmax_lse_triton` 供调用方提前分离 LSE 计算

### 2.9 与 SLI 的对比

| 特性 | Dense | SLI |
|------|-------|-----|
| KV 访问模式 | 全量 S2 循环（dense）| sparse_indices 随机 gather |
| 核心瓶颈 | teacher distribution + student softmax 重算 | 中间张量 HBM 流量 |
| 最大杠杆 | scalar→cube 路径升级 | kernel 融合 5→1 |
| di 复用 | workspace 物化（同 SFA Grad）| buf_i/buf_p L2-hot 复用 |
| Grid 设计 | 多 kernel + chunk 管理 | 单 kernel 全包含 |
| 共享 | `_ms_pyfunc` + infer_func | `_next_pow2` + dtype patch |

---

## 三、关键经验总结

| 经验 | 说明 |
|------|------|
| **di workspace 是中间结果复用的经典模式** | 同 SFA Grad 的 dS/P workspace——计算一次，多 kernel 共享，消除 2× 重算 |
| **scalar→cube 是 Ascend 优化的必修课** | `tl.sum`（vector）→ `tl.dot`（cube），让 Cube 单元参与计算，吞吐量数量级提升 |
| **Grid 降维减少 launch 开销** | 4D→3D→3D 每次降维消除一轮 grid 维度乘积，kernel launch 次数按 Nidx1 倍数减少 |
| **Causal skip 在 Dense 模式收益巨大** | S1=4096 时平均省 50% K-block 循环，Sparse 模式无此收益（topK 已过滤） |
| **online stats 是 softmax 相关计算的通用加速** | 两趟→单趟 chunked，与 SFA forward 的 online-softmax 同源 |
| **Grid 维度选择要考虑 shape 泛化** | S1 放在 grid 维度（高并行）vs 放在 kernel 循环（低 launch），需要根据 S1 大小权衡 |

---
