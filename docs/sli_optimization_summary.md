# SLI算子（SparseLightningIndexerGradKLLoss）Triton实现优化总结

---

## 一、整体优化效果概述

### 1.1 优化目标与结果

在 Ascend910_9382 + CANN 9.0 平台上，SLI算子的 Triton 实现经历了从基线 `816.71ms` 到最终 `48.34ms` 的迭代优化，端到端耗时降低 **16.9 倍**，达到了 **0.34× CANN**（CANN 端到端 16.67ms）的性能目标。

### 1.2 优化里程碑

| 阶段 | 耗时 | vs CANN | 关键推动力 |
|------|------|---------|-----------|
| 基线（5 kernel） | 816.71ms | 18.5× | 初始功能验证 |
| BLOCK 调优 | 694.21ms | 15.9× | 四维 BLOCK 参数放大 |
| Gather 融合 | 686.70ms | 15.7× | 3→1 kernel launch |
| 向量化 dQI/dKI | ~450ms | ~10× | tl.dot 走 cube path |
| K1+K2 融合 + 消中间张量 | 223.49ms | 5.4× | 省 9GB HBM 流量 |
| 5→4→3→2→1 kernel 融合 | 219→150→48ms | 5.0×→3.5×→1.15× | 全链路单 kernel |
| fp32 upcast + s_idx_buf | 48.34ms | **0.34×** | 精度/性能双重优化 |

### 1.3 优化维度全景图

优化围绕 **六大维度** 展开，各维度相互交织，最终形成了一条从"功能可用"到"性能达标"的完整优化链：

```
Kernel 融合 (5→1) ───────────────┐
BLOCK 参数调优 ──────────────────┤
向量化 & Cube 通路 ──────────────┼──→ 0.34× CANN
内存与中间张量消除 ──────────────┤
精度与数值路径优化 ──────────────┤
基础设施与 Shape 扩展 ───────────┘
```

---

## 二、优化点

### 2.1 Kernel 融合（Launch Overhead 消除）

**2.1.1 Gather Kernel 三合一**（commit `e8ea5e3`）

- **思路**：原 3 个独立的 `_gather_kv_kernel` 调用分别 gather key(D)、key_index(D_idx)、key_rope(D_rope)，每次 launch 都有 host-device 同步开销。合并为单次 launch，通过 grid 维度 `max(D, D_idx, D_rope)` 保证各 buffer 完整 gather。
- **收益**：每次 chunk 省 2 次 launch overhead，端到端 ~9ms（~1.3%）。

**2.1.2 Teacher + Gather 融合**（commit `ed82adf`）

- **思路**：`_teacher_distribution_kernel` 本身就要加载 `sparse_indices` 做 k/kr gather，顺带 gather `key_index`，省去独立的 `_gather_kv_kernel`。s_idx_buf 改 fp16 减半 HBM 带宽。
- **收益**：5→4 kernel，225ms→219ms（~2.7%）。

**2.1.3 K1+K2 融合 + 消除中间张量**（commit `42ebd08`）

- **思路**：方案 A — 消除 `key_gathered`/`key_rope_gathered` 两个中间张量（合计 9GB HBM）。`_teacher_distribution_kernel` 直接从原始 k/kr 按 sparse_indices 间接读，每 K-tile 一次性 load idx（~1KB UB）后复用。
- **收益**：239ms→223ms（1.07×），省 9GB HBM 读写流量。

**2.1.4 Indexer + KL + dI 融合**（commit `01f70e4`）

- **思路**：Stage 3+4 的三次 K-loop 合并为单次 full-VALID_K UB load（~8KB fp32），`i_full`/`exp_i_full`/`log_i_sum` 复用，避免 `buf_i` 重读 3 次。移除 `qi_tile`/`ki_tile` 的 `.to(tl.float32)` 强转，恢复 fp16 cube path。
- **收益**：282ms→239ms（1.18×），K3 cube_ratio 从 0.05% 提升。

**2.1.5 K1 并入 K23 → 单 kernel**（commit `bea6fae`）

- **思路**：将 `_teacher_indexer_kl_kernel` 和 `_indexer_grad_kernel` 合并为 `_sli_grad_fused_kernel`。Stage T/I/Final 之后直接接 Pass A (dW/dQI) 和 Pass B (dKI scatter)。`di`/`s_idx_buf`/`key_index_gathered`/`buf_p`/`buf_i` 全程在同一 program 内保持 L2-hot。
- **收益**：2→1 kernel，每次 chunk 省 1 次 launch + 1 次 host sync。

### 2.2 BLOCK 参数调优

**2.2.1 第一轮 BLOCK 放大**（commit `deddb34`）

- **思路**：四个维度的 BLOCK 参数系统性放大：
  - `BLOCK_K_GATHER`: 128→256，grid 维度减半
  - `BLOCK_K_QUERY_WEIGHT`: 64→128，K 分块数减半
  - `BLOCK_G_QUERY_WEIGHT`: 2→4，每 program 处理更多 group
  - `BLOCK_K_SCATTER`: 64→128，K 分块数减半
- **收益**：816ms→694ms（-15.0%）。
- **风险控制**：`BLOCK_K_MAIN=256` 导致 UB 溢出（保持 128），`BLOCK_D_GATHER=256` 同样溢出（保持 128）。

**2.2.2 Teacher BLOCK 独立解耦**（commit `77b9b9b`）

- **思路**：
  - `BLOCK_H_TEACHER`: 32→64，N1=64 时单次 H 循环完成，消除 K 重复加载
  - 新增独立 `BLOCK_D_TEACHER=128`/`BLOCK_K_TEACHER=128`
  - 快路径：`if BLOCK_H >= N1` 消除 broadcast/mask 开销
  - softmax 除法改乘倒数
- **收益**：3919us→2600us（1.5×），aiv_mte2_active_bw 39→119 GB/s。

**2.2.3 BLOCK 参数回退经验**

- `BLOCK_K_TEACHER`: 64→128 **失败**（Revert），UB 估算 130KB 仍 ok 但触发未知精度/UB 问题（commit `1f5ac83`）
- `BLOCK_G_QUERY_WEIGHT`: 64→32（9382 UB 溢出修复，commit `353758d`），单 chunk 耗时 995→1947us，但相对原始 6452us 仍有 3.3×
- `BLOCK_K_SCATTER`: 256 方案多轮迭代（方案1/方案3），最终回到 (256,128,32)（commit `1a0bc71`），证伪 9382 在短 M GEMM 上启用 cube 的假设

### 2.3 向量化 & Cube 通路

**2.3.1 dQueryIndex 向量化（6.4×加速）**（commit `92c2638`）

- **思路**：将 `_query_index_weight_grad_kernel` 中 `BLOCK_G` 上的 `tl.static_range` 标量展开改为二维 tile `[BLOCK_G, BLOCK_K]`，K 维归约用 `tl.dot` 走 cube。`BLOCK_D` 覆盖整个 `D_idx`，D 维 grid 消除。grid 从 131072 降至 4096。
- **收益**：6452us→995us（6.4×），kernel 升级为 mix_aic。

**2.3.2 dKeyIndex 向量化 + Cube 探索**（commit `82d2390`）

- **思路**：`_scatter_dkey_index_kernel` 内层 `for g in range(Nidx1)` 标量循环提升为 BLOCK_G tile，`[BLOCK_K, BLOCK_G] @ [BLOCK_G, BLOCK_D]` 走 `tl.dot` cube。BLOCK_D: 64→128 覆盖 D_idx。
- **收益**：4642us→2266us（2.05×）。
- **后续优化**：`ki_tile` 保持 fp16 进 `tl.dot`，省 32KB UB（multi-buffer 翻倍后），单 chunk 1947→972us（commit `0c69646`），累计 6.6×。

**2.3.3 Cube 路径 vs Vector 路径的实践经验**

- **9382 + CANN 9.0 的 cube 启用条件严格**：短 M 维 GEMM `[128,64]@[64,128]` 未触发 cube（commit `3a3cfbb`），需要 M ≥ 128 的方阵才能稳定启用
- **tl.dot 输入保持 fp16 是 cube 通路的前提**：显式 `.to(tl.float32)` 会强制走 vector path（commit `01f70e4`）
- **tl.dot(..., acc=acc) 在低 cube 利用率场景劣化**：4503→6104us（commit `49a9760`/`435b448`），L0C 累加引入额外上下文切换抵消费用

### 2.4 内存与中间张量消除

**2.4.1 中间 Buffer Hoist**（commit `62477ec`）

- **思路**：chunk 间可复用的中间 buffer（`di`/`s_idx_buf`/`buf_i`/`buf_p`/`key_index_gathered`）提升到 chunk loop 外，避免每次 chunk 分配。

**2.4.2 消除 key_gathered/key_rope_gathered 中间张量**（commit `42ebd08`）

- **思路**：如 2.1.3 所述，Teacher kernel 直接从原始 k/kr 按 sparse_indices 间接读取，省去 9GB HBM 中间张量。
- **收益**：HBM 流量显著降低，端到端 239→223ms。

**2.4.3 s_idx_buf 精度策略演进**

- **fp32 → fp16**（commit `ed82adf`）：减半 HBM 带宽，relu 值受限于 fp16 输入不损失精度，dW 累加时再升回 fp32
- **fp16 → fp32**（commit `a108e1d`）：消除 bf16 二次舍入损失 dW 精度（目标机 910_9382）
- **选择性 fp32**（commit `3df314e`）：UB overflow 后仅保留精度瓶颈的两处 dot 输入 fp32 + ds_idx 保持 fp32，其余回退

**2.4.4 `mint.zeros` vs `mint.empty`**

- 尝试 `mint.empty` 替代 `mint.zeros` 省初始化开销，但导致错误结果（MindSpore empty 可能无 device memory backing），保持 `zeros`。

### 2.5 精度与数值路径优化

**2.5.1 H 维度循环修复**（commit `d48d0d5`）

- **问题**：`_teacher_indexer_kl_kernel` 中 H 维度无 tile 循环，`BLOCK_H_TEACHER=64` 时 N1>64 场景后段 head 不参与计算，`p_tile` 偏差导致下游全部精度失败。
- **修复**：`sm_max`/`sm_sum` 与 scores 计算下沉到 H 外层循环内累加 `p_tile_acc`。

**2.5.2 dot 输入 fp32 upcast（目标机优化）**（commit `6fa6b2d`）

- **思路**：5 处 `tl.dot` 输入 upcast 到 fp32，面向 910_9382/CANN 9.0 的 fp32 cube 能力。s_idx_buf 改 fp32 存 relu 中间值。
- **风险**：910B3/CANN 8.5 上 cube 不吃 fp32（已验证），改动需目标机验证。
- **UB 溢出修复**（commit `3df314e`）：回退 4 处 dot + s_idx_buf 的 fp32 化，只保留精度瓶颈的两处。

**2.5.3 Stage I/II Numpy 参考修正**（commit `ac30b5b`）

- **思路**：修正 Stage 1/2 多余 fp16 rounding 使其与真机 MMA 语义对齐；Stage 1/2/5/6 全部向量化。
- **收益**：大 shape numpy 参考从 838s 降到 24s（35×），测试覆盖扩展到 13 条包含所有二元组合。

### 2.6 基础设施与可靠性

**2.6.1 S1 维度 Padding**（commit `444d944`/`c44ae07`/`7028985`/`2eead55`）

- **问题**：910C MTE burst 安全性要求，S1 维度的输入张量需 +1 row padding。
- **修复**：父 S1 维度、per-chunk 输入、中间 buffer 均加 +1 row padding。

**2.6.2 `runtime.synchronize()` 位置修复**（commit `444d944`）

- **问题**：synchronize 在 `_ms_pyfunc` 外部触发 GRAPH_MODE 错误。
- **修复**：移入 `_ms_pyfunc` 内部。

**2.6.3 边界保护**（commit `66fe12f`）

- **问题**：gather/scatter kernel 缺少 `k_offs < topK` 边界 guard。
- **修复**：增加 `k_offs < topK` boundary guard。

**2.6.4 Shape 扩展**（commit `ac30b5b`）

- **思路**：Pass A/B 新增 D_idx 外循环，dQI/dKI 全列可覆盖（原来仅 D_idx=128）。
- **收益**：支持 `Nidx1={32,64,128}`、`D/D_idx={128,256,512}` 全组合。

**2.6.5 Profiling Marker 清理**（commit `2348433`）
- 移除 `_profile_marker_*` 内核、`_block_dot_1xD_vs_KxD`、`_cast_dkey_index_kernel` 及 SPARSE_GRAD_PROFILE_MARKERS 相关代码，主文件精简到 580 行。

---

## 三、关键经验总结

| 经验 | 说明 |
|------|------|
| **Kernel 融合是最大杠杆** | 5→1 kernel 直接消除多次 launch + sync overhead，同时让中间 buffer 保持 L2-hot |
| **Cube 通路的前提条件严格** | fp16 输入 + M≥128 + 方阵 GEMM 才能稳定启用 cube；显式 fp32 cast 或短 M 维都会退化到 vector path |
| **BLOCK 参数调优是双刃剑** | 放大 BLOCK 减少 grid/循环次数，但 UB 192KB 是硬限制，需精确估算 multi-buffer 翻倍后的 UB 占用 |
| **精度与性能的 trade-off 需分平台对待** | fp16 s_idx_buf 在部分场景损精度、fp32 在旧平台损性能；需针对性做平台适配 |

---
