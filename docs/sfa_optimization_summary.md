# SFA算子（SparseFlashAttention）Triton实现优化总结

---

## 一、整体优化效果概述

### 1.1 优化目标与结果

在 Ascend910_9382 + CANN 9.0 平台上，SFA 算子的 Triton 实现经历了从基线实现到最终 **0.3× CANN** 的迭代优化。

### 1.2 优化里程碑

| 阶段 | 指标 | 关键推动力 |
|------|------|-----------|
| 基线（two-pass） | 内核 5181us | 初始功能验证 |
| SINGLE_BLOCK 快路径 | 内核 1971us（2.63×） | topK≤128 单次 score/P 计算 |
| Chunked online-softmax | 端到端 137ms→48ms（2.85×） | fp32 全局累加器 + last-block 融合 |
| Precision/UB 修复 | 67ms（bf16 验证通过） | BLOCK_DV 封顶 + D 依赖 UB 乘数 |
| Device-gather 路径 + 持久化缓存 | **0.3× CANN** | 双路径架构 + 消除分配开销 |
| S1=512/4096 兼容 | 0.3× CANN 保持 | 大 shape 最优配置选择 |

### 1.3 优化维度全景图

```
算法演进（two-pass → chunked online-softmax） ──┐
SINGLE_BLOCK 快路径 ────────────────────────────┤
双路径架构（inline vs device gather） ──────────┼──→ 0.3× CANN
BLOCK 参数 + autotune + UB 估算体系 ────────────┤
内存优化（持久化 workspace cache）──────────────┤
编译优化（load-order, loop-invariant） ─────────┘
```

---

## 二、优化点子章节

### 2.1 算法演进：Two-pass → Chunked online-softmax（最大收益 2.85×）

**2.1.1 基线 two-pass 设计的问题**（commit `d0edeb1`）

- **思路**：Pass 1 流式扫描 KV 计算 online-softmax 全局统计量（m_i, l_i，仅 `[BLOCK_G]` 向量）；Pass 2 按 BLOCK_DV tile 输出维度，每 tile 重新流式扫描 KV、recompute scores、累加 P@V。
- **问题**：对于 topK=2048, BLOCK_K=64 的场景，需要 32 个 K-block，每个 K-block 内又有 nope+rope 的 D 维度循环。Pass 2 中每个 dv-tile 都要完整重算 scores，导致 Q@K dot 次数为 `k_blocks * dv_tiles`（约 928 次 dot）。

**2.1.2 Chunked online-softmax 改造**（commit `fe3f515`，137ms→48ms）

- **思路**：单次流式扫描 KV，每个 chunk 内：
  1. 计算当前 chunk 的 scores 和 softmax
  2. 用 alpha_old/alpha_new 修正因子对 fp32 全局累加器做在线修正
  3. 直接累加 `P@V * alpha_new` 到 fp32_acc
  4. 最后一个 chunk 直接做归一化写入输出（省一次全 D 维读取回写）
- **关键技术点**：
  - `fp32_acc_ptr` 放在 Global Memory，不占 UB
  - `alpha_old = exp(m_i - m_new)`, `alpha_new = exp(m_blk - m_new)` 在线修正
  - Last-block fusion：`out = (acc * alpha_old + pv) / l_safe`，省一次独立的正常化 dv-tile pass
- **收益**：dot 次数从 928 降到 288（减少 ~69%），端到端 137ms→48ms（2.85×）
- **代价**：增加了每个 K-chunk 内的 dv-tile 累加（chunk 数 × dv_tiles 次 load-modify-store），但相比 two-pass 的 score 重算开销仍大幅净赢

**2.1.3 p_raw 驻留策略**

- chunked 路径中 `p_raw[BLOCK_G, BLOCK_K]`（~8KB fp16）在 dv loop 期间保持 UB 驻留，避免每次 dv-tile 重算 exp。这是 chunked 路径比 two-pass 快的关键：two-pass 中 Pass 2 每个 dv-tile 都要重算 scores+exp。

### 2.2 快路径：SINGLE_BLOCK（commit `1205994`，内核 2.63×）

- **思路**：生产 shape topK∈{128, 256}，对于 topK≤128（即 `_next_pow2(topK) ≤ 128`）的场景，整个 sparse 窗口可由单个 K-block 覆盖。此时 score/P 计算一次并保持驻留，然后 dv-tiled P@V，无需 K 维 chunk 循环。
- **实现**：编译期常量 `SINGLE_BLOCK` / `BLOCK_TOPK`，运行时分支选择 fast path vs chunked path
- **收益**：消除 K 维循环开销 + score 重算，内核 5181us→1971us（2.63×）
- **限制**：仅 topK≤128 时可用（生产 shape 的主要场景）

### 2.3 双路径架构：Inline-gather vs Device-gather（commit `f8d3624`）

**2.3.1 问题**

- **Inline-gather 路径**：原始设计中 `_sfa_kernel` 在每个 K-tile 内做 `index_select_simd`（随机 gather），对 Ascend 上大量轮询 CPU 的 MTE 不友好。
- **Device-gather 路径**：预先把 sparse KV 窗口 gather 成连续 buffer `[B*S1, topK, D]`，attention kernel 做连续顺序读。

**2.3.2 双路径自动选择**

- **实现**：`_sfa_gather_kernel` 将 sparse K/KR 按 index 预 gather 到连续 workspace，`_sfa_kernel_gathered` 在连续 buffer 上做 attention
- **选择逻辑**：`gather_workspace_bytes <= 2GB` 时走 device-gather 路径，否则 fallback 到 inline-gather
  - S1=512（典型生产 shape）：workspace ~1GB，走 device-gather
  - S1=4096：workspace > 2GB（超出限值），fallback 到 inline-gather
- **收益**：
  - 连续 K buffer 允许更大的 autotune tile（无 `index_select_simd` 的随机访存限制）
  - 连续 KV 访问更好地利用 MTE burst 传输

**2.3.3 持久化 Workspace Cache**（commit `f8d3624`）

- **思路**：`_GATHER_WS_CACHE` 字典按 `(B_S1, topK, D, D_ROPE, dtype)` 缓存 device-gather workspace，跨调用复用
- **收益**：消除每次调用 allocate/zero/free 大 workspace 的开销（对 benchmark 的 end-to-end 时间主导因素），是大 shape 场景从 ~66ms 降到 0.3× CANN 的关键推手

### 2.4 BLOCK 参数体系与 Autotune 演进

**2.4.1 Phased UB 估算**（commit `fe3f515`→`f8d3624`）

- **问题**：初版 UB 估算简单将所有 tile 求和 ×2.0（multi-buffer 翻倍防护），无法表示实际的阶段性 UB 峰值
- **改进**：分为 Phase 1（score 计算：q_tile, k_tile, qr_tile, kr_tile, scores, m_l）和 Phase 2（P@V：p_tile, v_tile, pv_tile, acc_dv, m_l），峰值取 max。两份 tile set 在编译器的生命周期不重叠（Phase 1 完成后可释放），真实峰值更低
- **效果**：解锁了更多激进 tile 配置

**2.4.2 D 依赖 UB 乘数**（commit `c40cb3f`）

- **问题**：D≤128 时编译器可能将 nope/rope tile 和 scores 同时保持活跃（BLOCK_D 覆盖完整 D/Dr 时仅一次迭代），导致真实 UB 峰值 > 2.0× 单次估算
- **改进**：`ub_multiplier = 3.0 (D≤128) / 2.0 (D≥256)`，D=256/512 使用 1.0（无额外乘数）
- **验证**：D=128 时 2.0× 乘数下 fp16/bf16 均出现 UB overflow（实测验证），3.0× 安全

**2.4.3 Autotune 架构**

- **内联 gather 路径**：20+ 配置，覆盖 BLOCK_G∈{8,16,32,64}, BLOCK_K∈{16,32,64,128,256}, BLOCK_D∈{64,128,256}, BLOCK_DV∈{64,128,256,512} 的组合
- **Device-gather 路径**：6 配置，主要用 `{BLOCK_G: 64, BLOCK_K: 128, BLOCK_D: 128, BLOCK_DV: 128}`（因连续 KV 读取效率高）
- **Config 裁剪规则**：
  - `BLOCK_D/DV > D` 时裁剪（`index_select_simd` 无 mask → 越界 MPU 访问）
  - `BLOCK_K > topK` 时裁剪（浪费 UB + last-block 逻辑读负偏移）
  - `BK*BD > 8192` 或 `BK*BDV > 8192` 裁剪（CUBE B-matrix MTE 地址越界，实测归纳）
  - D≤128 时 BLOCK_G>16 强制裁剪（UB overflow，BG=32/64 实测 crash）
  - Grid 总数 > 131072 裁剪（Ascend grid 限制）
  - 大 shape（BS1*N1*D > 80MB）仅保留 2 个最优配置对（避免 autotune 累积 VMM 贴满）

**2.4.4 冗余配置清理**（commit `ecd56bf`）

- 大量被注释掉的激进配置（BLOCK_DV=256/512, BLOCK_D=256/512 等）改回注释，仅保留实测安全通过的配置
- 效果：autotune 编译时间显著缩短（配置数从 ~50 降到 ~20）

### 2.5 精度与 UB 稳定性

**2.5.1 BLOCK_DV 封顶（64）**（commit `321f2dc`）

- **问题**：chunked fp32_acc correction 在编译器中产生约 4 个 `[BG,DV]` fp32 临时变量（tl.dot 结果 + load acc_dv + m_new × acc + pv_tile），编译器实际 ~4× 放大。BDV=128（每 temp 8KB，4×→32KB）在 D=128 场景下 UB 超限。
- **修复**：所有 autotune config 中 BDV 封顶 64（每 temp 4KB），峰值 UB 保持在 ~152KB < 192KB 限值
- **代价**：DV 迭代次数增多（D=512 时 8 次 vs 4 次），但被精度稳定性权衡接受

**2.5.2 精度基线变更**（commit `c40cb3f`）

- 从 fp16 基准切换到 bf16 生产精度基准
- bf16 下 UB 估算乘数从 2.0 改为 3.0（bf16 编译器对不同 tile layout 的 multi-buffer 策略不同）
- 最终 bf16 验证通过，67ms

### 2.6 编译优化（Load Order & Loop Invariant）

**2.6.1 Chunked 路径内联 Score 计算**（commit `f8d3624`）

- **思路**：chunked 分支中不再调用 `_sfa_scores_block` 独立函数，而是把 score 计算（Q@K + QR@KR）直接内联在 K 循环体内
- **效果**：编译器可以看到 Q/QR load 在 `blk_start` 循环中是 loop-invariant，可将其提升到循环外（loop-invariant hoisting），减少指令数

**2.6.2 Load 顺序优化**

- **Q before K**：Q load 不依赖 `tok_clamped`，可与上一迭代的 fp32_acc store 重叠
- **fp32_acc before V**：fp32_acc load 独立于 V 地址计算，可与上一迭代的 store 重叠

**2.6.3 Alignment Hints**

- 所有输入指针使用 `tl.multiple_of(ptr, 128)` 提示 128 字节对齐，允许编译器生成对齐向量访存指令

### 2.7 Shape 兼容性扩展

**2.7.1 S1=512 vs S1=4096 兼容**（commit `d84a17d`）

- **S1=512**（典型生产）：BS1*N1*D ≤ 80MB，走 device-gather 路径
- **S1=4096**（大 shape）：BS1*N1*D > 80MB，强制 inline-gather 路径 + 限制 autotune 配置为 2 个最优对
- **关键限制**：CUBE_TILE_LIMIT=8192 —— Ascend Cube B-matrix 单 tile 元素数上限，超过则 MTE 地址越界（实测归纳）
- **Grid pow2 padding**：Ascend 要求 kernel grid 每维都是 2 的幂，padding 多出的 program 在 kernel 内通过 `in_range` mask 空转（idle）

**2.7.2 Block-wise 稀疏索引展开**

- `_expand_block_indices`：block id → 逐 token 索引展开（`topK → topK*block_size`），在 kernel 外部完成（纯静态 tensor 操作，GRAPH_MODE 兼容）
- Kernel 内部始终使用 token-wise 路径，简化索引逻辑

### 2.8 基础设施

**2.8.1 Layout 标准化**

- 支持 TND/PA_BSND → BSND 的 host 端变换（PyNative 下），`_tnd_to_bsnd` / `_pa_to_bsnd` / `_bsnd_to_tnd`
- 返回 TND 结果的 softmax_max/sum 也做逆向变换

**2.8.2 与 SLI 的共性**

- 均使用 `_ms_pyfunc` + infer_func 做 MindSpore 图模式集成
- 均使用 `_patch_triton_ascend_mindspore_dtype_bytes()` 修复 triton-ascend autotuner dtype 查询
- Grid pow2 通用化：`_next_pow2` 工具函数
- 共享 UB 估算方法论（multi-buffer 翻倍, D 依赖乘数）

---

## 三、关键经验总结

| 经验 | 说明 |
|------|------|
| **算法创新是最大杠杆** | two-pass → chunked online-softmax 减少了 69% dot 次数，端到端 2.85× |
| **快路径设计要匹配生产 shape** | 生产 topK=128 使 SINGLE_BLOCK 成为常态路径，内核 2.63× |
| **双路径架构是 shape 泛化的关键** | device-gather 对小 shape 优化大 tile，inline-gather 对大 shape 兜底 |
| **Workspace 持久化对 benchmark 至关重要** | 消除 alloc/free 开销是达到 0.3× 的最后一公里 |
| **UB 估算是编译期安全的基石** | phased 估算 + D 依赖乘数 + 实测验证的三层防护体系 |
| **编译器行为需实测归纳** | CUBE_TILE_LIMIT=8192、BDV=128 的 UB overflow 等现象均为实测发现，非文档给出 |
| **autotune 配置过多有反效果** | 大 shape 场景多配置累积 VMM 贴满，最终只保留 2 个最优对 |

---
