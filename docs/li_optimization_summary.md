# LI算子（LightningIndexer）Triton实现优化总结

---

## 一、整体优化效果概述

### 1.1 优化目标与结果

在 Ascend910_9382 + CANN 9.0 平台上，LI 算子的 Triton 实现经历了从初始实现到最终 **0.3× CANN** 的迭代优化。

LI 算子的计算模式本质上是 **Q@K^T dot → ReLU → 加权求和** 的重复过程，相比 SFA/SLI 更为规律。优化重心集中在 Grid 并行架构和 Autotune 体系，而非算法大改。

### 1.2 优化里程碑

| 阶段 | speedup vs CANN | 关键推动力 |
|------|----------------|-----------|
| 初始实现 | ~0.10× | 1D grid，功能验证 |
| Autotune + 2D grid | 0.14× | S2 维并行 |
| 输入预加载 + early-return 修复 | 0.75× | 排除数据传输 |
| BLOCK_S1 分块 + autotune prune | 0.73× ~ 1.20× | 多 shape 自适应 |
| Kernel 代码精简 + Phased UB | 4.24ms kernel | G=64 fuse + UB 准确估算 |
| 最终优化 | **0.3× CANN** | 全部 shape 达标 |

### 1.3 优化维度全景图

```
Grid 架构演进（1D → 2D → BLOCK_S1 串行化） ─┐
K tile 复用（G 循环外提升） ─────────────────┤
Autotune 体系（UB 估算 + prune + pow2 grid）─┼──→ 0.3× CANN
性能测试方法（输入预加载）───────────────────┤
代码精简（分支合并、acc 操作融合）──────────┤
Shape 兼容性（多 S1/N1/N2/D 组合）──────────┘
```

---

## 二、优化点子章节

### 2.1 Grid 架构演进（最大杠杆）

**2.1.1 初始 1D Grid**（commit `be5d5fb` ~ `272b38f`）

- **设计**：grid = `(B*S1*N2,)`，每个 program 处理一个 (b, s1, n2) 位置的**完整 S2 行**
- **问题**：S2 维度没有并行度，每个 program 串行扫完整个 S2（最多 16384），仅利用 N2×S1 个 core 的并行度

**2.1.2 2D Grid：S2 维并行化**（commit `82d4ca2`，核心架构变更）

- **思路**：将 grid 从 `(B*S1*N2,)` 改为 `(B*S1*N2, cdiv(S2, BLOCK_S2))`，引入第二维并行 S2 拆分。每个 program 只处理一个 S2 tile（BLOCK_S2 个位置），多个 program 并行覆盖整个 S2 维度。
- **实现**：
  - `pid_bsn`（grid 第0维）：标识 (b, s1, n2) 位置
  - `pid_s2`（grid 第1维）：标识 S2 tile 偏移
  - 每个 program 内 `for s2_start in range(...)` → 直接用 `s2_offs = pid_s2 * BLOCK_S2 + tl.arange(...)`
- **收益**：S2 维度的并行度从 1 提升到 `cdiv(S2, BLOCK_S2)`（约 32~128 倍），大幅降低单个 program 的工作量
- **配套改动**：autotune config 从 {BLOCK_S2, BLOCK_D, BLOCK_G} 三维变为需要四维（后续引入 BLOCK_S1）

**2.1.3 BLOCK_S1 分块：Grid 总数控制**（commit `b52d652`，autotune prune 体系建立）

- **问题**：2D grid 化后，grid 第0维 = `B*S1*N2`，大 shape（如 B=1, S1=16384, N2=1）时 grid0=16384，加上 grid1≈128，总 program 数 16384×128=2M，远超 Ascend coreDim 上限 65535
- **思路**：引入 `BLOCK_S1` — 每个 program 串行处理 BLOCK_S1 个 bsn 位置，将 grid0 压缩为 `cdiv(B*S1*N2, BLOCK_S1)`
- **实现**：kernel 内 `for i in range(BLOCK_S1)` 外循环 + `bsn_in_range` mask 处理 padding 越界
- **shape 自适应**：
  - S1=128：BLOCK_S1=8 最优（grid0=num_cores 适中）
  - S1=4096：BLOCK_S1=1 最优（并行度最大化，grid0=4096 未超限）
  - S1=16384：BLOCK_S1=16 选中（grid0=1024，grid1×grid0 ≤ 65535）
- **autotune prune 规则**：`grid0 * grid1 > _GRID_LIMIT`（65535）则裁剪
- **收益**：大 shape（S1=16384）下通过增加 program 内串行工作量将 grid 控制在 Ascend 硬件限值内，speedup 0.73（S1=4096）~ 1.20（S1=16384）

### 2.2 K Tile 复用（commit `272b38f`/`96bba3c`）

- **核心洞察**：LI 的计算是 `score[n2, s2] = sum_g(ReLU(Q[g] · K[n2, s2]) * W[g])`。在 G 维度的迭代中，`K[n2, s2]` tile 与 g 无关但被每个 g-block 重复加载。
- **思路**：从伪代码 `for g in range(G): load Q[g]; load K[n2,s2]; dot` 变为 `load K[n2,s2] once; for g in range(G): load Q[g]; dot(K_cached)`
- **实现**：K tile 的加载放在 D 循环内、G 循环外（实际代码中 K 加载仍在 D 循环内，但 D 循环与 G 循环是内外嵌套，`k_tile` 对 G 迭代保持不变，编译器做 CSE）
- **注解**：代码注释明确标注 "k_tile 与 g 无关却每个 g-block 重 load 一遍，K 带宽吃紧时可提到 g 循环外复用"

### 2.3 Autotune 体系搭建

**2.3.1 UB 估算方法演进**

| 版本 | 方法 | 问题 |
|------|------|------|
| 初始（b52d652） | `(acc+k+q+trans+score) × 1.25`，粗估 multi-buffer + trans 额外开销 | 系数 1.25 欠估，实测 `(256,128,64)` 需要 ~262KB |
| Phased（fbc5325） | `max(phase1, phase2)`，Phase1=dot(q+k+acc+trans), Phase2=reduce(acc+score) | 精确反映编译器释放 Phase1 buffer 后的峰值 |

**2.3.2 Prune 规则**

- UB 上限：`_UB_LIMIT_BYTES = 192 * 1024`（单核 192KB）
- `BLOCK_S2 > S2` 或 `BLOCK_D > D` 或 `BLOCK_G > G` → 裁剪
- `grid0 * grid1 > _GRID_LIMIT`（65535 Ascend coreDim 硬上限）→ 裁剪
- 全部被裁后兜底：取 UB 占用最小的 config

**2.3.3 Autotune Config 策略**

- `BLOCK_S1 ∈ {1, 2, 4, 8, 16, 32, 128, 256, 512}`：小值并行度高、大值压缩 grid
- `BLOCK_S2 ∈ {128, 256, 512, 1024}`：越大单 tile 越饱满、但 UB 压力越大
- `BLOCK_D ∈ {16, 32, 64, 128}`：大值减少 D 循环次数、小值省 UB
- `BLOCK_G ∈ {16, 64}`：16 通用、64 单次 G 迭代 cover 所有 G（消除 K 重复加载）
- 总 16 个 config，覆盖 S1=128 到 16384 的全范围

**2.3.4 Grid Pow2 Padding**（commit `947e632`）

- **问题**：Ascend 要求 kernel grid 每维为 2 的幂，非 pow2 导致分核映射出错 → aicore trap（B=3 时实测触发）
- **思路**：`_next_pow2(cdiv(...))`，padding 多出的 program 在 kernel 内靠 `bsn_in_range` mask 空转
- **关键参数**：`_GRID_LIMIT = 65535`（Ascend coreDim 硬件上限，commit `f9012ac`），原本 131072 为过估计

### 2.4 性能测试方法优化

**2.4.1 输入预加载到 Ascend**（commit `35e93b1`，speedup 0.33→0.75）

- **问题**：性能测试中 benchmark 框架在每次 kernel launch 前隐式做 Host→Device 数据传输，这个传输时间被计入算子耗时，导致 speedup 仅 0.33
- **思路**：测试前显式 `tensor.to('Ascend')`，排除数据传输开销
- **收益**：speedup 从 0.33 跳变到 0.75（2.27×），说明早期性能瓶颈主要不在 kernel 本身

**2.4.2 内存管理**（commit `e4af012`）

- **问题**：LI 性能测试多次重复运行（rep=100）导致设备内存不足
- **修复**：减少 rep 100→50；do_bench 中强制释放 Python 侧内存（CANN 侧仍会残留）

### 2.5 Kernel 代码精简

**2.5.1 Early-return 合并**（commit `35e93b1`）

- **问题**：kernel 中 `if s1 >= act_q: return` 和 `if pid_s2*BLOCK_S2 >= visible_limit: return` 两个独立 early return，在 triton-ascend 上同时存在导致 score 写入 bug（部分 program 的 store 被错误跳过）
- **修复**：合并为一个 `if s1 >= act_q or pid_s2*BLOCK_S2 >= visible_limit` + 统一 `tl.store(-inf)`
- **关键约束**：triton-ascend 一个 kernel 内只能有**一个早退出口**

**2.5.2 Acc 操作融合**（commit `88b9d9f`，内核提升 7ms）

- **原始代码**：
  ```python
  acc = tl.maximum(acc, 0.0)
  acc = tl.where(g_valid[:, None], acc, 0.0)
  tile_scores += tl.sum(acc * w_g[:, None], axis=0)
  ```
- **优化后**：
  ```python
  tile_scores += tl.sum(tl.maximum(acc, 0.0) * w_g[:, None], axis=0)
  ```
- **效果**：将 3 个操作（ReLU + mask + sum）融合为单行，减少中间临时变量，编译器可做更激进的融合优化。内核整体提升 7ms

### 2.6 精度与 Bug 修复

**2.6.1 `sum → dot` 精度修复**（commit `9d517bc`）

- **问题**：初版用 `tl.sum` 做 Q@K 内积，精度不足
- **修复**：改为 `tl.dot(q_tile, tl.trans(k_tile))`，走 cube 路径获得 fp32 累加精度

**2.6.2 TopK Bug**（commit `374eb2b`）

- **问题**：S1=S2 时 causal mask 导致所有位置 -inf，topk 返回 inf 而非 -1
- **修复**：invalid 位置 "inf" → "-1"

**2.6.3 `_infer_core` 修复**（commit `6e72c98`）

- MindSpore `_ms_pyfunc` 的 infer_func 签名需与 core 函数严格对齐

### 2.7 Shape 兼容性

- **初始**（commit `a05e3e9`）：支持可变 N1, N2, D（GQA: N1 必须是 N2 的倍数，G = N1//N2）
- **布局**：BSND + TND（含 TND↔BSND 转换，PyNative only）
- **sparse_mode**：0（full）+ 3（rightDownCausal）
- **D 维度**：128/256/512（与 CANN 对齐）
- 不支持 PA_BSND（block_table）

### 2.8 与 SLI/SFA 的共性与差异

| 特性 | LI | SLI | SFA |
|------|----|-----|------|
| 计算模式 | Q@K + ReLU + weighted sum | Multi-stage fused | Online softmax flash attn |
| 核心杠杆 | Grid 架构 + Autotune | Kernel 融合 5→1 | 算法演进 two-pass→chunked |
| Grid 设计 | 2D (bsn-blocks, S2-tiles) | 1D (B*S1,) | 2D (BS1, N1-blocks) |
| 共享基础设施 | `_next_pow2`, `_patch_triton_ascend_mindspore_dtype_bytes`, `_ms_pyfunc` + infer_func |
| UB 估算 | Phased: max(phase1, phase2) | 总和对所有 tile | Phased: max(phase1, phase2) |

---

## 三、关键经验总结

| 经验 | 说明 |
|------|------|
| **Grid 设计决定并行度上限** | 1D→2D grid 让 S2 维度并行，是算法不变前提下的最大杠杆 |
| **性能测试要排除无关因素** | 输入预加载到 Ascend 后 speedup 跳变 2.27×，说明早期瓶颈在数据搬运而非 kernel |
| **UB 估算要从粗到细** | 系数法→Phased 法，逐步精化才能解锁更多 config |
| **triton-ascend 的 early-return 限制** | 多个 early return 导致 store 丢失 bug，必须合并为单一出口 |
| **K tile 复用是低垂果实** | k_tile 对 G 循环不变，编译器 CSE 可自动优化 |
| **autotune key 要包含足够维度** | 从 `["S2", "D", "G", "sparse_mode"]` 扩展到 `["B", "S1", "S2", "N1", "N2", "D"]`，因为 S1 决定了最优 BLOCK_S1 |

---
