# SFA Grad算子（SparseFlashAttentionGrad）Triton实现优化总结

---

## 一、整体优化效果概述

### 1.1 优化目标与结果

在 Ascend910_9382 + CANN 9.0 平台上，SFA Grad 算子的 Triton 实现经历了从初始到最终 **34ms（0.3× CANN）** 的迭代优化。

SFA Grad 核心洞察是：反向计算中大量操作可复用前向的 softmax 统计量，且 nope/rope 可提前拼接为单一维度，将 dot 操作数减半。

### 1.2 优化里程碑

| 阶段 | 端到端耗时 | 关键推动力 |
|------|-----------|-----------|
| 初始实现 | ~500ms+ | 多 kernel + autotune，功能/精度基线 |
| Agent 优化 | 精度通过 | 修复功能/精度问题 |
| 大重构 + 固定 config | **155ms** | Host concat nope\|rope，删 autotune，1D grid |
| BLOCK_K 拆分 + delta 预计算 | **130ms** | 独立 K_A/K_B + 消除 o 加载 |
| dS/P workspace + mask elision | **54ms** | 3 gathers→1, 8 dots→5 |
| GROUP_HC + cube co-location | **34ms** | Cube 利用率提升 + L0C 驻留 |
| 最终集成 | **0.3× CANN** | 全部 shape 达标 |

### 1.3 核心架构图

```
┌─────────────────────────────────────────────────────────┐
│ _sfa_grad_kernel (single launch, 1D grid)               │
│                                                         │
│ Stage A: dqcat 计算                                      │
│   for each head-chunk GROUP:                            │
│     load q_cat, do_pad, sm_max/sum, delta (host-pre)    │
│     for each K-tile BLOCK_K_A:                          │
│       ① gather k_cat (sparse fetch, once)               │
│       ② scores = q·k^T, dPv = do·k^T (cube co-located) │
│       ③ P = exp(scores-max)/sum, dS = P·(dPv-delta)·s  │
│       ④ acc_dqcat += dS·k (tl.dot acc=, L0C resident)  │
│       ⑤ store dS/P to GM workspace (bf16)              │
│     store dqcat (nope|rope)                             │
│                                                         │
│ Stage B1: dkcat scatter (reads dS from workspace)        │
│   for each K-tile BLOCK_K_B:                            │
│     for each head-chunk:                                │
│       load q_cat, load dS from workspace                │
│       acc_dkcat = dS^T·q (tl.dot acc=)                 │
│     atomic_add dkcat to fp32 output                     │
│                                                         │
│ Stage B2: dv scatter (reads P from workspace)            │
│   for each K-tile BLOCK_K_B:                            │
│     for each head-chunk:                                │
│       load do_tile, load P from workspace               │
│       acc_dv = P^T·do (tl.dot acc=)                    │
│     atomic_add dv to fp32 output                        │
└─────────────────────────────────────────────────────────┘
```

---

## 二、优化点子章节

### 2.1 Host 端数据预处理（收益：消除 2× dot + 1× 加载）

**2.1.1 nope|rope 拼接：D_TOT 单 dot 替换两段 dot**（commit `993b80a`/`a332492`）

- **思路**：反向过程中每个 K-tile 需要计算 `scores(q·k^T + qr·kr^T)`、`dPv(dO·k_nope)`、`acc_dq(dS·k)`。原始设计中对 nope 和 rope 分别做 dot（每项 2 次 dot）。通过 Host 端 `ops.cat((q_flat, qr_flat), axis=-1)` 将 nope 和 rope 拼接为 `D_TOT = D + D_ROPE` 的单一维度，每个 dot 从 2 次缩减为 1 次。
- **dO 补齐技巧**：`triton-ascend` 不支持列切片（无法从 `k_cat[..., :D]` 取 nope 子集），dPv 需要 `dO·k_nope`。将 dO 加载时 pad 到 D_TOT（rope 列填 0），对完整 k_cat 做 dot——rope 列贡献 0，结果等价于 `dO·k_nope`。
- **收益**：scores 计算 2 dots→1 dot，dPv 计算 2 dots→1 dot，dQ 计算 2 dots→1 dot。总计每个 K-tile 省 3 次 tl.dot。

**2.1.2 delta = rowsum(dO*O) 预计算**（commit `8f85811`）

- **原始设计**：kernel 内每个 head-chunk 加载 `o_pad`，计算 `delta = sum(dO * o_pad)`，每 head-chunk 重复
- **优化**：Host 端一次性计算 `delta_flat = (do_flat * o_flat).sum(axis=-1)`，kernel 只需 `tl.load(delta_ptr)`
- **收益**：
  - 消除 `o_pad` 的加载（~32KB 带宽/head-chunk）
  - 消除 kernel 内 fp32 归约操作（`tl.sum(axis=1)` 是 vector 操作，阻塞 cube）
  - delta 对整个 K-tile 循环是常量，只在 head-chunk 入口加载一次

### 2.2 dS/P Workspace：Gather 3→1，Dot 8→5（commit `8f85811`，130ms→54ms）

- **原始设计**：Stage A 计算 dqcat，Stage B 重新 gather k_cat 并重新计算 scores/P/dS 来做 dk/dv scatter。
- **问题**：每个 K-tile 需要 3 次 sparse gather（Stage A: k_cat gather 1 次 + Stage B1: k_cat gather 1 次 + Stage B2: k_cat gather 1 次），以及 8 次 tl.dot（Stage A: 3 dots + Stage B1: 3 dots + Stage B2: 2 dots）
- **优化**：Stage A 在计算 dqcat 的同时将 dS 和 P 物化写入 GM workspace（bf16，`[B*S1, N1, topK]`），Stage B1/B2 直接从 workspace 读取 dS/P，**无需重做 gather 或 recompute**
- **收益**：gather 3→1（省 2 次），dot 8→5（省 3 次）。130ms→54ms（2.4×）
- **约束**：workspace 额外占用 `2 × N1 × topK × sizeof(bf16)` ≈ 512KB/program（N1=64, topK=2048），可接受

### 2.3 单 Kernel 架构：L2-hot Workspace（commit `8e8b943`）

- **问题**：将 Stage A 和 Stage B 拆为两个独立 kernel 会导致 dS/P workspace 冷读——Stage A 写入后 kernel 退出，Stage B 需从 GM 重新加载（~77ms 回退）
- **设计**：所有三个阶段（A: dqcat, B1: dkcat, B2: dv）统一在单个 `_sfa_grad_kernel` 内执行
  - Stage A 写的 dS/P 在 L2 中保持热度
  - Stage B1/B2 读取时命中 L2，无需回 GM
- **Grid**：1D `(_next_pow2(B*S1),)`，每个 program 负责一个 (b,s1) 完整行，无跨 program 竞争

### 2.4 固定 Block Config，放弃 Autotune（commit `a332492`）

- **问题**：SFA Grad 的 dkcat/dv 输出通过 `tl.atomic_add` 写入共享 buffer。Autotune 每个 config 会跑多次 benchmark（do_bench），多次 atomic_add 叠加导致结果 double-count，无法正确 benchmark
- **方案**：放弃 autotune，使用 `_select_block_config(D, N1)` 固定配置
  - `BLOCK_G = min(16, max(8, N1))`：N1=64→16，N1<8→8，平衡 cube M 维 16×16 tile
  - `BLOCK_K_A = 32`：Stage A 使用。上限 32（64 时 k_cat 需在两种 dot orientation 中同时存在 = 144KB UB overflow）
  - `BLOCK_K_B = 32`：Stage B1/B2 使用。上限 32（`[BK_B, D_TOT]` fp32 累加器 64 时溢出）
  - `BLOCK_D = 128`：D/D_ROPE 分块粒度

### 2.5 Head Chunk 合并（GROUP_HC）：Cube 利用率提升（commit `8e8b943`，54ms→34ms）

- **问题**：Ascend Cube 微 tile 是 16×16。`BLOCK_G=8` 或 16 时，dot 的 M 维半填充微 tile，cube_ratio 仅 0.14%，且 dot 调用次数多（issue/sync-bound）
- **思路**：当 `NUM_HC >= 2` 时，Stage A 将 `GROUP_HC=2` 个 head chunk 合并为一次 dot，M 维放大到 `MG = 2*BLOCK_G`
  - M=16（GROUP_HC=2, BLOCK_G=8）或 M=32（GROUP_HC=2, BLOCK_G=16）时 cube 微 tile 全填充
  - dot 调用次数减半（k_cat 在合并的 heads 间共享 —— 同一个 sparse token set）
- **UB 约束**：`MG` 上限为 `2*BLOCK_G`。因为 `acc_dqcat[MG, D_TOT]` fp32 + `q_cat/do_pad[MG, D_TOT]` bf16 + cube 内部 fp32 upcast 拷贝必须与 `k_cat[blk, D_TOT]` + trans 共存于 192KB UB
- **HC_LOOP 兜底**：Triton 编译器在 `for hc in range(1)` 时 crash（scf.For assertion failure in ttir_to_linalg），当 `NUM_HC==1` 时设置 `HC_LOOP=2`，第二个迭代 g_valid 全 False 空转

### 2.6 Cube 操作协同定位（commit `8e8b943`）

- **问题**：原始代码 flow 为 `scores=dot → P=exp → dPv=dot → dS`，Cube（dot）和 Vector（exp/sub/mul）交替执行，产生 2 次额外的 cube→vector→cube 同步
- **优化**：scores 和 dPv 两个 `tl.dot` 连在一起在 cube 上执行，之后才切换到 vector 做 P/dS 计算
  ```python
  # Before: cube→vector→cube→vector (4 syncs)
  scores = tl.dot(q, kT)
  P = tl.exp(scores - sm_max) * inv_sum  # vector
  dPv = tl.dot(do, kT)
  dS = P * (dPv - delta)
  
  # After: cube→vector (2 syncs)
  scores = tl.dot(q, kT) * scale_value   # cube
  dPv = tl.dot(do, kT)                    # cube (still on cube)
  P = tl.exp(scores - sm_max) * inv_sum  # vector
  dS = P * (dPv - delta)                 # vector
  ```
- **收益**：Cube↔Vector 同步次数减半，每条指令的 issue/sync 开销显著降低

### 2.7 `tl.dot(acc=)` L0C 累加器驻留（commit `8e8b943`）

- **原始**：`acc += tl.dot(...).to(tl.float32)`，每次迭代 dot 结果从 Cube L0C→Vector 寄存器→fp32 UB，下一次迭代再从 UB 加载
- **优化**：`tl.dot(..., acc=acc)`，fp32 累加器直接驻留在 Cube 侧 L0C，避免每轮 cube→vector 同步 + UB 写回再读取
- **收益**：消除 3 处累加（acc_dqcat, dkcat_acc, dv_acc）的往返开销
- **与 SLI 的对比**：SLI 尝试 `tl.dot(acc=)` 反而劣化（cube 利用率 0.15% 太低），但 SFA Grad 的 cube 利用率提升后（GROUP_HC），L0C 驻留带来净收益

### 2.8 Mask Elision（commit `8f85811`）

- **问题**：Ascend 上 i32 LT/GT 比较被编译器标量 lowering，每个 K-tile 的 `blk_offs < topK` 是矢量→标量的昂贵操作
- **思路**：当 `topK % BLOCK_K_A == 0` 时，`blk_offs < topK` 对所有 K-tile 恒真——此时通过编译期常量 `NEED_BLK_MASK_A=False` 完全消除这个 mask 检查
- **实现**：
  ```python
  if NEED_BLK_MASK_A:  # False at compile time → dead code elimination
      blk_in_count = blk_offs < topK
      tok = tl.load(sparse_ptr + sp_base + blk_offs, mask=blk_in_count, other=-1)
  else:
      tok = tl.load(sparse_ptr + sp_base + blk_offs)  # bare load, no mask
  ```
- **效果**：eliminate per-iteration `blk_in_count` mask 的计算和分支

### 2.9 无 Autotune 但保留代码简洁性

- **Grid pow2 padding**：沿用 `_next_pow2` 确保 Ascend 分核映射正确
- **dtype patch**：沿用 `_patch_triton_ascend_mindspore_dtype_bytes`（与 SFA/SLI/LI 共享）
- **`TRITON_ENABLE_TASKQUEUE=false`**（commit `a332492`）：修复 Ascend 上 taskqueue 导致的 kernel 排序问题
- **无 early-return**：triton-ascend 上 early-return 会导致 store 被优化器丢弃。非活跃行通过 `tok_valid` 掩码自然处理（dS/P 变 0，无贡献）
- **g_offs 不 clamp**：`tl.where(g_valid, g_offs, 0)` 会让 BiShengHIR 编译器对 N1 做 specialize，head tile 缩成 size-1 导致 N1=1 编译失败。直接使用未 clamp 的 g_offs，靠 `mask=g_valid` 屏蔽越界访问（Ascend 上 masked 越界访问安全）

### 2.10 依赖关系管理（commit `506dc69`/`a332492`）

- **问题**：`from sparse_flash_attention_triton import ...` 导致框架接入时的循环依赖报错
- **解决**：先 inline 所有 helper 函数进 grad 文件（506dc69），后续大重构时恢复 import（a332492）——依赖链最终理顺

### 2.11 与 SFA 反向的数学恒等式

SFA Grad 利用了一个关键数学简化（已体现在初始设计中）：

```
delta = rowsum(dO * O)
dS = P * (dPv - delta) * scale
```

`delta` 是 `dO` 和 `O` 的内积（标量），host 端一次性计算。这个简化避免了 kernel 加载 `out` 张量并做 per-head-chunk 归约的开销。

---

## 三、关键经验总结

| 经验 | 说明 |
|------|------|
| **Host 预处理是免费午餐** | concat nope\|rope + delta 预计算，从 kernel 中移除了 2×dot + out 加载 + reduction，全部在 host 完成 |
| **Workspace 物化是一次性投资的智慧** | dS/P workspace 额外 512KB/program，但省了 2 次 gather + 3 次 dot 重算，净赢 2.4× |
| **单 Kernel 架构对 L2 复用至关重要** | 拆成 2 kernel 导致 workspace 冷读（~77ms），L2-hot 单 kernel 完成是正确选择 |
| **Cube 利用率是沉默的性能杀手** | GROUP_HC 合并让 M 维填满 16×16 微 tile，cube 利用率从 0.14% 大幅提升 |
| **Cube↔Vector 同步是主要瓶颈** | co-locate scores+dPv 在 cube 上连续执行，同步减半 |
| **tl.dot(acc=) 在 cube 利用率提升后有效** | SLI 上失败、SFA Grad 上成功——关键前置条件是 cube 计算密度先提升 |
| **Autotune 对 atomic_add 算子有天然冲突** | atomic_add 的 non-idempotent 特性使 autotune 的重复 benchmark 产生错误结果 |
| **编译器优化行为需实测验证** | g_offs clamp 导致 BiShengHIR 错误 specialize N1、mask elision 对标量 lowering 的消除——均需实测发现 |

---
