# DSA Triton 算子优化方法论总结

> 基于五个 DSA 算子（LI、SLI、SFA Forward、SFA Grad、Dense）在 Ascend910_9382 + CANN 9.0 达成 0.3× CANN 目标的实战经验提炼

---

## 前言

DSA（Dynamic Sparse Attention）是 MindFormers 大模型推理框架中的核心注意力模块。本项目将五个 DSA 算子从 CANN 原生实现移植到 Triton-Ascend，目标是在 Ascend910_9382 + CANN 9.0 平台达成 **0.3× CANN** 的端到端性能，同时支持额外的 shape 组合。

五个算子端到端耗时从数百毫秒压缩至 CANN 的 0.3 倍以内。本文从五个实战案例中提炼出**五大优化方向**，阐述其理论基础、收益来源，并结合具体实例说明。

---

## 一、Kernel 融合与 Launch 削减

### 1.1 理论基础

Ascend NPU 的 kernel launch 有两大开销：

- **Host-Device 同步开销**：每次 `<<<grid, block>>>` launch 需要 host 端准备参数、下发任务、等待 device 确认。多个 kernel 串行时，前一个 kernel 完成 → 同步信号回 host → host 下发下一个 kernel，形成 ping-pong 延迟
- **GM（Global Memory）数据冷读开销**：kernel A 的输出写入 GM → kernel B 从 GM 读取。如果数据在 A 退出后未停留在 L2 cache，B 必须从 HBM 重新加载（~TB/s 级别带宽 vs L2 ~数十 TB/s）

**融合的核心价值**：将多个逻辑阶段合并为单个 kernel launch，中间数据保持在 program 局部（UB/L2），消除 launch 同步 + GM 往返。

### 1.2 收益来源（重点）

| 收益来源 | 数量级 | 说明 |
|----------|--------|------|
| **Launch overhead 消除** | 每次 ~10-50μs | N 次 launch 串行累加，对高频调用场景显著 |
| **L2-hot 数据复用** | 带宽 ~10× vs HBM | 中间 buffer 在单 kernel 内保持 L2 热度，后续 stage 读 L2 而非 HBM |
| **GM 读写消除** | 每次 ~数百 MB | 融合前 A 写 → B 读的两趟 GM 变成零趟 |

### 1.3 DSA 实例

**SLI（5→1 kernel 融合）**

SLI 最极致的优化：将 5 个独立 kernel 逐步融合为单个 `_sli_grad_fused_kernel`。

```
5 kernel: gather_kv → teacher → indexer → kl_loss → query_weight → scatter
  ↓ (K1+K2 融合, 省 key_gathered/key_rope_gathered 中间张量 9GB HBM)
4 kernel
  ↓ (teacher + gather 融合)
3 kernel
  ↓ (indexer + kl_loss + dI 融合)
2 kernel: teacher_indexer_kl → indexer_grad
  ↓ (K1 并入 K23)
1 kernel: _sli_grad_fused_kernel (Stage T → I → Final → Pass A → Pass B)
```

融合后 `di`/`s_idx_buf`/`key_index_gathered`/`buf_p`/`buf_i` 五个中间 buffer 全程在同一 program 的 L2 中保持热度，消除 4 次 launch overhead + 4 轮 GM 读写。

**SFA Grad（单 kernel L2-hot workspace）**

将 dS/P workspace 拆出独立 kernel → GM 冷读退回 ~77ms。所有 Stage（A: dqcat, B1: dkcat, B2: dv）统一在单个 `_sfa_grad_kernel` 内执行，Stage A 写入的 dS/P 在 L2 保持热度，Stage B1/B2 命中 L2。

---

## 二、算法重构（计算模式革新）

### 2.1 理论基础

Trition-Ascend 的编译器和硬件特性决定了：**同样的数学结果，不同的计算编排，硬件执行效率可能相差数倍**。核心影响因素：

- **dot 调用次数**：每次 `tl.dot` 触发 Cube 单元，有 issue/sync 开销。减少 dot 次数直接降低指令开销
- **Cube↔Vector 切换**：Cube dot 的结果需同步回 Vector 侧才能做 exp/max/sum 等标量操作，每次切换 ≈ 数十 cycles
- **数据流方向**：重算 vs 缓存。在 UB 有限（192KB）的约束下，有时重算比分存更优

### 2.2 收益来源（重点）

| 收益来源 | 实例 | 数量级 |
|----------|------|--------|
| **Dot 次数削减** | SFA fwd: two-pass 928 dots → chunked 288 dots | **3.2×** dot 减少 |
| **Cube↔Vector 同步减半** | SFA Grad: scores+dPv co-located on cube | 每 K-tile 省 2 次同步 |
| **重算→缓存 trade-off** | SFA fwd chunked: p_raw 驻留 UB（~8KB）vs two-pass 每 dv-tile 重算 | dv_tiles× 次 exp 计算消除 |

### 2.3 DSA 实例

**SFA Forward：two-pass → chunked online-softmax（137ms→48ms, 2.85×）**

| | Two-pass（原始） | Chunked（优化后） |
|---|---|---|
| Pass 1 | 流式扫 KV，算 global m/l | — |
| Pass 2 | 每 dv-tile 重算 scores + exp + P@V | — |
| Chunked | — | 单趟扫 KV，每 chunk 内：scores→P→alpha_correct→fp32_acc |
| K-tile 循环 | 32 chunks | 32 chunks |
| 每个 K-tile dot 数 | Pass1: 2nope+2rope, Pass2: (2nope+2rope)×dv_tiles | 2nope+2rope+1dv |
| 总 dot 数 | 32×(4+4×8)=1056 (pass1+2) | 32×9=288 |
| Last-block | 独立 dv-tile 循环读回 fp32_acc | **直接归一化写入 output**，省全 D 维读+写 |

**SFA Grad：scores+dPv cube co-location**

```python
# 优化前：cube→vector→cube→vector (4 次同步)
scores = tl.dot(q, kT)     # cube
P = exp(scores - sm_max)   # vector
dPv = tl.dot(do, kT)       # cube（等待 vector 完成）
dS = P * (dPv - delta)     # vector

# 优化后：cube→vector (2 次同步)
scores = tl.dot(q, kT)     # cube
dPv = tl.dot(do, kT)       # cube（连续执行，仍在 cube 侧）
P = exp(scores - sm_max)   # vector
dS = P * (dPv - delta)     # vector
```

**Dense LSE：two-pass → single-pass chunked online stats**

```python
# 两趟：Pass1 找 max → Pass2 用 safe max 算 sum
# 单趟 chunked：每 K-tile 同时更新 max 和 sum
m_new = maximum(i_max, max(i_masked))
alpha = exp(i_max - m_new)
i_sum = i_sum * alpha + sum(exp_i * alpha)
i_max = m_new
```

---

## 三、Grid 架构与并行度设计

### 3.1 理论基础

Ascend NPU 的 kernel grid 设计受以下约束：

- **coreDim 上限 65535**：`grid[0] × grid[1] × grid[2]` 不能超过此值（`GRID_LIMIT`）
- **grid 每维必须是 2 的幂**：非 pow2 导致分核映射出错 → aicore trap
- **program 间无共享 UB**：每个 program 独立 UB 192KB，无跨 program 通信（除 atomic_add）

**Grid 设计的核心 trade-off**：
- 高并行度（grid 大）→ 每个 program 工作量小，但 launch 开销大、可能超 coreDim
- 低并行度（grid 小）→ 程序内串行工作多，但 launch 次数少

### 3.2 收益来源（重点）

| 收益来源 | 说明 |
|----------|------|
| **维度并行化** | 将循环维度提升为 grid 维度，让多个 core 同时工作 |
| **串行化压缩 grid** | 引入 BLOCK_S1/BLOCK_G 等分块参数，将 grid 控制在 coreDim 限值内 |
| **pow2 padding** | 多余 program 空转（idle），避免 aicore trap |

### 3.3 DSA 实例

**LI：1D→2D Grid**

```
1D grid: (B*S1*N2,)           — S2 维度无并行，每个 program 串行扫完整 S2
2D grid: (B*S1*N2, cdiv(S2, BLOCK_S2))  — S2 维度并行，并行度 ×32~128
```

这是 LI 算子从 0.10× 跃升到 0.14× CANN 的核心变更。S2 维度并行化后，每个 program 从处理 4096 个 token 降至处理 128 个 token。

**LI：BLOCK_S1 分块控制 grid**

```
S1=128:   BLOCK_S1=8,  grid0=128*1/8=16     ✓ grid 适中
S1=4096:  BLOCK_S1=1,  grid0=4096*1/1=4096   ✓ 并行最大化
S1=16384: BLOCK_S1=16, grid0=16384*1/16=1024  ✓ grid 压缩
```

**Dense dkey Grid 降维（S1=4096 兼容）**

```
S1=512:  grid (bs1_chunk, num_k_blocks, num_d_blocks)    ✓ 512 个 bs1
S1=4096: grid (B, num_k_blocks, num_d_blocks) + s1-loop  ✓ B=1, grid 不再爆炸
```

---

## 四、内存层级优化

### 4.1 理论基础

Ascend 910 系列的内存层级：

```
HBM (32-64 GB, ~1-2 TB/s)
  ↕
L2 Cache (~数 MB, ~数十 TB/s)
  ↕
UB (192 KB, ~数百 TB/s, 每 core 独占)
```

**优化金字塔**（越上层收益越高）：
1. **消除中间张量**：不分配即最优（省 HBM 分配 + 写入 + 读取）
2. **L2 复用**：单 kernel 内多 stage 共享中间结果（省 HBM 读写）
3. **Workspace 持久化**：跨调用复用 device buffer（省分配/清零开销）
4. **UB 预算管理**：精算每 phase 的 UB 占用（解锁更大 tile）

### 4.2 收益来源（重点）

| 优化层级 | 收益 | DSA 实例 |
|----------|------|---------|
| **消除中间张量** | 省 9GB HBM 读写 | SLI: K1 不 gather k/kr，teacher kernel 按 sparse_indices 直接读 |
| **L2-hot 复用** | 省 512KB×N 次 GM 读写 | SFA Grad: dS/P workspace 单 kernel 内 L2-hot |
| **Workspace 持久化** | 省 alloc/zero/free | SFA fwd: `_GATHER_WS_CACHE` 跨调用复用 |
| **UB 精算** | 解锁更大 tile → dot 次数减少 | SFA fwd: phased UB 估算解锁 BLOCK_D=256 等激进 config |

### 4.3 DSA 实例

**SLI：消除 key_gathered/key_rope_gathered 中间张量（1.07×）**

```
优化前：K1(_gather_kv_kernel) 写 key_gathered(8GB) + key_rope_gathered(1GB)
        → K2(_teacher_distribution_kernel) 读
优化后：K2 直接按 sparse_indices 间接读原始 k/kr，每 K-tile load idx (~1KB UB)
```

这是**零字节分配**的极致：省 9GB HBM 写 + 9GB HBM 读 = 18GB 流量，仅用 1KB UB 换。

**SFA Forward：_GATHER_WS_CACHE 持久化工作区**

```python
_GATHER_WS_CACHE = {}  # 全局字典，按 shape/dtype 缓存
cache_key = (B_S1, topK, D, D_ROPE, dtype)
cached = _GATHER_WS_CACHE.get(cache_key)
if cached is None:
    gk  = empty(...); gkr = empty(...); gvalid = empty(...)
    _GATHER_WS_CACHE[cache_key] = (gk, gkr, gvalid)
```

Benchmark 中 allocate/zero/free 大 workspace（~1GB）的开销主导端到端时间，持久化后直接消除。

**Phased UB 估算（SFA Forward）**

```
Phase 1 (scores):  q+k+qr+kr+scores+m_l     = ~150KB
Phase 2 (P@V):     p+v+pv+acc_dv+m_l         = ~140KB
Peak = max(150, 140) 而非 sum(150+140=290)   ← phased 估算解锁配置
```

---

## 五、计算路径升级（硬件适配）

### 5.1 理论基础

Ascend 910 系列的两种计算单元：

| 单元 | 擅长 | 调用方式 | 吞吐量 |
|------|------|---------|--------|
| **Cube** | 矩阵乘（MMA） | `tl.dot()` → fp16×fp16→fp32 | 256 TFLOPS (fp16) |
| **Vector** | 逐元素操作 | `tl.sum()`, `tl.exp()`, add/mul | ~数十 TFLOPS |

**关键适配原则**：
- 将 `tl.sum(k * q)` 的 vector 乘加替换为 `tl.dot(q, kT)` 的 cube 矩阵乘
- 保持 dot 输入为 fp16（Cube 原生精度），避免 `.to(tl.float32)` 强制走 vector path
- 利用 `tl.dot(..., acc=acc)` 让 fp32 累加器驻留 L0C，避免 cube→vector→UB 往返

### 5.2 收益来源（重点）

| 收益来源 | 吞吐量提升 | DSA 实例 |
|----------|-----------|---------|
| **tl.sum → tl.dot** | Vector(数十TF) → Cube(256TF) 10×+ | Dense: scalar 6×→cube 直接持平 CANN |
| **L0C 累加器驻留** | 每 K-tile 省一次 cube→vector 同步 | SFA Grad: acc_dqcat/dkcat_acc/dv_acc 三处 |
| **Cube 微 tile 填充** | M维 全填充 16×16 → 利用率 0.14%→接近 100% | SFA Grad: GROUP_HC 合并 |
| **fp16 保持 Cube path** | 避免 `.to(fp32)` 降级到 vector | SLI: K3 恢复 fp16 path, cube_ratio 提升 |

### 5.3 DSA 实例

**Dense：scalar→cube 路径升级（最大单次收益，直抵 CANN 持平）**

```
优化前（scalar/vector 路径）:
  for g in range(Nidx1):                    # G 循环
    for d_start in range(0, D, BLOCK_D):    # D 循环
      qi = tl.load(...)  # [BLOCK_D]
      ki = tl.load(...)  # [BLOCK_K, BLOCK_D]
      acc += tl.sum(ki * qi[None,:], axis=1)  # vector mul-add
  # 总操作: Nidx1 × cdiv(D,BLOCK_D) 次 vector 乘加

优化后（cube 路径）:
  qi = tl.load(...)  # [NIDX1, D_idx]     ← 全量 QI
  ki = tl.load(...)  # [BLOCK_K, D_idx]    ← 全量 KI
  dot = tl.dot(qi, tl.trans(ki))          # cube! [NIDX1, BLOCK_K]
  # 总操作: 1 次 cube dot
```

G×D 维度的数十次 vector 迭代合并为一次 cube 矩阵乘法，硬件利用率从 ~数% 跃升到接近 Cube 的 256 TFLOPS 理论峰值。

**SFA Grad：GROUP_HC 合并 → Cube 微 tile 填充**

```
问题: BLOCK_G=8 时 dot M 维 = 8，Cube 微 tile 是 16×16
      → M 维半填充，cube_ratio 仅 0.14%

解决: GROUP_HC=2, MG=16 → 两个 head chunk 合并为一次 dot
      → M 维 = 16，16×16 微 tile 全填充
      → dot 调用次数减半（k_cat 共享），cube 利用率大幅提升
```

**tl.dot(acc=) 的 L0C 驻留**

```
优化前: acc += tl.dot(...).to(tl.float32)
        → 每轮: Cube L0C → Vector寄存器 → fp32 UB, 下轮 UB → L0C

优化后: acc = tl.dot(..., acc=acc)
        → 每轮: L0C 原地累加, 无往返
```

**注意**：`tl.dot(acc=)` 仅在 cube 利用率较高时有效。SLI 尝试此优化反而劣化（4503→6104us），因为其时 cube_ratio 仅 0.15%，L0C 累加引入的上下文切换开销超过了省下的同步开销。这是一个重要的**硬件特性依赖**教训。

---

## 六、五大方向关系图与决策指南

### 6.1 方向关系

```
                ┌──────────────────┐
                │  3. Grid 架构     │  ← 并行度基础设施
                └────────┬─────────┘
                         │ 为后续优化提供并行空间
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌────────────────┐ ┌──────────┐ ┌──────────────────┐
│ 1. Kernel 融合  │ │2. 算法重构│ │ 5. 计算路径升级   │
│ (减少 launch)   │ │(减少 dot) │ │ (scalar→cube)    │
└───────┬────────┘ └─────┬────┘ └────────┬─────────┘
        │                │               │
        └────────────────┼───────────────┘
                         │ 中间数据生命周期变长
                         ▼
                ┌──────────────────┐
                │ 4. 内存层级优化   │  ← 支撑融合后的数据局部性
                └──────────────────┘
```

### 6.2 决策指南

| 场景特征 | 优先方向 | 参考算子 |
|----------|---------|---------|
| 多 kernel 串行，中间数据量大 | **Kernel 融合**（方向一） | SLI |
| 算法有明显重算冗余（如 two-pass） | **算法重构**（方向二） | SFA Forward |
| 某个维度无并行（如 S2 串行扫描） | **Grid 架构**（方向三） | LI |
| 大量中间张量分配/释放 | **内存优化**（方向四） | SLI / SFA Forward |
| 计算用 tl.sum 而非 tl.dot | **计算路径升级**（方向五） | Dense |
| 精度问题伴随 UB overflow | **UB 精算**（方向四·子项） | SFA Forward |
| 多 shape 通用性需求 | Grid 维度伸缩 + autotune（三+五） | LI / SFA |
| Cube 利用率极低（<1%） | GROUP_HC 合并 + cube co-location（五） | SFA Grad |

### 6.3 各算子的方向权重

| 算子 | 方向一 | 方向二 | 方向三 | 方向四 | 方向五 |
|------|--------|--------|--------|--------|--------|
| LI | ○ | — | ● | ○ | ○ |
| SLI | ● | ○ | — | ● | ○ |
| SFA Forward | ○ | ● | ○ | ● | ○ |
| SFA Grad | ● | ○ | — | ● | ● |
| Dense | ○ | — | ● | ○ | ● |

> ● 主要杠杆  ○ 辅助优化  — 不适用

---

## 七、关键教训与反模式

### 7.1 跨算子的共性教训

1. **triton-ascend 的 early-return 只能有一个出口**（LI + SFA Grad）：多个 early-return 导致 store 被编译器丢弃，必须合并为单一出口或完全不用

2. **tl.dot(acc=) 不是万能药**（SLI 劣化 vs SFA Grad 收益）：仅在 cube 利用率足够高（>10%?）时 L0C 驻留才净赢，低利用率时上下文切换开销反噬

3. **fp32 upcast 谨慎使用**（SLI + SFA Forward）：`.to(tl.float32)` 强制走 vector path，破坏 cube 通路；仅保留精度瓶颈处

4. **Grid pow2 padding 是 Ascend 硬约束**（全部算子）：非 pow2 grid → aicore trap，padding program 靠 mask 空转

### 7.2 常见反模式

| 反模式 | 症状 | 正确做法 |
|--------|------|---------|
| 盲目加 autotune config | 编译时间爆炸 + VMM 贴满 | 按 shape 分类裁剪，大 shape 只保留 2 个最优对 |
| 忽略 UB 估算直接上大 tile | UB overflow 编译失败 | 分 phase 估算，multi-buffer 翻倍防护 |
| 性能测试含 Host→Device 传输 | speedup 严重偏低 | 显式 `.to('Ascend')` 排除传输 |
| 两个 kernel 共享中间 buffer 不检查 L2 | 第二个 kernel 从 HBM 冷读 | 合并为单 kernel 或确认 L2 大小 |

---
