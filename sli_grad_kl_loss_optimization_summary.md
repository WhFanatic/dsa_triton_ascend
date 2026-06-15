# SparseLightningIndexerGradKLLoss Triton-Ascend 优化复盘

日期：2026-06-10

对象文件：`sparse_lightning_indexer_grad_kl_loss_triton.py`

## 1. 结论先行

当前最终保留代码已经补到 full profile 实测值。用户在 2026-06-11 贴出的最新当前版本 profile 为：

```text
total kernel duration: 4079.548 ms / 4 ops
当前最终版本实测：4079.548 / 4 = 1019.887 ms / op
```

相对用户给出的原始全链路：

```text
原始全链路：4032.93 ms / op
当前实测：  1019.887 ms / op
整体加速：  约 3.95x
耗时下降：  约 74.7%
```

需要特别区分一个容易混淆的 profile：用户最后贴出的 full profile 为：

```text
total kernel duration: 4845.873 ms
main calls: 32
CANN calls: 4
```

这份 profile 等价于 4 次 op 调用，因此同口径单次 Triton op 为：

```text
4845.873 / 4 = 1211.468 ms / op
```

但这份 profile 里还有：

```text
_query_weight_grad_kernel_0
_query_weight_grad_kernel_1
```

说明它对应的是“拆 Stage 5”实验版本，而不是当前最终保留版本。该实验已经判断为链路负收益并回滚。因此：

```text
1211.47 ms/op 是已回滚实验版本的 full profile 实测值。
1019.887 ms/op 是当前最终保留版本的 full profile 实测值。
```

建议最终确认命令：

```bash
cd /workspace/chen_dsa/dsa_triton_ascend_latest
./run_sli_profile.sh prof-triton
python summarize_latest_profile.py \
    profiler_data_sli_grad_kl_loss \
    profiler_data_sli_grad_kl_loss_cann
```

重点看：

```text
total kernel duration / 4
_indexer_grad_kl_loss_kernel_0 avg
_scatter_dkey_index_kernel_0 avg
_gather_kv_kernel_0 avg
```

## 2. 算子执行链路

Public API：

```text
sparse_lightning_indexer_grad_kl_loss_triton(...)
```

主要链路：

```text
sparse_lightning_indexer_grad_kl_loss_triton
  -> query_rope/key_rope 默认值处理
  -> actual_seq_qlen/actual_seq_klen 处理
  -> 如果 S1 > SPARSE_GRAD_S1_CHUNK，按 S1 chunk 切分
      -> _sparse_lightning_indexer_grad_kl_loss_core
          -> reshape / contiguous
          -> 分配中间 buffer
          -> _gather_kv_kernel       gather key
          -> _gather_kv_kernel       gather key_index
          -> _gather_kv_kernel       gather key_rope
          -> _indexer_grad_kl_loss_kernel
          -> _scatter_dkey_index_kernel
          -> reshape outputs
      -> chunk 间累加 d_key_index
      -> concat d_query_index / d_weights
      -> cast d_key_index 到 key_index dtype
```

每个 chunk 内的 Triton 主 kernel：

```text
3 x _gather_kv_kernel
1 x _indexer_grad_kl_loss_kernel
1 x _scatter_dkey_index_kernel
```

主要计算语义：

```text
Stage 1: I[k] = sum_g W[g] * ReLU(qi[g] @ ki[idx[k]]^T)
Stage 2: p[k] = mean_h softmax(score_h)[k]
Stage 3: softmax(I)
Stage 4: KL(p || softmax(I)), dI = softmax(I) - p
Stage 5: dW, dQueryIndex
Scatter: dKeyIndex scatter-add
```

## 3. 优化前瓶颈判断

用户最早给的 profile 核心数据：

```text
_scatter_dkey_index_kernel_0: total 24364.445 ms / 64 calls, avg 380.694 ms
_indexer_grad_kl_loss_kernel_0: total 7641.454 ms / 64 calls, avg 119.398 ms
_gather_kv_kernel_0: total 210.633 ms / 192 calls, avg 约 1.097 ms
CANN baseline: avg 约 20.8 ms
```

按一次 op 约 8 个 S1 chunk 估算：

```text
scatter: 8 * 380.694 = 3045.552 ms
main:    8 * 119.398 = 955.184 ms
gather:  24 * 1.097  = 26.328 ms
aux:     少量 MindSpore 辅助 kernel
合计：   约 4032.93 ms / op
```

初始瓶颈排序：

```text
scatter >>> main >>> gather / auxiliary
```

这个判断决定了第一阶段应优先优化 scatter，而不是一开始拆 main kernel。

## 4. 有效优化一：scatter 空 K block 提前返回

### 做了什么

在 `_scatter_dkey_index_kernel` 中，如果当前 program 对应的 `s1_global` 无效，或者当前 `k_block` 完全超过 `s2_real`，直接返回：

```python
if s1_global >= act_q:
    return

s2_real = tl.minimum(topK, tl.maximum(act_k - act_q + s1_global + 1, 0))
s2_bound = tl.minimum(s2_real, valid_k)
if k_block * BLOCK_K >= s2_bound:
    return
```

### 为什么这样优化

rightDownCausal 下，早期 token 的有效 K 范围很小。但 scatter grid 原来仍按固定：

```text
(B * S1, ceil(valid_k / BLOCK_K_SCATTER))
```

发射。很多 `(s1, k_block)` 实际全是 `k_mask=False`。

原始代码虽然不会写错结果，但仍会进入：

```text
for g in range(Nidx1)
for d_start in range(0, D_idx, BLOCK_D)
masked tl.load
masked tl.atomic_add
```

提前返回可以删除这些空 program 的循环骨架。

### 效果

这一步单独收益有限，但风险低，并为后续 scatter 优化打基础。语义风险很低，因为被跳过的 program 原本所有 atomic mask 都为 false。

## 5. 有效优化二：scatter 先本地累加，再一次 atomic_add

### 做了什么

原始 scatter 逻辑近似为：

```python
for g in range(Nidx1):
    ...
    for d_start in range(0, D_idx, BLOCK_D):
        tl.atomic_add(d_key_index, dki_vals)
```

优化后改为：

```python
for d_start in range(0, D_idx, BLOCK_D):
    dki_acc = tl.zeros([BLOCK_K, BLOCK_D], dtype=tl.float32)

    for g in range(Nidx1):
        dki_acc += dki_contrib[:, None] * qi_g[None, :]

    tl.atomic_add(d_key_index, dki_acc)
```

### 为什么这样优化

`dKeyIndex` 的梯度公式是：

```text
dki[b, target_k, d] += sum_g dI[k] * W[g] * 1_{relu_gk > 0} * qi[g, d]
```

原始实现对每个 `g` 都做一次全局 `atomic_add`。当 `Nidx1=64` 时，同一个 `(target_k, d)` 会被大量 atomic 写竞争。

优化后，先在一个 Triton program 内把 `g` 维度求和，再对同一个输出位置只做一次 atomic。数学上等价，但全局 atomic 次数约减少 `Nidx1` 倍。

### 效果

scatter 从最大瓶颈变成次要瓶颈：

```text
_scatter_dkey_index_kernel_0
avg 380.694 ms -> avg 约 23.96 ms
```

加速比：

```text
380.694 / 23.96 = 15.89x
```

耗时下降：

```text
(380.694 - 23.96) / 380.694 = 93.7%
```

按一次 op 8 个 chunk：

```text
scatter 原始：8 * 380.694 = 3045.552 ms
scatter 优化：8 * 23.96  = 191.68 ms
单 scatter 部分节省：约 2853.872 ms
```

这一步是今天最确定的链路级大收益。

## 6. 有效优化三：main kernel 跳过空 K tile

### 做了什么

在 `_indexer_grad_kl_loss_kernel` 的每个 K tile loop 内加：

```python
if k_start < s2_real:
    ...
```

覆盖位置包括：

```text
Stage 1: 计算 I[k] 和 s_idx_buf
Stage 2: 初始化和累加 teacher p[k]
Stage 3/4: softmax(I), KL, dI
Stage 5: dW, dQueryIndex
```

### 为什么这样优化

main kernel 原来固定遍历：

```python
for k_start in range(0, VALID_K, BLOCK_K):
```

即使 `k_start >= s2_real`，也只是依赖：

```python
k_mask = k_offs < s2_real
```

来避免写错。问题是：

```text
mask=false 不等于这段代码完全免费
```

在 Triton-Ascend 上，masked load/store、dot helper、向量表达式、exp/log、Stage5 循环骨架仍可能产生指令和调度开销。

rightDownCausal 下，早期 token 的有效 K tile 很少：

```text
topK = 2048
BLOCK_K_MAIN = 128
总 K tile = 16
早期 token 可能只需要 1-4 个 tile
```

跳过空 tile 可以减少大量无效向量计算和 GM->UB 搬运。

### 单 kernel profile 证据

优化前，main 单 kernel profile 中前段 block：

```text
aiv_time:              约 10500 us/block
aiv_mte2_instructions: 25424
GM_to_UB_datas block0: 1568.625 KB
```

优化后，用户贴的 profile 中前段 block：

```text
aiv_time:              约 850-930 us/block
aiv_mte2_instructions: 4304
GM_to_UB_datas block0: 248.625 KB
```

示例对比：

```text
block0 aiv_time:
10502.811 us -> 849.878 us
约 12.36x 下降

block0 GM_to_UB_datas:
1568.625 KB -> 248.625 KB
约 84.15% 下降

aiv_mte2_instructions:
25424 -> 4304
约 83.07% 下降
```

这说明空 K tile skip 确实减少了 main kernel 的实际执行路径，而不是只改变了代码结构。

### 对全链路的估算

main 原始按一次 op 估算为：

```text
8 * 119.398 = 955.184 ms
```

根据 S1 chunk 和 rightDownCausal 的有效 K tile 比例粗估：

```text
chunk0 平均约 2.5 / 16 tile
chunk1 平均约 6.5 / 16 tile
chunk2 平均约 10.5 / 16 tile
chunk3 平均约 14.5 / 16 tile
chunk4-7 基本满 16 / 16 tile

总有效 tile 比例：
(2.5 + 6.5 + 10.5 + 14.5 + 16 + 16 + 16 + 16) / (16 * 8)
= 98 / 128
= 76.56%
```

因此 main 估算：

```text
955.184 * 0.7656 = 731.4 ms
```

当前最终版本全链路估算：

```text
main:    约 731 ms
scatter: 约 192 ms
gather:  约 26 ms
aux:     约 5-10 ms
合计：   约 950 ms / op
```

## 7. 已回滚或负收益实验

这些实验虽然没有保留，但很适合面试时说明 profiling-driven 的探索过程：不是每个看起来合理的优化都会变快，关键是用数据判断并回滚。

### 7.1 scatter relu mask 合到 atomic mask

想法：

```text
把 relu_gk > 0 合并进 atomic_add 的 mask，减少无效 atomic。
```

结果：

```text
scatter 变慢，回滚。
```

原因推测：

```text
额外 mask 计算和更复杂 predicate 可能比少量无效写更贵。
Ascend 后端对复杂 mask 的代码生成也可能更差。
```

### 7.2 拆 Stage 5 为 `_query_weight_grad_kernel_0/_1`

想法：

```text
把 dW/dQueryIndex 从 main kernel 拆出来，降低 main kernel duration。
```

结果：

用户贴出的 full profile：

```text
total kernel duration: 4845.873 ms / 4 op = 1211.468 ms/op

_indexer_grad_kl_loss_kernel_0: 97.673 ms avg
_scatter_dkey_index_kernel_0:   23.962 ms avg
_query_weight_grad_kernel_0:    14.249 ms avg
_query_weight_grad_kernel_1:    11.457 ms avg
```

虽然 main 从约 119 ms 降到约 97 ms，但新增两个 query/weight grad kernel 合计约 25.7 ms/launch，链路总账变差，因此回滚。

面试表达：

```text
拆 kernel 不能只看原 kernel 变短，必须看端到端 launch 链路、额外 GM 读写、同步边界和新增 kernel 总耗时。
```

### 7.3 Stage 2 改 K-outer 本地累加

想法：

```text
把 teacher p[k] 计算从 h-outer 改成 K-outer，减少 buf_p 反复 load/store。
```

结果：

```text
main 变慢到约 116.565 ms，回滚。
```

原因推测：

```text
局部 accumulator 和循环重排增加寄存器/UB 压力，后端向量表达式和指令数变多。
```

### 7.4 调 BLOCK_K_MAIN

实验：

```text
BLOCK_K_MAIN = 64  -> main 变慢到约 138.065 ms
BLOCK_K_MAIN = 256 -> UB out of bounds
```

结论：

```text
当前 BLOCK_K_MAIN = 128 是已知最稳选择。
更小 BLOCK_K 增加 tile 数和循环开销。
更大 BLOCK_K 超过 UB/向量临时量承载能力。
```

### 7.5 Stage5 dW 合并进 dQI 第一个 d block

想法：

```text
减少一次 dW 独立 K pass。
```

结果：

```text
main 变慢到约 117.403 ms，回滚。
```

原因推测：

```text
合并后控制流和局部变量生命周期变长，增加寄存器/UB 压力，收益不足以抵消。
```

### 7.6 helper/shape constexpr codegen 实验

想法：

```text
把 shape/stride 标成 tl.constexpr，期望改善代码生成。
```

结果：

```text
main 变慢到 117000.992 us，回滚。
```

结论：

```text
Triton-Ascend 后端不一定因更多 constexpr 标注变快。
需要实测，而不是按 GPU Triton 经验直接推断。
```

## 8. 面试可讲的核心知识点

### 8.1 Profiling-driven optimization

面试官可能问：你怎么确定先优化哪里？

回答要点：

```text
我先看 full profile 的 total/calls/avg，而不是凭感觉改代码。
初始 scatter avg 380.694 ms，占 Triton kernel 时间大头；
main avg 119.398 ms；
gather avg 约 1.1 ms。
因此第一阶段优先优化 scatter。
scatter 降到约 24 ms 后，瓶颈转移到 main，再优化 main。
```

核心原则：

```text
先全链路定位，再单 kernel profile。
每轮只改一个因素。
看端到端总账，不只看单 kernel 变短。
负收益实验及时回滚。
```

### 8.2 Mask 不等于免费

面试官可能问：`k_mask` 已经 false 了，为什么还要加 early skip？

回答要点：

```text
mask=false 保证语义正确，但不保证后端完全不生成或不执行相关指令。
在 Triton-Ascend 上，masked load/store、向量表达式、循环骨架、exp/log 路径仍可能消耗 AIV 时间和 MTE 指令。
rightDownCausal 下大量 K tile 全 false，所以 early skip 可以减少实际执行路径。
```

证据：

```text
aiv_mte2_instructions: 25424 -> 4304
block0 GM_to_UB_datas: 1568.625 KB -> 248.625 KB
block0 aiv_time: 10502.811 us -> 849.878 us
```

### 8.3 Atomic add 优化

面试官可能问：scatter 为什么慢？

回答要点：

```text
dKeyIndex 是 scatter-add，多数 sparse index 会产生不规则写和 atomic 冲突。
原始实现每个 g 都 atomic_add 一次。
Nidx1=64 时，同一 target_k/d 上 atomic 次数很高。
```

优化思想：

```text
把 g 维度在 program 内先 reduce 到 dki_acc，再一次 atomic_add。
减少全局 atomic 次数和冲突。
```

效果：

```text
scatter avg 380.694 ms -> 约 23.96 ms，约 15.9x。
```

### 8.4 UB、BLOCK_K、tiling

面试官可能问：为什么不直接调大 BLOCK_K？

回答要点：

```text
BLOCK_K 大可以减少 tile 数，但会增加 [BLOCK_K, BLOCK_D] 临时张量、mask、load tile 的 UB 占用。
在 Ascend AIV 上 UB 容量有限。
实验 BLOCK_K_MAIN=256 出现 UB out of bounds。
BLOCK_K_MAIN=64 又因为 tile 数增加和循环开销变慢。
所以 BLOCK_K_MAIN=128 是当前平衡点。
```

### 8.5 Causal sparse pattern

面试官可能问：为什么 rightDownCausal 对优化很重要？

回答要点：

```text
s2_real = min(topK, max(act_k - act_q + s1_global + 1, 0))
早期 s1 的有效 key 很少，后期逐渐增加，最终 topK 饱和。
如果 kernel 固定遍历 topK，就会在早期产生大量空 K tile。
early skip 利用了 causal sparsity。
```

### 8.6 数值正确性

面试官可能问：优化会不会影响 KL loss 和梯度？

回答要点：

```text
只跳过 k_start >= s2_real 的 tile。
这些 tile 原本 k_mask 全 false，对 buf_i、buf_p、di、loss、dW、dQueryIndex、dKeyIndex 都没有有效贡献。
有效 tile 的计算公式、dtype、softmaxMax/softmaxSum 复用逻辑不变。
```

scatter 本地累加的正确性：

```text
原公式是对 g 求和后写入同一个 dKeyIndex 位置。
把多个 atomic_add 改为 program 内 fp32 累加后一次 atomic_add，只改变求和顺序，不改变数学表达式。
梯度本来就是浮点归约，允许存在极小求和顺序差异；测试容忍度没有修改。
```

### 8.7 为什么不改 public API 和测试容忍度

回答要点：

```text
这是 drop-in replacement for aclnnSparseLightningIndexerGradKLLoss。
优化必须保持 public API、输出 dtype、shape、CANN/numpy 对齐逻辑不变。
测试容忍度不能放宽，否则性能优化可能掩盖数值错误。
```

### 8.8 单 kernel profile 与 full profile 的关系

面试官可能问：为什么既看 op-main，又看 full profile？

回答要点：

```text
单 kernel profile 能定位内部瓶颈，例如 AIV、MTE、UB、bank conflict。
full profile 能判断端到端是否真的变快。
拆 Stage5 的实验就是反例：main 变短了，但新增 kernel 让 full chain 变慢。
```

### 8.9 读 Ascend profile 指标

常用指标解释：

```text
Task Duration(us): kernel 总执行时间。
Block Dim: kernel program/block 数量。
aiv_time: vector core 上每个 block 的执行时间。
aiv_vec_ratio: 向量计算占比。
aiv_mte2_ratio / aiv_mte2_instructions: GM->UB 搬运压力。
aiv_mte3_ratio: UB->GM 写回压力。
GM_to_UB_datas: 主存读取到 UB 的数据量。
UB read/write BW: UB 内部读写带宽。
ResourceConflictRatio: bank/bankgroup/vector wait 等冲突。
```

本次优化如何使用这些指标：

```text
scatter 优化主要看 kernel duration 和 atomic 写冲突导致的耗时。
main early skip 主要看 aiv_time、MTE2 instruction、GM_to_UB_datas 是否下降。
```

## 9. 面试问答模板

### Q1：你这次优化最大收益来自哪里？

答：

```text
最大确定收益来自 dKeyIndex scatter。
原来每个 g 都 atomic_add，Nidx1=64 导致大量全局 atomic 冲突。
我改成 program 内先对 g 累加到 dki_acc，再一次 atomic_add。
scatter avg 从 380.694 ms 降到约 23.96 ms，约 15.9x。
```

### Q2：main kernel 为什么后面还能优化？

答：

```text
scatter 降下来后，main 成为第一瓶颈。
main 原来对所有 token 固定扫 full topK tile。
但 rightDownCausal 下早期 token 的 s2_real 很小，很多 K tile 是全 false mask。
我在每个 K tile loop 外加 k_start < s2_real 的 early skip。
单 kernel profile 显示 block0 aiv_time 从 10502 us 降到 850 us 左右，MTE2 指令和 GM_to_UB 数据量也大幅下降。
```

### Q3：为什么拆 kernel 没有效果？

答：

```text
拆 Stage5 后 main avg 从约 119 ms 降到约 97 ms，但新增 query_weight_grad_kernel_0/1 合计约 25.7 ms/launch。
full profile 总耗时是 4845.873 ms / 4 = 1211.47 ms/op，链路总账变差。
所以我回滚了拆 kernel。
```

### Q4：你怎么保证优化正确？

答：

```text
优化只跳过原本 k_mask 全 false 的 tile，不改变有效 tile 的公式。
scatter 本地累加只改变求和组织方式，仍是 sum_g 后写入 dKeyIndex。
public API、输出 shape、dtype、CANN/numpy 对齐逻辑和测试容忍度都没有改。
每轮跑 precision 和 large correctness，再看 profile。
```

### Q5：如果继续优化下一步做什么？

答：

```text
先跑当前最终版本 full profile，确认 main 和 scatter 谁是新的第一瓶颈。
如果 scatter 重新成为第一瓶颈，继续优化 scatter 的并行粒度和 atomic 冲突。
如果 main 仍然第一，考虑更结构化地拆 Stage2 teacher 或 Stage1/5 的中间 buffer 复用，但必须用 full profile 判断新增 kernel 是否值得。
```

## 10. 下一步必须补的证据

当前 full profile 已经确认当前最终版本为 `1019.887 ms/op`，且瓶颈仍然是 `_indexer_grad_kl_loss_kernel_0`。下一步最缺的是当前最终版本的 main 单 kernel profile：

```bash
./run_sli_profile.sh op-main
```

当前 full profile 分解如下：

```text
total kernel duration:              4079.548 ms / 4 ops = 1019.887 ms/op
_indexer_grad_kl_loss_kernel_0:     3180.195 ms / 4 ops = 795.049 ms/op
_scatter_dkey_index_kernel_0:        768.298 ms / 4 ops = 192.075 ms/op
_gather_kv_kernel_0:                 105.026 ms / 4 ops = 26.257 ms/op
other auxiliary kernels:              26.029 ms / 4 ops = 6.507 ms/op
```

最终复盘时建议使用三种口径：

```text
原始：4032.93 ms/op
已回滚 Stage5 拆分实验：1211.47 ms/op，不作为最终版本
当前最终保留版本实测：1019.887 ms/op
```
