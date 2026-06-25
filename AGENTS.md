# AGENTS.md — dsa_triton_ascend_v2

## Repo Overview

Triton-Ascend implementation of DSA (DeepSeek Sparse Attention) operators on Huawei Ascend NPU.
Interfaces aligned with `ops.lightning_indexer`, `aclnnSparseLightningIndexerGradKLLoss`,
and `ops.sparse_flash_attention` via `ops._ms_pyfunc()` into MindSpore static graph.

**Hardware**: NPU 910B (current) → 910C (target). 910B UB: 192KB/core.
**Stack**: CANN 9.0.0, MindSpore 2.9.0, triton-ascend 3.2.1.

## Key Files

| File | Role |
|------|------|
| `sparse_lightning_indexer_grad_kl_loss_triton.py` | **SLI backward** — main optimization target |
| `lightning_indexer_triton.py` | LightningIndexer forward |
| `sparse_flash_attention_triton.py` | SparseFlashAttention forward |
| `sparse_flash_attention_grad_triton.py` | SFA backward |
| `sli_grad_kl_loss_cann.py` | CANN reference (CANN aclnn custom op wrapper) |
| `sli_grad_kl_loss_numpy.py` | Numpy golden reference |
| `test_sli_grad_kl_loss_triton.py` | SLI correctness tests |
| `perf_sli_grad_kl_loss_triton.py` | SLI benchmarking + profiling driver |
| `script/` | Test/profile/diagnostic shell scripts (see SCRIPT_GUIDE.md) |

## Dev Commands

```bash
# NPU device: default 6. Required env:
export ASCEND_RT_VISIBLE_DEVICES=6
export TRITON_END=mindspore
export TRITON_BACKEND=mindspore
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TRITON_CACHE_DIR=./my_triton_cache

# Clear Triton cache after ANY code change (stale cache causes silent failures)
rm -rf ./my_triton_cache

# Smoke test (single shape, triton vs CANN)
bash script/test_sparse.sh smoke

# Accuracy test (multiple small shapes)
bash script/test_sparse.sh accuracy

# End-to-end timing (triton vs CANN)
bash script/profile_sparse.sh timing

# Triton profiler (ms.profiler)
bash script/profile_sparse.sh triton

# Per-kernel msprof profiling (5 kernels, --kernel-name= prefix)
bash script/profile_sparse_detail.sh 6 ./profiler_data_sli_detail

# Extract text report from profiling data
bash script/extract_sparse_profile.sh
```

## SLI Operator Architecture

### 5 Triton kernels (per chunk)

| Kernel | Grid | Time % | Role |
|--------|------|--------|------|
| `_gather_kv_kernel` ×3 | (B*S1, K_blocks, D_blocks) | 16% | Gather key/key_idx/key_rope at sparse positions |
| `_teacher_distribution_kernel` | (B*S1, K_blocks) | 18% | Teacher p[k] from forward softmax stats |
| `_indexer_grad_kl_loss_kernel` | (B*S1,) | 7% | I[k], dI, KL loss (main compute) |
| `_query_index_weight_grad_kernel` | (B*S1, G_blocks, D_blocks) | 35% | dW + dQueryIndex chain rule |
| `_scatter_dkey_index_kernel` | (B*S1, K_blocks, D_blocks) | 24% | Scatter-add dKeyIndex (atomic) |

### Chunking

`SPARSE_GRAD_S1_CHUNK = 512` — S1 dimension is chunked when >512.
Each chunk independently runs `_sparse_lightning_indexer_grad_kl_loss_core`.
**S1 per core call MUST be ≤ 1024** — values ≥1024 cause silent incorrect results.
The unchunked path (S1 ≤ CHUNK) works for small S1 but fails for large S1 due to
an uninvestigated correctness bug (possibly grid/tile size related).

### Per-chunk kernel time breakdown (S1_chunk=512, B=1, topK=2048)

| Kernel | us | % |
|--------|-----|-----|
| gather_kv ×3 | 3,551 | 15.9% |
| teacher_dist | 3,932 | 17.6% |
| indexer_main | 1,551 | 6.9% |
| query_idx_weight | 7,922 | 35.4% |
| scatter_dkey | 5,411 | 24.2% |
| **Total** | **22,367** | **100%** |

## BLOCK Parameter Constraints (910B, 192KB UB)

### Tried-and-safe values (current code)
```
BLOCK_K_GATHER = 256     (128→256: 15% improvement, safe)
BLOCK_D_GATHER = 128     (256: UB overflow, FAIL)
BLOCK_K_MAIN = 128       (256: wrong results, FAIL)
BLOCK_D_MAIN = 64        (128: UB overflow in teacher, FAIL)
BLOCK_K_QUERY_WEIGHT = 128  (64→128: safe)
BLOCK_G_QUERY_WEIGHT = 4    (2→4: safe)
BLOCK_K_SCATTER = 128    (64→128: safe)
```

### UB overflow threshold
- `BLOCK_K_GATHER=256 + BLOCK_D_GATHER=256` → UB overflow (~233KB needed, 192KB available)
- Teacher kernel: BLOCK_K=256 + BLOCK_D=128 → wrong results (likely UB overflow)

## Optimization Notes

### Attempted and failed
- **Chunk size increase** (512→1016/2048/4096): Larger per-chunk S1 scales kernel compute linearly,
  offsetting fewer chunks. 1016 was ~11% *slower*. Root cause: per-chunk kernel time scales
  with S1, overhead reduction insufficient.
- **Remove `runtime.synchronize()` on line 606**: Causes incorrect results (race condition
  between indexer_main → query_weight). Must keep.
- **`mint.zeros` → `mint.empty`**: Causes incorrect results even with masked reads.
  MindSpore empty tensors may not have device memory backing.
- **Merge 3 gather calls → 1 fused kernel**: ✅ Completed. Uses `D_IDX`/`D_ROPE` constexpr
  guards with per-dimension `d_block * BLOCK_D < D_x` checks. Grid uses `max(D, D_idx, D_rope)`.
  Eliminates 2 kernel launches per chunk. (commit `e8ea5e3`)
- **Merge teacher_dist into indexer_main**: Not attempted (Oracle estimated modest gains).

### Future directions (per Oracle review)
1. Inline teacher_dist into indexer_main (eliminate buf_p, reduce GM round-trips)
2. Merge query_idx_weight + scatter_dkey (both read s_idx_buf/di; combine to avoid double-read)
3. BLOCK parameter re-tuning after kernel fusion
4. Persistent output buffer across chunks (eliminate concat/reduce overhead)

### 910B → 910C transition
- 910C has larger UB (256KB+ vs 192KB) → larger BLOCK_K/BLOCK_D possible
- 910C has more AI cores → better scaling for parallel kernels
- Re-tune all BLOCK parameters on 910C hardware

## Constraints (SLI operator)

- N2=1 (MQA), Nidx2=1
- sparse_mode=3 only (rightDownCausal)
- layout="BSND" only
- pre_tokens/next_tokens: only default INT64_MAX
- N1 ∈ {32, 64, 128}, D ∈ {128, 256, 512}, D_idx ∈ {128}

## Code Style

- Address user as '主人' (from CLAUDE.md)
- No comment/formatting changes unrelated to code modifications
- Keep code as simple as possible
- No runtime env on dev machine; write code, don't run tests (trivial verification excluded)
- Python files: LF line endings only (`.gitattributes`)
- Match existing code style exactly — don't "improve" adjacent code
- Every changed line must trace directly to the request
