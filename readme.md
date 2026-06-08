# Triton-Ascend implementation of DSA

This directory contains correctness-first Triton-Ascend implementations for the
DSA (DeepSeek Sparse Attention) related operators. The wrappers follow the same
style as the existing `lightning_indexer_triton.py` and
`sparse_lightning_indexer_grad_kl_loss_triton.py`: expose a MindSpore-friendly
function plus an `ms.nn.Cell` wrapper, keep unsupported official features as
explicit argument checks, and connect kernels through `ops._ms_pyfunc()`.

Current priority is correctness and interface alignment. These kernels are not
performance tuned.

## Environment

- CANN 9.0.0
- MindSpore 2.9.0
- triton-ascend 3.2.1
- pytest-forked, for full pytest runs with `pytest --forked ...`

Recommended environment variables:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export TRITON_END=mindspore
export TRITON_BACKEND=mindspore
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

## Files

| File | Description |
| --- | --- |
| `lightning_indexer_triton.py` | Triton-Ascend `lightning_indexer` implementation |
| `sparse_lightning_indexer_grad_kl_loss_triton.py` | Triton-Ascend sparse LightningIndexer grad KL loss implementation |
| `sparse_flash_attention_triton.py` | Triton-Ascend sparse flash attention forward/backward implementation |
| `dense_loss_backward_triton.py` | Triton-Ascend dense LightningIndexer softmax LSE and grad KL loss implementation |
| `test_li_triton.py` | `lightning_indexer` correctness test |
| `test_sli_grad_kl_loss_triton.py` | sparse LightningIndexer grad KL loss correctness test |
| `test_sparse_flash_attention_triton.py` | SFA forward/backward correctness test |
| `test_dense_loss_backward_triton.py` | dense LightningIndexer softmax LSE + grad KL loss correctness test |
| `perf_li_triton.py` | `lightning_indexer` timing/profiling helper |
| `perf_sli_grad_kl_loss_triton.py` | sparse LightningIndexer grad KL loss timing/profiling helper |
| `perf_sparse_flash_attention_triton.py` | SFA forward/backward timing/profiling helper |
| `perf_dense_loss_backward_triton.py` | dense LightningIndexer softmax LSE + grad KL loss timing/profiling helper |

## Supported Shapes

| Operator | Parameter | Supported values |
| --- | --- | --- |
| `lightning_indexer` | `N1` / indexer heads | 32, 64, 128 |
|  | `D` / indexer head dim | 128, 256, 512 |
| `sparse_lightning_indexer_grad_kl_loss` | `Nidx1` / indexer heads | 32, 64, 128 |
|  | `D_idx` / indexer head dim | 128, 256, 512 |
|  | `D` / attention head dim | 128, 256, 512 |
| `sparse_flash_attention` | `D` / attention head dim | 128, 256, 512 |
| `dense_lightning_indexer_softmax_lse` | `Nidx1` / indexer heads | 32, 64, 128 |
|  | `D_idx` / indexer head dim | 128, 256, 512 |
| `dense_lightning_indexer_grad_kl_loss` | `Nidx1` / indexer heads | 32, 64, 128 |
|  | `D_idx` / indexer head dim | 128, 256, 512 |
|  | `D` / attention head dim | 128, 256, 512 |

## `lightning_indexer_triton`

### Interface

```python
lightning_indexer_triton(
    query,                          # [B,S1,N1,D] or [T1,N1,D], fp16/bf16
    key,                            # [B,S2,N2,D] or [T2,N2,D], fp16/bf16
    weights,                        # [B,S1,N1] or [T1,N1], fp16/bf16/fp32
    actual_seq_lengths_query=None,  # [B] int32, list/tuple, or None
    actual_seq_lengths_key=None,    # [B] int32, list/tuple, or None
    block_table=None,               # PA_BSND is not supported
    layout_query="BSND",
    layout_key="BSND",
    sparse_count=2048,
    sparse_mode=3,
    pre_tokens=INT64_MAX,
    next_tokens=INT64_MAX,
    return_value=False,
) -> (sparseIndicesOut, sparseValuesOut)
```

The implementation follows `ops.lightning_indexer` for the supported BSND/TND
paths. Unsupported PA_BSND features are rejected.

## `sparse_lightning_indexer_grad_kl_loss_triton`

### Interface

```python
sparse_lightning_indexer_grad_kl_loss_triton(
    query,                   # [B,S1,N1,D], fp16/bf16
    key,                     # [B,S2,1,D], fp16/bf16
    query_index,             # [B,S1,Nidx1,D_idx], fp16/bf16
    key_index,               # [B,S2,1,D_idx], fp16/bf16
    weights,                 # [B,S1,Nidx1], fp16/bf16/fp32
    sparse_indices,          # [B,S1,1,topK], int32
    softmax_max,             # [B,1,S1,N1], fp32, from full attention forward
    softmax_sum,             # [B,1,S1,N1], fp32, from full attention forward
    query_rope=None,
    key_rope=None,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    scale_value=1.0,
    layout="BSND",
    sparse_mode=3,
    pre_tokens=INT64_MAX,
    next_tokens=INT64_MAX,
    deterministic=False,
) -> (dQueryIndex, dKeyIndex, dWeights, loss)
```

This operator is aligned with the sparse CANN grad KL loss path for the supported
BSND, MQA (`N2=1`, `Nidx2=1`) case.

## `sparse_flash_attention_triton`

### Forward Interface

```python
sparse_flash_attention_triton(
    query,                          # [B,S1,N1,D], fp16/bf16
    key,                            # [B,S2,N2,D], fp16/bf16
    value,                          # [B,S2,N2,D], fp16/bf16
    sparse_indices,                 # [B,S1,N2,K], int32
    scale_value=1.0,
    actual_seq_lengths_query=None,
    actual_seq_lengths_key=None,
    block_table=None,
    query_rope=None,
    key_rope=None,
    sparse_block_size=1,
    layout_query="BSND",
    layout_key="BSND",
    sparse_mode=3,
    pre_tokens=INT64_MAX,
    next_tokens=INT64_MAX,
    attention_mode=0,
    return_softmax_lse=False,
) -> (attention_out, softmax_max, softmax_sum)
```

### Backward Interface

```python
sparse_flash_attention_backward_triton(
    dout,                           # same shape as query
    query,
    key,
    value,
    sparse_indices,
    attention_out,
    softmax_max,
    softmax_sum,
    scale_value=1.0,
    actual_seq_lengths_query=None,
    actual_seq_lengths_key=None,
    block_table=None,
    query_rope=None,
    key_rope=None,
    sparse_block_size=1,
    layout_query="BSND",
    layout_key="BSND",
    sparse_mode=3,
    pre_tokens=INT64_MAX,
    next_tokens=INT64_MAX,
    attention_mode=0,
) -> (dquery, dkey, dvalue)
```

`SparseFlashAttentionTriton` and `SparseFlashAttentionGradTriton` are the
corresponding `ms.nn.Cell` wrappers.

Current limitations:

- Only BSND is supported.
- `D` must be one of 128, 256, 512.
- `N1 % N2 == 0`.
- `block_table` / PA_BSND is not supported.
- TND, RoPE, dropout, and `attention_mode != 0` are not supported.
- Non-default `pre_tokens` / `next_tokens` are rejected.

## `dense_loss_backward_triton`

The dense path is exposed as two official-style stages.

### Dense Index Softmax LSE

```python
dense_lightning_indexer_softmax_lse_triton(
    query_index,             # [B,S1,Nidx1,D_idx], fp16/bf16
    key_index,               # [B,S2,1,D_idx], fp16/bf16
    weights,                 # [B,S1,Nidx1], fp16/bf16/fp32
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout="BSND",
    sparse_mode=3,
    pre_tokens=INT64_MAX,
    next_tokens=INT64_MAX,
) -> (softmax_max_index, softmax_sum_index)
```

The softmax LSE helper returns official-style dense DLI index stats with shape
`[B,1,S1]`.

### Dense Grad KL Loss

```python
dense_lightning_indexer_grad_kl_loss_triton(
    query,                   # [B,S1,N1,D], fp16/bf16
    key,                     # [B,S2,N2,D], fp16/bf16
    query_index,             # [B,S1,Nidx1,D_idx], fp16/bf16
    key_index,               # [B,S2,1,D_idx], fp16/bf16
    weights,                 # [B,S1,Nidx1], fp16/bf16/fp32
    softmax_max,             # [B,N2,S1,G], fp32, from full attention forward
    softmax_sum,             # [B,N2,S1,G], fp32, from full attention forward
    softmax_max_index,       # [B,1,S1], [B,S1], [B*S1], or [B,1,S1,1], fp32
    softmax_sum_index,       # [B,1,S1], [B,S1], [B*S1], or [B,1,S1,1], fp32
    scale_value,
    query_rope=None,
    key_rope=None,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout="BSND",
    sparse_mode=3,
    pre_tokens=INT64_MAX,
    next_tokens=INT64_MAX,
) -> (dQueryIndex, dKeyIndex, dWeights, loss)
```

`DenseLightningIndexerSoftmaxLseTriton` and
`DenseLightningIndexerGradKLLossTriton` are the corresponding `ms.nn.Cell`
wrappers. `dense_loss_backward_triton` is kept as a compatibility helper; if the
index softmax stats are omitted, it computes them internally.

Current limitations:

- Only BSND is supported.
- Only `sparse_mode=3` rightDownCausal is supported.
- `key_index` is still limited to `Nidx2=1`.
- `key` supports `N2` values that divide query `N1`; `G = N1 / N2`.
- `Nidx1` must be 32, 64, or 128.
- `D` and `D_idx` must be 128, 256, or 512.
- Optional `query_rope` / `key_rope` are supported together for teacher-score
  computation when `Drope=64`.
- TND and non-default `pre_tokens` / `next_tokens` are not supported.

## Tests

```bash
pytest --forked test_li_triton.py -v
pytest --forked test_sli_grad_kl_loss_triton.py -v
pytest --forked test_sparse_flash_attention_triton.py -v
pytest --forked test_dense_loss_backward_triton.py -v
```

Quick smoke runs:

```bash
python test_sparse_flash_attention_triton.py
python test_dense_loss_backward_triton.py
```

## Performance Helpers

These scripts are simple timing/profiling helpers. They are not autotune scripts
and do not imply a performance target. When an official MindSpore/CANN baseline
is available for the tested shape, the helper prints the official timing and
`speedup = official_ms / triton_ms`. Unsupported official shapes or unavailable
official Python APIs are printed as `skipped` instead of failing the script.

```bash
python perf_li_triton.py
python perf_sli_grad_kl_loss_triton.py
python perf_sparse_flash_attention_triton.py
python perf_dense_loss_backward_triton.py
```

Notes:

- `perf_li_triton.py` compares against `ms.ops.lightning_indexer`.
- `perf_sli_grad_kl_loss_triton.py` compares against the local CANN wrapper
  `SparseLightningIndexerGradKLLoss` when the shape is CANN-comparable.
- `perf_sparse_flash_attention_triton.py` compares SFA forward against
  `ms.ops.sparse_flash_attention` only for the official-comparable
  `D=512, N2=1, K=2048` path. SFA backward is reported as skipped because this
  helper does not use an explicit official backward baseline.
- `perf_dense_loss_backward_triton.py` tries the official dense lightning
  experimental APIs from `hyper_parallel.custom_ops.experimental` first, then
  `mindspore.ops`. If neither path is present, official dense timings are
  skipped.

Profiler entry points are provided as commented `run_profiling()` calls inside
each perf file.

## Integration Notes

In MindFormers or other higher-level code, replace only the supported official
calls:

- `ops.lightning_indexer` -> `lightning_indexer_triton`
- `aclnnSparseLightningIndexerGradKLLoss` -> `sparse_lightning_indexer_grad_kl_loss_triton`
- sparse flash attention forward/backward -> `sparse_flash_attention_triton` and `sparse_flash_attention_backward_triton`
- dense LightningIndexer two-stage path -> `dense_lightning_indexer_softmax_lse_triton` and `dense_lightning_indexer_grad_kl_loss_triton`

For dense loss backward, `softmax_max` and `softmax_sum` must come from the full
visible-key attention teacher distribution. Sparse topK-only SFA statistics are
not equivalent unless the sparse index list covers the full visible key range.

## References

- Triton-Ascend `ops._ms_pyfunc()` integration:
  <https://gitcode.com/Ascend/triton-ascend/issues/283>
- CANN lightning_indexer:
  <https://gitcode.com/cann/ops-transformer/tree/master/attention/lightning_indexer>
- CANN sparse_lightning_indexer_grad_kl_loss:
  <https://gitcode.com/cann/ops-transformer/tree/master/attention/sparse_lightning_indexer_grad_kl_loss>
- MindSpore `ops.lightning_indexer`:
  <https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.lightning_indexer.html>
