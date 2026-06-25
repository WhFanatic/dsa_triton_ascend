# Script 目录使用指导

## 脚本概览

| 脚本 | 用途 | 典型场景 |
|------|------|---------|
| `test_sparse.sh` | Sparse 算子功能/精度测试 | 验证算子正确性、精度对齐 |
| `profile_sparse.sh` | Sparse 算子性能采集 | Triton vs CANN 性能对比 |
| `profile_sparse_detail.sh` | Triton 单 kernel 细粒度 profiling | 定位 Triton 各子 kernel 耗时占比 |
| `extract_sparse_profile.sh` | profiling 数据文本提取 | 从 profiling 目录提取可读报告 |
| `diag_env.sh` | NPU 内存/环境诊断 | 排查 OOM、VMM 耗尽、环境异常 |

---

## 1. test_sparse.sh — Sparse 算子功能/精度测试

### 命令

```bash
./script/test_sparse.sh [test_type]
```

### 参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `test_type` | 否 | `all` | 测试类型：`smoke` / `accuracy` / `all` |

### test_type 说明

| 类型 | 说明 | pytest filter |
|------|------|--------------|
| `smoke` | 单 shape 冒烟测试（CANN 兼容 shape） | `test_sparse_grad_kl_loss_large_precision[1-4096-4096-64-512-64-128-2048-fp16]` |
| `accuracy` | 精度对齐测试（多个小 shape） | `test_sparse_grad_kl_loss_precision and not large` |
| `all` | smoke + accuracy | 依次执行以上两项 |

### 依赖

- Ascend NPU（默认 device 6）
- mindspore 2.9.0
- triton-ascend 3.2.1
- `test_sli_grad_kl_loss_triton.py`（项目根目录）

### 使用场景

| 场景 | 命令 |
|------|------|
| 修改 kernel 后快速验证 | `./script/test_sparse.sh smoke` |
| 精度对齐检查 | `./script/test_sparse.sh accuracy` |
| 完整回归测试 | `./script/test_sparse.sh all` |

---

## 2. profile_sparse.sh — Sparse 算子性能采集

### 命令

```bash
./script/profile_sparse.sh [mode]
```

### 参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `mode` | 否 | `all` | 采集模式：`timing` / `triton` / `cann` / `all` |

### mode 说明

| mode | 行为 | 产出 |
|------|------|------|
| `timing` | Triton vs CANN 端到端耗时对比 | 标准输出中的耗时结果 |
| `triton` | Triton kernel profiling（msprof） | `./profiler_data_sli_grad_kl_loss/` |
| `cann` | CANN 算子 profiling（msprof） | `./profiler_data_sli_grad_kl_loss_cann/` |
| `all` | 以上三项全部执行 | 全部产出 |

### 依赖

- Ascend NPU（默认 device 6）
- mindspore 2.9.0
- triton-ascend 3.2.1
- `perf_sli_grad_kl_loss_triton.py`（项目根目录）

### 使用场景

| 场景 | 命令 |
|------|------|
| 只对比耗时，不做 profiling | `./script/profile_sparse.sh timing` |
| 只采集 Triton 侧 profile | `./script/profile_sparse.sh triton` |
| 只采集 CANN 侧 profile | `./script/profile_sparse.sh cann` |
| 首次完整性能评估 | `./script/profile_sparse.sh all` |

---

## 3. profile_sparse_detail.sh — Triton 单 kernel 细粒度 profiling

### 命令

```bash
./script/profile_sparse_detail.sh [device_id] [output_dir]
```

### 参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `device_id` | 否 | `6` | Ascend NPU 设备 ID |
| `output_dir` | 否 | `./profiler_data_sli_detail` | msprof 输出目录 |

### 输出

在 `$output_dir/OPPROF_*/` 下按 kernel 生成子目录，每个 kernel 包含：

| CSV 文件 | 说明 |
|----------|------|
| `OpBasicInfo.csv` | kernel 名称 / 耗时 / block dim |
| `PipeUtilization.csv` | 流水线利用率 |
| `ArithmeticUtilization.csv` | 计算利用率 |
| `Memory.csv` / `MemoryUB.csv` | 显存 / UB 使用 |
| `L2Cache.csv` | L2 缓存 |

采集的 5 个 Triton kernel：
- `_gather_kv_kernel`
- `_teacher_distribution`
- `_indexer_grad_kl_loss`
- `_query_index_weight_grad`
- `_scatter_dkey_index`

### 依赖

- msprof（CANN toolkit）
- 环境变量：`ASCEND_RT_VISIBLE_DEVICES`、`TRITON_END=mindspore` 等（脚本内已设置）

### 使用场景

| 场景 | 命令 |
|------|------|
| 定位哪个子 kernel 耗时最大 | `./script/profile_sparse_detail.sh` |
| 分析 kernel 硬件利用率瓶颈 | `./script/profile_sparse_detail.sh 6` |
| 指定输出路径 | `./script/profile_sparse_detail.sh 6 ./my_detail_prof` |
| 新 shape 下 kernel 耗时分解 | `./script/profile_sparse_detail.sh` |

---

## 4. extract_sparse_profile.sh — Profiling 数据文本提取

### 命令

```bash
./script/extract_sparse_profile.sh [triton_dir] [cann_dir] [msprof_dir] [output_file]
```

### 参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `triton_dir` | 否 | `./profiler_data_sli_grad_kl_loss` | profile_sparse.sh triton 模式输出 |
| `cann_dir` | 否 | `./profiler_data_sli_grad_kl_loss_cann` | profile_sparse.sh cann 模式输出 |
| `msprof_dir` | 否 | `./profiler_data_sli_detail` | profile_sparse_detail.sh 输出 |
| `output_file` | 否 | `./sparse_profile_report.txt` | 文本报告路径 |

### 报告内容（5 个 Part）

| Part | 内容 | 数据来源 |
|------|------|---------|
| Part 1 | Triton kernel 详情、API 统计、AICore 指标、op 耗时排序、日志 | `triton_dir` |
| Part 2 | CANN kernel 详情、API 统计、AICore 指标、op 耗时排序、日志 | `cann_dir` |
| Part 3 | msprof 单 kernel 耗时、stage 分解、硬件利用率 | `msprof_dir` |
| Part 4 | Triton vs CANN 对比（标注来自标准输出） | 手动参考 |
| Part 5 | 汇总统计（各 stage 耗时占比） | 综合 |

### 使用场景

| 场景 | 命令 |
|------|------|
| 采集完成后生成可读报告 | `./script/extract_sparse_profile.sh` |
| 指定不同 profiling 数据目录 | `./script/extract_sparse_profile.sh ./triton_data ./cann_data ./detail ./report.txt` |
| 只看 Triton 侧（不提供 CANN/msprof） | `./script/extract_sparse_profile.sh ./triton_data /none /none` |
| 生成后归档报告 | `./script/extract_sparse_profile.sh && cp sparse_profile_report.txt reports/20260625.txt` |

---

## 5. diag_env.sh — NPU 环境诊断

### 命令

```bash
./script/diag_env.sh [device_id]
```

### 参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `device_id` | 否 | `6` | Ascend NPU 设备 ID |

### 输出

- 文件：`diag_env_device{id}.log`
- 内容：NPU 硬件信息、HBM 使用率、进程列表、驱动/CANN 版本、VMM 内核信息、
  MindSpore 内存 API、小分配压力测试、系统内存限制

### 使用场景

| 场景 | 命令 |
|------|------|
| OOM / VMM handle 耗尽 | `./script/diag_env.sh 6` |
| 查看 NPU 进程内存占用 | `./script/diag_env.sh` |
| 升级 CANN 前存档环境信息 | `./script/diag_env.sh 0 > diag_before_upgrade.log` |
| A2/A3 服务器通用诊断 | `./script/diag_env.sh 6` |

---

## 典型工作流

### 初次性能评估

```bash
# 1. 环境诊断（确认 NPU 可用）
./script/diag_env.sh 6

# 2. 冒烟测试（确认算子可用）
./script/test_sparse.sh smoke

# 3. 完整 profiling
./script/profile_sparse.sh all

# 4. Triton 细粒度分析
./script/profile_sparse_detail.sh 6

# 5. 生成文本报告
./script/extract_sparse_profile.sh
```

### 迭代优化流程

```bash
# 1. 修改 kernel 代码
vim ...

# 2. 快速冒烟验证
./script/test_sparse.sh smoke

# 3. 重新 profiling 看效果
./script/profile_sparse.sh timing     # 只看耗时
./script/profile_sparse_detail.sh     # 看 kernel 级别瓶颈

# 4. 提取报告对比
./script/extract_sparse_profile.sh . . ./profiler_data_sli_detail ./report_v2.txt
```

### 精度回归

```bash
# 精度测试（多个 shape）
./script/test_sparse.sh accuracy
```
