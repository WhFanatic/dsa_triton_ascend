#!/bin/bash
# export ASCEND_RT_VISIBLE_DEVICES=4
# export TRITON_END=mindspore
# export TRITON_BACKEND=mindspore
# export TORCH_DEVICE_BACKEND_AUTOLOAD=0
# export TRITON_CACHE_DIR=./my_triton_cache

# ####################
# LightningIndexer 算子测试
# ####################

# ---- lightning_indexer_triton ----
# 基础调试（__main__，跑固定 shape）
# python test_li_triton.py
# 功能测试（triton vs numpy golden，验算法正确性，覆盖 GQA + fp16/bf16）
# pytest --forked test_li_triton.py -v -k test_golden "$@"
# 精度测试（triton vs CANN ops.lightning_indexer）
# pytest --forked test_li_triton.py -v -k test_accuracy "$@"
# 功能自检（shape/dtype/indices 范围，超 CANN 约束的大 shape）
# pytest --forked test_li_triton.py -v -k test_basic "$@"

# ---- 日常快速回归（改完代码先跑这条：smoke，精选 3 个典型 case）----
# 命中 GQA / multi-batch / 不同 dtype 场景
# pytest --forked test_li_triton.py -v -k "smoke" "$@"

# ---- 全量功能+精度回归（结构/逻辑大改后才跑：golden & accuracy）----
# pytest --forked test_li_triton.py -v -k "test_golden or test_accuracy" "$@"

# ---- 性能 / profiling ----
# 计时 + speedup（triton vs CANN）
# TRITON_PRINT_AUTOTUNING=1 python perf_li_triton.py
# 内核性能测试（msprof op 指定 kernel，避免全量采集与 triton driver 冲突导致 segfault）
# msprof op --kernel-name="_lightning_indexer_score_kernel" --output=./profilers python perf_li_triton.py --kernel-only

# ####################
# SparseLightningIndexerGradKLLoss 算子测试----脚本待调试
# ####################

# 基础调试
# python test_sli_grad_kl_loss_triton.py
# 全量测试
# pytest --forked test_sli_grad_kl_loss_triton.py -v "$@"
# 性能测试（triton vs CANN 计时 + speedup）
# TRITON_PRINT_AUTOTUNING=1 python perf_sli_grad_kl_loss_triton.py
# ./script/profile_sparse.sh all
# 内核性能测试（msprof op 指定 kernel，3个 kernel 耗时汇总为 triton 总耗时）


# ####################
# Dense 算子测试（DenseLightningIndexerSoftmaxLse / GradKLLoss）
# ####################

# ---- 基础调试（__main__，跑 DENSE_TEST_CONFIGS 4 个 CANN 范围配置 × bf16，不依赖 CANN import）----
# python test_dense_loss_backward_triton.py

# ---- LSE 精度（分流：D_idx=128/Nidx1∈{32,64} vs CANN，其余 vs NumPy；bf16 全 29 配置 + fp16 smoke 4，共 33）----
# pytest test_dense_loss_backward_triton.py -v -k test_dense_softmax_lse_precision "$@"

# ---- LSE 接口守卫（不支持参数须 raise，不跑算子、不依赖 NPU，本机可跑）----
# pytest test_dense_loss_backward_triton.py -v -k test_dense_softmax_lse_guards "$@"

# ---- grad 功能+精度（分流：CANN 范围 vs CANN grad，其余 vs NumPy golden；两段式 LSE→grad；bf16 全 29 配置 + fp16 smoke 4，共 33）----
# pytest test_dense_loss_backward_triton.py -v -k test_dense_grad_kl_loss_triton_supported_shapes "$@"

# ---- grad 严格 CANN golden 对比（纯 CANN baseline，4 配置，需 CANN 环境）----
# pytest test_dense_loss_backward_triton.py -v -k test_dense_grad_kl_loss_precision "$@"

# ---- NumPy LSE 与 CANN LSE 校准（默认跑，确认 NumPy golden 可信，需 CANN 环境）----
# pytest test_dense_loss_backward_triton.py -v -k test_dense_lse_numpy_matches_cann "$@"

# ---- 全量回归（LSE + grad 分流 + CANN golden，所有配置）----
# pytest test_dense_loss_backward_triton.py -v "$@"

# ---- 性能 / profiling ----
# 计时 + speedup（triton vs CANN）
# TRITON_PRINT_AUTOTUNING=1 python perf_dense_loss_backward_triton.py



# ####################
# SparseFlashAttention 算子测试
# ####################

# ---- 前向 sparse_flash_attention_triton ----
# 基础调试（__main__，跑固定几组 golden）
# python test_sfa_triton.py
# 功能测试（triton vs numpy golden，验算法正确性，覆盖 D 128/256/512 + fp16/bf16）
# pytest --forked test_sfa_triton.py -v -k test_golden "$@"
# 精度测试（triton vs CANN ops.sparse_flash_attention）
# pytest --forked test_sfa_triton.py -v -k test_accuracy "$@"
# 功能自检（shape/dtype/finiteness，超 CANN 约束的大 shape）
# pytest --forked test_sfa_triton.py -v -k test_basic "$@"

# ---- 反向 sparse_flash_attention_grad_triton ----
# 基础调试（__main__，诊断 smoke：逐梯度打印 maxabs/over-tol/worst）
# python test_sfa_grad_triton.py
# 功能测试（triton vs numpy backward golden）
# pytest --forked test_sfa_grad_triton.py -v -k test_golden "$@"
# 功能测试只跑 cann 前向路径
# pytest --forked test_sfa_grad_triton.py -v -k "test_golden and cann"
# 只功能测试跑 triton 前向路径
# pytest --forked test_sfa_grad_triton.py -v -k "test_golden and triton"
# 精度测试（triton vs ms.grad(ops.sparse_flash_attention)，CANN backward）
# pytest --forked test_sfa_grad_triton.py -v -k test_accuracy "$@"
# 功能自检（shape/dtype/finiteness）
# pytest --forked test_sfa_grad_triton.py -v -k test_basic "$@"
# 接口守卫（不支持参数须 raise ValueError，无需 NPU）
# pytest --forked test_sfa_grad_triton.py -v -k test_guards "$@"
# smoke（已混合覆盖两条路径）
# pytest --forked test_sfa_grad_triton.py -v -k smoke

# ---- 日常快速回归（改完代码先跑这条：前向+反向 smoke，精选 7+7 个典型 case）----
# 命中 BLOCK_S1 合核风险点（跨 batch / 尾部 padding / mode0/3 / S1=1 / block-wise）+ fp16/bf16/D 覆盖
# pytest --forked test_sfa_triton.py test_sfa_grad_triton.py -v -k "smoke" "$@"

# ---- 全量功能+精度回归（结构/逻辑大改后才跑：前向+反向 golden & accuracy，约 90+ case）----
# pytest --forked test_sfa_triton.py test_sfa_grad_triton.py -v -k "test_golden or test_accuracy" "$@"

# ---- 性能 / profiling ----
# 计时 + speedup（triton vs CANN）
# TRITON_PRINT_AUTOTUNING=1 python perf_sfa_triton.py
# 内核性能测试（msprof op 指定 kernel，避免全量采集与 triton driver 冲突导致 segfault）
# msprof op --kernel-name="_sfa_kernel" --output=./profilers python perf_sfa_triton.py --kernel-only

# 计时 + speedup（triton vs CANN）
# TRITON_PRINT_AUTOTUNING=1 python perf_sfa_grad_triton.py
# 内核性能测试（msprof op 指定 kernel，避免全量采集与 triton driver 冲突导致 segfault）
# msprof op --kernel-name="_sfa_grad_kernel" --output=./profilers python perf_sfa_grad_triton.py --kernel-only
