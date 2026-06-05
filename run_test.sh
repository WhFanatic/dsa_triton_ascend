#!/bin/bash
export ASCEND_RT_VISIBLE_DEVICES=7
export TRITON_END=mindspore
export TRITON_BACKEND=mindspore
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# ####################
# LightningIndexer 算子测试
# ####################

# 基础调试
# python test_li_triton.py
# 全量测试
# pytest --forked test_li_triton.py -v "$@"
# 性能测试
# TRITON_PRINT_AUTOTUNING=1 python perf_li_triton.py
# 内核性能测试
# python test_li_triton.py
# msprof --output=./profilers/prof_arith --aicpu=on --ai-core=on --ascendcl=on --aic-metrics=ArithmeticUtilization python test_li_triton.py
# msprof --output=./profilers/prof_pipe  --aicpu=on --ai-core=on --ascendcl=on --aic-metrics=PipeUtilization       python test_li_triton.py
# msprof --output=./profilers/prof_mem   --aicpu=on --ai-core=on --ascendcl=on --aic-metrics=Memory                python test_li_triton.py
# msprof --output=./profilers/prof_ub    --aicpu=on --ai-core=on --ascendcl=on --aic-metrics=MemoryUB              python test_li_triton.py

# ####################
# SparseLightningIndexerGradKLLoss 算子测试
# ####################

# python test_sli_grad_kl_loss_triton.py

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
# 精度测试（triton vs ms.grad(ops.sparse_flash_attention)，CANN backward）
# pytest --forked test_sfa_grad_triton.py -v -k test_accuracy "$@"
# 功能自检（shape/dtype/finiteness）
# pytest --forked test_sfa_grad_triton.py -v -k test_basic "$@"
# 接口守卫（不支持参数须 raise ValueError，无需 NPU）
# pytest --forked test_sfa_grad_triton.py -v -k test_guards "$@"

# ---- 一键功能+精度回归（前向+反向 golden & accuracy，改完代码跑这条）----
# pytest --forked test_sfa_triton.py test_sfa_grad_triton.py -v -k "test_golden or test_accuracy" "$@"

# ---- 性能 / profiling ----
# 计时 + speedup（triton vs CANN）
TRITON_PRINT_AUTOTUNING=1 python perf_sfa_triton.py
# 内核性能测试（msprof op 指定 kernel，避免全量采集与 triton driver 冲突导致 segfault）
msprof op --kernel-name="_sfa_kernel" --output=./profilers python perf_sfa_triton.py --kernel-only

