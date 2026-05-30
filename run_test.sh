#!/bin/bash
export ASCEND_RT_VISIBLE_DEVICES=8
export TRITON_END=mindspore
export TRITON_BACKEND=mindspore
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# ####################
# LightningIndexer 算子测试
# ####################

# 基础调试
# python test_li_triton.py
# 全量测试
# pytest test_li_triton.py -v "$@"
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
