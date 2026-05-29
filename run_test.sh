#!/bin/bash
export ASCEND_RT_VISIBLE_DEVICES=8
export TRITON_END=mindspore
export TRITON_BACKEND=mindspore
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

pytest test_li_triton.py -v "$@"
# python test_li_triton.py
# python perf_li_triton.py
# python test_sli_grad_kl_loss_triton.py

# python test_li_triton.py
# msprof --output=./profilers/prof_arith --aicpu=on --ai-core=on --ascendcl=on --aic-metrics=ArithmeticUtilization python test_li_triton.py
# msprof --output=./profilers/prof_pipe  --aicpu=on --ai-core=on --ascendcl=on --aic-metrics=PipeUtilization       python test_li_triton.py
# msprof --output=./profilers/prof_mem   --aicpu=on --ai-core=on --ascendcl=on --aic-metrics=Memory                python test_li_triton.py
# msprof --output=./profilers/prof_ub    --aicpu=on --ai-core=on --ascendcl=on --aic-metrics=MemoryUB              python test_li_triton.py