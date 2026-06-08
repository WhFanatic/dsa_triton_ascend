#!/bin/bash
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export TRITON_END=mindspore
export TRITON_BACKEND=mindspore
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# ####################
# LightningIndexer tests
# ####################

# Basic debug
# python test_li_triton.py
# Full correctness test
# pytest --forked test_li_triton.py -v "$@"
# Performance helper
# python perf_li_triton.py

# ####################
# SparseLightningIndexerGradKLLoss tests
# ####################

# Basic debug
# python test_sli_grad_kl_loss_triton.py
# Full correctness test
# pytest --forked test_sli_grad_kl_loss_triton.py -v "$@"
# Performance helper
# python perf_sli_grad_kl_loss_triton.py

# ####################
# SparseFlashAttention forward/backward tests
# ####################

# Basic debug
# python test_sparse_flash_attention_triton.py
# Full correctness test
# pytest --forked test_sparse_flash_attention_triton.py -v "$@"
# Performance helper
# python perf_sparse_flash_attention_triton.py

# ####################
# Dense LightningIndexer softmax_lse + grad_kl_loss tests
# ####################

# Basic debug
# python test_dense_loss_backward_triton.py
# Full correctness test
# pytest --forked test_dense_loss_backward_triton.py -v "$@"
# Performance helper
# python perf_dense_loss_backward_triton.py

# ####################
# Common command groups
# ####################

# Full correctness sweep
# pytest --forked test_li_triton.py -v "$@"
# pytest --forked test_sli_grad_kl_loss_triton.py -v "$@"
# pytest --forked test_sparse_flash_attention_triton.py -v "$@"
# pytest --forked test_dense_loss_backward_triton.py -v "$@"

# New generalized DSA operators only
# pytest --forked test_sparse_flash_attention_triton.py -v "$@"
# pytest --forked test_dense_loss_backward_triton.py -v "$@"

# New perf helpers only
# python perf_sparse_flash_attention_triton.py
# python perf_dense_loss_backward_triton.py

# ####################
# Profiling examples
# ####################

# msprof --output=./profilers/prof_li --aicpu=on --ai-core=on --ascendcl=on --aic-metrics=PipeUtilization python test_li_triton.py
# msprof --output=./profilers/prof_sli --aicpu=on --ai-core=on --ascendcl=on --aic-metrics=PipeUtilization python test_sli_grad_kl_loss_triton.py
# msprof --output=./profilers/prof_sfa --aicpu=on --ai-core=on --ascendcl=on --aic-metrics=PipeUtilization python test_sparse_flash_attention_triton.py
# msprof --output=./profilers/prof_dense --aicpu=on --ai-core=on --ascendcl=on --aic-metrics=PipeUtilization python test_dense_loss_backward_triton.py
