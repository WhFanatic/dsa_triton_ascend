#!/bin/bash
export ASCEND_RT_VISIBLE_DEVICES=8
export TRITON_END=mindspore
export TRITON_BACKEND=mindspore
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

pytest test_li_triton.py -v "$@"
# python test_li_triton.py
# python test_sli_grad_kl_loss_triton.py