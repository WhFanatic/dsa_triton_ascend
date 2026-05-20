#!/bin/bash
export ASCEND_RT_VISIBLE_DEVICES=4
export TRITON_END=mindspore
##pytest test_lightning_indexer_triton.py -v "$@"
python add_kernel.py