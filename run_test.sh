#!/bin/bash
export ASCEND_RT_VISIBLE_DEVICES=8
export TRITON_END=mindspore
pytest test_lightning_indexer_triton.py -v "$@"
# python test_lightning_indexer_triton.py