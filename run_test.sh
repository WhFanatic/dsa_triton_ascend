#!/bin/bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
pytest test_lightning_indexer_triton.py -v "$@"
