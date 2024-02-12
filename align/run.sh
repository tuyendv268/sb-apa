#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

. ./path.sh

serve run align:app --host 0.0.0.0 --port 9999
# python test.py