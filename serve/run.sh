#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

. ./path.sh

CUDA_VISIBLE_DEVICES=1 serve run main:app --host 0.0.0.0 --port 9999