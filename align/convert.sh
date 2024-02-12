#!/usr/bin/env bash
. ./path.sh

nnet3-copy --binary=false \
    exp/chain_cleaned/tdnn_1d_sp/final.mdl \
    exp/chain_cleaned/tdnn_1d_sp/final.txt

cp -r src/convert.py .
python convert.py \
    --libri_chain_txt_path exp/chain_cleaned/tdnn_1d_sp/final.txt \
    --acoustic_model_path exp/torch/acoustic_model.pt

rm convert.py
