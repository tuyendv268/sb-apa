#!/usr/bin/env bash
. ./path.sh

nnet3-copy --binary=false \
    kaldi/exp/chain_cleaned/tdnn_1d_sp/final.mdl \
    kaldi/exp/chain_cleaned/tdnn_1d_sp/final.txt

cp -r src/convert.py .
python convert.py \
    --libri_chain_txt_path kaldi/exp/chain_cleaned/tdnn_1d_sp/final.txt \
    --acoustic_model_path kaldi/torch/acoustic_model.pt
rm convert.py

