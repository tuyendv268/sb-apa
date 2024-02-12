#!/bin/bash

STAGE=2
MODEL_TYPE="w2v2"

DATA_DIR=data  # Change this path to the path where you keep your data.
APR_DATA_FOLDER=$DATA_DIR/apr/
SOCRING_DATA_FOLDER=$DATA_DIR/scoring/

# Result logging
RESULTS_FOLDER=$DATA_DIR/results/
EXP_METADATA_FILE=${RESULTS_FOLDER}/exp_metadata.csv

APR_RESULTS_FILE=${RESULTS_FOLDER}/results_apr.csv
PREP_SCORING_RESULTS_FILE=${RESULTS_FOLDER}/results_scoring.csv

EPOCH_RESULTS_DIR=${RESULTS_FOLDER}/epoch_results
PARAMS_DIR=${RESULTS_FOLDER}/params
EXP_DESCRIPTION=""

# Train the scorer (Best version).
if [ $STAGE -le 2 ]; then
    SCORING_MODEL_DIR="results/scoring"
    PRETRAINED_MODEL_DIR="results/scoring"
    SCORING_HPARAM_FILE="hparams/scoring.yml"

    python3 eval.py $SCORING_HPARAM_FILE \
        --data_folder=$SOCRING_DATA_FOLDER \
        --batch_size=4 \
        --pretrained_model_folder=$PRETRAINED_MODEL_DIR \
        --use_augmentation=True \
        --round_scores=False \
        --exp_folder=$SCORING_MODEL_DIR \
        --exp_metadata_file=$EXP_METADATA_FILE \
        --results_file=$PREP_SCORING_RESULTS_FILE\
        --epoch_results_dir=$EPOCH_RESULTS_DIR \
        --params_dir=$PARAMS_DIR \
        --exp_description="Step 1 then 3 directly (aug, no round)" \

fi