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

# EXP_DESCRIPTION="Train APR then scorer"
# Train phoneme recognition model.
if [ $STAGE -le 1 ]; then
    echo "##### Training phoneme recognition model on TIMIT. ######"

    APR_MODEL_DIR="results/apr"
    APR_HPARAM_FILE="hparams/apr.yml"

    [ -d $APR_MODEL_DIR ] && rm -r $APR_MODEL_DIR && echo "Removed already existing $APR_MODEL_DIR directory.";

    python3 train.py $APR_HPARAM_FILE \
        --data_folder=$APR_DATA_FOLDER \
        --exp_folder=$APR_MODEL_DIR \
        --batch_size=2 \
        --exp_metadata_file=$EXP_METADATA_FILE \
        --results_file=$APR_RESULTS_FILE \
        --epoch_results_dir=$EPOCH_RESULTS_DIR \
        --params_dir=$PARAMS_DIR \
        --exp_description="$EXP_DESCRIPTION";

    # OMP_NUM_THREADS=1 python3 -m torch.distributed.launch \
    #     --nproc_per_node=2 \
    #         train.py $APR_HPARAM_FILE \
    #             --distributed_launch \
    #             --distributed_backend='nccl' \
    #             --data_folder=$APR_DATA_FOLDER \
    #             --exp_folder=$APR_MODEL_DIR \
    #             --batch_size=2 \
    #             --exp_metadata_file=$EXP_METADATA_FILE \
    #             --results_file=$APR_RESULTS_FILE \
    #             --epoch_results_dir=$EPOCH_RESULTS_DIR \
    #             --params_dir=$PARAMS_DIR \
    #             --exp_description="$EXP_DESCRIPTION";
fi

# Train the scorer (Best version).
if [ $STAGE -le 2 ]; then
    SCORING_MODEL_DIR="results/scoring"
    PRETRAINED_MODEL_DIR="results/apr"
    SCORING_HPARAM_FILE="hparams/scoring.yml"
    LABEL_ENCODER_PATH="pretrained/apr/label_encoder.txt"

    [ -d $SCORING_MODEL_DIR ] && rm -r $SCORING_MODEL_DIR && echo "Removed existing $SCORING_MODEL_DIR directory.";

    # OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 train.py $SCORING_HPARAM_FILE \
    #     --data_folder=$SOCRING_DATA_FOLDER \
    #     --batch_size=4 \
    #     --pretrained_model_folder=$PRETRAINED_MODEL_DIR \
    #     --label_encoder_path= $LABEL_ENCODER_PATH \
    #     --use_augmentation=True \
    #     --round_scores=False \
    #     --exp_folder=$SCORING_MODEL_DIR \
    #     --exp_metadata_file=$EXP_METADATA_FILE \
    #     --results_file=$PREP_SCORING_RESULTS_FILE\
    #     --epoch_results_dir=$EPOCH_RESULTS_DIR \
    #     --params_dir=$PARAMS_DIR \
    #     --exp_description="Step 1 then 3 directly (aug, no round)" \
    #     --find_unused_parameters \
    #     --distributed_launch \
    #     --distributed_backend='nccl' \

    python3 train.py $SCORING_HPARAM_FILE \
        --data_folder=$SOCRING_DATA_FOLDER \
        --batch_size=4 \
        --pretrained_model_folder=$PRETRAINED_MODEL_DIR \
        --label_encoder_path= $LABEL_ENCODER_PATH \
        --use_augmentation=True \
        --round_scores=False \
        --exp_folder=$SCORING_MODEL_DIR \
        --exp_metadata_file=$EXP_METADATA_FILE \
        --results_file=$PREP_SCORING_RESULTS_FILE\
        --epoch_results_dir=$EPOCH_RESULTS_DIR \
        --params_dir=$PARAMS_DIR \
        --exp_description="Step 1 then 3 directly (aug, no round)" \

fi