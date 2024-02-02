#!/bin/bash

STAGE=0
MODEL_TYPE="w2v2"

SSL_TYPE="wavlm"  # The type of the SSL. Choose from ["hubert", "wavlm", "wav2vec2"]
# The HuggingFace name of the SSL. Choose from ["facebook/hubert-large-ls960-ft", "microsoft/wavlm-large",
# "facebook/wav2vec2-large-lv60"]
WAV2VEC2_HUB="microsoft/wavlm-large"

DATA_DIR=data  # Change this path to the path where you keep your data.
TIMIT_DATA_FOLDER=$DATA_DIR/TIMIT/
PREP_DATA_FOLDER=$DATA_DIR/speechocean762/

# Result logging
RESULTS_FOLDER=$DATA_DIR/results/
EXP_METADATA_FILE=${RESULTS_FOLDER}/exp_metadata.csv

TIMIT_APR_RESULTS_FILE=${RESULTS_FOLDER}/results_apr_timit.csv
PREP_APR_RESULTS_FILE=${RESULTS_FOLDER}/results_apr_so762.csv
PREP_SCORING_RESULTS_FILE=${RESULTS_FOLDER}/results_scoring_so762.csv

EPOCH_RESULTS_DIR=${RESULTS_FOLDER}/epoch_results
PARAMS_DIR=${RESULTS_FOLDER}/params
EXP_DESCRIPTION=""
TRAINING_TYPE="apr"

SCORING_TYPE=""

# Prepare datasets

# if [ ! -d $DATA_DIR/speechocean762 ]; then
#     echo "Preparing speechocean762 dataset.";
#     mkdir $DATA_DIR/speechocean762
#     tar -xzmf $DATA_DIR/speechocean762.tar.gz -C $DATA_DIR/speechocean762;
#     echo "Finished preparing speechocean762 dataset.";
# fi

# if [ ! -d $DATA_DIR/TIMIT ]; then
#     echo "Preparing TIMIT dataset.";
#     unzip -DDq $DATA_DIR/TIMIT.zip -d $DATA_DIR/timit;
#     echo "Finished preparing TIMIT dataset.";
# fi

#######################################################################################################################

# Experiments


# Experiment #1: Train APR on native speech and train scorer using transfer learning.

# EXP_DESCRIPTION="Train APR then scorer"
# Train phoneme recognition model.
# if [ $STAGE -le 1 ]; then
#     echo "##### Training phoneme recognition model on TIMIT. ######"

#     APR_MODEL_DIR="results/apr/${MODEL_TYPE}/crdnn_${MODEL_TYPE}_timit_apr"
#     APR_HPARAM_FILE="hparams/apr/${MODEL_TYPE}/train_${MODEL_TYPE}_timit_apr.yaml"

#     [ -d $APR_MODEL_DIR ] && rm -r $APR_MODEL_DIR && echo "Removed already existing $APR_MODEL_DIR directory.";

#     python3 train.py $APR_HPARAM_FILE \
#         --data_folder=$TIMIT_DATA_FOLDER \
#         --exp_folder=$APR_MODEL_DIR \
#         --batch_size=2 \
#         --exp_metadata_file=$EXP_METADATA_FILE \
#         --results_file=$TIMIT_APR_RESULTS_FILE \
#         --epoch_results_dir=$EPOCH_RESULTS_DIR \
#         --params_dir=$PARAMS_DIR \
#         --exp_description="$EXP_DESCRIPTION";
# fi

# Train the scorer (Best version).
if [ $STAGE -le 2 ]; then
    echo "##### Training pronunciation scoring on speechocean762 with no adaptation to non-native speech. ######"

    SCORING_MODEL_DIR="results/scoring/${MODEL_TYPE}/crdnn_${MODEL_TYPE}_so762${SCORING_TYPE}_scoring_aug_no_round_no_pre_train"
    PRETRAINED_MODEL_DIR="results/apr/${MODEL_TYPE}/crdnn_${MODEL_TYPE}_timit_apr/1234"
    SCORING_HPARAM_FILE="hparams/scoring/${MODEL_TYPE}/train_${MODEL_TYPE}_so762${SCORING_TYPE}_scoring.yaml"

    [ -d $SCORING_MODEL_DIR ] && rm -r $SCORING_MODEL_DIR && echo "Removed existing $SCORING_MODEL_DIR directory.";

    CUDA_VISIBLE_DEVICES=1 python3 train.py $SCORING_HPARAM_FILE \
        --data_folder=$PREP_DATA_FOLDER \
        --batch_size=4 \
        --pretrained_model_folder=$PRETRAINED_MODEL_DIR \
        --use_augmentation=True \
        --round_scores=False \
        --exp_folder=$SCORING_MODEL_DIR \
        --exp_metadata_file=$EXP_METADATA_FILE \
        --results_file=$PREP_SCORING_RESULTS_FILE\
        --epoch_results_dir=$EPOCH_RESULTS_DIR \
        --params_dir=$PARAMS_DIR \
        --exp_description="Step 1 then 3 directly (aug, no round)";
fi