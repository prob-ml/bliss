#!/bin/bash

# Download dataset from scratch
echo "start downloading scratch!"
export SAVE_NAME="desc_dc2_run2.2i_dr6_truth_nona"         # User define
export OUT_DIR="$HOME/bliss/data/redshift/dc2/"       # User define
export OUT_PATH="${OUT_DIR}${SAVE_NAME}.pkl"

mkdir -p "$OUT_DIR"

if [ ! -f "$OUT_PATH" ]; then
    echo "start reading dataset, about 1-2hrs"
    python "$HOME/bliss/case_studies/redshift/preprocess_rs.py" \
    --out="$OUT_PATH"
    echo "finish downloading scratch without nan!"
else
    echo "scratch exists!"
fi

# split train val test dataset
export SOURCE=$OUT_PATH
export OUT_NAME="${OUT_DIR}${SAVE_NAME}"
export TRAIN_OUT_PATH="${OUT_NAME}_train.pkl"
export VAL_OUT_PATH="${OUT_NAME}_val.pkl"
export TEST_OUT_PATH="${OUT_NAME}_test.pkl"

if [ ! -f "$TRAIN_OUT_PATH" ] || [ ! -f "$VAL_OUT_PATH" ] || [ ! -f "$TEST_OUT_PATH" ]; then
    echo "start spliting dataset, about a few minutes"
    python "$HOME/bliss/case_studies/redshift/split_rs_train.py" \
        --source="$SOURCE" \
        --outname="$OUT_NAME"
    echo "finish spliting train-val-test!"
else
    echo "train-val-test dataset exist!"
fi
