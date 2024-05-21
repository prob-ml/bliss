#!/bin/bash

# train model
export NUM_EPOCH=5000                   #
export RESUME_PATH=""                   # Resume path
export NICK=""                          # Name for memorize
export OUT_DIR="${HOME}/bliss/case_studies/redshift/training_runs/"
# export TRAIN_PATH="${HOME}/bliss/data/redshift/dc2/desc_dc2_run2.2i_dr6_truth_nona_train_set_small.pkl"
# export VAL_PATH="${HOME}/bliss/data/redshift/dc2/desc_dc2_run2.2i_dr6_truth_nona_val_set_small.pkl"
export TRAIN_PATH="${HOME}/bliss/data/redshift/dc2/desc_dc2_run2.2i_dr6_truth_nona_train.pkl"
export VAL_PATH="${HOME}/bliss/data/redshift/dc2/desc_dc2_run2.2i_dr6_truth_nona_val.pkl"
echo "start training!"

if [ -z "$RESUME_PATH" ]; then
    python "$HOME/bliss/case_studies/redshift/train_rs_light.py" \
    --epoch="$NUM_EPOCH" \
    --nick="$NICK" \
    --outdir="$OUT_DIR" \
    --train_path="$TRAIN_PATH" \
    --val_path="$VAL_PATH"
else
    python "$HOME/bliss/case_studies/redshift/train_rs_light.py" \
    --resume="$RESUME_PATH" \
    --epoch="$NUM_EPOCH" \
    --nick="$NICK" \
    --outdir="$OUT_DIR" \
    --train_path="$TRAIN_PATH" \
    --val_path="$VAL_PATH"
fi

echo "finish training!"
