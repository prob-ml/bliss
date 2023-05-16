#!/bin/bash

# This script repeats the experiment $N_REPEATS times and aggregates the results.
EXPERIMENT_NAME=$1
N_REPEATS=${2:-5}
BLISS_HOME=${BLISS_HOME:-$HOME/727-optimize-encoder}
CONFIG_PATH=${3:-$BLISS_HOME/case_studies/optimize_encoder_727}
RANDOM_SEED=${4:-42}

if [ -z "$EXPERIMENT_NAME" ]; then
    echo "Please provide an experiment name."
    echo "Usage: repeat_and_aggregate.sh <experiment_name> [<n_repeats (default: 5)>] [<config_path (default: $HOME/bliss/case_studies/optimize_encoder_727)>]"
    exit 1
fi

# Experiment: pretraining with crowded starfield
# Ensure pretrained weights `sdss-pretrained-0.02.pt` are present in the pretrained_weights folder `data/pretrained_models/`
pretrained_weights_file=$BLISS_HOME/data/pretrained_models/sdss-pretrained-0.02.pt
if [ ! -f $pretrained_weights_file ]; then
    echo "Pretrained weights $pretrained_weights_file not found. Please train the model on the crowded starfield (mean_sources=0.02) dataset first."
    exit 1
fi

# Repeat the experiment $N_REPEATS times
# Create the logs directory if it doesn't exist
mkdir -p $CONFIG_PATH/logs/$EXPERIMENT_NAME/
RANDOM=$RANDOM_SEED
pids=()
for ((i=0; i<$N_REPEATS; i++));
do
    bliss -cp $CONFIG_PATH 'mode=train' "training.seed=$RANDOM" |& tee $CONFIG_PATH/logs/$EXPERIMENT_NAME/rep-$i.log &
    pids+=($!)
done
for pid in ${pids[*]}; do
    wait $pid
done

# Aggregate the results
# TODO
