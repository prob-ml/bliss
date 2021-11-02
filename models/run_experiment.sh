#!/usr/bin/env bash
OUTPUT_DIR=$1
EXPERIMENT=$2

echo "Started $EXPERIMENT..."
bliss +experiment=$EXPERIMENT training.save_top_k=1 paths.output=$OUTPUT_DIR training.version=$EXPERIMENT
CKPT=`find $OUTPUT_DIR/default/$EXPERIMENT/checkpoints | tail -n 1` 
cp $CKPT ./models/$EXPERIMENT.ckpt
echo $EXPERIMENT >> ./models/results
basename $CKPT >> ./models/results
echo >> ./models/results