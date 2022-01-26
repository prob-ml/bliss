#!/usr/bin/env bash
OUTPUT_DIR=$1
EXPERIMENT=$2


echo "Starting $EXPERIMENT..."
EXP_DIR=$OUTPUT_DIR/default/$EXPERIMENT
rm -rf $EXP_DIR
bliss mode=train training=$EXPERIMENT training.save_top_k=1 paths.output=`realpath $OUTPUT_DIR` training.version=$EXPERIMENT
