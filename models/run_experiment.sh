#!/usr/bin/env bash
set -e
OUTPUT_DIR=$1
EXPERIMENT=$2

NEXT_AVAILABLE_GPU=`./get_available_gpu.sh`
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:=$NEXT_AVAILABLE_GPU}
export CUDA_VISIBLE_DEVICES


echo "Starting $EXPERIMENT..."
echo "CUDA_VISIBLE_DEVICES is $CUDA_VISIBLE_DEVICES"
EXP_DIR=$OUTPUT_DIR/default/$EXPERIMENT
rm -rf $EXP_DIR
bliss +experiment=$EXPERIMENT training.save_top_k=1 paths.output=`realpath $OUTPUT_DIR` training.version=$EXPERIMENT
CKPT=`find $EXP_DIR/checkpoints | tail -n 1`
cp $CKPT ./$EXPERIMENT.ckpt
RESULTS=${EXPERIMENT}_results
echo $EXPERIMENT > ./${RESULTS}
basename $CKPT >> ./${RESULTS}
echo >> ./${RESULTS}
