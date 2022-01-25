#!/usr/bin/env bash
OUTPUT_DIR=$1
EXPERIMENT=$2


echo "Starting $EXPERIMENT..."
EXP_DIR=$OUTPUT_DIR/default/$EXPERIMENT
rm -rf $EXP_DIR
bliss mode=train training=$EXPERIMENT training.save_top_k=1 paths.output=`realpath $OUTPUT_DIR` training.version=$EXPERIMENT
CKPT=`find $EXP_DIR/checkpoints | tail -n 1`
cp $CKPT ./$EXPERIMENT.ckpt
RESULTS=${EXPERIMENT}_results
echo $EXPERIMENT > ./${RESULTS}
basename $CKPT >> ./${RESULTS}
echo >> ./${RESULTS}
