#!/usr/bin/env bash
EXPERIMENT=$1
GPU=$2
export CUDA_VISIBLE_DEVICES=$GPU
echo "Starting $EXPERIMENT..."
./main.py mode=train training=$EXPERIMENT training.save_top_k=1 training.version=$EXPERIMENT
