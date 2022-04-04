#!/usr/bin/env bash
EXPERIMENT=$1
echo "Starting $EXPERIMENT..."
./main.py mode=train training=$EXPERIMENT training.save_top_k=1 training.experiment=$EXPERIMENT
