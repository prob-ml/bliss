#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="5"

echo >> log.txt
cmd="./bin/run_autoencoder_train.py -o -s 42 -t "2""
echo $cmd >> log.txt
eval $cmd
