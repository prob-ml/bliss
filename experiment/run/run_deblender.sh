#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="6"

echo >> log.txt
cmd="./bin/run_deblender_train.py -o -s 42 -t "17_42""
echo $cmd >> log.txt
eval $cmd
