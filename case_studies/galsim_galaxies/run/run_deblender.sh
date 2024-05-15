#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="5"

echo >> log.txt
cmd="./bin/run_deblender_train.py -o -s 44 -t "15_44""
echo $cmd >> log.txt
eval $cmd
