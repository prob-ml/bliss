#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="6"

echo >> log.txt
cmd="./bin/run_binary_train.py -s 44 -t "12_43""
echo $cmd >> log.txt
eval $cmd
