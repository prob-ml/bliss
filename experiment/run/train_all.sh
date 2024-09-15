#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="5"


echo >> log.txt
cmd="./bin/run_detection_train.py -s 42 -t "42_1""
echo $cmd >> log.txt
eval $cmd


echo >> log.txt
cmd="./bin/run_binary_train.py -s 42 -t "42_1""
echo $cmd >> log.txt
eval $cmd



echo >> log.txt
cmd="./bin/run_deblender_train.py -s 42 -t "42_1""
echo $cmd >> log.txt
eval $cmd
