#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="5"

for i in {43..45};
do
    echo >> log.txt
    cmd="./bin/run_detection_train_script.py -o -s $i -t "12_${i}""
    echo $cmd >> log.txt
    eval $cmd
done
