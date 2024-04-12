#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="6"

for i in {43..44};
do
    echo >> log.txt
    cmd="./bin/run_detection_train_script.py -o -s $i --star-density 0 -t "11_${i}""
    echo $cmd >> log.txt
    eval $cmd
done
