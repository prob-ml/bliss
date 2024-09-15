#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="1"

echo >> log.txt
cmd="./bin/run_autoencoder_train.py -s 42 --lr 1e-5 -t "42_1""
echo $cmd >> log.txt
eval $cmd

echo >> log.txt
cmd="./bin/run_autoencoder_train.py -s 42 --lr 5e-5 -t "42_1""
echo $cmd >> log.txt
eval $cmd

echo >> log.txt
cmd="./bin/run_autoencoder_train.py -s 42 --lr 1e-6 -t "42_1""
echo $cmd >> log.txt
eval $cmd

echo >> log.txt
cmd="./bin/run_autoencoder_train.py -s 42 --lr 5e-6 -t "42_1""
echo $cmd >> log.txt
eval $cmd

echo >> log.txt
cmd="./bin/run_autoencoder_train.py -s 42 --lr 1e-4 -t "42_1""
echo $cmd >> log.txt
eval $cmd

echo >> log.txt
cmd="./bin/run_autoencoder_train.py -s 42 --lr 1e-3 -t "42_1""
echo $cmd >> log.txt
eval $cmd
