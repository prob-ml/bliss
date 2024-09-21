#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="2"

echo >> log.txt
cmd="./bin/run_autoencoder_train.py -s 42 --train-file ../data/datasets/train_ae_ds_42_20240920164841.pt --val-file ../data/datasets/val_ae_ds_42_20240920164841.pt"
echo $cmd >> log.txt
eval $cmd

echo >> log.txt
cmd="./bin/run_autoencoder_train.py -s 43 --train-file ../data/datasets/train_ae_ds_42_20240920164841.pt --val-file ../data/datasets/val_ae_ds_42_20240920164841.pt"
echo $cmd >> log.txt
eval $cmd

echo >> log.txt
cmd="./bin/run_autoencoder_train.py -s 44 --train-file ../data/datasets/train_ae_ds_42_20240920164841.pt --val-file ../data/datasets/val_ae_ds_42_20240920164841.pt"
echo $cmd >> log.txt
eval $cmd
