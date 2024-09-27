#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="2"

# echo >> log.txt
# cmd="./bin/run_deblender_train.py --seed 40 --train-file ../data/datasets/train_ds_42_20240925144602.pt  --val-file ../data/datasets/val_ds_42_20240925144602.pt --ae-model-path ../models/autoencoder_42.pt  --lr 1e-4"
# echo $cmd >> log.txt
# eval $cmd

# echo >> log.txt
# cmd="./bin/run_deblender_train.py --seed 42 --train-file ../data/datasets/train_ds_42_20240925144602.pt --ae-model-path ../models/autoencoder_42.pt  --val-file ../data/datasets/val_ds_42_20240925144602.pt --lr 1e-3"
# echo $cmd >> log.txt
# eval $cmd

echo >> log.txt
cmd="./bin/run_deblender_train.py --seed 41 --train-file ../data/datasets/train_ds_42_20240925144602.pt --ae-model-path ../models/autoencoder_42.pt  --val-file ../data/datasets/val_ds_42_20240925144602.pt --lr 1e-5"
echo $cmd >> log.txt
eval $cmd
