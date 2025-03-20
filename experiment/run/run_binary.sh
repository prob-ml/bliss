#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="1"

./bin/run_binary_train.py --seed 23 --ds-seed 23 --train-file ../data/datasets/train_ds_23.npz --val-file ../data/datasets/val_ds_23.npz
