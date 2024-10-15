#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="1"

./bin/run_detection_train.py --seed 43 --ds-seed 42 --train-file ../data/datasets/train_ds_42.npz --val-file ../data/datasets/val_ds_42.npz
