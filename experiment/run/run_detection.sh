#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="3"

./bin/run_detection_train.py --seed 19 --ds-seed 23 --train-file ../data/datasets/train_ds_23.npz --val-file ../data/datasets/val_ds_23.npz
