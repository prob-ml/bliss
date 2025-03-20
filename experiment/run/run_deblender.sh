#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="5"

./bin/run_deblender_train.py --seed 23 --ds-seed 23 --ae-model-path ../models/autoencoder_42_42.pt --train-file ../data/datasets/train_ds_deblend_23.npz --val-file ../data/datasets/val_ds_deblend_23.npz
