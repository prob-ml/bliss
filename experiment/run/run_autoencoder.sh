#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="3"
export SEED="42"

./run_autoencoder_train.py --seed $SEED --ds-seed $SEED --train-file ../data/datasets/train_ae_ds_${SEED}.npz --val-file ../data/datasets/val_ae_ds_${SEED}.npz
