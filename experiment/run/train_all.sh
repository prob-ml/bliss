#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0"
export SEED="44" # for training models
export DS_SEED="42" # keep test data set fixed, possibly re run models to estimate model bias.
export AE_VERSION="31" # make sure no overlap with existing version number

# ../get_single_galaxies_datasets.py --seed $SEED

# ../get_blends_datasets.py --seed $SEED

./bin/run_autoencoder_train.py --seed $SEED --ds-seed $DS_SEED --train-file ../data/datasets/train_ae_ds_${DS_SEED}.npz --val-file ../data/datasets/val_ae_ds_${DS_SEED}.npz --version ${AE_VERSION}

../get_model_from_checkpoint.py -m "autoencoder" --seed $SEED --ds-seed $DS_SEED --checkpoint-dir ../data/out/autoencoder/version_${AE_VERSION}/checkpoints

./bin/run_detection_train.py --seed $SEED --ds-seed $DS_SEED --train-file ../data/datasets/train_ds_${DS_SEED}.npz --val-file ../data/datasets/val_ds_${DS_SEED}.npz

./bin/run_binary_train.py --seed $SEED --ds-seed $DS_SEED --train-file ../data/datasets/train_ds_${DS_SEED}.npz --val-file ../data/datasets/val_ds_${DS_SEED}.npz

./bin/run_deblender_train.py --seed $SEED --ds-seed $DS_SEED --ae-model-path ../models/autoencoder_${DS_SEED}_${SEED}.pt --train-file ../data/datasets/train_ds_${DS_SEED}.npz --val-file ../data/datasets/val_ds_${DS_SEED}.npz
