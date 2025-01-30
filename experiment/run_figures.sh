#!/usr/bin/env bash

# ./get_figures.py "toy" "test1" --detection-fpath models/detection_42_42.pt --ae-fpath models/autoencoder_42_42.pt --deblend-fpath models/deblender_42_42.pt

./get_figures.py "deblend" "test1" --ae-fpath models/autoencoder_42_42.pt --deblend-fpath models/deblender_42_42.pt --test-file-blends data/datasets/test_ds_42.npz
