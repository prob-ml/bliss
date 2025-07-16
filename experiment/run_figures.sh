#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="3"

# ./get_figures.py "detection" "test2" --test-file-blends data/datasets/test_ds_23.npz --detection-fpath models/detection_23_23.pt --seed 23

# ./get_figures.py "toy" "test2" --detection-fpath models/detection_23_23.pt --ae-fpath models/autoencoder_42_42.pt --deblend-fpath models/deblender_23_22.pt

# ./get_figures.py "toy_samples" "test2" --detection-fpath models/detection_23_23.pt --ae-fpath models/autoencoder_42_42.pt --deblend-fpath models/deblender_23_22.pt --overwrite

# ./get_figures.py "deblend" "test2" --ae-fpath models/autoencoder_42_42.pt --deblend-fpath models/deblender_23_22.pt --test-file-blends data/datasets/test_ds_23.npz

./get_figures.py "scores" "test2" --ae-fpath models/autoencoder_42_42.pt --deblend-fpath models/deblender_23_22.pt  --detection-fpath models/detection_23_23.pt --test-file-blends data/datasets/test_ds_251.npz --overwrite
#
# ./get_figures.py "binary" "test2" --binary-fpath models/binary_23_23.pt --test-file-blends data/datasets/test_ds_23.npz

# ./get_figures.py "samples" "test2" --ae-fpath models/autoencoder_42_42.pt --deblend-fpath models/deblender_23_22.pt --detection-fpath models/detection_23_23.pt --test-file-blends data/datasets/test_ds_25.npz  --overwrite
