#!/bin/bash

# This script runs the experiments from the BLISS spatially-variant PSF paper.

# Generate new data
bash ~/bliss/scripts/generate_data_in_parallel.sh -n 32 -cp ~/bliss/case_studies/psf_variation/conf -cn psf_aware
bash ~/bliss/scripts/generate_data_in_parallel.sh -n 32 -cp ~/bliss/case_studies/psf_variation/conf -cn single_field

# Train single-field model
bliss -cp ~/bliss/case_studies/psf_variation/conf -cn single_field mode=train

# Train psf-unaware model
bliss -cp ~/bliss/case_studies/psf_variation/conf -cn psf_unaware mode=train

# Train psf-aware model
bliss -cp ~/bliss/case_studies/psf_variation/conf -cn psf_aware mode=train

# Run evaluation and generate figures
python evaluate_models.py --run_eval --plot_eval --run_calibration --plot_calibration --data_path=/data/scratch/aakash/multi_field
python evaluate_models.py --run_eval --plot_eval --run_calibration --plot_calibration --data_path=/data/scratch/aakash/single_field