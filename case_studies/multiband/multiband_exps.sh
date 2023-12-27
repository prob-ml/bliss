#!/bin/bash

# Runs the experiments for the BLISS multiband paper.

# train single-band model
bliss -cp ~/bliss/case_studies/multiband -cn config mode=train \
    train.trainer.logger.name=MULTIBAND \
    train.trainer.logger.version=r_band \
    encoder.image_normalizer.bands=[2] \
    cached_simulator.cached_data_path=/data/scratch/regier/2percent

# train three-band model
bliss -cp ~/bliss/case_studies/multiband -cn config mode=train \
    train.trainer.logger.name=MULTIBAND \
    train.trainer.logger.version=gri_band \
    encoder.image_normalizer.bands=[1,2,3] \
    cached_simulator.cached_data_path=/data/scratch/regier/2percent

# train five-band model
bliss -cp ~/bliss/case_studies/multiband -cn config mode=train \
    train.trainer.logger.name=MULTIBAND \
    train.trainer.logger.version=ugriz_band \
    encoder.image_normalizer.bands=[0,1,2,3,4] \
    cached_simulator.cached_data_path=/data/scratch/regier/2percent

# run the evaluation notebook
jupyter nbconvert --execute multiband_exp.ipynb --to html
