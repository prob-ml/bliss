#!/bin/bash

# Runs the experiments for the BLISS multiband paper.
export cached_data_path="/data/scratch/regier/2percent"
if [ "$#" -gt 0 ]; then
    cached_data_path="$1"
fi
echo $cached_data_path

# train single-band model
bliss -cp ~/bliss/case_studies/multiband -cn config mode=train \
    train.trainer.logger.name=MULTIBAND \
    train.trainer.logger.version=r_band \
    encoder.image_normalizer.bands=[2] \
    cached_simulator.cached_data_path=$cached_data_path

# train three-band model
bliss -cp ~/bliss/case_studies/multiband -cn config mode=train \
    train.trainer.logger.name=MULTIBAND \
    train.trainer.logger.version=gri_band \
    encoder.image_normalizer.bands=[1,2,3] \
    cached_simulator.cached_data_path=$cached_data_path

# train five-band model
bliss -cp ~/bliss/case_studies/multiband -cn config mode=train \
    train.trainer.logger.name=MULTIBAND \
    train.trainer.logger.version=ugriz_band \
    encoder.image_normalizer.bands=[0,1,2,3,4] \
    cached_simulator.cached_data_path=$cached_data_path

# run the evaluation notebook(s) to generate figures
jupyter nbconvert --execute multiband_exp.ipynb --to html
jupyter nbconvert --execute alignment.ipynb --to html
jupyter nbconvert --execute color_model.ipynb --to html
