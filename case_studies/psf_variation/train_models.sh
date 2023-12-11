#!/bin/bash

# Runs the experiments for the BLISS PSF variation paper.

# train base model
bliss -cp ~/bliss/case_studies/psf_variation -cn config mode=train \
    train.trainer.logger.name=PSF_MODELS \
    train.trainer.logger.version=base_model \
    encoder.image_normalizer.use_deconv_channel=False \
    encoder.image_normalizer.concat_psf_params=False \
    cached_simulator.cached_data_path=/data/scratch/aakash/train_single_94-1-12

# train psf-unaware model
bliss -cp ~/bliss/case_studies/psf_variation -cn config mode=train \
    train.trainer.logger.name=PSF_MODELS \
    train.trainer.logger.version=multi_field_psf_unaware \
    encoder.image_normalizer.use_deconv_channel=False \
    encoder.image_normalizer.concat_psf_params=False \
    cached_simulator.cached_data_path=/data/scratch/aakash/train_multi_field

# train deconv-only model
bliss -cp ~/bliss/case_studies/psf_variation -cn config mode=train \
    train.trainer.logger.name=PSF_MODELS \
    train.trainer.logger.version=multi_field_deconv_only \
    encoder.image_normalizer.use_deconv_channel=True \
    encoder.image_normalizer.concat_psf_params=False \
    cached_simulator.cached_data_path=/data/scratch/aakash/train_multi_field

# train concat params only model
bliss -cp ~/bliss/case_studies/psf_variation -cn config mode=train \
    train.trainer.logger.name=PSF_MODELS \
    train.trainer.logger.version=multi_field_psf_params_only \
    encoder.image_normalizer.use_deconv_channel=False \
    encoder.image_normalizer.concat_psf_params=True \
    cached_simulator.cached_data_path=/data/scratch/aakash/train_multi_field

# train both params only model
bliss -cp ~/bliss/case_studies/psf_variation -cn config mode=train \
    train.trainer.logger.name=PSF_MODELS \
    train.trainer.logger.version=multi_field_deconv_and_psf_param \
    encoder.image_normalizer.use_deconv_channel=True \
    encoder.image_normalizer.concat_psf_params=True \
    cached_simulator.cached_data_path=/data/scratch/aakash/train_multi_field


# run the evaluation notebook
jupyter nbconvert --execute evaluate_models.ipynb --to html
