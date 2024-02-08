#!/bin/bash

# run the galaxy parameters and psf parameters extration notebook
jupyter nbconvert --execute DC2_galaxy_psf_params.ipynb --to html

# run the deconvolution image generation notebook
jupyter nbconvert --execute DC2_deconv.ipynb --to html

# train base model
bliss -cp ~/bliss/case_studies/bliss_dc2 -cn config mode=train \
    train.trainer.logger.name=DC2_MODELS \
    train.trainer.logger.version=base_model \
    encoder.image_normalizer.use_deconv_channel=False \
    encoder.image_normalizer.concat_psf_params=False \
    encoder.do_data_augmentation = False \
    train.pretrained_weights = None

# train both PSF params and deconvolution channel model
bliss -cp ~/bliss/case_studies/bliss_dc2 -cn config mode=train \
    train.trainer.logger.name=DC2_MODELS \
    train.trainer.logger.version=dc2_psf_deconv \
    encoder.image_normalizer.use_deconv_channel=True \
    encoder.image_normalizer.concat_psf_params=True \
    encoder.do_data_augmentation = False \
    train.pretrained_weights = None

# train with data augmentation
bliss -cp ~/bliss/case_studies/bliss_dc2 -cn config mode=train \
    train.trainer.logger.name=DC2_MODELS \
    train.trainer.logger.version=dc2_data_aug \
    encoder.image_normalizer.use_deconv_channel=False \
    encoder.image_normalizer.concat_psf_params=False \
    encoder.do_data_augmentation = True \
    train.pretrained_weights = None

# train with data augmentation and psf, deconvolution
bliss -cp ~/bliss/case_studies/bliss_dc2 -cn config mode=train \
    train.trainer.logger.name=DC2_MODELS \
    train.trainer.logger.version=dc2_psf_deconv_da \
    encoder.image_normalizer.use_deconv_channel=True \
    encoder.image_normalizer.concat_psf_params=True \
    encoder.do_data_augmentation = True \
    train.pretrained_weights = None
