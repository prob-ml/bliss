#!/bin/bash
jupyter nbconvert --execute DC2_galaxy_psf_params.ipynb --to html
cp merged_catalog_with_flux_over_100.pkl ../../data/dc2/merged_catalog/
nohup bliss -cp ~/bliss/case_studies/dc2 -cn full_train_config > DC2_psf_aug_asinh.out 2>&1 &
jupyter nbconvert --execute DC2_exp.ipynb --to html
