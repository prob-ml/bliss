#!/bin/bash
jupyter nbconvert --execute DC2_generate_catalog.ipynb --to html
mkdir merged_catalog
cp ./DC2_generate_catalog_output/merged_catalog_with_flux_over_100.pkl ./merged_catalog/
nohup bliss -cp ~/bliss/case_studies/dc2 -cn full_train_config > DC2_psf_aug_asinh.out 2>&1 &
jupyter nbconvert --execute DC2_exp.ipynb --to html