#!/bin/bash
jupyter nbconvert --execute generate_catalog.ipynb --to html
mkdir merged_catalog
cp ./generate_catalog_output/merged_catalog_with_flux_over_50.pkl ./merged_catalog/
nohup bliss -cp ~/bliss/case_studies/dc2_cataloging -cn train_config > DC2_exp.out 2>&1 &
jupyter nbconvert --execute pre_exp.ipynb --to html
jupyter nbconvert --execute bootstrap_testing_full_image.ipynb --to html
