#!/bin/bash
nohup bliss -cp ~/bliss/case_studies/dc2_cataloging -cn train_config > DC2_exp.out 2>&1 &
jupyter nbconvert --execute cataloging_exp.ipynb --to html
