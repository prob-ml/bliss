#!/bin/bash
nohup bliss -cp ~/bliss/case_studies/dc2_cataloging -cn train_config > DC2_exp.out 2>&1 &
jupyter nbconvert --execute cataloging_exp.ipynb --to html

bliss -cp ~/bliss/case_studies/dc2_cataloging -cn train_config \
    train.trainer.logger.version=new_baseline \
    train.trainer.devices=[6]
nohup jupyter nbconvert --execute blendedness_exp.ipynb --to html > blendedness_exp.out 2>&1 &