#!/bin/bash
bliss -cp ~/bliss/case_studies/dc2_mdt/mdt_config -cn mdt_train_config
bliss -cp ~/bliss/case_studies/dc2_mdt/mdt_config -cn simple_net_train_config
bliss -cp ~/bliss/case_studies/dc2_mdt/mdt_config -cn simple_net_speed_train_config
bliss -cp ~/bliss/case_studies/dc2_mdt/mdt_config -cn simple_ar_net_train_config
bliss -cp ~/bliss/case_studies/dc2_mdt/mdt_config -cn simple_cond_true_net_train_config
# run inference
nohup jupyter nbconvert --execute --to html case_studies/dc2_mdt/inference.ipynb > infer_notebook.out 2>&1 &