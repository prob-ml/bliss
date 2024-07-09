#!/bin/bash
DIRNAME="DC2_redshift_only_bin_allmetrics"

# Create the directory if it does not already exist
if [ ! -d "/home/qiaozhih/bliss/case_studies/redshift/redshift_from_img/DC2_redshift_training/$DIRNAME" ]; then
  mkdir -p "/home/qiaozhih/bliss/case_studies/redshift/redshift_from_img/DC2_redshift_training/$DIRNAME"
else
  echo "Directory already exists: /home/qiaozhih/bliss/case_studies/redshift/redshift_from_img/DC2_redshift_training/$DIRNAME"
fi

nohup bliss -cp ~/bliss/case_studies/redshift/redshift_from_img -cn full_train_config_redshift > "/home/qiaozhih/bliss/case_studies/redshift/redshift_from_img/DC2_redshift_training/$DIRNAME/DC2_psf_aug_asinh.out" 2>&1 &
