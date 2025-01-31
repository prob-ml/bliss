#!/bin/bash

export OUT_DIR="/data/scratch/declan/redshift/dc2"

# Produce data artifacts 
echo "producing data artifacts for BLISS and RAIL from DC2"
python data_preprocessing/data_generation.py

# Run BLISS (discrete variational distribution)
DIRNAME="BLISS_DC2_redshift_discrete_results"

if [ ! -d "$OUT_DIR/$DIRNAME" ]; then
  mkdir -p "$OUT_DIR/$DIRNAME"
  echo "BLISS training logs/checkpoints will be saved to $OUT_DIR/$DIRNAME"
else
  echo "BLISS training logs/checkpoints will be saved to $OUT_DIR/$DIRNAME"
fi

nohup bliss -cp ~/bliss/case_studies/redshift/redshift_from_img -cn full_train_config_redshift > "$OUT_DIR/$DIRNAME/output.out" 2>&1 &

# Run BLISS (continuous variational distribution)
if [ ! -d "$OUT_DIR/$DIRNAME" ]; then
  mkdir -p "$OUT_DIR/$DIRNAME"
  echo "BLISS training logs/checkpoints will be saved to $OUT_DIR/$DIRNAME"
else
  echo "BLISS training logs/checkpoints will be saved to $OUT_DIR/$DIRNAME"
fi

nohup bliss -cp ~/bliss/case_studies/redshift/redshift_from_img -cn full_train_config_redshift > "$OUT_DIR/$DIRNAME/output.out" 2>&1 &

# Run RAIL
# TODO

# Create plots
# TODO
