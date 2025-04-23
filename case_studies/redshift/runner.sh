#!/bin/bash

export OUT_DIR="/data/scratch/declan/redshift/dc2/training_logs"
export OMP_NUM_THREADS="16"
export MKL_NUM_THREADS="16"
export NUMEXPR_NUM_THREADS="16"
export CUDA_VISIBLE_DEVICES=3

timestamp=$(date "+%Y-%m-%d-%H-%M-%S")

# # Produce data artifacts
# echo "producing data artifacts for BLISS and RAIL from DC2"
# python artifacts/data_generation.py

# Run BLISS (discrete variational distribution)
# DIRNAME="$OUT_DIR/discrete_$timestamp"

# if [ ! -d "$DIRNAME" ]; then
#   mkdir -p "$DIRNAME"
#   echo "BLISS training logs/checkpoints will be saved to $DIRNAME"
# else
#   echo "BLISS training logs/checkpoints will be saved to $DIRNAME"
# fi

# nohup python bliss/main.py -cp ~/bliss/case_studies/redshift -cn redshift_discrete timestamp=$timestamp > "$DIRNAME/output.out" 2>&1 &

# Run BLISS (continuous variational distribution)
# DIRNAME="$OUT_DIR/continuous_$timestamp"

# if [ ! -d "$DIRNAME" ]; then
#   mkdir -p "$DIRNAME"
#   echo "BLISS training logs/checkpoints will be saved to $DIRNAME"
# else
#   echo "BLISS training logs/checkpoints will be saved to $DIRNAME"
# fi

# nohup python bliss/main.py -cp ~/bliss/case_studies/redshift -cn redshift_continuous timestamp=$timestamp > "$DIRNAME/output.out" 2>&1 &

DIRNAME="$OUT_DIR/bspline_$timestamp"

if [ ! -d "$DIRNAME" ]; then
  mkdir -p "$DIRNAME"
  echo "BLISS training logs/checkpoints will be saved to $DIRNAME"
else
  echo "BLISS training logs/checkpoints will be saved to $DIRNAME"
fi

nohup python bliss/main.py -cp ~/bliss/case_studies/redshift -cn redshift_bspline > "$DIRNAME/output.out" 2>&1 &

# DIRNAME="$OUT_DIR/mdn_$timestamp"

# if [ ! -d "$DIRNAME" ]; then
#   mkdir -p "$DIRNAME"
#   echo "BLISS training logs/checkpoints will be saved to $DIRNAME"
# else
#   echo "BLISS training logs/checkpoints will be saved to $DIRNAME"
# fi

# nohup python bliss/main.py -cp ~/bliss/case_studies/redshift -cn redshift_mdn > "$DIRNAME/output.out" 2>&1 &

# # # Run RAIL
# # # TODO

# # # Create plots
# echo "creating plots for BLISS and RAIL from DC2"
# python evaluation/evaluate_cts.py
# python evaluation/evaluate_discrete.py
# python evaluation/plots_bliss.py
