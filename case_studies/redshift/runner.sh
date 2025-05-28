#!/bin/bash

set -e

# Check that BLISS_HOME is set
if [ -z "$BLISS_HOME" ]; then
  echo "Error: BLISS_HOME environment variable is not set. Please export BLISS_HOME before running this script."
  exit 1
fi

if [ -z "$BLISS_DATA_DIR" ]; then
  echo "Error: BLISS_DATA_DIR environment variable is not set. Please export BLISS_HOME before running this script."
  exit 1
fi

# Change to BLISS_HOME
cd "$BLISS_HOME"

# Environment setup
export OUT_DIR="$BLISS_DATA_DIR/training_logs"
export OMP_NUM_THREADS="16"
export MKL_NUM_THREADS="16"
export NUMEXPR_NUM_THREADS="16"
export CUDA_VISIBLE_DEVICES=3

timestamp=$(date "+%Y-%m-%d-%H-%M-%S")

# Usage function
usage() {
  echo "Usage: $0 [--discrete] [--continuous] [--bspline] [--mdn] [--all]"
  exit 1
}

# Argument parsing
DISCRETE=false
CONTINUOUS=false
BSPLINE=false
MDN=false

if [ "$#" -eq 0 ]; then
  usage
fi

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --discrete) DISCRETE=true ;;
    --continuous) CONTINUOUS=true ;;
    --bspline) BSPLINE=true ;;
    --mdn) MDN=true ;;
    --all)
      DISCRETE=true
      CONTINUOUS=true
      BSPLINE=true
      MDN=true
      ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
  shift
done

run_bliss() {
  local name=$1
  local config=$2
  local dirname="$OUT_DIR/${name}_$timestamp"

  mkdir -p "$dirname"
  echo "BLISS training logs/checkpoints will be saved to $dirname"

  nohup python bliss/main.py \
    -cp ~/bliss/case_studies/redshift \
    -cn "$config" timestamp="$timestamp" \
    paths.root="$BLISS_HOME" \
    paths.data_dir="$BLISS_DATA_DIR" \
    > "$dirname/output.out" 2>&1 &
}

# Run selected configurations
$DISCRETE   && run_bliss "discrete"   "redshift_discrete"
$CONTINUOUS && run_bliss "continuous" "redshift_continuous"
$BSPLINE    && run_bliss "bspline"    "redshift_bspline"
$MDN        && run_bliss "mdn"        "redshift_mdn"
