#!/bin/bash

#SBATCH --mail-user=regier@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --account=regier0
#SBATCH --partition=spgpu

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a40:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=32GB

## wall time hours:minutes:seconds
#SBATCH --time=16:00:00

echo "running on: $SLURM_JOB_NODELIST"

module load python/3.10.4
module load poetry

export NUMEXPR_MAX_THREADS=8

echo "running bliss"
poetry -C $HOME/bliss run bliss \
  -cp $HOME/bliss/case_studies/spatial_tiling \
  -cn m2_config \
  cached_simulator.cached_data_path=/scratch/regier_root/regier0/regier/m2_full_cat \
  cached_simulator.batch_size=64 \
  cached_simulator.num_workers=16 \
  train.matmul_precision=medium \
  train.trainer.precision=32-true \
  train.trainer.logger.name=batchsize64
