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

echo "loading modules"
module load python/3.10.4
module load poetry

echo "setting environment variables"
export NUMEXPR_MAX_THREADS=8

echo "copying data from remote server to /tmp_data"
mkdir -p /tmp_data/regier/
rsync -av --info=progress2 \
 deeplearning-01.stat.lsa.umich.edu:/data/scratch/regier/m2_full_cat \
 /tmp_data/regier/

echo "running bliss"
poetry -C $HOME/bliss run bliss \
  -cp $HOME/bliss/case_studies/spatial_tiling \
  -cn m2_config \
  cached_simulator.cached_data_path=/tmp_data/regier/m2_full_cat \
  cached_simulator.batch_size=64 \
  cached_simulator.num_workers=4 \
  train.matmul_precision=medium \
  train.trainer.precision=16-mixed \
  train.trainer.logger.name=test
