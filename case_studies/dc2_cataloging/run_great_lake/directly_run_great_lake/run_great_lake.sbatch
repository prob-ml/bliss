#!/bin/bash

#SBATCH --mail-user=pduan@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --account=regier0
#SBATCH --partition=spgpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a40:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64GB

## wall time hours:minutes:seconds
#SBATCH --time=24:00:00

###   Load software modules

module load python/3.10.4 poetry/1.6.1 cuda/12.1.1 cudnn/12.1-v8.9.0
module list

####  Commands your job should run follow this line

echo "running on: $SLURM_JOB_NODELIST"

EXP_POSTFIX=$1
echo "experiment postfix is: ${EXP_POSTFIX}"

USER_SEED=$2
echo "seed is ${USER_SEED}"

GPUS_WE_USE=$3
echo "use ${GPUS_WE_USE} GPUs"

echo "prepare data"
mkdir -p /tmpssd/pduan/dc2local
rsync -av --info=progress2 \
 /scratch/regier_root/regier0/pduan/dc2local/dc2_cached_data \
 /tmpssd/pduan/dc2local/

echo "go to bliss"
cd ~/bliss/

echo "run the bliss"
poetry run bliss \
 -cp ~/bliss/case_studies/dc2_cataloging \
 -cn train_config_great_lake \
  train.trainer.logger.version=great_lake_exp_${EXP_POSTFIX} \
  train.seed=${USER_SEED} \
  train.trainer.devices=${GPUS_WE_USE}
