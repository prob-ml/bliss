#!/bin/bash

#SBATCH --mail-user=pduan@umich.edu
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

###   Load software modules

module load python/3.10.4 poetry/1.6.1 cuda/12.1.1 cudnn/12.1-v8.9.0
module list

####  Commands your job should run follow this line

echo "running on: $SLURM_JOB_NODELIST"

EXP_POSTFIX=$1
echo "experiment postfix is: $EXP_POSTFIX"

echo "go to bliss"
cd ~/bliss/

GO_TO_BANCH=$2
echo "go to branch ${GO_TO_BRANCH}"
git pull
eval "git checkout ${GO_TO_BRANCH}"

echo "create output folder if not exists"
mkdir -p ~/bliss_output

echo "copy dc2local from remote server to /tmp_data/"
mkdir -p /tmp_data/pduan/dc2local/
rsync -av --info=progress2 \
 -e "ssh -i ~/.ssh/jeffbox_great_lake_side_rsa" \
 pduan@deeplearning-01.stat.lsa.umich.edu:/data/scratch/dc2local/dc2_split_results.tar \
 /tmp_data/pduan/dc2local/
pv /tmp_data/pduan/dc2local/dc2_split_results.tar | tar -xf - -C /tmp_data/pduan/dc2local/
ln -sf /tmp_data/pduan/dc2local/dc2_split_results ~/bliss_output/dc2_split_results

echo "run the bliss"
poetry run bliss \
 -cp ~/bliss/case_studies/dc2_cataloging \
 -cn train_config_great_lake \
  train.trainer.logger.version=great_lake_exp_${EXP_POSTFIX}

echo "clean directory"
rm -r /tmp_data/pduan/dc2local/dc2_split_results
