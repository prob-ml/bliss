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

module load singularity/4.1.2 cuda/12.1.1 cudnn/12.1-v8.9.0
module list

####  Commands your job should run follow this line

echo "running on: $SLURM_JOB_NODELIST"

EXP_POSTFIX=$1
echo "experiment postfix is: $EXP_POSTFIX"

EXP_CUR_PATH=`pwd`
echo "current directory is: ${EXP_CUR_PATH}"

echo "create output folder if not exists"
mkdir -p ./output
echo "create overlay image folder if not exists"
mkdir -p ./overlay_imgs

echo "copy dc2local from remote server to /tmp_data/"
mkdir -p /tmp_data/pduan/dc2local/
rsync -av --info=progress2 \
 -e "ssh -i ~/.ssh/jeffbox_great_lake_side_rsa" \
 pduan@deeplearning-01.stat.lsa.umich.edu:/data/scratch/dc2local/dc2_split_results.tar \
 /tmp_data/pduan/dc2local/
pv /tmp_data/pduan/dc2local/dc2_split_results.tar | tar -xf - -C /tmp_data/pduan/dc2local/
ln -sf /tmp_data/pduan/dc2local/dc2_split_results /home/pduan/bliss_related/output/dc2_split_results

echo "create overlay image"
singularity overlay create --sparse --size 5120 "./overlay_imgs/bliss_machine_overlay_${EXP_POSTFIX}.img"

echo "run the bliss"
srun singularity run --nv \
 --bind /tmp_data/:/tmp_data/ \
 --overlay "./overlay_imgs/bliss_machine_overlay_${EXP_POSTFIX}.img" \
 bliss_machine.sif \
 yd/test_container \
 "bliss -cp /home/container_files/bliss/case_studies/dc2_cataloging -cn full_train_config_great_lake train.trainer.logger.version=great_lake_exp_${EXP_POSTFIX}"

echo "clean directory"
rm -r /tmp_data/pduan/dc2local/dc2_split_results
