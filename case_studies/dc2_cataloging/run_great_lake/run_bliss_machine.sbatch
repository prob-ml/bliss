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
echo "running in: $(pwd)"

EXP_POSTFIX=$1
echo "postfix is: $EXP_POSTFIX"

singularity overlay create --sparse --size 5120 "bliss_machine_overlay_${EXP_POSTFIX}.img"

srun singularity run --nv --overlay "bliss_machine_overlay_${EXP_POSTFIX}.img" bliss_machine.sif \
 yd/test_container \
 "bliss -cp /home/container_files/bliss/case_studies/dc2_cataloging/run_great_lake/configs -cn full_train_config_great_lake_$EXP_POSTFIX"
