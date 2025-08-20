#!/bin/bash

#SBATCH --mail-user=pduan@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --account=regier0
#SBATCH --partition=spgpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=a40:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32GB

## wall time hours:minutes:seconds
#SBATCH --time=168:00:00

module load python/3.12.1 poetry/1.6.1 cuda/12.8.1 cudnn/12.8-v9.10.0
module list

function print_info() {
  echo -e "\033[1;34m[INFO]\033[0m $1"
}

function print_warn() {
  echo -e "\033[1;33m[WARNING]\033[0m $1"
}

function print_error() {
  echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_info "running on: $SLURM_JOB_NODELIST"

mkdir -p /tmpssd/pduan/
mkdir -p logs

print_info "prepare data"
rsync -a \
 /nfs/turbo/lsa-regier/scratch/pduan/dc2_cached_data \
 /scratch/regier_root/regier0/pduan/
rsync -a \
   /scratch/regier_root/regier0/pduan/dc2_cached_data \
   /tmpssd/pduan/

print_info "go to bliss"
cd ~/bliss/

function run_exp() {
    task_name=$1
    exp_name=$2
    cuda_devices=$3
    config_tag=$4
    cfg_name="${task_name}_train_config_gl"

    {
        set -e
        print_info "training starts"
        srun poetry run bliss \
            -cp ~/bliss/case_studies/dc2_mdt/mdt_config \
            -cn ${cfg_name} \
            train.trainer.logger.version=${exp_name} \
            train.trainer.devices=${cuda_devices} \
            ${config_tag}
        print_info "training ends"
    } > logs/task_${task_name}_${exp_name}.log 2>&1 &
}


task_name=$1
config_tag=$2
print_info "run ${task_name}"
exp_name="great_lake_exp_$(date +"%m-%d-%H-%M-%S")"
cuda_devices=4
run_exp "${task_name}" "${exp_name}" "${cuda_devices}" "${config_tag}"


print_info "wait to complete"
wait

print_info "done"
