#!/bin/bash

#SBATCH --mail-user=pduan@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --account=regier0
#SBATCH --partition=spgpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a40:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32GB

## wall time hours:minutes:seconds
#SBATCH --time=48:00:00

module load python/3.10.4 poetry/1.6.1 cuda/12.6.3 cudnn/12.6-v9.6.0
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

task_name=$1
print_info "task_name: ${task_name}"
exp_name=$2
print_info "exp_name: ${exp_name}"
infer_batch_size=$3
print_info "infer_batch_size: ${infer_batch_size}"
infer_total_iters=$4
print_info "infer_total_iters: ${infer_total_iters}"


print_info "running on: $SLURM_JOB_NODELIST"

mkdir -p /tmpssd/pduan/dc2local
mkdir -p /scratch/regier_root/regier0/pduan/posterior_cached_files
mkdir -p logs

print_info "prepare data"
rsync -a \
   /scratch/regier_root/regier0/pduan/dc2local/dc2_cached_data \
   /tmpssd/pduan/dc2local/

print_info "go to bliss"
cd ~/bliss/

function run_exp() {
    task_name=$1
    exp_name=$2
    cuda_device=$3
    cfg_name="${task_name}_train_config_gl"

    {
        set -e
        best_ck_point_path=$(find ~/bliss_output/DC2_mdt_exp/${exp_name}/checkpoints/ -name '*.ckpt' | sort | head -n1)
        best_ck_point=$(basename "${best_ck_point_path}")
        print_info "best checkpoint is [${best_ck_point}] at [${best_ck_point_path}]"

        print_info "inference starts"
        cd ~/bliss/case_studies/dc2_mdt/
        poetry run python inference.py \
            --model-tag-name=${task_name} \
            --exp-name=${exp_name} \
            --exp-check-point-name=${best_ck_point} \
            --cfg-name=${cfg_name} \
            --cached-data-path=/scratch/regier_root/regier0/pduan/posterior_cached_files \
            --cuda-idx=${cuda_device} \
            --infer-batch-size=${infer_batch_size} \
            --infer-total-iters=${infer_total_iters}
        print_info "inference ends"
    } > logs/infer_task_${task_name}_${exp_name}.log 2>&1 &
}

cuda_device=0
run_exp "${task_name}" "${exp_name}" "${cuda_device}"

print_info "wait to complete"
wait

print_info "done"
