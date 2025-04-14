#!/bin/bash

#SBATCH --mail-user=pduan@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --account=regier0
#SBATCH --partition=spgpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a40:5
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32GB

## wall time hours:minutes:seconds
#SBATCH --time=72:00:00

module load python/3.10.4 poetry/1.6.1 cuda/12.6.3 cudnn/12.6-v9.6.0
module list

echo "running on: $SLURM_JOB_NODELIST"

mkdir -p /tmpssd/pduan/dc2local
mkdir -p /scratch/regier_root/regier0/pduan/posterior_cached_files
mkdir -p logs

echo "prepare data"
rsync -av --info=progress2 \
   /scratch/regier_root/regier0/pduan/dc2local/dc2_cached_data \
   /tmpssd/pduan/dc2local/

echo "go to bliss"
cd ~/bliss/

function run_exp() {
    task_name=$1
    exp_name=$2
    cuda_device=$3
    cuda_device_tag="[${cuda_device}]"
    cfg_name="${task_name}_train_config_gl"

    {
        set -e

        poetry run bliss \
            -cp ~/bliss/case_studies/dc2_mdt/mdt_config \
            -cn ${cfg_name} \
            train.trainer.logger.version=${exp_name} \
            train.trainer.devices=${cuda_device_tag}

        best_ck_point=$(find ~/bliss_output/DC2_mdt_exp/${exp_name}/checkpoints/ -name '*.ckpt' | sort | head -n1 | xargs -- basename)

        poetry run python case_studies/dc2_mdt/inference.py \
            --model-tag-name=${task_name} \
            --exp-name=${exp_name} \
            --exp-check-point-name=${best_ck_point} \
            --cfg-name=${cfg_name} \
            --cached-data-path=/scratch/regier_root/regier0/pduan/posterior_cached_files \
            --cuda-idx=${cuda_device} \
            --infer-batch-size=800 \
            --infer-total-iters=500
    } > logs/task_${task_name}.log 2>&1 &
}

tasks=("mdt" "simple_net" "simple_net_speed" "simple_ar_net" "simple_cond_true_net")
for i in "${!tasks[@]}"; do
    task_name="${tasks[i]}"
    echo "run ${task_name}"
    exp_name="great_lake_exp_$(date +"%m-%d")-$((i+1))"
    cuda_device=$i
    run_exp "${task_name}" "${exp_name}" "${cuda_device}"
done

echo "wait to complete"
wait

echo "done"
