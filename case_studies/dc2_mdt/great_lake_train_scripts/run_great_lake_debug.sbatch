#!/bin/bash

#SBATCH --mail-user=pduan@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --account=regier0
#SBATCH --partition=spgpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=a40:2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32GB

## wall time hours:minutes:seconds
#SBATCH --time=02:00:00

module load python/3.10.4 poetry/1.6.1 cuda/12.1.1 cudnn/12.1-v8.9.0
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
    cfg_name="${task_name}_train_config_gl"

    srun -N1 -n1 bash -c "
        set -e

        poetry run bliss \
            -cp ~/bliss/case_studies/dc2_mdt/mdt_config \
            -cn ${cfg_name} \
            train.trainer.logger.version=${exp_name} \
            train.trainer.devices=${cuda_device} \
            train.trainer.max_epochs=1

        best_ck_point=\$(find ~/bliss_output/DC2_mdt_exp/${exp_name}/checkpoints/ -name '*.ckpt' | xargs -- basename)

        poetry run python inference.py \
            --model-tag-name=${task_name} \
            --exp-name=${exp_name} \
            --exp-check_point_name=\${best_ck_point} \
            --cfg-name=${cfg_name} \
            --cached-data-path=/scratch/regier_root/regier0/pduan/posterior_cached_files \
            --cuda-idx=${cuda_device} \
            --infer-batch-size=32 \
            --infer-total-iters=10
    " >> logs/task_${task_name}.log 2>&1 &
}

echo "run simple_net"
task_name="simple_net"
exp_name="debug_great_lake_exp_$(date +"%m-%d")-1"
cuda_device=0
run_exp ${task_name} ${exp_name} ${cuda_device}

echo "run simple_net_speed"
task_name="simple_net_speed"
exp_name="debug_great_lake_exp_$(date +"%m-%d")-2"
cuda_device=1
run_exp ${task_name} ${exp_name} ${cuda_device}

echo "wait to complete"
wait

echo "done"
