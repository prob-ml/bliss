#!/bin/bash

cd ~/bliss/case_studies/dc2_mdt
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 \
    m2_multi_gpus_infer.py \
    --model-tag-name "m2_simple_net" \
    --exp-check-point-path "/home/pduan/bliss_output/M2_mdt_exp/exp_04-20-1/checkpoints/encoder_32.ckpt" \
    --cfg-path "./m2_mdt_config/m2_simple_net_train_config" \
    --cached-data-path "/data/scratch/pduan/posterior_cached_files" \
    --infer-batch-size 800 \
    --infer-total-iters 500


CUDA_VISIBLE_DEVICES=3,4 torchrun --nproc_per_node=2 \
    --rdzv_endpoint=localhost:29400 \
    m2_multi_gpus_infer.py \
    --model-tag-name "m2_mdt" \
    --exp-check-point-path "/home/pduan/bliss_output/M2_mdt_exp/exp_04-19-1/checkpoints/encoder_76.ckpt" \
    --cfg-path "./m2_mdt_config/m2_mdt_train_config" \
    --cached-data-path "/data/scratch/pduan/posterior_cached_files" \
    --infer-batch-size 400 \
    --infer-total-iters 500


CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 \
    --rdzv_endpoint=localhost:29400 \
    m2_multi_gpus_infer.py \
    --model-tag-name "m2_mdt_speed" \
    --exp-check-point-path "/home/pduan/bliss_output/M2_mdt_exp/exp_04-21-2/checkpoints/encoder_156.ckpt" \
    --cfg-path "./m2_mdt_config/m2_mdt_train_config" \
    --cached-data-path "/data/scratch/pduan/posterior_cached_files" \
    --infer-batch-size 400 \
    --infer-total-iters 500
