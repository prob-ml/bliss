#!/bin/bash
bliss -cp ~/bliss/case_studies/dc2_mdt/mdt_config -cn mdt_train_config
bliss -cp ~/bliss/case_studies/dc2_mdt/mdt_config -cn simple_net_train_config
bliss -cp ~/bliss/case_studies/dc2_mdt/mdt_config -cn simple_net_speed_train_config
bliss -cp ~/bliss/case_studies/dc2_mdt/mdt_config -cn simple_ar_net_train_config
bliss -cp ~/bliss/case_studies/dc2_mdt/mdt_config -cn simple_cond_true_net_train_config
# run inference
nohup jupyter nbconvert --execute --to html case_studies/dc2_mdt/inference.ipynb > infer_notebook.out 2>&1 &

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 \
                                multi_gpus_infer.py \
                                --model-tag-name="mdt" \
                                --exp-name="exp_04-12-1" \
                                --exp-check-point-name="encoder_101.ckpt" \
                                --cfg-name="mdt_train_config" \
                                --cached-data-path="/data/scratch/pduan/posterior_cached_files" \
                                --infer-batch-size=400 \
                                --infer-total-iters=500