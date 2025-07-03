#!/bin/bash
bliss -cp ~/bliss/case_studies/dc2_mdt/mdt_config -cn mdt_train_config
bliss -cp ~/bliss/case_studies/dc2_mdt/mdt_config -cn simple_net_train_config
bliss -cp ~/bliss/case_studies/dc2_mdt/mdt_config -cn simple_net_speed_train_config
bliss -cp ~/bliss/case_studies/dc2_mdt/mdt_config -cn simple_ar_net_train_config
bliss -cp ~/bliss/case_studies/dc2_mdt/mdt_config -cn simple_cond_true_net_train_config
bliss -cp ~/bliss/case_studies/dc2_mdt/m2_mdt_config -cn m2_mdt_train_config
bliss -cp ~/bliss/case_studies/dc2_mdt/m2_mdt_config -cn m2_simple_net_train_config
bliss -cp ~/bliss/case_studies/dc2_mdt/m2_mdt_config -cn m2_ori_bliss_train_config
bliss -cp ~/bliss/case_studies/dc2_mdt/m2_mdt_config -cn m2_cond_true_bliss_train_config
bliss -cp ~/bliss/case_studies/dc2_mdt/m2_mdt_config -cn m2_mdt_rml_train_config
bliss -cp ~/bliss/case_studies/dc2_mdt/mdt_config -cn flux_only_mdt_rml_train_config
bliss -cp ~/bliss/case_studies/dc2_mdt/m2_mdt_config -cn m2_mdt_full_rml_train_config \
    train.trainer.logger.version=exp_07-03_full_rml \
    train.trainer.devices=[3]
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

bliss -cp ~/bliss/case_studies/dc2_mdt/other_config -cn ori_bliss_train_config \
    encoder.use_double_detect=false \
    encoder.minimalist_conditioning=false \
    encoder.use_checkerboard=false \
    train.trainer.logger.version=exp_06-09-1 \
    train.trainer.devices=[2]

bliss -cp ~/bliss/case_studies/dc2_mdt/other_config -cn ori_bliss_train_config \
    encoder.use_double_detect=false \
    encoder.minimalist_conditioning=false \
    encoder.use_checkerboard=true \
    train.trainer.logger.version=exp_06-09-2 \
    train.trainer.devices=[3]