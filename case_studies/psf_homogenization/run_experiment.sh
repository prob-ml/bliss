#!/usr/bin/env bash
GPU=3
export CUDA_VISIBLE_DEVICES=$GPU
echo "Starting $EXPERIMENT..."
python main.py mode=train training=sdss_detection_encoder_full_decoder_unhomo training.save_top_k=1 training.experiment=sdss_detection_encoder_full_decoder_unhomo