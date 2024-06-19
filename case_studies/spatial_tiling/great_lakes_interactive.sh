#!/bin/bash

salloc --job-name=interactive \
       --account=regier0 \
       --partition=spgpu \
       --nodes=1 \
       --ntasks=1 \
       --gpus-per-node=a40:1 \
       --cpus-per-gpu=16 \
       --mem-per-gpu=32GB \
       --time=02:00:00
