#!/bin/sh
GPU_FROM_CUDA=`nvidia-smi -q -d Memory |grep -A4 GPU|grep Used | grep -n "1 MiB" | head -n 1 | sed "s/:.*//"`
GPU=`expr $GPU_FROM_CUDA - 1`
echo $GPU
