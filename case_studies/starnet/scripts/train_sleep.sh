#!/bin/bash

outfolder='../fits/'
encoder_name='starnet-m2-sum_logprobs'
# encoder_name='tmp'

python train_sleep.py \
  --outfolder $outfolder \
  --outfilename $encoder_name
