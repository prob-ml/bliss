#!/bin/bash

outfolder='../fits/'
encoder_name='starnet-one_source_one_hot_tile'
# encoder_name='tmp'

python train_sleep.py \
  --outfolder $outfolder \
  --outfilename $encoder_name
