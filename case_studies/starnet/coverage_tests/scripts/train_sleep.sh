#!/bin/bash

outfolder='../fits/'
encoder_name='starnet-many_sources_padded2-sumlogprobs'
# encoder_name='tmp'

python train_sleep.py \
  --outfolder $outfolder \
  --outfilename $encoder_name
