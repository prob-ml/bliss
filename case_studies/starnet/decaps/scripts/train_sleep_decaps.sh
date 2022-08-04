#!/bin/bash

outfolder='../fits/'
encoder_name='starnet-tmp'

python train_sleep_decaps.py \
  --outfolder $outfolder \
  --outfilename $encoder_name
