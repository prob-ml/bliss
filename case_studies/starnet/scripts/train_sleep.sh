#!/bin/bash

outfolder='../fits/'
encoder_name='starnet-random_bg-random_pois'

python train_sleep.py \
  --outfolder $outfolder \
  --outfilename $encoder_name
