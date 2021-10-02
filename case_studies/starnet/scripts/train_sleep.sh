#!/bin/bash

outfolder='../fits/'
encoder_name='starnet'

python train_sleep.py \
  --outfolder $outfolder \
  --outfilename $encoder_name
