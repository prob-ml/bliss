#!/bin/bash

outfolder='../fits/'
encoder_name='starnet-m2'

python train_sleep.py \
  --outfolder $outfolder \
  --outfilename $encoder_name
