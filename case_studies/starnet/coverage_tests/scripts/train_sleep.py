import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--outfolder', type=str, default='../fits/')
parser.add_argument('--outfilename', type=str, default='sleepnet')

parser.add_argument('--config_path', type=str, default='./')

parser.add_argument('--seed', type=int, default=23423)

parser.add_argument('--cuda_no', type=int, default=2)

args = parser.parse_args()

import os
assert os.path.isdir(args.outfolder)

import torch
import pytorch_lightning as pl
print('torch version: ', torch.__version__)

cuda_no = "cuda:" + str(args.cuda_no)
device = torch.device(cuda_no if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
print(device)

from hydra.utils import instantiate
from hydra.experimental import initialize, compose

import numpy as np
import time 

from bliss.datasets import simulated
from bliss import sleep

torch.manual_seed(args.seed)
np.random.seed(args.seed)

print("Training sleep phase")
###################
# load config parameters 
###################
with initialize(config_path=args.config_path):
    cfg = compose("many_sources")

print('config: ')
print(cfg)

###################
# initialize data set and model
###################
dataset = instantiate(cfg.training.dataset)
sleep_net = instantiate(cfg.training.model,
                        optimizer_params=cfg.training.optimizer_params)
trainer = instantiate(cfg.training.trainer)

###################
# train and save!
###################
t0 = time.time()
trainer.fit(sleep_net, datamodule = dataset)

out_filename = os.path.join(args.outfolder, args.outfilename)

print('saving into: ', out_filename)
torch.save(sleep_net.image_encoder.state_dict(), out_filename)

print('TOTAL TIME ELAPSED: {:.3f}secs'.format(time.time() - t0))
