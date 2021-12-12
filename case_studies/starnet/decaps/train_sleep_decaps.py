import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--outfolder', type=str, default='./fits/')
parser.add_argument('--outfilename', type=str, default='tmp')

parser.add_argument('--config_path', type=str, default='../../../config')

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

from hydra.experimental import initialize, compose

import numpy as np
import time 

from bliss import sleep
from bliss.datasets import simulated, sdss

import sys
sys.path.append('../starnet_utils/')
from starnet_sleep_dataset import SimulatedStarnetDataset

torch.manual_seed(args.seed)
np.random.seed(args.seed)


print("Training sleep phase")
###################
# load config parameters 
###################
overrides = dict(
    model="sleep_m2",
    dataset="m2",
    training="m2",
    optimizer="m2"
)

print('config overrides: ')
print(overrides)

overrides = [f"{key}={value}" for key, value in overrides.items()]

with initialize(config_path=args.config_path):
    cfg = compose("config", overrides=overrides)

    
    
cfg.model.decoder.kwargs.update({'n_bands': 1, 
                                 'slen': 300, 
                                 'tile_slen': 10, 
                                 'ptile_slen': 30, 
                                 'border_padding': 5, 
                                 'background_values': [680], 
                                 'psf_params_file': './psf/zband_psf_simple.npy'})

cfg.model.encoder.kwargs.update({'ptile_slen': 20})

cfg.dataset.kwargs.update({'n_batches': 40, 'batch_size': 5})

print('config: ')
print(cfg)


###################
# initialize data set and model
###################
# dataset = simulated.SimulatedDataset(**cfg.dataset.kwargs)

# cfg.dataset.kwargs['mean_background_vals'] = [680.]
# cfg.dataset.kwargs['background_sd'] = 15

dataset = SimulatedStarnetDataset(mean_background_vals = [680.], 
                                  background_sd = 15, 
                                  **cfg.dataset.kwargs)
sleep_net = sleep.SleepPhase(**cfg.model.kwargs)
trainer = pl.Trainer(**cfg.training.trainer)


###################
# train and save!
###################
t0 = time.time()
trainer.fit(sleep_net, datamodule = dataset)

out_filename = os.path.join(args.outfolder, args.outfilename)

print('saving into: ', out_filename)
torch.save(sleep_net.image_encoder.state_dict(), out_filename)

print('TOTAL TIME ELAPSED: {:.3f}secs'.format(time.time() - t0))
