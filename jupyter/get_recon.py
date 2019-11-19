import numpy as np
import torch

import json

import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../')
import simulated_datasets_lib
import sdss_dataset_lib
import sdss_psf
import image_utils

import starnet_vae_lib
import inv_kl_objective_lib as inv_kl_lib
import plotting_utils
import wake_sleep_lib

import psf_transform_lib
import image_statistics_lib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(34534)

bands = [2]

slen = 801
sdss_hubble_data = sdss_dataset_lib.SDSSHubbleData(x0 = 600, x1 = 0, slen = slen,
                                                   bands = bands, fudge_conversion=1.0)

# Get reconstruction
import fitsio

psf_dir = '../data/'
psf_r = fitsio.FITS(psf_dir + 'sdss-002583-2-0136-psf-r.fits')[0].read()

psf_og = np.array([psf_r])
sky_intensity = torch.Tensor([686.]).to(device)

# get reconstruction
simulator = simulated_datasets_lib.StarSimulator(psf=psf_og,
                                                slen = slen,
                                                transpose_psf = False,
                                                sky_intensity = sky_intensity)

truth_recon = simulator.draw_image_from_params(locs = sdss_hubble_data.locs.unsqueeze(0).to(device),
                            fluxes = sdss_hubble_data.fluxes.unsqueeze(0).to(device),
                            n_stars = torch.Tensor([len(sdss_hubble_data.fluxes)]).type(torch.LongTensor).to(device),
                            add_noise = False).squeeze()

np.savetxt('truth_recon', truth_recon.cpu().numpy())
print('done')
