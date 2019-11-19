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

np.random.seed(34534)

bands = [2]

sdss_hubble_data = sdss_dataset_lib.SDSSHubbleData(x0 = 600, x1 = 0, slen = 800,
                                                   bands = bands, fudge_conversion=1.0)

# image
full_image = sdss_hubble_data.sdss_image.unsqueeze(0)
full_background = sdss_hubble_data.sdss_background.unsqueeze(0)

# true parameters
true_locs = sdss_hubble_data.locs
true_fluxes = sdss_hubble_data.fluxes

slen0 = full_image.shape[2]
slen1 = full_image.shape[3]

# Get reconstruction
import fitsio

psf_dir = '../data/'
psf_r = fitsio.FITS(psf_dir + 'sdss-002583-2-0136-psf-r.fits')[0].read()

psf_og = np.array([psf_r])
sky_intensity = torch.Tensor([686.])

# get reconstruction
simulator = simulated_datasets_lib.StarSimulator(psf=psf_og,
                                                slen = full_image.shape[-1],
                                                transpose_psf = False,
                                                sky_intensity = sky_intensity)

truth_recon = simulator.draw_image_from_params(locs = sdss_hubble_data.locs.unsqueeze(0),
                            fluxes = sdss_hubble_data.fluxes.unsqueeze(0),
                            n_stars = torch.Tensor([len(sdss_hubble_data.fluxes)]).type(torch.LongTensor),
                            add_noise = False).squeeze()

np.savetxt('truth_recon', truth_recon)
