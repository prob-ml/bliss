import numpy as np
import fitsio

import torch
import torch.optim as optim

import sdss_dataset_lib

import starnet_vae_lib

from wake_sleep_lib import run_joint_wake

import psf_transform_lib

import time

import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)

print('torch version: ', torch.__version__)

# set seed
np.random.seed(4534)
_ = torch.manual_seed(2534)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# get sdss data
sdss_hubble_data = sdss_dataset_lib.SDSSHubbleData(sdssdir='../celeste_net/sdss_stage_dir/',
                                       hubble_cat_file = './hubble_data/NCG7089/' + \
                                        'hlsp_acsggct_hst_acs-wfc_ngc7089_r.rdviq.cal.adj.zpt.txt',
                                        bands = [2])

# sdss image
full_image = sdss_hubble_data.sdss_image.unsqueeze(0).to(device)
full_background = sdss_hubble_data.sdss_background.unsqueeze(0).to(device) + 88.


# define VAE
star_encoder = starnet_vae_lib.StarEncoder(full_slen = full_image.shape[-1],
                                           stamp_slen = 7,
                                           step = 2,
                                           edge_padding = 2,
                                           n_bands = full_image.shape[1],
                                           max_detections = 2,
                                           fmin = 1000.)

star_encoder.to(device)

# freeze batchnorm layers
# code taken from https://discuss.pytorch.org/t/freeze-batchnorm-layer-lead-to-nan/8385/2
# def set_bn_eval(m):
#     classname = m.__class__.__name__
#     if classname.find('BatchNorm') != -1:
#       m.eval()
#
# star_encoder.apply(set_bn_eval);

# load psf
psf_dir = './data/'
psf_r = fitsio.FITS(psf_dir + 'sdss-002583-2-0136-psf-r.fits')[0].read()
psf_i = fitsio.FITS(psf_dir + 'sdss-002583-2-0136-psf-i.fits')[0].read()

psf_og = torch.Tensor(np.array([psf_r])).to(device)

# define psf transform
psf_transform = psf_transform_lib.PsfLocalTransform(psf_og,
									full_image.shape[-1],
									kernel_size = 3, init_bias = 10)
psf_transform.to(device)

# filename = './fits/results_11042019/wake_sleep-loc630x310-11042019'

########################
# Initial training of encoder
########################
# init_encoder = './fits/results_11202019/kl_starnet'
# print('loading encoder from: ', init_encoder)
# star_encoder.load_state_dict(torch.load(init_encoder,
#                                map_location=lambda storage, loc: storage));
star_encoder.to(device)

# load optimizer
encoder_lr = 1e-4
vae_optimizer = optim.Adam([
                    {'params': star_encoder.parameters(),
                    'lr': encoder_lr}],
                    weight_decay = 1e-5)


run_joint_wake(full_image, full_background, star_encoder, psf_transform,
                    optimizer = vae_optimizer,
                    n_epochs = 4000,
                    n_samples = 40,
                    encoder_outfile = './fits/results_11202019/kl_starnet2',
                    psf_outfile = '././fits/results_11202019/identity_psf')
