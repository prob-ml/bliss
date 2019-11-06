import numpy as np

import torch
import torch.optim as optim

import sdss_dataset_lib

import simulated_datasets_lib
import starnet_vae_lib
import inv_kl_objective_lib as inv_kl_lib

from wake_sleep_lib import run_wake, run_sleep

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
                                        'hlsp_acsggct_hst_acs-wfc_ngc7089_r.rdviq.cal.adj.zpt.txt')

# sdss image
full_image = sdss_hubble_data.sdss_image.unsqueeze(0).to(device)
full_background = sdss_hubble_data.sdss_background.unsqueeze(0).to(device)

# simulated data parameters
with open('./data/default_star_parameters.json', 'r') as fp:
    data_params = json.load(fp)

print(data_params)
data_params['sky_intensity'] = 179.
full_background = full_background * 0.0 + data_params['sky_intensity']

# draw data
print('generating data: ')
n_images = 200
t0 = time.time()
star_dataset = \
    simulated_datasets_lib.load_dataset_from_params(str(sdss_hubble_data.psf_file),
                            data_params,
                            n_images = n_images,
                            add_noise = True)

print('data generation time: {:.3f}secs'.format(time.time() - t0))
# get loader
batchsize = 20

loader = torch.utils.data.DataLoader(
                 dataset=star_dataset,
                 batch_size=batchsize,
                 shuffle=True)

# define VAE
star_encoder = starnet_vae_lib.StarEncoder(full_slen = data_params['slen'],
                                           stamp_slen = 7,
                                           step = 2,
                                           edge_padding = 2,
                                           n_bands = 1,
                                           max_detections = 2)

star_encoder.to(device)

# freeze batchnorm layers
# code taken from https://discuss.pytorch.org/t/freeze-batchnorm-layer-lead-to-nan/8385/2
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
      m.eval()

star_encoder.apply(set_bn_eval);

# define psf transform
from copy import deepcopy
psf_og = deepcopy(loader.dataset.simulator.psf_og)
psf_transform = psf_transform_lib.PsfLocalTransform(torch.Tensor(psf_og).to(device),
									data_params['slen'],
									kernel_size = 3)
psf_transform.to(device)



filename = './fits/results_11042019/wake_sleep-loc630x310-11042019'

########################
# Initial training of encoder
########################
init_encoder = './fits/results_11052019/starnet'
print('loading encoder from: ', init_encoder)
star_encoder.load_state_dict(torch.load(init_encoder,
                               map_location=lambda storage, loc: storage));
star_encoder.to(device)

# load optimizer
encoder_lr = 1e-5
vae_optimizer = optim.Adam([
                    {'params': star_encoder.parameters(),
                    'lr': encoder_lr}],
                    weight_decay = 1e-5)


run_joint_wake(full_image, full_background, star_encoder, psf_transform,
                    optimizer = vae_optimizer,
                    n_epochs = 200,
                    n_samples = 10,
                    out_filename = './fits/results_11052019/kl_starnet')
