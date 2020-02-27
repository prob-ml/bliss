import numpy as np

import torch
import torch.optim as optim

import simulated_datasets_lib
import starnet_vae_lib
import sleep_lib
import psf_transform_lib
import wake_lib

import json
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)

print('torch version: ', torch.__version__)

###############
# set seed
###############
np.random.seed(65765)
_ = torch.manual_seed(3453453)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

###############
# data parameters
###############
with open('./data/default_star_parameters.json', 'r') as fp:
    data_params = json.load(fp)

print(data_params)

###############
# load psf
###############
bands = [2, 3]
psfield_file = './../celeste_net/sdss_stage_dir/2583/2/136/psField-002583-2-0136.fit'
init_psf_params = psf_transform_lib.get_psf_params(
                                    psfield_file,
                                    bands = bands)
# init_psf_params = torch.Tensor(np.load('./data/fitted_powerlaw_psf_params.npy'))
# power_law_psf = psf_transform_lib.PowerLawPSF(init_psf_params.to(device))
psf_og = power_law_psf.forward().detach()

###############
# sky intensity: for the r and i band
###############
init_background_params = torch.zeros(len(bands), 3).to(device)
init_background_params[:, 0] = torch.Tensor([686., 1123.])
# init_background_params = torch.Tensor(np.load('./data/fitted_planar_backgrounds.npy'))
# planar_background = wake_lib.PlanarBackground(image_slen = data_params['slen'],
                            # init_background_params = init_background_params.to(device))
background = planar_background.forward().detach()

###############
# draw data
###############
print('generating data: ')
n_images = 200
t0 = time.time()
star_dataset = \
    simulated_datasets_lib.load_dataset_from_params(psf_og,
                            data_params,
                            background = background,
                            n_images = n_images,
                            transpose_psf = False,
                            add_noise = True)

print('data generation time: {:.3f}secs'.format(time.time() - t0))
# get loader
batchsize = 20

loader = torch.utils.data.DataLoader(
                 dataset=star_dataset,
                 batch_size=batchsize,
                 shuffle=True)

###############
# define VAE
###############
star_encoder = starnet_vae_lib.StarEncoder(slen = data_params['slen'],
                                           patch_slen = 7,
                                           step = 2,
                                           edge_padding = 2,
                                           n_bands = psf_og.shape[0],
                                           max_detections = 2,
                                           estimate_flux = True)

star_encoder.to(device)

###############
# define optimizer
###############
learning_rate = 1e-3
weight_decay = 1e-5
optimizer = optim.Adam([
                    {'params': star_encoder.parameters(),
                    'lr': learning_rate}],
                    weight_decay = weight_decay)


###############
# Train!
###############
n_epochs = 201
print_every = 20
print('training')

out_filename = './fits/results_2020-02-26/starnet_ri'

sleep_lib.run_sleep(star_encoder, loader, optimizer, n_epochs,
                        out_filename = out_filename,
                        print_every = print_every)
