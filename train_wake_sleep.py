import numpy as np

import torch
import torch.optim as optim

import sdss_dataset_lib

import simulated_datasets_lib
import starnet_vae_lib
import inv_kl_objective_lib as inv_kl_lib
from kl_objective_lib import sample_normal, sample_class_weights

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
                                        'hlsp_acsggct_hst_acs-wfc_ngc7089_r.rdviq.cal.adj.zpt.txt',
					x0 = 650, x1 = 120)

# sdss image
full_image = sdss_hubble_data.sdss_image.unsqueeze(0).to(device)
full_background = sdss_hubble_data.sdss_background.unsqueeze(0).to(device)

# true parameters
true_full_locs = sdss_hubble_data.locs.unsqueeze(0).to(device)
true_full_fluxes = sdss_hubble_data.fluxes.unsqueeze(0).to(device)

# simulated data parameters
with open('./data/default_star_parameters.json', 'r') as fp:
    data_params = json.load(fp)

data_params['slen'] = full_image.shape[-1]
data_params['min_stars'] = 2000
data_params['max_stars'] = 2000
data_params['alpha'] = 0.5

print(data_params)

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
batchsize = 10

loader = torch.utils.data.DataLoader(
                 dataset=star_dataset,
                 batch_size=batchsize,
                 shuffle=True)

# define VAE
star_encoder = starnet_vae_lib.StarEncoder(full_slen = data_params['slen'],
                                           stamp_slen = 9,
                                           step = 2,
                                           edge_padding = 3,
                                           n_bands = 1,
                                           max_detections = 4)

star_encoder.to(device)

# define psf transform
psf_transform = psf_transform_lib.PsfLocalTransform(torch.Tensor(loader.dataset.simulator.psf_og).to(device),
									data_params['slen'],
									kernel_size = 3)
psf_transform.to(device)



filename = './fits/wake_sleep-alt_m2-101320129'
for iteration in range(0, 6):
    print('RUNNING WAKE PHASE. ITER = ' + str(iteration))
    # load encoder
    if iteration == 0:
        encoder_file = './fits/starnet_invKL_encoder-10092019-reweighted_samples'
    else:
        encoder_file = filename + '-encoder-iter' + str(iteration)

        psf_transform_file = filename + 'psf_transform' + '-iter' + str(iteration - 1)
        print('loading psf_transform from: ', psf_transform_file)
        psf_transform.load_state_dict(torch.load(psf_transform_file,
                                    map_location=lambda storage, loc: storage))
        psf_transform.to(device)

    print('loading encoder from: ', encoder_file)
    star_encoder.load_state_dict(torch.load(encoder_file,
                                   map_location=lambda storage, loc: storage)); star_encoder.to(device); star_encoder.eval();

    # get optimizer
    learning_rate = 0.5
    weight_decay = 1e-5
    psf_optimizer = optim.Adam([
                        {'params': psf_transform.parameters(),
                        'lr': learning_rate}],
                        weight_decay = weight_decay)

    run_wake(full_image, full_background, star_encoder, psf_transform,
                    simulator = loader.dataset.simulator,
                    optimizer = psf_optimizer,
                    n_epochs = 41,
                    out_filename = filename + 'psf_transform',
                    iteration = iteration)

    print('RUNNING SLEEP PHASE. ITER = ' + str(iteration + 1))

    # load encoder
    if iteration == 0:
        encoder_file = './fits/starnet_invKL_encoder-10092019-reweighted_samples'
    else:
        encoder_file = filename + '-encoder-iter' + str(iteration)
    print('loading encoder from: ', encoder_file)
    star_encoder.load_state_dict(torch.load(encoder_file,
                                   map_location=lambda storage, loc: storage)); star_encoder.to(device)

    psf_transform_file = filename + 'psf_transform' + '-iter' + str(iteration)
    print('loading psf_transform from: ', psf_transform_file)

    # load trained transform
    psf_transform.load_state_dict(torch.load(psf_transform_file,
                                map_location=lambda storage, loc: storage)); psf_transform.to(device)
    loader.dataset.simulator.psf = psf_transform.forward().detach()

    # load optimizer
    learning_rate = 5e-4
    weight_decay = 1e-5
    vae_optimizer = optim.Adam([
                        {'params': star_encoder.parameters(),
                        'lr': learning_rate}],
                        weight_decay = weight_decay)

    run_sleep(star_encoder,
                loader,
                vae_optimizer,
                n_epochs = 21,
                out_filename = filename + '-encoder',
                iteration = iteration + 1)
