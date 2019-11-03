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
                                        'hlsp_acsggct_hst_acs-wfc_ngc7089_r.rdviq.cal.adj.zpt.txt',
                                        x0 = 650, x1 = 120)

# sdss image
full_image = sdss_hubble_data.sdss_image.unsqueeze(0).to(device)
full_background = sdss_hubble_data.sdss_background.unsqueeze(0).to(device)

# simulated data parameters
with open('./data/default_star_parameters.json', 'r') as fp:
    data_params = json.load(fp)

print(data_params)
data_params['sky_intensity'] = 170
full_background = full_background * 0.0 + data_params['sky_intensity']

# draw data
print('generating data: ')
n_images =1
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
                                           max_detections = 2)

star_encoder.to(device)

# define psf transform
from copy import deepcopy
psf_og = deepcopy(loader.dataset.simulator.psf_og)
psf_transform = psf_transform_lib.PsfLocalTransform(torch.Tensor(psf_og).to(device),
									data_params['slen'],
									kernel_size = 3)
psf_transform.to(device)



filename = './fits/results_10302019/wake_sleep-loc650x120-11022019'

########################
# Initial training of encoder
########################
init_encoder = './fits/results_10302019/starnet-10302019'
print('loading encoder from: ', init_encoder)
star_encoder.load_state_dict(torch.load(init_encoder,
                               map_location=lambda storage, loc: storage));
star_encoder.to(device)
star_encoder.eval(); 

images_full = star_dataset.images.detach()
backgrounds_full = torch.ones(star_dataset.images.shape) * star_dataset.sky_intensity

true_full_locs = star_dataset.locs[:, star_dataset.fluxes.squeeze() > 0, :]
true_full_fluxes = star_dataset.fluxes[:, star_dataset.fluxes.squeeze() > 0]

# check loss
loss, counter_loss, locs_loss, fluxes_loss, perm_indx = \
    inv_kl_lib.get_encoder_loss(star_encoder, images_full, backgrounds_full,
                                true_full_locs, true_full_fluxes)
print('loss: {:06f}'.format(loss))
print(counter_loss.mean())
print(locs_loss.mean())
print(fluxes_loss.mean())

# load optimizer
encoder_lr = 0e-5
vae_optimizer = optim.Adam([
                    {'params': star_encoder.parameters(),
                    'lr': encoder_lr}],
                    weight_decay = 1e-5)

run_sleep(star_encoder,
            loader,
            vae_optimizer,
            n_epochs = 11,
            out_filename = filename + '-encoder',
            iteration = 0)

for iteration in range(0, 6):
    ########################
    # wake phase training
    ########################
    print('RUNNING WAKE PHASE. ITER = ' + str(iteration))
    # load encoder
    encoder_file = filename + '-encoder-iter' + str(iteration)

    if iteration > 0:
        # load psf transform
        psf_transform_file = filename + '-psf_transform' + '-iter' + str(iteration - 1)
        print('loading psf_transform from: ', psf_transform_file)
        psf_transform.load_state_dict(torch.load(psf_transform_file,
                                    map_location=lambda storage, loc: storage))
        psf_transform.to(device)

    print('loading encoder from: ', encoder_file)
    star_encoder.load_state_dict(torch.load(encoder_file,
                                   map_location=lambda storage, loc: storage));
    star_encoder.to(device);
    star_encoder.eval();

    # get optimizer
    psf_lr = 0.05 / (1 + 80 * iteration)
    psf_optimizer = optim.Adam([
                        {'params': psf_transform.parameters(),
                        'lr': psf_lr}], weight_decay = 1e-5)

    run_wake(full_image, full_background, star_encoder, psf_transform,
                    optimizer = psf_optimizer,
                    n_epochs = 20,
                    n_samples = 1000,
                    out_filename = filename + '-psf_transform',
                    iteration = iteration,
                    use_iwae = True)

    ########################
    # sleep phase training
    ########################
    print('RUNNING SLEEP PHASE. ITER = ' + str(iteration + 1))

    # load encoder
    encoder_file = filename + '-encoder-iter' + str(iteration)
    print('loading encoder from: ', encoder_file)
    star_encoder.load_state_dict(torch.load(encoder_file,
                                   map_location=lambda storage, loc: storage));
    star_encoder.to(device)

    # load trained transform
    psf_transform_file = filename + '-psf_transform' + '-iter' + str(iteration)
    print('loading psf_transform from: ', psf_transform_file)
    psf_transform.load_state_dict(torch.load(psf_transform_file,
                                map_location=lambda storage, loc: storage));
    psf_transform.to(device)
    loader.dataset.simulator.psf = psf_transform.forward().detach()

    # load optimizer
    encoder_lr = 5e-5
    vae_optimizer = optim.Adam([
                        {'params': star_encoder.parameters(),
                        'lr': encoder_lr}],
                        weight_decay = 1e-5)

    run_sleep(star_encoder,
                loader,
                vae_optimizer,
                n_epochs = 11,
                out_filename = filename + '-encoder',
                iteration = iteration + 1)
