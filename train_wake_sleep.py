import numpy as np

import torch
import torch.optim as optim

import sdss_dataset_lib

import simulated_datasets_lib
import starnet_vae_lib

import sleep_lib
from sleep_lib import run_sleep
import wake_lib

import psf_transform_lib

import time

import fitsio

import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)

print('torch version: ', torch.__version__)

#######################
# set seed
########################
np.random.seed(32090275)
_ = torch.manual_seed(120457)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#######################
# get sdss data
#######################
bands = [2, 3]
sdss_hubble_data = sdss_dataset_lib.SDSSHubbleData(sdssdir='../celeste_net/sdss_stage_dir/',
                                       hubble_cat_file = './hubble_data/NCG7089/' + \
                                        'hlsp_acsggct_hst_acs-wfc_ngc7089_r.rdviq.cal.adj.zpt.txt',
                                        bands = bands)

full_image = sdss_hubble_data.sdss_image.unsqueeze(0).to(device)

#######################
# simulated data parameters
#######################
with open('./data/default_star_parameters.json', 'r') as fp:
    data_params = json.load(fp)
print(data_params)


###############
# load psf
###############
psfield_file = './../celeste_net/sdss_stage_dir/2583/2/136/psField-002583-2-0136.fit'
init_psf_params = psf_transform_lib.get_psf_params(
                                    psfield_file,
                                    bands = bands).to(device)
power_law_psf = psf_transform_lib.PowerLawPSF(init_psf_params.to(device))
psf_og = power_law_psf.forward().detach()

###############
# sky intensity: for the r and i band
###############
init_background_params = torch.zeros(len(bands), 3).to(device)
init_background_params[:, 0] = torch.Tensor([686., 1123.])
planar_background = wake_lib.PlanarBackground(image_slen = data_params['slen'],
                            init_background_params = init_background_params.to(device))
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
                            n_images = n_images,
                            background = background,
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
star_encoder = starnet_vae_lib.StarEncoder(full_slen = data_params['slen'],
                                           stamp_slen = 7,
                                           step = 2,
                                           edge_padding = 2,
                                           n_bands = len(bands),
                                           max_detections = 2,
                                           estimate_flux = True)

init_encoder = './fits/results_2020-02-18/starnet_ri'
star_encoder.load_state_dict(torch.load(init_encoder,
                                   map_location=lambda storage, loc: storage))
star_encoder.to(device)
star_encoder.eval();

####################
# optimzer
#####################
encoder_lr = 1e-5
sleep_optimizer = optim.Adam([
                    {'params': star_encoder.parameters(),
                    'lr': encoder_lr}],
                    weight_decay = 1e-5)

# initial loss:
test_loss, test_counter_loss, test_locs_loss, test_fluxes_loss = \
    sleep_lib.eval_star_encoder_loss(star_encoder,
                                    loader, train = False)

print('**** INIT test loss: {:.3f}; counter loss: {:.3f}; locs loss: {:.3f}; fluxes loss: {:.3f} ****'.format(\
    test_loss, test_counter_loss, test_locs_loss, test_fluxes_loss))

# file header to save results
outfolder = './fits/results_2020-02-18/'

n_iter = 6
map_losses = torch.zeros(n_iter)
for iteration in range(0, n_iter):
    #######################
    # wake phase training
    #######################
    print('RUNNING WAKE PHASE. ITER = ' + str(iteration))
    if iteration == 0:
        encoder_file = init_encoder
        powerlaw_psf_params = init_psf_params
        planar_background_params = None
    else:
        encoder_file = outfolder + 'wake-sleep-encoder-iter' + str(iteration)
        powerlaw_psf_params = \
            torch.Tensor(np.load(outfolder + 'iter' + str(iteration - 1) +\
                                    '-powerlaw_psf_params.npy')).to(device)
        planar_background_params = \
            torch.Tensor(np.load(outfolder + 'iter' + str(iteration - 1) +\
                                    '-planarback_params.npy')).to(device)

    print('loading encoder from: ', encoder_file)
    star_encoder.load_state_dict(torch.load(encoder_file,
                                   map_location=lambda storage, loc: storage))
    star_encoder.to(device);
    star_encoder.eval();

    model_params, map_losses[iteration] = wake_lib.run_wake(full_image, star_encoder, powerlaw_psf_params,
                        planar_background_params,
                        n_samples = 60,
                        out_filename = outfolder + 'iter' + str(iteration),
                        lr = 1e-3,
                        run_map = False)

    print(list(model_params.planar_background.parameters())[0])
    print(list(model_params.power_law_psf.parameters())[0])
    print(map_losses[iteration])
    np.save(outfolder + 'map_losses', map_losses.cpu().detach())

    ########################
    # sleep phase training
    ########################
    print('RUNNING SLEEP PHASE. ITER = ' + str(iteration + 1))

    # update psf
    loader.dataset.simulator.psf = model_params.get_psf().detach()
    loader.dataset.simulator.background = model_params.get_background().squeeze(0).detach()

    run_sleep(star_encoder,
                loader,
                sleep_optimizer,
                n_epochs = 11,
                out_filename = outfolder + 'wake-sleep-encoder-iter' + str(iteration + 1))
