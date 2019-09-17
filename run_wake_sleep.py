import torch
import torch.optim as optim

import numpy as np

import sdss_dataset_lib
import residuals_vae_lib
import simulated_datasets_lib
import starnet_vae_lib

import time

import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)

print('torch version: ', torch.__version__)

# set seed
np.random.seed(43534)
_ = torch.manual_seed(24534)

max_stars = 20

########################
# Get Hubble data
########################
hubble_cat_file='./hubble_data/NCG7078/hlsp_acsggct_hst_acs-wfc_ngc7078_r.rdviq.cal.adj.zpt.txt'
sdss_hubble_data = \
    sdss_dataset_lib.SDSSHubbleData(hubble_cat_file=hubble_cat_file,
                                    sdssdir = './../celeste_net/sdss_stage_dir/',
                                    slen = 11,
                                    run = 2566,
                                    camcol = 6,
                                    field = 65,
                                    max_detections = max_stars)
batchsize = len(sdss_hubble_data)
sdss_loader = torch.utils.data.DataLoader(
                 dataset=sdss_hubble_data,
                 batch_size=batchsize,
                 shuffle=False)


sky_intensity = sdss_hubble_data.sdss_background_full.mean()

########################
# Get data simulator
########################
# data parameters
with open('./data/default_star_parameters.json', 'r') as fp:
    data_params = json.load(fp)

data_params['min_stars'] = 0
data_params['max_stars'] = max_stars
data_params['sky_intensity'] = sky_intensity

print(data_params)

# dataset
simulated_dataset = \
    simulated_datasets_lib.load_dataset_from_params(sdss_hubble_data.psf_file,
                            data_params,
                            n_stars = 60000,
                            add_noise = True)

simulated_loader = torch.utils.data.DataLoader(
                                 dataset=simulated_dataset,
                                 batch_size=2048,
                                 shuffle=True)

########################
# Get VAEs
########################
star_encoder = starnet_vae_lib.StarEncoder(data_params['slen'],
                                           n_bands = 1,
                                          max_detections = max_stars)

# load iteration 0 results: i.e. encoder trained on simulated data only
encoder_init = './fits/starnet_invKL_encoder_twenty_stars'
print('loading encoder from: ', encoder_init)
star_encoder.load_state_dict(torch.load(encoder_init,
                               map_location=lambda storage, loc: storage))

resiudal_vae = residuals_vae_lib.ResidualVAE(slen = sdss_hubble_data.slen,
                                            n_bands = 1,
                                            f_min = 2000.)

star_encoder.to(device)
resiudal_vae.to(device)

#####################
# Define losses
#####################
def run_wake(residual_vae, star_encoder, loader,
                simulator, optimizer, cycle, n_epochs):

    star_encoder.eval();

    for epoch in range(n_epochs):
        t0 = time.time()

        avg_loss = \
            residuals_vae_lib.eval_residual_vae(residual_vae, loader, simulator,
                                optimizer = optimizer, train = True ,
                                star_encoder = star_encoder)

        elapsed = time.time() - t0
        print('[{}] loss: {:.6E}; \t[{:.1f} seconds]'.format(\
                        epoch, avg_loss, elapsed))

        if (epoch % 5) == 0:

            test_loss = \
                residuals_vae_lib.eval_residual_vae(residual_vae, loader, simulator,
                                    optimizer = None, train = False,
                                    star_encoder = star_encoder)

            print('**** test loss: {:.6E} ****'.format(test_loss))

            outfile = './fits/residual_vae_wake' + str(cycle)
            print("writing the residual vae parameters to " + outfile)
            torch.save(resiudal_vae.state_dict(), outfile)


def run_sleep(residual_vae, star_encoder, loader, optimizer, cycle, n_epochs):

    residual_vae.eval();

    n_epochs = 500

    for epoch in range(n_epochs):
        t0 = time.time()

        avg_loss, counter_loss = objectives_lib.eval_star_encoder_loss(star_encoder, loader,
                                                        optimizer, train = True,
                                                        resiudal_vae = residual_vae)

        elapsed = time.time() - t0
        print('[{}] loss: {:0.4f}; counter loss: {:0.4f} \t[{:.1f} seconds]'.format(\
                        epoch, avg_loss, counter_loss, elapsed))

        # draw fresh data
        loader.dataset.set_params_and_images()

        if (epoch % 20) == 0:
            _, _ = \
                objectives_lib.eval_star_encoder_loss(star_encoder,
                                                loader, train = True,
                                                resiudal_vae = residual_vae)

            loader.dataset.set_params_and_images()
            test_loss, test_counter_loss = \
                objectives_lib.eval_star_encoder_loss(star_encoder,
                                                loader, train = False,
                                                resiudal_vae = residual_vae)

            print('**** test loss: {:.3f}; counter loss: {:.3f} ****'.format(test_loss, test_counter_loss))

            outfile = './fits/starnet_encoder_sleep' + str(cycle)
            print("writing the encoder parameters to " + outfile)
            torch.save(star_encoder.state_dict(), outfile)


##############################
# Train!
##############################
# get optimizers
learning_rate = 1e-3
weight_decay = 1e-5
residual_optimizer = optim.Adam([
                    {'params': resiudal_vae.parameters(),
                    'lr': learning_rate}],
                    weight_decay = weight_decay)

learning_rate = 1e-3
weight_decay = 1e-5
encoder_optimizer = optim.Adam([
                    {'params': star_encoder.parameters(),
                    'lr': learning_rate}],
                    weight_decay = weight_decay)

# run_wake(resiudal_vae, star_encoder, sdss_loader,
#             simulated_dataset.simulator, residual_optimizer, cycle = 1, n_epochs = 150)

resiudal_vae.load_state_dict(torch.load('./fits/residual_vae_wake1',
                               map_location=lambda storage, loc: storage))

run_sleep(resiudal_vae, star_encoder, simulated_loader, encoder_optimizer, cycle = 1)
