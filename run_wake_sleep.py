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

resid_vae = residuals_vae_lib.ResidualVAE(slen = sdss_hubble_data.slen,
                                            n_bands = 1,
                                            f_min = 2000.)

star_encoder.to(device)
resid_vae.to(device)

#####################
# Define losses
#####################
def eval_wake_loss(residual_vae, star_encoder, loader, simulator,
                optimizer = None, train = False):

    avg_loss = 0.0

    for _, data in enumerate(loader):
        # infer parameters
        images = data['image'].to(device)
        backgrounds = data['background'].to(device)

        logit_loc_mean, logit_loc_logvar, \
            log_flux_mean, log_flux_logvar, log_probs = \
                    star_encoder(images, backgrounds)

        locs = torch.sigmoid(\
            residuals_vae_lib.sample_normal(logit_loc_mean.detach(),
                                            logit_loc_logvar.detach()))
        fluxes = torch.exp(\
            residuals_vae_lib.sample_normal(log_flux_mean.detach(),
                                            log_flux_logvar.detach()))
        n_stars = torch.multinomial(torch.exp(log_probs), num_samples = 1).squeeze()

        # reconstruct image
        simulated_images = simulator.draw_image_from_params(locs, fluxes, n_stars,
                                add_noise = False)

        if train:
            residual_vae.train()
            assert optimizer is not None
            optimizer.zero_grad()
        else:
            residual_vae.eval()

        # get loss
        loss = residuals_vae_lib.get_resid_vae_loss(images, simulated_images, residual_vae)

        if train:
            (loss / images.shape[0]).backward()
            optimizer.step()

        avg_loss += loss.item() / len(loader.dataset)

    return avg_loss

def run_wake(residual_vae, star_encoder, loader,
                simulator, optimizer, cycle):

    n_epochs = 500

    for epoch in range(n_epochs):
        t0 = time.time()

        avg_loss = eval_wake_loss(residual_vae, star_encoder, loader, simulator,
                                    optimizer = optimizer, train = True)

        elapsed = time.time() - t0
        print('[{}] loss: {:.6E}; \t[{:.1f} seconds]'.format(\
                        epoch, avg_loss, elapsed))

        if (epoch % 5) == 0:

            test_loss = eval_wake_loss(residual_vae, star_encoder, loader, simulator,
                                        optimizer = None, train = False)

            print('**** test loss: {:.6E} ****'.format(test_loss))

            outfile = './fits/residual_vae_wake' + str(cycle)
            print("writing the residual vae parameters to " + outfile)
            torch.save(resid_vae.state_dict(), outfile)

##############################
# Train!
##############################
learning_rate = 1e-3
weight_decay = 1e-5
optimizer = optim.Adam([
                    {'params': resid_vae.parameters(),
                    'lr': learning_rate}],
                    weight_decay = weight_decay)

run_wake(resid_vae, star_encoder, sdss_loader,
            simulated_dataset.simulator, optimizer, cycle = 1)
