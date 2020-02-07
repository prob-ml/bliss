import numpy as np

import torch
import torch.optim as optim

import sdss_dataset_lib

import simulated_datasets_lib
import starnet_vae_lib
import inv_kl_objective_lib as inv_kl_lib

import wake_lib

import psf_transform_lib
import psf_transform_lib2

import time

import fitsio

import json

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print('device: ', device)

print('torch version: ', torch.__version__)

#######################
# set seed
########################
np.random.seed(34532534)
_ = torch.manual_seed(5435)
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
# psf_dir = './data/'
# psf_r = fitsio.FITS(psf_dir + 'sdss-002583-2-0136-psf-r.fits')[0].read()
# psf_i = fitsio.FITS(psf_dir + 'sdss-002583-2-0136-psf-i.fits')[0].read()
# psf_og = np.array([psf_r, psf_i])

bands = [2, 3]
psfield_file = './../celeste_net/sdss_stage_dir/2583/2/136/psField-002583-2-0136.fit'
powerlaw_psf_params = torch.zeros(len(bands), 6).to(device)
for i in range(len(bands)):
    powerlaw_psf_params[i] = psf_transform_lib2.get_psf_params(
                                    psfield_file,
                                    band = bands[i])
power_law_psf = psf_transform_lib2.PowerLawPSF(powerlaw_psf_params)
psf_og = power_law_psf.forward().detach()

###############
# sky intensity: for the r and i band
###############
planar_background_params = torch.zeros(len(bands), 3).to(device)
planar_background_params[:, 0] = torch.Tensor([686., 1123.])
planar_background = wake_lib.PlanarBackground(image_slen = data_params['slen'],
                            init_background_params = planar_background_params)

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
                                           estimate_flux = False)

init_encoder = './fits/results_2020-02-06/starnet_ri'
star_encoder.load_state_dict(torch.load(init_encoder,
                                   map_location=lambda storage, loc: storage))
star_encoder.to(device)

####################
# optimzers
#####################
encoder_lr = 1e-5
sleep_optimizer = optim.Adam([
                    {'params': star_encoder.parameters(),
                    'lr': encoder_lr}],
                    weight_decay = 1e-5)

# initial loss:
test_loss, test_counter_loss, test_locs_loss, test_fluxes_loss = \
    inv_kl_lib.eval_star_encoder_loss(star_encoder,
                                    loader, train = False)

print('**** INIT test loss: {:.3f}; counter loss: {:.3f}; locs loss: {:.3f}; fluxes loss: {:.3f} ****'.format(\
    test_loss, test_counter_loss, test_locs_loss, test_fluxes_loss))

# file header to save results
filename = './fits/results_2020-02-06/wake-sleep_630x310_ri'

for iteration in range(0, 6):
    #######################
    # wake phase training
    #######################
    print('RUNNING WAKE PHASE. ITER = ' + str(iteration))
    if iteration == 0:
        encoder_file = init_encoder
    else:
        encoder_file = filename + '-encoder-iter' + str(iteration)
        powerlaw_psf_params = \
            torch.Tensor(np.load('./fits/results_2020-02-06/powerlaw_psf_params-iter' + \
                                    str(iteration - 1) + '.npy')).to(device)
        planar_background_params = \
            torch.Tensor(np.load('./fits/results_2020-02-06/planarback_params-iter' + \
                                    str(iteration - 1) + '.npy')).to(device)

    print('loading encoder from: ', encoder_file)
    star_encoder.load_state_dict(torch.load(encoder_file,
                                   map_location=lambda storage, loc: storage))
    star_encoder.to(device)

    map_locs_full_image, _, map_n_stars_full = \
        star_encoder.sample_star_encoder(full_image,
                                            torch.ones(full_image.shape).to(device),
                                            return_map_n_stars = True,
                                            return_map_star_params = True)[0:3]

    estimator = wake_lib.EstimateModelParams(full_image,
                                            map_locs_full_image,
                                            map_n_stars_full,
                                            init_psf_params = powerlaw_psf_params,
                                            init_background_params = planar_background_params,
                                            init_fluxes = None,
                                            fmin = data_params['f_min'])
    estimator.run_coordinate_ascent()

    np.save('../fits/results_2020-02-06/powerlaw_psf_params-iter' + str(iteration),
        list(estimator.power_law_psf.parameters())[0].data.numpy())
    np.save('../fits/results_2020-02-06/planarback_params-iter' + str(iteration),
        list(estimator.planar_background.parameters())[0].data.numpy())

    ########################
    # sleep phase training
    ########################
    print('RUNNING SLEEP PHASE. ITER = ' + str(iteration + 1))

    # update psf
    loader.dataset.simulator.psf = estimator.get_psf().detach()
    loader.dataset.simulator.background = estimator.get_background().squeeze(0).detach()

    run_sleep(star_encoder,
                loader,
                sleep_optimizer,
                n_epochs = 11,
                out_filename = filename + '-encoder',
                iteration = iteration + 1)
