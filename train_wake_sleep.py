import numpy as np

import torch
import torch.optim as optim

import sdss_dataset_lib

import simulated_datasets_lib
import starnet_vae_lib
import inv_kl_objective_lib as inv_kl_lib

from wake_sleep_lib import run_joint_wake, run_wake, run_sleep

import psf_transform_lib

import time

import fitsio

import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)

print('torch version: ', torch.__version__)

# set seed
np.random.seed(34)
_ = torch.manual_seed(34)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# get sdss data
bands = [2, 3]
sdss_hubble_data = sdss_dataset_lib.SDSSHubbleData(sdssdir='../celeste_net/sdss_stage_dir/',
                                       hubble_cat_file = './hubble_data/NCG7089/' + \
                                        'hlsp_acsggct_hst_acs-wfc_ngc7089_r.rdviq.cal.adj.zpt.txt',
                                        bands = bands)

# sdss image
full_image = sdss_hubble_data.sdss_image.unsqueeze(0).to(device)
full_background = sdss_hubble_data.sdss_background.unsqueeze(0).to(device)

# simulated data parameters
with open('./data/default_star_parameters.json', 'r') as fp:
    data_params = json.load(fp)

print(data_params)

sky_intensity = full_background.reshape(full_background.shape[1], -1).mean(1)

# load psf
psf_dir = './data/'
psf_r = fitsio.FITS(psf_dir + 'sdss-002583-2-0136-psf-r.fits')[0].read()
psf_i = fitsio.FITS(psf_dir + 'sdss-002583-2-0136-psf-i.fits')[0].read()

psf_og = np.array([psf_r, psf_i])
# psf_og = np.array([psf_r])
# psf_og = np.loadtxt('./data/my_r_psf2.txt')[None]

# draw data
print('generating data: ')
n_images = 200
t0 = time.time()
star_dataset = \
    simulated_datasets_lib.load_dataset_from_params(psf_og,
                            data_params,
                            n_images = n_images,
                            sky_intensity = sky_intensity,
                            transpose_psf = False,
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
                                           n_bands = len(bands),
                                           max_detections = 2)

star_encoder.to(device)

# define psf transform
psf_transform = psf_transform_lib.PsfLocalTransform(torch.Tensor(psf_og).to(device),
									data_params['slen'],
									kernel_size = 3)
psf_transform.to(device)

# filename = './fits/debugging/still_debugging'
# filename = './fits/results_11122019/true_psf_encoder_630x310_r'
filename = './fits/results_11172019/wake-sleep_630x310_ri'
init_encoder = './fits/results_11172019/starnet_ri'
# init_encoder = './fits/results_11052019/starnet3'

# optimzers
psf_lr = 0.1
wake_optimizer = optim.Adam([
                    {'params': psf_transform.parameters(),
                    'lr': psf_lr},
                    {'params': star_encoder.enc_final.parameters(),
					'lr': 5e-5}],
                    weight_decay = 1e-5)
# wake_optimizer = optim.Adam([
#                     {'params': psf_transform.parameters(),
#                     'lr': psf_lr}],
#                     weight_decay = 1e-5)

sleep_optimizer = optim.Adam([
                    {'params': star_encoder.parameters(),
                    'lr': 5e-5}],
                    weight_decay = 1e-5)

for iteration in range(0, 6):
    #######################
    # wake phase training
    #######################
    print('RUNNING WAKE PHASE. ITER = ' + str(iteration))

    if iteration > 0:
        # load psf transform
        psf_transform_file = filename + '-psf_transform' + '-iter' + str(iteration - 1)
        print('loading psf_transform from: ', psf_transform_file)
        psf_transform.load_state_dict(torch.load(psf_transform_file,
                                    map_location=lambda storage, loc: storage))
        psf_transform.to(device)

        encoder_file = filename + '-encoder-iter' + str(iteration)
    else:
        encoder_file = init_encoder

    # load encoder
    print('loading encoder from: ', encoder_file)
    star_encoder.load_state_dict(torch.load(encoder_file,
                                   map_location=lambda storage, loc: storage));
    star_encoder.to(device);
    star_encoder.eval();

    # reset learning rate
    wake_optimizer.param_groups[0]['lr'] = psf_lr / (1 + 80 * iteration)
    run_wake(full_image, full_background, star_encoder, psf_transform,
                    optimizer = wake_optimizer,
                    n_epochs = 80,
                    n_samples = 50,
                    out_filename = filename,
                    iteration = iteration,
                    train_encoder_fluxes = True,
                    use_iwae = True)

    ########################
    # sleep phase training
    ########################
    print('RUNNING SLEEP PHASE. ITER = ' + str(iteration + 1))

    # load encoder
    # if iteration == 0:
    #     encoder_file = init_encoder
    # else:
    #     encoder_file = filename + '-encoder-iter' + str(iteration)
    encoder_file = filename + '-encoder-iter' + str(iteration)
    print('loading encoder from: ', encoder_file)
    star_encoder.load_state_dict(torch.load(encoder_file,
                                   map_location=lambda storage, loc: storage));
    star_encoder.to(device)

    # load trained transform
    psf_transform_file = filename + '-psf_transform' + '-iter' + str(iteration)
    # psf_transform_file = './fits/results_11122019/true_psf_transform_630x310_r'
    print('loading psf_transform from: ', psf_transform_file)
    psf_transform.load_state_dict(torch.load(psf_transform_file,
                                map_location=lambda storage, loc: storage));
    psf_transform.to(device)
    loader.dataset.simulator.psf = psf_transform.forward().detach()

    run_sleep(star_encoder,
                loader,
                sleep_optimizer,
                n_epochs = 11,
                out_filename = filename + '-encoder',
                iteration = iteration + 1)
