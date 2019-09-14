import numpy as np

import torch
import torch.optim as optim

import sdss_psf
import simulated_datasets_lib
import starnet_vae_lib
import objectives_lib

import time

import json

from torch.distributions import normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)

print('torch version: ', torch.__version__)

# load PSF
psf_fit_file = './../celeste_net/sdss_stage_dir/2566/6/65/psField-002566-6-0065.fit'
print('psf file: \n', psf_fit_file)

# set seed
np.random.seed(43534)
_ = torch.manual_seed(24534)

# data parameters
with open('./data/default_star_parameters.json', 'r') as fp:
    data_params = json.load(fp)

data_params['min_stars'] = 0
data_params['max_stars'] = 4

print(data_params)

# draw data
n_stars = 60000
star_dataset = \
    simulated_datasets_lib.load_dataset_from_params(psf_fit_file,
                            data_params,
                            n_stars = n_stars,
                            use_fresh_data = False,
                            add_noise = True)

# get loader
batchsize = 2048

loader = torch.utils.data.DataLoader(
                 dataset=star_dataset,
                 batch_size=batchsize,
                 shuffle=True)

# define VAE
star_encoder = starnet_vae_lib.StarEncoder(data_params['slen'],
                                                n_bands = 1,
                                                max_detections = data_params['max_stars'])

star_encoder.to(device)

# define optimizer
learning_rate = 1e-3
weight_decay = 1e-5
optimizer = optim.Adam([
                    {'params': star_encoder.parameters(),
                    'lr': learning_rate}],
                    weight_decay = weight_decay)


n_epochs = 500

for epoch in range(n_epochs):
    t0 = time.time()

    avg_loss, counter_loss = objectives_lib.eval_star_encoder_loss(star_encoder, loader,
                                                    optimizer, train = True)

    elapsed = time.time() - t0
    print('[{}] loss: {:0.4f}; counter loss: {:0.4f} \t[{:.1f} seconds]'.format(\
                    epoch, avg_loss, counter_loss, elapsed))

    # draw fresh data
    loader.dataset.set_params_and_images()

    if (epoch % 5) == 0:
        _, _ = \
            objectives_lib.eval_star_encoder_loss(star_encoder,
                                            loader, train = True)

        loader.dataset.set_params_and_images()
        test_loss, test_counter_loss = \
            objectives_lib.eval_star_encoder_loss(star_encoder,
                                            loader, train = False)

        print('**** test loss: {:.3f}; counter loss: {:.3f} ****'.format(test_loss, test_counter_loss))

        outfile = './fits/starnet_invKL_encoder_four_stars'
        print("writing the encoder parameters to " + outfile)
        torch.save(star_encoder.state_dict(), outfile)

print('done')
