import numpy as np
import timeit

import torch
import torch.optim as optim

import sdss_psf
import star_datasets_lib
import starnet_vae_lib

import time

import json

from torch.distributions import normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('torch version: ', torch.__version__)

# load PSF
psf_fit_file = './sdss_stage_dir/3900/6/269/psField-003900-6-0269.fit'
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
n_stars = 2048
star_dataset = \
    star_datasets_lib.load_dataset_from_params(psf_fit_file,
                            data_params,
                            n_stars = n_stars,
                            use_fresh_data = True,
                            add_noise = True)

# get loader
batchsize = n_stars

loader = torch.utils.data.DataLoader(
                 dataset=star_dataset,
                 batch_size=batchsize,
                 shuffle=True)

# define rnn
star_rnn = starnet_vae_lib.StarRNN(\
                n_bands=1, slen=data_params['slen']).to(device)

# define optimizer
learning_rate = 1e-3
weight_decay = 1e-5
optimizer = optim.Adam([
                    {'params': star_rnn.parameters(),
                    'lr': learning_rate}],
                    weight_decay = weight_decay)



# loss function
def get_loss():
    avg_loss = 0.

    for _, data in enumerate(loader):
        # get data
        true_fluxes = data['fluxes'][:, 0]
        true_locs = data['locs'][:, 0, :]
        true_n_stars = data['n_stars']
        images = data['image']

        # optimizer
        optimizer.zero_grad()

        # forward
        logit_locs_mean, logit_locs_logvar, \
            log_flux_mean, log_flux_logvar = \
                star_rnn.forward_once(images, \
                                        h_i = torch.zeros(images.shape[0], 180))

        # get loss
        logit_locs_q = normal.Normal(loc = logit_locs_mean, \
                                    scale = torch.exp(0.5 * logit_locs_logvar))
        log_flux_q = normal.Normal(loc = log_flux_mean, \
                                    scale = torch.exp(0.5 * log_flux_logvar))

        loss = -logit_locs_q.log_prob(true_locs).sum(dim = 1) - \
                    log_flux_q.log_prob(true_fluxes)

        loss.mean().backward()
        optimizer.step()

        avg_loss += loss.sum() / loader.dataset.n_images

    return avg_loss

print('training')

for epoch in range(200):

    t0 = time.time()
    avg_loss = get_loss()
    elapsed = time.time() - t0

    print('[{}] loss: {:0.4f} \t[{:.1f} seconds]'.format(\
                    epoch, avg_loss, elapsed))

    torch.save(star_rnn.state_dict(), './fits/test_fit_one_detection')

print('done')
