import numpy as np

import torch
import torch.optim as optim

import json
import sdss_dataset_lib
import sdss_psf
import simulated_datasets_lib
import starnet_vae_lib
from psf_transform_lib import PsfLocalTransform, get_psf_loss

import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)

print('torch version: ', torch.__version__)

# set seed
np.random.seed(4534)
_ = torch.manual_seed(2534)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# data parameters
with open('./data/default_star_parameters.json', 'r') as fp:
    data_params = json.load(fp)

use_real_data = True
if use_real_data:
	# get sdss data
	sdss_hubble_data = sdss_dataset_lib.SDSSHubbleData(sdssdir='../celeste_net/sdss_stage_dir/',
						hubble_cat_file = './hubble_data/NCG7089/' + \
	                                        'hlsp_acsggct_hst_acs-wfc_ngc7089_r.rdviq.cal.adj.zpt.txt')

	# load psf
	psf_og = sdss_psf.psf_at_points(0, 0, psf_fit_file = str(sdss_hubble_data.psf_file))

	psf = torch.Tensor(simulated_datasets_lib._expand_psf(psf_og, sdss_hubble_data.slen)).to(device)

	# image
	sdss_image = sdss_hubble_data.sdss_image.unsqueeze(0).to(device)
	sdss_background = sdss_hubble_data.sdss_background.unsqueeze(0).to(device) * 0.0 + 179.

else:
	print('simulating data; data params: ')
	print(data_params)
	psf_fit_file = './../celeste_net/sdss_stage_dir/2583/2/136/psField-002583-2-0136.fit'

	n_images = 20
	star_dataset = \
	    simulated_datasets_lib.load_dataset_from_params(psf_fit_file,
	                            data_params,
	                            n_images = n_images,
	                            add_noise = True)

	psf = star_dataset.simulator.psf.to(device)


# define encoder
star_encoder = starnet_vae_lib.StarEncoder(full_slen = data_params['slen'],
                                            stamp_slen = 7,
                                            step = 2,
                                            edge_padding = 2,
                                            n_bands = 1,
                                            max_detections = 2)

star_encoder.load_state_dict(torch.load('./fits/results_11042019/starnet-11042019',
                               map_location=lambda storage, loc: storage))

star_encoder.to(device)
# define optimizer
weight_decay = 1e-5
optimizer = optim.Adam([
					{'params': star_encoder.enc_final.parameters(),
					'lr': 5e-5}],
                    weight_decay = weight_decay)


n_epochs = 1001
print_every = 10
print('training')

test_losses = np.zeros(n_epochs)

cached_grid = simulated_datasets_lib._get_mgrid(star_encoder.full_slen).to(device)

for epoch in range(n_epochs):
	t0 = time.time()

	optimizer.zero_grad()

	if use_real_data:
		full_image = sdss_image
		full_background = sdss_background
	else:
		star_dataset.set_params_and_images()
		full_image = star_dataset.images
		full_background = torch.ones(star_dataset.images.shape).to(device) * star_dataset.sky_intensity

	# get params
	locs, fluxes, n_stars = \
		star_encoder.sample_star_encoder(full_image,
			                                full_background,
			                                n_samples = 100,
			                                return_map = False,
			                                return_log_q = False,
			                                training = True)[0:3]

	# get loss
	loss = get_psf_loss(full_image, full_background,
	                    locs.detach(), fluxes, n_stars.detach(),
						psf,
	                    pad = 5,
	                    grid = cached_grid)[1]

	avg_loss = loss.mean()

	avg_loss.backward()
	optimizer.step()

	elapsed = time.time() - t0
	print('[{}] loss: {:0.4f} \t[{:.1f} seconds]'.format(\
	                epoch, avg_loss, elapsed))

	test_losses[epoch] = avg_loss

	if (epoch % print_every) == 0:
		outfile = './fits/results_11042019/starnet_testing-11042019';
		print("writing the encoder parameters to " + outfile);
		torch.save(star_encoder.state_dict(), outfile)

print('done')
