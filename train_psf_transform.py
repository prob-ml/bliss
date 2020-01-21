import numpy as np

import torch
import torch.optim as optim

import sdss_dataset_lib
import simulated_datasets_lib
import starnet_vae_lib
from psf_transform_lib import PsfLocalTransform, get_psf_loss

import time

import fitsio

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
					bands = [2])

# image
full_image = sdss_hubble_data.sdss_image.unsqueeze(0).to(device)
full_background = sdss_hubble_data.sdss_background.unsqueeze(0).to(device)

# true paramters
true_full_locs = sdss_hubble_data.locs.unsqueeze(0).to(device)
true_full_fluxes = sdss_hubble_data.fluxes.unsqueeze(0).to(device)

# load psf
psf_dir = './data/'
psf_r = fitsio.FITS(psf_dir + 'sdss-002583-2-0136-psf-r.fits')[0].read()
psf_i = fitsio.FITS(psf_dir + 'sdss-002583-2-0136-psf-i.fits')[0].read()

psf_og = np.array([psf_r])

# define transform
psf_transform = PsfLocalTransform(torch.Tensor(psf_og).to(device),
									full_image.shape[-1],
									kernel_size = 3)
psf_transform.to(device)
# define optimizer
learning_rate = 0.5
weight_decay = 1e-5
optimizer = optim.Adam([
                    {'params': psf_transform.parameters(),
                    'lr': learning_rate}],
                    weight_decay = weight_decay)


n_epochs = 101
print_every = 10
print('training')

test_losses = np.zeros(n_epochs)

cached_grid = simulated_datasets_lib._get_mgrid(full_image.shape[-1]).to(device)

for epoch in range(n_epochs):
	t0 = time.time()

	optimizer.zero_grad()

	# get params: these normally would be the variational parameters.
	# using true parameters atm
	locs = true_full_locs
	fluxes = true_full_fluxes
	n_stars = torch.sum(true_full_fluxes[:, :, 0] > 0, dim = 1);

	psf_trained = psf_transform.forward()

	# get loss
	loss = get_psf_loss(full_image, full_background,
	                    locs, fluxes, n_stars,
						psf_trained,
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
	    outfile = './fits/results_11202019/true_psf_transform_630x310_r'
	    print("writing the psf transform parameters to " + outfile)
	    torch.save(psf_transform.state_dict(), outfile)

print('done')
