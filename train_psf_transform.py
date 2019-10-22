import numpy as np

import torch
import torch.optim as optim

import sdss_dataset_lib
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

# get sdss data
sdss_hubble_data = sdss_dataset_lib.SDSSHubbleData(sdssdir='../celeste_net/sdss_stage_dir/',
					hubble_cat_file = './hubble_data/NCG7089/' + \
                                        'hlsp_acsggct_hst_acs-wfc_ngc7089_r.rdviq.cal.adj.zpt.txt',
					x0 = 650, x1 = 120)

# image
full_image = sdss_hubble_data.sdss_image.unsqueeze(0).to(device)
full_background = sdss_hubble_data.sdss_background.unsqueeze(0).to(device)

# true paramters
true_full_locs = sdss_hubble_data.locs.unsqueeze(0).to(device)
true_full_fluxes = sdss_hubble_data.fluxes.unsqueeze(0).to(device)

# simulator
simulator = simulated_datasets_lib.StarSimulator(
                    psf_fit_file=str(sdss_hubble_data.psf_file),
                    slen = full_image.shape[-1],
                    sky_intensity = 0.)

# define VAE
star_encoder = starnet_vae_lib.StarEncoder(full_slen = full_image.shape[-1],
                                           stamp_slen = 9,
                                           step = 2,
                                           edge_padding = 3,
                                           n_bands = 1,
                                           max_detections = 4)
encoder_file = './fits/starnet-10172019-no_reweighting'
star_encoder.load_state_dict(torch.load(encoder_file,
							   map_location=lambda storage, loc: storage));
star_encoder.to(device);
star_encoder.eval();

# We want to set batchnorm to eval
# https://discuss.pytorch.org/t/freeze-batchnorm-layer-lead-to-nan/8385
# def set_bn_eval(m):
#     classname = m.__class__.__name__
#     if classname.find('BatchNorm') != -1:
#       m.eval()
#
# star_encoder.apply(set_bn_eval)

# define transform
psf_transform = PsfLocalTransform(torch.Tensor(simulator.psf_og).to(device),
									simulator.slen,
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
	# locs = true_full_locs
	# fluxes = true_full_fluxes
	# n_stars = torch.sum(true_full_fluxes > 0, dim = 1)
	locs, fluxes, n_stars = \
		sample_star_encoder(star_encoder, full_image, full_background,
								n_samples = 100, return_map = False)

	psf_trained = psf_transform.forward()

	# get loss
	loss = get_psf_loss(full_image.squeeze(), full_background.squeeze(),
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

	    outfile = './fits/psf_transform-altm2-real_params-10222019'
	    print("writing the psf transform parameters to " + outfile)
	    torch.save(psf_transform.state_dict(), outfile)


print('done')
