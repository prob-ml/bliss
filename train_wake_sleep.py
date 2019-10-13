import numpy as np

import torch
import torch.optim as optim

import sdss_dataset_lib

import simulated_datasets_lib
import starnet_vae_lib
import inv_kl_objective_lib as inv_kl_lib

import psf_transform_lib

import time

import json

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

# sdss image
full_image = sdss_hubble_data.sdss_image.unsqueeze(0).to(device)
full_background = sdss_hubble_data.sdss_background.unsqueeze(0).to(device)

# true parameters
true_full_locs = sdss_hubble_data.locs.unsqueeze(0).to(device)
true_full_fluxes = sdss_hubble_data.fluxes.unsqueeze(0).to(device)

# simulated data parameters
with open('./data/default_star_parameters.json', 'r') as fp:
    data_params = json.load(fp)

data_params['slen'] = full_image.shape[-1]
data_params['min_stars'] = 2000
data_params['max_stars'] = 2000
data_params['alpha'] = 0.5

print(data_params)

# draw data
print('generating data: ')
n_images = 200
t0 = time.time()
star_dataset = \
    simulated_datasets_lib.load_dataset_from_params(str(sdss_hubble_data.psf_file),
                            data_params,
                            n_images = n_images,
                            add_noise = True)

print('data generation time: {:.3f}secs'.format(time.time() - t0))
# get loader
batchsize = 10

loader = torch.utils.data.DataLoader(
                 dataset=star_dataset,
                 batch_size=batchsize,
                 shuffle=True)

# define VAE
star_encoder = starnet_vae_lib.StarEncoder(full_slen = data_params['slen'],
                                           stamp_slen = 9,
                                           step = 2,
                                           edge_padding = 3,
                                           n_bands = 1,
                                           max_detections = 4)

star_encoder.to(device)

# load init
vae_file = './fits/starnet_invKL_encoder-10092019-reweighted_samples'
print('loading vae from: ', vae_file)
star_encoder.load_state_dict(torch.load(vae_file,
                               map_location=lambda storage, loc: storage))


# define psf transform
psf_transform = psf_transform_lib.PsfLocalTransform(torch.Tensor(loader.dataset.simulator.psf_og).to(device),
									data_params['slen'],
									kernel_size = 3)
psf_transform.to(device)

def run_sleep(star_encoder, loader, optimizer, n_epochs, filename, iteration):
    print_every = 10

    test_losses = np.zeros((4, n_epochs // print_every + 1))

    for epoch in range(n_epochs):
        t0 = time.time()

        avg_loss, counter_loss, locs_loss, fluxes_loss \
            = inv_kl_lib.eval_star_encoder_loss(star_encoder, loader,
                                                        optimizer, train = True)

        elapsed = time.time() - t0
        print('[{}] loss: {:0.4f}; counter loss: {:0.4f}; locs loss: {:0.4f}; fluxes loss: {:0.4f} \t[{:.1f} seconds]'.format(\
                        epoch, avg_loss, counter_loss, locs_loss, fluxes_loss, elapsed))

        # draw fresh data
        loader.dataset.set_params_and_images()

        if (epoch % print_every) == 0:
            _ = \
                inv_kl_lib.eval_star_encoder_loss(star_encoder,
                                                loader, train = True)

            loader.dataset.set_params_and_images()
            test_loss, test_counter_loss, test_locs_loss, test_fluxes_loss = \
                inv_kl_lib.eval_star_encoder_loss(star_encoder,
                                                loader, train = False)

            print('**** test loss: {:.3f}; counter loss: {:.3f}; locs loss: {:.3f}; fluxes loss: {:.3f} ****'.format(\
                test_loss, test_counter_loss, test_locs_loss, test_fluxes_loss))

            outfile = filename + '-iter' + str(iteration)
            print("writing the encoder parameters to " + outfile)
            torch.save(star_encoder.state_dict(), outfile)

            test_losses[:, epoch // print_every] = np.array([test_loss, test_counter_loss, test_locs_loss, test_fluxes_loss])
            np.savetxt(filename + '-test_losses-' + 'iter' + str(iteration),
                        test_losses)

def run_wake(star_encoder, psf_transform, simulator, optimzer,
                n_epochs, filename, iteration):

    for epoch in range(n_epochs):
    	t0 = time.time()

    	optimizer.zero_grad()

    	# get params: these normally would be the variational parameters.
    	# using true parameters atm
    	_, subimage_locs, subimage_fluxes, _, _ = \
    		star_encoder.get_image_stamps(full_image, true_full_locs, true_full_fluxes,
    										trim_images = False)

    	# get loss
    	loss = get_psf_transform_loss(full_image, full_background,
    	                            subimage_locs,
    	                            subimage_fluxes,
    	                            star_encoder.tile_coords,
    	                            star_encoder.stamp_slen,
    	                            star_encoder.edge_padding,
    	                            simulator,
    	                            psf_transform)[1]

    	avg_loss = loss.mean()

    	avg_loss.backward()
    	optimizer.step()

    	elapsed = time.time() - t0
    	print('[{}] loss: {:0.4f} \t[{:.1f} seconds]'.format(\
    	                epoch, avg_loss, elapsed))

    	if (epoch % print_every) == 0:
    	    outfile = filename + '-iter' + str(iteration)
    	    print("writing the psf parameters to " + outfile)
    	    torch.save(psf_transform.state_dict(), outfile)


# define optimizer
learning_rate = 5e-4
weight_decay = 1e-5
vae_optimizer = optim.Adam([
                    {'params': star_encoder.parameters(),
                    'lr': learning_rate}],
                    weight_decay = weight_decay)



print('running sleep. Loading psf transform from')
# load trained transform
psf_transform.load_state_dict(torch.load('./fits/psf_transform-real_params-10112019',
                                         map_location=lambda storage, loc: storage))
loader.dataset.simulator.psf = psf_transform.forward().detach()
run_sleep(star_encoder,
            loader,
            vae_optimizer,
            n_epochs = 101,
            filename = 'starnet_invKL_encoder-trained_psf-10122019',
            iteration = 0)
