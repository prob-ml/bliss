import numpy as np

import torch
import torch.optim as optim

import sdss_dataset_lib
import residuals_vae_lib
import simulated_datasets_lib

import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)

print('torch version: ', torch.__version__)

# set seed
np.random.seed(43534)
_ = torch.manual_seed(24534)

# Get Hubble data
hubble_cat_file='./hubble_data/NCG7078/hlsp_acsggct_hst_acs-wfc_ngc7078_r.rdviq.cal.adj.zpt.txt'
sdss_hubble_data = sdss_dataset_lib.SDSSHubbleData(hubble_cat_file=hubble_cat_file,
                                                    sdssdir = './../celeste_net/sdss_stage_dir/',
                                                   slen = 11,
                                                   run = 2566,
                                                   camcol = 6,
                                                   field = 65,
                                                max_detections = 20)

# get simulator
sky_intensity = sdss_hubble_data.sdss_background_full.mean()

simulator = simulated_datasets_lib.StarSimulator(psf_fit_file=sdss_hubble_data.psf_file,
                                    slen = sdss_hubble_data.slen,
                                    sky_intensity = sky_intensity)
# get loader
batchsize = 64
loader = torch.utils.data.DataLoader(
                 dataset=sdss_hubble_data,
                 batch_size=batchsize,
                 shuffle=True)

# define VAE
resid_vae = residuals_vae_lib.ResidualVAE(slen = sdss_hubble_data.slen,
                                            n_bands = 1,
                                            f_min = 1300.)

resid_vae.to(device)

# define optimizer
learning_rate = 1e-3
weight_decay = 1e-5
optimizer = optim.Adam([
                    {'params': resid_vae.parameters(),
                    'lr': learning_rate}],
                    weight_decay = weight_decay)


n_epochs = 100

for epoch in range(n_epochs):
    t0 = time.time()

    avg_loss = \
        residuals_vae_lib.eval_residual_vae(resid_vae, loader, simulator,
                                            optimizer, train = True)

    elapsed = time.time() - t0
    print('[{}] loss: {:0.4f}; \t[{:.1f} seconds]'.format(\
                    epoch, avg_loss, elapsed))

    if (epoch % 5) == 0:

        test_loss = \
            residuals_vae_lib.eval_residual_vae(resid_vae, loader, simulator,
                                                optimizer = None, train = False)

        print('**** test loss: {:.3f} ****'.format(test_loss))

        outfile = './fits/residual_vae'
        print("writing the vae parameters to " + outfile)
        torch.save(resid_vae.state_dict(), outfile)


print('done')
