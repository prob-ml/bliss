import torch
from torch import nn
from torch.distributions.normal import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import pytest

from bliss import wake
import pytorch_lightning as pl

from copy import deepcopy

class TestWake:
    @pytest.fixture(scope="class")
    def overrides(self, devices):
        overrides = dict(
            model="sleep_star_basic",
            dataset="default" if devices.use_cuda else "cpu",
            training="unittest" if devices.use_cuda else "cpu",
        )
        return overrides

    @pytest.fixture(scope="class")
    def trained_sleep(self, overrides, sleep_setup):
        return sleep_setup.get_trained_sleep(overrides)

    def test_simulated(self, overrides, trained_sleep, sleep_setup, devices):
        overrides.update({"testing": "default"})
        
        # the original decoder
        image_decoder = trained_sleep.image_decoder.to(device)
        
        # draw some catalogs from the prior
        torch.manual_seed(23421)
        batch = trained_sleep.image_decoder.sample_prior(batch_size=100)
        
        # pick the catalog with the most stars. 
        # we will use this to construct an image
        which_batch = batch['n_sources'].sum(1).argmax()
        batch_i = dict()
        for key in batch.keys(): 
            if key != 'slen': 
                batch_i[key] = batch[key][which_batch:(which_batch+1)]
        
        # the "observed" image
        obs_image = image_decoder.render_images(batch_i['n_sources'], 
                                           batch_i['locs'], 
                                           batch_i['galaxy_bool'] * 0., 
                                           batch_i['galaxy_params'], 
                                           batch_i['fluxes'])
        
        # now perturb psf parameters and get new decoder
        image_decoder_perturbed = deepcopy(image_decoder)
        true_psf_params = image_decoder.params
        psf_params_perturbed = true_psf_params * (1.1)
        image_decoder_perturbed.params = nn.Parameter(psf_params_perturbed)
        psf_init = image_decoder_perturbed.forward().detach()
        
        def eval_decoder_loss(image_decoder): 
            # evaluate loss of a decoder at the **true** catalog
            recon = image_decoder.render_images(batch_i['n_sources'], 
                                               batch_i['locs'], 
                                               batch_i['galaxy_bool'] * 0., 
                                               batch_i['galaxy_params'], 
                                               batch_i['fluxes'], 
                                               add_noise = False)
            loss = -Normal(recon, recon.sqrt()).log_prob(obs_image).mean()
            return loss
        
        # loss of true decoder
        target_loss = eval_decoder_loss(image_decoder)

        # loss of perurbed decoder
        init_loss = eval_decoder_loss(image_decoder_perturbed)
        assert (init_loss - target_loss) > 0
        
        # define wake-phase and train starting from the perturbed decoder
        wake_net = wake.WakeNet(trained_sleep.image_encoder,
                                image_decoder_perturbed,
                                obs_image,
                                torch.Tensor([686.]),
                                hparams = dict({'lr':1e-5, 
                                           'n_samples':2000}));
        wake_net.to(device);
        
        n_epochs = 3000 if torch.cuda.is_available() else 2
        wake_trainer = pl.Trainer(check_val_every_n_epoch = 10000, 
                                     max_epochs = n_epochs, 
                                     min_epochs = n_epochs)
        
        wake_trainer.fit(wake_net)
        
        # loss after training
        trained_loss = eval_decoder_loss(wake_net.image_decoder)
        
        # check loss went down
        print(target_loss)
        print(init_loss)
        print(trained_loss)
        diff0 = init_loss - target_loss
        diff1 = trained_loss - target_loss
        assert diff1 < (diff0 * 0.75)
        
        
        # now compare PSFs
        psf_true = image_decoder.forward().detach()
        psf_fitted = wake_net.image_decoder.forward().detach()
        
        init_psf_mse = ((psf_init - psf_true)**2).mean()
        trained_psf_mse = ((psf_fitted - psf_true)**2).mean()
        
        # check if mse of psf improved
        print(init_psf_mse)
        print(trained_psf_mse)
        assert trained_psf_mse < (init_psf_mse * 0.75)
        
# def test_star_wake(sleep_setup, paths, devices):
#     device = devices.device
#     overrides = dict(model="sleep_star_one_tile", training="cpu", dataset="cpu")
#     sleep_net = sleep_setup.get_trained_sleep(overrides)

#     # load the test image
#     test_path = paths["data"].joinpath("star_wake_test1.pt")
#     test_star = torch.load(test_path, map_location="cpu")
#     test_image = test_star["images"][0].unsqueeze(0).to(device)
#     test_slen = test_star["slen"].item()
#     image_decoder = sleep_net.image_decoder.to(device)
#     background_value = image_decoder.background.mean().item()

#     # initialize background params, which will create the true background
#     init_background_params = torch.zeros(1, 3, device=device)
#     init_background_params[0, 0] = background_value

#     n_samples = 1
#     hparams = {"n_samples": n_samples, "lr": 0.001}
#     assert image_decoder.slen == test_slen
#     wake_phase_model = wake.WakeNet(
#         sleep_net.image_encoder,
#         image_decoder,
#         test_image,
#         init_background_params,
#         hparams,
#     )

#     # run the wake-phase training
#     n_epochs = 1

#     wake_trainer = pl.Trainer(
#         gpus=devices.gpus,
#         profiler=None,
#         logger=False,
#         checkpoint_callback=False,
#         min_epochs=n_epochs,
#         max_epochs=n_epochs,
#         reload_dataloaders_every_epoch=True,
#     )

#     wake_trainer.fit(wake_phase_model)
