from copy import deepcopy

import pytest
import pytorch_lightning as pl
import torch
from torch import nn
from torch.distributions.normal import Normal

from bliss import wake


class TestWake:
    @pytest.fixture(scope="class")
    def overrides(self, devices):
        return {
            "model": "sleep_star_basic",
            "dataset": "default" if devices.use_cuda else "cpu",
            "training": "unittest" if devices.use_cuda else "cpu",
            "optimizer": "m2",
        }

    @pytest.fixture(scope="class")
    def trained_sleep(self, overrides, model_setup):
        return model_setup.get_trained_model(overrides)

    def test_simulated(self, trained_sleep, devices):

        # the original decoder
        image_decoder = trained_sleep.image_decoder.to(devices.device)

        # draw some catalogs from the prior
        torch.manual_seed(23421)
        batch = image_decoder.sample_prior(batch_size=100)

        # pick the catalog with the most stars.
        # this will be the ground truth catalog.
        # we will use this to construct an image
        which_batch = batch["n_sources"].sum(1).argmax()
        true_params = {}
        for key in batch.keys():
            if key != "slen":
                true_params[key] = batch[key][which_batch : (which_batch + 1)]

        # the observed image
        obs_image, _ = image_decoder.render_images(
            true_params["n_sources"],
            true_params["locs"],
            true_params["galaxy_bool"],
            true_params["galaxy_params"],
            true_params["fluxes"],
        )

        # now perturb psf parameters and get new decoder
        image_decoder_perturbed = deepcopy(image_decoder)
        true_psf_params = image_decoder.star_tile_decoder.params
        psf_params_perturbed = true_psf_params * 1.1
        image_decoder_perturbed.star_tile_decoder.params = nn.Parameter(psf_params_perturbed)
        psf_init = image_decoder_perturbed.forward().detach()

        def eval_decoder_loss(decoder):
            # evaluate loss of a decoder at the true catalog
            recon, _ = decoder.render_images(
                true_params["n_sources"],
                true_params["locs"],
                true_params["galaxy_bool"],
                true_params["galaxy_params"],
                true_params["fluxes"],
                add_noise=False,
            )
            return -Normal(recon, recon.sqrt()).log_prob(obs_image).mean()

        # loss of true decoder
        target_loss = eval_decoder_loss(image_decoder)

        # loss of perturbed decoder
        init_loss = eval_decoder_loss(image_decoder_perturbed)
        assert (init_loss - target_loss) > 0

        # the trained encoder:
        # note that this encoder was trained using the
        # "true" decoder (trained using image_decoder,
        # not image_decoder_perturbed)
        trained_encoder = trained_sleep.image_encoder

        wake_net = wake.WakeNet(
            trained_encoder,
            image_decoder_perturbed,
            obs_image,
            hparams=dict({"lr": 1e-5, "n_samples": 2000}),
        ).to(devices.device)

        n_epochs = 3000 if devices.use_cuda else 2
        wake_trainer = pl.Trainer(
            check_val_every_n_epoch=10000,
            max_epochs=n_epochs,
            min_epochs=n_epochs,
            gpus=devices.gpus,
            logger=False,
            checkpoint_callback=False,
            deterministic=True,
        )

        wake_trainer.fit(wake_net)

        # loss after training
        trained_loss = eval_decoder_loss(wake_net.image_decoder.to(devices.device))

        # check loss went down
        print("target loss: ", target_loss)
        print("initial loss: ", init_loss)
        print("trained loss: ", trained_loss)
        diff0 = init_loss - target_loss
        diff1 = trained_loss - target_loss
        if devices.use_cuda:
            assert diff1 < (diff0 * 0.5)

        # now compare PSFs
        psf_true = image_decoder.forward().detach()
        psf_fitted = wake_net.image_decoder.forward().detach()

        init_psf_mse = ((psf_init - psf_true) ** 2).mean()
        trained_psf_mse = ((psf_fitted - psf_true) ** 2).mean()

        # check if mse of psf improved
        print("initial psf mse: ", init_psf_mse)
        print("trained psf mse: ", trained_psf_mse)
        if devices.use_cuda:
            assert trained_psf_mse < (init_psf_mse * 0.5)
