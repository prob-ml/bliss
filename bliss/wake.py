import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
import pytorch_lightning as pl

import numpy as np

from .models import encoder
from .models.decoder import get_mgrid
from . import device


# workaround for avoiding error of profile decorator
if type(__builtins__) is not dict or "profile" not in __builtins__:
    profile = lambda f: f


def _fit_plane_to_background(background):
    assert len(background.shape) == 3
    n_bands = background.shape[0]
    slen = background.shape[-1]

    planar_params = np.zeros((n_bands, 3))
    for i in range(n_bands):
        # can we make numpy to torch?
        y = background[i].flatten().detach().cpu().numpy()
        grid = get_mgrid(slen).detach().cpu().numpy()

        x = np.ones((slen ** 2, 3))
        x[:, 1:] = np.array(
            [grid[:, :, 0].flatten(), grid[:, :, 1].flatten()]
        ).transpose()

        xtx = np.einsum("ki, kj -> ij", x, x)
        xty = np.einsum("ki, k -> i", x, y)

        planar_params[i, :] = np.linalg.solve(xtx, xty)

    return planar_params


class PlanarBackground(nn.Module):
    def __init__(self, init_background_params, image_slen=101):
        super(PlanarBackground, self).__init__()

        assert len(init_background_params.shape) == 2
        self.n_bands = init_background_params.shape[0]

        self.init_background_params = init_background_params.clone()

        self.image_slen = image_slen

        # get grid
        # can we use cached grid to replace?
        _mgrid = get_mgrid(image_slen).to(device)
        self.mgrid = torch.stack([_mgrid for _ in range(self.n_bands)], dim=0)

        # initial weights
        # why do we clone the parameter here?
        self.params = nn.Parameter(init_background_params.clone())

    def forward(self):
        return (
            self.params[:, 0][:, None, None]
            + self.params[:, 1][:, None, None] * self.mgrid[:, :, :, 0]
            + self.params[:, 2][:, None, None] * self.mgrid[:, :, :, 1]
        )


class WakeNet(pl.LightningModule):

    # ---------------
    # Model
    # ----------------

    def __init__(
        self,
        star_encoder,
        image_decoder,
        observed_img,
        init_background_params,
        hparams,
        pad=0,
    ):
        super(WakeNet, self).__init__()

        self.star_encoder = star_encoder
        self.image_decoder = image_decoder

        # observed image is batch_size (or 1) x n_bands x slen x slen
        assert len(observed_img.shape) == 4
        self.observed_img = observed_img
        self.pad = pad

        # hyper-parameters
        self.hparams = hparams
        self.n_samples = self.hparams["n_samples"]
        self.lr = self.hparams["lr"]
        self.slen = observed_img.shape[-1]
        assert self.image_decoder.slen == self.slen, "cached grid won't match."

        # get n_bands
        self.n_bands = self.image_decoder.n_bands

        # set up initial background parameters
        assert init_background_params.shape[0] == self.n_bands
        self.init_background_params = init_background_params
        self.planar_background = PlanarBackground(init_background_params, self.slen)

        # self.init_background = self.planar_background.forward()

    @profile
    def forward(self, obs_img, run_map=False):

        with torch.no_grad():
            self.star_encoder.eval()
            (
                n_stars_sampled,
                locs_sampled,
                galaxy_params_sampled,
                log_fluxes_sampled,
                galaxy_bool_sampled,
            ) = self.star_encoder.sample_encoder(
                obs_img,
                n_samples=self.n_samples,
                return_map_n_sources=run_map,
                return_map_source_params=run_map,
            )

        max_stars = log_fluxes_sampled.shape[1]
        is_on_array = encoder.get_is_on_from_n_sources(n_stars_sampled, max_stars)
        is_on_array = is_on_array.unsqueeze(-1).float()
        fluxes_sampled = log_fluxes_sampled.exp() * is_on_array

        background = self.planar_background.forward().unsqueeze(0).detach()
        stars = self.image_decoder.render_multiple_stars(
            n_stars_sampled, locs_sampled, fluxes_sampled,
        )

        recon_mean = stars + background

        return recon_mean

    # ---------------
    # Data
    # ----------------

    def train_dataloader(self):
        return DataLoader(self.observed_img, batch_size=None)

    def val_dataloader(self):
        return DataLoader(self.observed_img, batch_size=None)

    # ---------------
    # Optimizer
    # ----------------

    def configure_optimizers(self):
        return optim.Adam(
            [{"params": self.image_decoder.power_law_psf.parameters(), "lr": self.lr}]
        )

    # ---------------
    # Training
    # ----------------
    @profile
    def training_step(self, batch, batch_idx):
        img = batch.unsqueeze(0)
        recon_mean = self.forward(img)
        error = -Normal(recon_mean, recon_mean.sqrt()).log_prob(img)

        last = self.slen - self.pad
        loss = error[:, :, self.pad : last, self.pad : last].sum((1, 2, 3)).mean()
        logs = {"train_loss": loss}

        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        img = batch.unsqueeze(0)
        recon_mean = self.forward(img)
        error = -Normal(recon_mean, recon_mean.sqrt()).log_prob(img)

        last = self.slen - self.pad
        loss_val = error[:, :, self.pad : last, self.pad : last].sum((1, 2, 3)).mean()

        return {"val_loss": loss_val}

    def validation_epoch_end(self, outputs):
        return {"val_loss": outputs[-1]["val_loss"]}

    def _get_init_background(self, sample_every=25):
        sampled_background = self._sample_image(sample_every)
        self.init_background_params = torch.tensor(
            _fit_plane_to_background(sampled_background)
        ).to(device)
        self.planar_background = PlanarBackground(
            self.init_background_params, self.slen
        )

    def _sample_image(self, sample_every=10):
        batch_size = self.observed_image.shape[0]
        n_bands = self.observed_image.shape[1]
        slen = self.observed_image.shape[-1]

        samples = torch.zeros(
            n_bands,
            int(np.floor(slen / sample_every)),
            int(np.floor(slen / sample_every)),
        )

        for i in range(samples.shape[1]):
            for j in range(samples.shape[2]):
                x0 = i * sample_every
                x1 = j * sample_every
                samples[:, i, j] = (
                    self.observed_image[
                        :, :, x0 : (x0 + sample_every), x1 : (x1 + sample_every)
                    ]
                    .reshape(batch_size, n_bands, -1)
                    .min(2)[0]
                    .mean(0)
                )

        return samples
