import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
import pytorch_lightning as pl

import numpy as np

from .models.decoder import get_mgrid
from . import device


class WakeNet(pl.LightningModule):

    # ---------------
    # Model
    # ----------------

    def __init__(
        self,
        star_encoder,
        image_decoder,
        observed_img,
        init_background_values,
        hparams,
    ):
        super(WakeNet, self).__init__()

        self.star_encoder = star_encoder
        self.image_decoder = image_decoder
        self.image_decoder.requires_grad_(True)
        
        self.slen = image_decoder.slen 
        self.border_padding = image_decoder.border_padding 
        
        # observed image is batch_size (or 1) x n_bands x slen x slen
        self.slen_plus_padding = self.slen + 2 * self.border_padding
        assert len(observed_img.shape) == 4
        assert observed_img.shape[-1] == self.slen_plus_padding, "cached grid won't match."
        
        self.observed_img = observed_img

        # hyper-parameters
        self.hparams = hparams
        self.n_samples = self.hparams["n_samples"]
        self.lr = self.hparams["lr"]
        self.slen = observed_img.shape[-1]

        # get n_bands
        self.n_bands = self.image_decoder.n_bands

    def forward(self, obs_img):

        with torch.no_grad():
            self.star_encoder.eval()
            sample = self.star_encoder.sample_encoder(obs_img, self.n_samples)
        
        recon_mean = self.image_decoder.render_images(
            sample["n_sources"].contiguous(),
            sample["locs"].contiguous(),
            sample["galaxy_bool"].contiguous(),
            sample["galaxy_params"].contiguous(),
            sample["fluxes"].contiguous(),
            add_noise = False
        )
        
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
        return optim.Adam([{"params": self.image_decoder.parameters(), "lr": self.lr}])

    # ---------------
    # Training
    # ----------------

    def get_loss(self, batch):
        img = batch.unsqueeze(0)
        recon_mean = self.forward(img)
        error = -Normal(recon_mean, recon_mean.sqrt()).log_prob(img)

        last = self.slen - self.border_padding
        loss = error[:, :, self.border_padding : last, self.border_padding : last].sum((1, 2, 3)).mean()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log("validation_loss", loss)
