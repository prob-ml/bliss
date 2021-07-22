import numpy as np

import pytorch_lightning as pl
import torch
from torch import nn

import torch.nn.functional as F
from torch.distributions import normal


from bliss.models import decoder
from bliss.models.encoder import get_star_bool, tile_images
from bliss.optimizer import get_optimizer


def _trim_images(images, trim_slen):

    # crops an image to be the center
    # trim_slen x trim_slen pixels

    slen = images.shape[-1]

    diff = slen - trim_slen
    assert diff >= 0

    indx0 = int(np.floor(diff / 2))
    indx1 = indx0 + trim_slen

    return images[:, :, indx0:indx1, indx0:indx1]


class FluxEncoder(nn.Module):
    def __init__(self, ptile_slen=52, tile_slen=4, flux_tile_slen=20, n_bands=1, max_sources=1):

        super(FluxEncoder, self).__init__()

        # image / model parameters
        self.ptile_slen = ptile_slen
        self.tile_slen = tile_slen
        self.n_bands = n_bands

        self.max_sources = max_sources

        if max_sources > 1:
            # need to implement a triangular array of
            # fluxes as before
            raise NotImplementedError()

        # output dimension
        outdim = 2 * self.n_bands

        # the size of the ptiles passed to this encoder
        self.flux_tile_slen = flux_tile_slen

        # the network
        self.conv1 = nn.Conv2d(self.n_bands, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 16, 3)

        # compute output dimension
        conv_out_dim = (self.flux_tile_slen - 6) ** 2 * 16

        latent_dim = 64
        self.fc1 = nn.Linear(conv_out_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc4 = nn.Linear(latent_dim, outdim)

    def forward(self, images):

        batch_size = images.shape[0]

        # tile the image
        image_ptiles = self._get_ptiles_from_images(images)

        # pass through nn
        mean, sd = self._forward_ptiles(image_ptiles)

        # sample
        z = torch.randn(mean.shape, device=mean.device)
        samples = mean + z * sd

        # save everything in a dictionary
        out_dict = dict(mean=mean, sd=sd, samples=samples)

        # reshape
        n_tiles_per_image = int(image_ptiles.shape[0] / batch_size)

        for k in out_dict.keys():
            out_dict[k] = out_dict[k].view(
                batch_size, n_tiles_per_image, self.max_sources, self.n_bands
            )

        return out_dict

    def _conv_layers(self, image_ptiles):
        # pass through conv layers
        h = F.relu(self.conv1(image_ptiles))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))

        return h.flatten(1, -1)

    def _forward_ptiles(self, image_ptiles):

        # pass through conv layers
        h = self._conv_layers(image_ptiles)

        # pass through fully connected
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))

        indx0 = self.n_bands
        indx1 = 2 * indx0

        mean = h[:, 0:indx0]
        sd = F.softplus(h[:, indx0:indx1]) + 1e-6

        return mean, sd

    def _trim_ptiles(self, image_ptiles):

        return _trim_images(image_ptiles, self.flux_tile_slen)

    def _get_ptiles_from_images(self, images):
        image_ptiles = tile_images(images, tile_slen=self.tile_slen, ptile_slen=self.ptile_slen)

        return self._trim_ptiles(image_ptiles)


class FluxEstimator(pl.LightningModule):

    # ---------------
    # Model
    # ----------------

    def __init__(
        self,
        decoder_kwargs,
        flux_tile_slen=20,
        optimizer_params: dict = None,
    ):

        super().__init__()

        # the image decoder: we need this to
        # compute the kl-q-p loss
        self.image_decoder = decoder.ImageDecoder(**decoder_kwargs)
        self.image_decoder.requires_grad_(False)

        # save some image parameters that we need
        # to define the flux encoder
        n_bands = self.image_decoder.n_bands
        ptile_slen = self.image_decoder.ptile_slen
        tile_slen = self.image_decoder.tile_slen
        max_sources = self.image_decoder.max_sources

        self.enc = FluxEncoder(
            ptile_slen=ptile_slen,
            tile_slen=tile_slen,
            flux_tile_slen=flux_tile_slen,
            n_bands=n_bands,
            max_sources=max_sources,
        )

        self.optimizer_params = optimizer_params

    def forward(self, images):
        # input images is the full scene.
        # tiling is done under the hood here.
        # output are flux parameters (mean, sd, samples) on tiles

        return self.enc.forward(images)

    def kl_qp_flux_loss(self, batch, est_flux, est_flux_sd):

        batchsize = batch["images"].shape[0]
        assert est_flux.shape == batch["fluxes"].shape

        # get reconstruction
        recon, _ = self.image_decoder.render_images(
            batch["n_sources"],
            batch["locs"],
            batch["galaxy_bool"],
            batch["galaxy_params"],
            est_flux,
            add_noise=False,
        )

        # log likelihood
        scale = torch.sqrt(recon.clamp(min=1.0))
        norm = normal.Normal(loc=recon, scale=scale)
        loglik = norm.log_prob(batch["images"]).view(batchsize, -1).sum(1)

        # entropy
        star_bool = get_star_bool(batch["n_sources"], batch["galaxy_bool"])
        entropy = torch.log(est_flux_sd) * star_bool
        entropy = entropy.view(batchsize, -1).sum(1)

        # negative elbo
        kl = -(loglik + entropy)

        return kl, recon

    def get_loss(self, batch):
        out = self.enc(batch["images"])

        # get loss
        kl, recon = self.kl_qp_flux_loss(batch, out["samples"], out["sd"])

        return kl.mean()

    # ---------------
    # Optimizer
    # ----------------

    def configure_optimizers(self):
        assert self.optimizer_params is not None, "Need to specify `optimizer_params`."
        name = self.optimizer_params["name"]
        kwargs = self.optimizer_params["kwargs"]
        return get_optimizer(name, self.parameters(), kwargs)

    # ---------------
    # Training
    # ----------------

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        loss = self.get_loss(batch)
        self.log("train_loss", loss)
        return loss

    # ---------------
    # Validation
    # ----------------

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        out = self.enc(batch["images"])

        # get loss
        kl, recon = self.kl_qp_flux_loss(batch, out["samples"], out["sd"])

        # metrics
        self.log("val_loss", kl)

        return {"images": batch["images"], "recon_mean": recon}
