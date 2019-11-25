import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions import Normal, Categorical, Bernoulli


class ColorEncoder(nn.Module):  # recognition, inference

    def __init__(self, num_bands, latent_dim, hidden=256):
        """

        :rtype: NoneType
        """
        super(ColorEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.num_bands = num_bands

        self.features = nn.Sequential(

            nn.Linear(self.num_bands, hidden),

            nn.ReLU(),

            nn.BatchNorm1d(hidden, track_running_stats=False),

            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Linear(hidden, hidden),
            nn.ReLU(),

            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.fc_mean = nn.Linear(hidden, self.latent_dim)
        self.fc_var = nn.Linear(hidden, self.latent_dim)

    def forward(self, color_sample):
        """
        1e-4 here is to avoid NaNs, .exp gives you positive and variance increase quickly.
        Exp is better matched for logs. (trial and error, but makes big difference)
        :param subimage: image to be encoded.
        :return:
        """
        z = self.features(color_sample)
        z_mean = self.fc_mean(z)
        z_var = 1e-4 + torch.exp(self.fc_var(
            z))
        return z_mean, z_var


class ColorDecoder(nn.Module):  # generator

    def __init__(self, num_bands, latent_dim, hidden=256):
        super(ColorDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.num_bands = num_bands

        self.deconv = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),

            nn.Linear(hidden, hidden),  # shrink dimensions
            nn.ReLU(),

            nn.Linear(hidden, self.num_bands * 2)
        )

    def forward(self, z):
        z = self.deconv(z)
        recon_mean = f.relu(z[:, :self.num_bands])

        # sometimes nn can get variance to be really large, if sigma gets really large then small learning
        # this avoids variance getting too large.
        var_multiplier = 1 + 10 * torch.sigmoid(z[:, self.num_bands:(2 * self.num_bands)])
        recon_var = 1e-4 + var_multiplier * recon_mean

        # reconstructed mean and variance, these are per pixel.
        return recon_mean, recon_var


class ColorNet(nn.Module):

    def __init__(self, latent_dim=12, num_bands=6):
        super(ColorNet, self).__init__()

        assert num_bands == 6, "Not implemented non-LSST bands"

        self.latent_dim = latent_dim
        self.num_bands = num_bands

        self.enc = ColorEncoder(latent_dim, num_bands)
        self.dec = ColorDecoder(latent_dim, num_bands)

        self.register_buffer("zero", torch.zeros(1))
        self.register_buffer("one", torch.ones(1))

    def forward(self, color_sample):
        z_mean, z_var = self.enc.forward(color_sample)  # shape = [nsamples, latent_dim]

        q_z = Normal(z_mean, z_var.sqrt())
        z = q_z.rsample()

        log_q_z = q_z.log_prob(z).sum(1)
        p_z = Normal(self.zero, self.one)
        log_p_z = p_z.log_prob(z).sum(1)  # using stochastic optimization by sampling only one z from prior.
        kl_z = (log_q_z - log_p_z)

        recon_mean, recon_var = self.dec.forward(z)  # this reconstructed mean/variances images (per pixel quantities)

        return recon_mean, recon_var, kl_z

    def loss(self, color_sample):
        recon_mean, recon_var, kl_z = self.forward(color_sample)
        return (recon_mean - color_sample).sum()

        # recon_losses = -Normal(recon_mean, recon_var.sqrt()).log_prob(color_sample)
        # recon_losses = recon_losses.sum(1)
        #
        # loss = (recon_losses + kl_z).sum()

        # return loss
