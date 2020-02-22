import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions import Normal


class CatalogEncoder(nn.Module):

    def __init__(self, num_params, latent_dim, hidden=256):
        """

        :rtype: NoneType
        """
        super(CatalogEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.num_params = num_params

        self.features = nn.Sequential(

            nn.Linear(self.num_params, hidden),

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

    def forward(self, data):
        """
        1e-4 here is to avoid NaNs, .exp gives you positive and variance increase quickly.
        Exp is better matched for logs. (trial and error, but makes big difference)
        :param subimage: image to be encoded.
        :return:
        """
        z = self.features(data)
        z_mean = self.fc_mean(z)
        z_var = 1e-4 + torch.exp(self.fc_var(z))  # 1e-4 prevents having too small a variance.
        return z_mean, z_var


class CatalogDecoder(nn.Module):  # generator

    def __init__(self, num_params, latent_dim, hidden=256):
        super(CatalogDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.num_params = num_params

        self.deconv = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),

            nn.Linear(hidden, hidden),  # shrink dimensions
            nn.ReLU(),

            nn.Linear(hidden, self.num_params * 2)
        )

    def forward(self, z):
        z = self.deconv(z)
        recon_mean = f.relu(z[:, :self.num_params])
        recon_var = 1e-4 + 10 * torch.sigmoid(z[:, self.num_params:(2 * self.num_params)])  # 1e-4 prevent too small.

        # reconstructed mean and variance, these are per pixel.
        return recon_mean, recon_var


class catalogNet(nn.Module):

    def __init__(self, num_params, latent_dim=20):
        super(catalogNet, self).__init__()

        self.latent_dim = latent_dim
        self.num_params = num_params

        self.enc = CatalogEncoder(self.num_params, self.latent_dim)
        self.dec = CatalogDecoder(self.num_params, self.latent_dim)

        self.register_buffer("zero", torch.zeros(1))
        self.register_buffer("one", torch.ones(1))

    def forward(self, data):
        z_mean, z_var = self.enc.forward(data)  # shape = [nsamples, latent_dim]
        raise NotImplementedError("If plan to use it, make sure to factorize it with the lines in galaxy_net.")
        # q_z = Normal(z_mean, z_var.sqrt())
        # z = q_z.rsample()
        #
        # log_q_z = q_z.log_prob(z).sum(1)
        # p_z = Normal(self.zero, self.one)
        #
        # log_p_z = p_z.log_prob(z).sum(1)  # using stochastic optimization by sampling only one z from prior.
        # kl_z = (log_q_z - log_p_z)
        #
        # recon_mean, recon_var = self.dec.forward(z)  # this reconstructed mean/variances images (per pixel quantities)
        #
        # return recon_mean, recon_var, kl_z

    def loss(self, data):
        recon_mean, recon_var, kl_z = self.forward(data)

        recon_losses = -Normal(recon_mean, recon_var.sqrt()).log_prob(data)
        recon_losses = recon_losses.sum(1)

        loss = (recon_losses + kl_z).sum()

        return loss

    def get_sample(self):
        p_z = Normal(self.zero, self.one)
        z = p_z.rsample()
        recon_mean, _ = self.dec.forward(z)
        return recon_mean
