import torch
from torch import nn

from torch.distributions.normal import Normal

device = 'cuda:0'


class MLPEncoder(nn.Module):
    def __init__(self, latent_dim = 5,
                    slen = 51):

        super(MLPEncoder, self).__init__()

        # image / model parameters
        self.slen = slen
        self.n_pixels = self.slen ** 2
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(self.n_pixels, 528)
        self.fc2 = nn.Linear(528, 528)
        self.fc3 = nn.Linear(528, self.latent_dim * 2)

        self.tanh = torch.nn.Tanh()

    def forward(self, image):

        h = image.view(-1, self.n_pixels)

        h = self.tanh(self.fc1(h))
        h = self.tanh(self.fc2(h))
        h = self.fc3(h)

        latent_mean = h[:, 0:self.latent_dim]
        latent_log_std = h[:, self.latent_dim:(2 * self.latent_dim)]

        return latent_mean, latent_log_std

class MLPDecoder(nn.Module):
    def __init__(self, 
                 latent_dim = 5,
                 slen = 51):

        super(MLPDecoder, self).__init__()

        # image / model parameters
        self.slen = slen
        self.n_pixels = self.slen ** 2
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(self.latent_dim, 528)
        self.fc2 = nn.Linear(528, 528)
        self.fc3 = nn.Linear(528, self.n_pixels)

        self.tanh = torch.nn.Tanh()

    def forward(self, latent_samples):

        h = self.tanh(self.fc1(latent_samples))
        h = self.tanh(self.fc2(h))
        h = self.fc3(h)

        return h.view(-1, self.slen, self.slen)

class VAE(nn.Module):
    def __init__(self,
                 latent_dim = 8,
                 slen = 51,
                 background = 686.):

        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.slen = slen

        self.encoder = MLPEncoder(self.latent_dim, self.slen)
        self.decoder = MLPDecoder(self.latent_dim, self.slen)
        
        self.background = background

    def forward(self, image):

        # get latent means and std
        latent_mean, latent_log_std = self.encoder(image - self.background)

        # sample latent params
        latent_samples = torch.randn(latent_mean.shape, device = device) * \
                            torch.exp(latent_log_std) + latent_mean

        # pass through decoder
        recon_mean = self.decoder(latent_samples) + self.background

        return recon_mean, latent_mean, latent_log_std, latent_samples

    def get_loss(self, image):

        recon_mean, latent_mean, latent_log_std, _ = \
            self.forward(image)

        # kl term
        kl_q = get_kl_q_standard_normal(latent_mean, latent_log_std)

        # gaussian likelihood
        recon_var = recon_mean.clamp(min = 1)
        recon_losses = -Normal(recon_mean, recon_var.sqrt()).log_prob(image)
        recon_losses = recon_losses.view(image.size(0), -1).sum(1)
        
        return recon_losses + kl_q 


def get_kl_q_standard_normal(mu, log_sigma):
    # The KL between a Gaussian variational distribution
    # and a standard normal
    mu = mu.view(mu.shape[0], -1)
    log_sigma = log_sigma.view(log_sigma.shape[0], -1)
    return 0.5 * torch.sum(-1 - 2 * log_sigma + \
                                mu**2 + torch.exp(log_sigma)**2, dim = 1)
