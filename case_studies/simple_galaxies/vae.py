import torch
from torch import nn

from torch.distributions.normal import Normal

device = 'cuda:0'

class Flatten(nn.Module):
    def forward(self, tensor):
        return tensor.view(tensor.size(0), -1)


class Encoder(nn.Module):
    def __init__(self, 
                 slen = 51,
                 num_bands = 1, 
                 latent_dim = 5):

        super(Encoder, self).__init__()

        # image / model parameters
        self.slen = slen
        self.n_pixels = self.slen ** 2
        self.latent_dim = latent_dim

        hidden = 256
        
        self.features = nn.Sequential(
            nn.Conv2d(num_bands, 16, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),

            Flatten(),
            nn.Linear(16 * self.slen ** 2, hidden),
            nn.ReLU(),

            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden, track_running_stats=False),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Linear(hidden, hidden),
            nn.ReLU(),

            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        
        self.fc1 = nn.Linear(hidden, self.latent_dim)
        self.fc2 = nn.Linear(hidden, self.latent_dim)

    def forward(self, image):

        z = self.features(image)
        
        latent_mean = self.fc1(z)
        latent_log_std = self.fc2(z)

        return latent_mean, latent_log_std

class Decoder(nn.Module):
    def __init__(self, 
                 slen = 51,
                 num_bands = 1, 
                 latent_dim = 5):

        super(Decoder, self).__init__()

        # image / model parameters
        self.slen = slen
        self.n_pixels = self.slen ** 2
        self.latent_dim = latent_dim
        
        hidden = 64

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),

            nn.Linear(hidden, 64 * (slen // 2 + 1) ** 2),
            nn.ReLU())

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, padding=0, stride=2),
            nn.ConvTranspose2d(64, num_bands, 3, padding=0))


        self.tanh = torch.nn.Tanh()

    def forward(self, latent_samples):

        z = self.fc(latent_samples)
        z = z.view(-1, 64, self.slen // 2 + 1, self.slen // 2 + 1)
        z = self.deconv(z)
        
        # the output of deconv isnt exactly "slen"
        # we just take what we need
        
        recon_mean = z[:, :, :self.slen, :self.slen]
        return recon_mean

class VAE(nn.Module):
    def __init__(self,
                 slen = 51,
                 num_bands = 1, 
                 latent_dim = 8,
                 background = 686.):

        super(VAE, self).__init__()
        
        self.slen = slen
        self.num_bands = num_bands
        self.latent_dim = latent_dim
        
        self.encoder = Encoder(slen = self.slen,
                               num_bands = self.num_bands, 
                               latent_dim = self.latent_dim)
        
        self.decoder = Decoder(slen = self.slen,
                               num_bands = self.num_bands, 
                               latent_dim = self.latent_dim)
        
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
        recon_var = recon_mean.clamp(min = 1) # variance equals mean
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
