import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions import Normal


class Flatten(nn.Module):

    @staticmethod
    def forward(tensor):
        return tensor.view(tensor.size(0), -1)


class CenteredGalaxyEncoder(nn.Module):  # recognition, inference

    def __init__(self, slen, latent_dim, num_bands, hidden=256):
        """

        :rtype: NoneType
        """
        super(CenteredGalaxyEncoder, self).__init__()

        self.slen = slen
        self.latent_dim = latent_dim
        self.num_bands = num_bands

        self.features = nn.Sequential(
            nn.Conv2d(self.num_bands, 16, 3, padding=1),
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

        self.fc_mean = nn.Linear(hidden, self.latent_dim)
        self.fc_var = nn.Linear(hidden, self.latent_dim)

    def forward(self, subimage):
        """
        1e-4 here is to avoid NaNs, .exp gives you positive and variance increase quickly.
        Exp is better matched for logs. (trial and error, but makes big difference)
        :param subimage: image to be encoded.
        :return:
        """
        z = self.features(subimage)
        z_mean = self.fc_mean(z)
        z_var = 1e-4 + torch.exp(self.fc_var(
            z))
        return z_mean, z_var


class CenteredGalaxyDecoder(nn.Module):  # generator

    # ToDo: Maybe add its own zeros and one? although messes up previous state dicts.
    def __init__(self, slen, latent_dim, num_bands, hidden=256):
        super(CenteredGalaxyDecoder, self).__init__()

        self.slen = slen  # side-length.
        self.latent_dim = latent_dim
        self.num_bands = num_bands

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),

            nn.Linear(hidden, 64 * (slen // 2 + 1) ** 2),  # shrink dimensions
            nn.ReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, padding=0, stride=2),  # this will increase size back to twice.
            nn.ConvTranspose2d(64, 2 * self.num_bands, 3, padding=0)  # why channels=2 * num bands?
        )

    def forward(self, z):
        """

        :param z: Has shape = latent_dim.
        :return:
        """
        z = self.fc(z)

        # view takes in -1 and automatically determines that dimension. This dimension is the number of samples.
        z = z.view(-1, 64, self.slen // 2 + 1, self.slen // 2 + 1)
        z = self.deconv(z)
        z = z[:, :, :self.slen, :self.slen]

        # first half of the bands is now used.
        # expected number of photons has to be positive, this is why we use f.relu here.
        recon_mean = f.relu(z[:, :self.num_bands])

        # sometimes nn can get variance to be really small, if sigma gets really small then small learning
        # this is what the 1e-4 is for.
        # We also want var >= mean because of the poisson noise, which is also imposed here.
        var_multiplier = 1 + 10 * torch.sigmoid(z[:, self.num_bands:(2 * self.num_bands)])
        recon_var = 1e-4 + var_multiplier * recon_mean

        # reconstructed mean and variance, these are per pixel.
        return recon_mean, recon_var

    def get_sample(self, num_samples, return_latent=False):
        p_z = Normal(torch.zeros(1), torch.ones(1))
        z = p_z.rsample(torch.tensor([num_samples, self.latent_dim])).view(-1)  # shape = (8,)
        samples, _ = self.dec.forward(z)

        if return_latent:
            return z, samples

        else:
            return samples  # shape = (num_samples, num_bands, slen, slen)


class OneCenteredGalaxy(nn.Module):

    def __init__(self, slen, latent_dim=8, num_bands=1):
        super(OneCenteredGalaxy, self).__init__()

        self.slen = slen  # The dimensions of the image slen * slen
        self.latent_dim = latent_dim
        self.num_bands = num_bands

        self.enc = CenteredGalaxyEncoder(slen, latent_dim, num_bands)
        self.dec = CenteredGalaxyDecoder(slen, latent_dim, num_bands)

        self.register_buffer("zero", torch.zeros(1))
        self.register_buffer("one", torch.ones(1))

    def forward(self, image, background):
        z_mean, z_var = self.enc.forward(image - background)  # shape = [nsamples, latent_dim]

        q_z = Normal(z_mean, z_var.sqrt())
        z = q_z.rsample()

        log_q_z = q_z.log_prob(z).sum(1)
        p_z = Normal(self.zero, self.one)
        log_p_z = p_z.log_prob(z).sum(1)  # using stochastic optimization by sampling only one z from prior.
        kl_z = (log_q_z - log_p_z)

        recon_mean, recon_var = self.dec.forward(z)  # this reconstructed mean/variances images (per pixel quantities)

        recon_mean = recon_mean + background  # w/out kl can behave wildly if not background.
        recon_var = recon_var + background

        return recon_mean, recon_var, kl_z

    def loss(self, image, background, k=1):
        """

        :param image: The complete image that includes the background.
        :param background:
        :param k:
        :return:
        """
        # TODO LATER: use k
        # sampling images from the real distribution
        recon_mean, recon_var, kl_z = self.forward(image, background)  # z | x ~ decoder

        # -log p(x | a, z), dimensions: torch.Size([ nsamples, num_bands, slen, slen])
        # assuming covariance is diagonal.
        recon_losses = -Normal(recon_mean, recon_var.sqrt()).log_prob(image)

        # image.size(0) = first dimension = number of samples, .sum(1) sum over all dimensions except sample.
        recon_losses = recon_losses.view(image.size(0), -1).sum(1)  # shape = [nsamples]

        # the expectation is subtle and implicit bc we are using stochastic optimization multiple times.
        # sum here is over the samples (only remaining dimensions)
        loss = (recon_losses + kl_z).sum()  # this is actually just ELBO, shape=[nsamples]

        return loss

    def rmse_pp(self, image, background):
        """
        Per pixel avg. rmse.
        :param image:
        :param background:
        :return:
        """
        recon_mean, recon_var, kl_z = self.forward(image, background)
        return torch.sqrt(((recon_mean - image) ** 2).sum()) / self.slen ** 2
