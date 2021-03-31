import torch
import torch.nn as nn
import torch.nn.functional as F


def get_enc_dec(name, params):
    if name == "simple":
        assert params["slen"] % 2 == 1
        return SimpleEncoder(**params), SimpleDecoder(**params)

    if name == "jeff":
        assert params["slen"] % 2 == 0
        return JeffEncoder(**params), JeffDecoder(**params)

    if name == "bastien":
        assert params["slen"] % 2 == 0
        return BastienEncoder(**params), BastienDecoder(**params)

    raise NotImplementedError(f"Encoder-Decoder architecture '{name}' has not been implemented.")


class SimpleEncoder(nn.Module):
    def __init__(self, slen=41, latent_dim=8, n_bands=1, hidden=256):
        super().__init__()

        self.slen = slen
        self.latent_dim = latent_dim
        self.n_bands = n_bands

        self.features = nn.Sequential(
            nn.Conv2d(self.n_bands, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(1, -1),
            nn.Linear(16 * self.slen ** 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
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
        z = self.features(subimage)
        z_mean = self.fc_mean(z)
        z_var = 1e-4 + torch.exp(self.fc_var(z))
        return z_mean, z_var


class SimpleDecoder(nn.Module):
    max_channel = 64

    def __init__(self, slen=41, latent_dim=8, n_bands=1, hidden=256):
        super().__init__()

        self.slen = slen
        self.latent_dim = latent_dim
        self.n_bands = n_bands

        self.min_slen = slen // 2 + 1

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.max_channel * self.min_slen ** 2),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, padding=0, stride=2),
            nn.ConvTranspose2d(64, 2 * self.n_bands, 3, padding=0),
        )

    def forward(self, z):
        n_batches = z.shape[0]
        z = self.fc(z)
        z = z.view(n_batches, self.max_channel, self.min_slen, self.min_slen)
        z = self.deconv(z)
        z = z[:, :, : self.slen, : self.slen]
        assert z.shape[-1] == self.slen and z.shape[-2] == self.slen
        recon_mean = F.relu(z[:, : self.n_bands])
        var_multiplier = 1 + 10 * torch.sigmoid(z[:, self.n_bands : (2 * self.n_bands)])
        recon_var = 1e-4 + var_multiplier * recon_mean
        return recon_mean, recon_var


class JeffEncoder(SimpleEncoder):
    def __init__(self, slen, latent_dim, n_bands, hidden):
        super().__init__(slen=slen, latent_dim=latent_dim, n_bands=n_bands, hidden=hidden)
        self.features = nn.Sequential(
            nn.BatchNorm2d(self.n_bands, momentum=0.5),
            nn.Conv2d(self.n_bands, 16, 3, stride=1, padding=1),  # e.g. input=64, 64x64
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),  # 4x4
            nn.ReLU(),
            nn.Flatten(1, -1),
            nn.Linear(16 * 4 ** 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )


class JeffDecoder(SimpleDecoder):
    max_channel = 16

    def __init__(self, slen, latent_dim, n_bands, hidden):
        super().__init__(slen=slen, latent_dim=latent_dim, n_bands=n_bands, hidden=hidden)
        self.min_slen = slen / 2 ** 4  # e.g. slen = 64, min_slen = 4
        assert self.min_slen % 1 == 0

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.max_channel * self.min_slen ** 2),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 3, padding=1, stride=2, output_padding=1),  # e.g.  8x8
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 3, padding=1, stride=2, output_padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 3, padding=1, stride=2, output_padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 3, padding=1, stride=2, output_padding=1),  # 64x64
            nn.ConvTranspose2d(16, 2 * self.n_bands, 3, padding=1, stride=1),  # 6x64x64
        )


class BastienEncoder(SimpleEncoder):
    def __init__(self, slen, latent_dim, n_bands, hidden):
        super().__init__(slen=slen, latent_dim=latent_dim, n_bands=n_bands, hidden=hidden)
        self.features = nn.Sequential(
            nn.BatchNorm2d(self.n_bands, momentum=0.5),
            nn.Conv2d(self.n_bands, 32, 3, stride=1, padding=1),  # e.g. input=64, 64x64
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),  # 4x4
            nn.ReLU(),
            nn.Flatten(1, -1),
            nn.Linear(256 * 4 ** 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )


class BastienDecoder(SimpleDecoder):
    max_channel = 256

    def __init__(self, slen, latent_dim, n_bands, hidden):
        super().__init__(slen=slen, latent_dim=latent_dim, n_bands=n_bands, hidden=hidden)
        self.min_slen = slen / 2 ** 4  # e.g. slen = 64, min_slen = 4
        assert self.min_slen % 1 == 0

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.max_channel * self.min_slen ** 2),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, padding=1, stride=2, output_padding=1),  # e.g.  8x8
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, padding=1, stride=2, output_padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, padding=1, stride=2, output_padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, padding=1, stride=2, output_padding=1),  # 64x64
            nn.ConvTranspose2d(32, 2 * self.n_bands, 3, padding=1, stride=1),  # 6x64x64
        )
