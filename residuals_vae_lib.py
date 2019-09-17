import torch
import torch.nn as nn

import numpy as np

from inv_KL_objective_lib import eval_normal_logprob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Flatten(nn.Module):
    def forward(self, tensor):
        return tensor.view(tensor.size(0), -1)

def CenterCrop(tensor, edge):
        assert len(tensor.shape) == 4
        w = tensor.shape[2]
        h = tensor.shape[3]

        return tensor[:, :, edge:(w - edge),edge:(h - edge)]

def sample_normal(mean, logvar):
    return mean + torch.exp(0.5 * logvar) * torch.randn(mean.shape).to(device)


class ResidualVAE(nn.Module):
    def __init__(self, slen, n_bands, f_min, latent_dim = 8):

        super(ResidualVAE, self).__init__()

        # image parameters
        self.slen = slen
        self.n_bands = n_bands
        self.f_min = f_min

        self.latent_dim = latent_dim

        # convolutional NN paramters
        enc_kern = 3
        enc_hidden = 128
        self.conv_channels = 64

        # convolutional NN
        self.enc_conv = nn.Sequential(
            nn.Conv2d(self.n_bands, 32, enc_kern,
                        stride=1, padding=1),
            nn.BatchNorm2d(32, track_running_stats=True),
            nn.ReLU(),

            nn.Conv2d(32, self.conv_channels, enc_kern,
                        stride=1, padding=1),
            nn.BatchNorm2d(self.conv_channels, track_running_stats=True),
            nn.ReLU(),
            Flatten()
        )

        # output dimension of convolutions
        self.conv_out_dim = \
            self.enc_conv(torch.zeros(1, n_bands, slen, slen)).size(1)
        self.h_slen = np.sqrt(self.conv_out_dim / self.conv_channels)
        assert (self.h_slen % 1) == 0
        self.h_slen = int(self.h_slen)

        # fully connected layers
        self.enc_fc = nn.Sequential(
            nn.Linear(self.conv_out_dim, enc_hidden),
            nn.BatchNorm1d(enc_hidden, track_running_stats=True),
            nn.ReLU(),

            nn.Linear(enc_hidden, enc_hidden),
            nn.BatchNorm1d(enc_hidden, track_running_stats=True),
            nn.ReLU(),

            nn.Linear(enc_hidden, 2 * latent_dim)
        )


        # fully connected layers for generative model
        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim, enc_hidden),
            nn.ReLU(),

            nn.Linear(enc_hidden, enc_hidden),
            nn.ReLU(),

            nn.Linear(enc_hidden, self.conv_out_dim),
            nn.ReLU()
        )

        # self.dec_conv_mean = nn.Sequential(
        #     nn.ConvTranspose2d(self.conv_channels, 32, enc_kern, stride=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, self.n_bands, enc_kern, stride=1)
        # )
        # self.dec_conv_var = nn.Sequential(
        #     nn.ConvTranspose2d(self.conv_channels, 32, enc_kern, stride=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, self.n_bands, enc_kern, stride=1)
        # )

        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(self.conv_channels, 32, enc_kern, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.n_bands * 2, enc_kern, stride=1)
        )

        eta = self.encode(torch.zeros(5, n_bands, slen, slen))[0]
        out1, out2 = self.decode(eta)
        assert out1.shape[2] == slen
        assert out1.shape[3] == slen
        assert out1.shape == out2.shape

    def encode(self, residual):
        h = self.enc_fc(self.enc_conv(residual))

        eta_mean = h[:, 0:self.latent_dim]
        eta_logvar = h[:, self.latent_dim:(2 * self.latent_dim)]

        return eta_mean, eta_logvar

    def decode(self, eta):

        h = self.dec_fc(eta)

        h = h.view(eta.shape[0], self.conv_channels, self.h_slen, self.h_slen)

        h = self.dec_conv(h)
        h = CenterCrop(h, 2)

        recon_mean = h[:, 0:self.n_bands, :, :]
        recon_logvar = h[:, self.n_bands:(2 * self.n_bands), :, :]
        #
        # recon_mean = CenterCrop(self.dec_conv_mean(h), 2)
        # recon_logvar = CenterCrop(self.dec_conv_var(h), 2)

        return recon_mean, recon_logvar

    def forward(self, image, simulated_image, sample = True, return_unnormalized = False):

        residual_clamped = (image - simulated_image).clamp(min = -self.f_min,
                                                max = self.f_min)

        normalized_residual = residual_clamped / simulated_image

        # encode
        eta_mean, eta_logvar = self.encode(normalized_residual)

        # reparameterize
        if sample:
            eta = sample_normal(eta_mean, eta_logvar)
        else:
            eta = eta_mean

        # decode
        recon_mean, recon_logvar = self.decode(eta)

        # unnormalize
        if return_unnormalized:
            recon_mean = recon_mean * simulated_image
            recon_logvar = recon_logvar + torch.log(simulated_image**2)
            residual = normalized_residual * simulated_image
        else:
            residual = normalized_residual


        return recon_mean, recon_logvar, eta_mean, eta_logvar, residual

def normalize_image(image):
    assert len(image.shape) == 4
    image_mean = image.view(image.shape[0], -1).mean(1)
    image_var = image.view(image.shape[0], -1).var(1)

    _image_mean = image_mean.view(image.shape[0], 1, 1, 1)
    _image_var = image_var.view(image.shape[0], 1, 1, 1)

    return (image - _image_mean) / torch.sqrt(_image_var + 1e-5), _image_mean, _image_var

def get_kl_prior_term(mean, logvar):
    return - 0.5 * (1 + logvar - mean.pow(2) - logvar.exp())

def get_residual_vae_loss(image, simulated_image, resiudal_vae):

    recon_mean, recon_logvar, eta_mean, eta_logvar, normalized_residual = \
            resiudal_vae(image, simulated_image)

    assert normalized_residual.shape == recon_mean.shape
    assert normalized_residual.shape == recon_logvar.shape

    recon_loss = - eval_normal_logprob(normalized_residual, recon_mean, recon_logvar)

    kl_prior = get_kl_prior_term(eta_mean, eta_logvar)

    return recon_loss.sum() + kl_prior.sum()

def eval_residual_vae(residual_vae, loader, simulator,
                        optimizer = None, train = False ,
                        star_encoder = None):
    avg_loss = 0.0

    for _, data in enumerate(loader):
        images = data['image'].to(device)
        backgrounds = data['background'].to(device)

        if star_encoder is None:
            # use true parameters
            fluxes = data['fluxes'].to(device).type(torch.float)
            locs = data['locs'].to(device).type(torch.float)
            n_stars = data['n_stars'].to(device)
        else:
            # use estimated parameters
            star_encoder.eval()

            logit_loc_mean, logit_loc_logvar, \
                log_flux_mean, log_flux_logvar, log_probs = \
                        star_encoder(images, backgrounds)

            locs = torch.sigmoid(\
                sample_normal(logit_loc_mean.detach(),
                                                logit_loc_logvar.detach()))
            fluxes = torch.exp(\
                sample_normal(log_flux_mean.detach(),
                                                log_flux_logvar.detach()))
            n_stars = torch.multinomial(torch.exp(log_probs), num_samples = 1).squeeze()

        # reconstruction
        simulated_images = \
            simulator.draw_image_from_params(locs = locs,
                                             fluxes = fluxes,
                                             n_stars = n_stars,
                                             add_noise = False)

        if train:
            residual_vae.train()
            assert optimizer is not None
            optimizer.zero_grad()
        else:
            residual_vae.eval()


        loss = get_residual_vae_loss(images, simulated_images, residual_vae)

        if train:
            (loss / images.shape[0]).backward()
            optimizer.step()

        avg_loss += loss.item() / len(loader.dataset)

    return avg_loss
