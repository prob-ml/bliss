import torch
import torch.nn as nn
from torch.nn.functional import pad
from astropy.io import fits

from .models import decoder
from . import device


def get_psf_params(psfield_fit_file, bands: list):
    psfield = fits.open(psfield_fit_file)
    psf_params = torch.zeros(len(bands), 6)
    psf_params[:, 0] = psfield[6]["psf_sigma1"][0][bands] ** 2
    psf_params[:, 1] = psfield[6]["psf_sigma2"][0][bands] ** 2
    psf_params[:, 2] = psfield[6]["psf_sigmap"][0][bands] ** 2
    psf_params[:, 3] = psfield[6]["psf_beta"][0][bands]
    psf_params[:, 4] = psfield[6]["psf_b"][0][bands]
    psf_params[:, 5] = psfield[6]["psf_p0"][0][bands]
    return psf_params.log()


class PowerLawPSF(nn.Module):
    def __init__(self, init_psf_params, psf_slen=25, image_slen=101):

        super(PowerLawPSF, self).__init__()
        assert len(init_psf_params.shape) == 2
        assert image_slen % 2 == 1, "image_slen must be odd"

        self.n_bands = init_psf_params.shape[0]
        self.init_psf_params = init_psf_params.clone()

        self.psf_slen = psf_slen
        self.image_slen = image_slen

        grid = decoder.get_mgrid(self.psf_slen) * (self.psf_slen - 1) / 2
        self.cached_radii_grid = (grid ** 2).sum(2).sqrt().to(device)

        # initial weights
        self.params = nn.Parameter(init_psf_params.clone(), requires_grad=True)

        # get normalization_constant
        self.normalization_constant = torch.zeros(self.n_bands)
        for i in range(self.n_bands):
            psf_i = self.get_psf_single_band(self.init_psf_params[i])
            self.normalization_constant[i] = 1 / psf_i.sum()

        # initial psf
        self.init_psf = self.get_psf()
        self.init_psf_sum = self.init_psf.sum(-1).sum(-1).detach()

    @staticmethod
    def psf_fun(r, sigma1, sigma2, sigmap, beta, b, p0):
        term1 = torch.exp(-(r ** 2) / (2 * sigma1))
        term2 = b * torch.exp(-(r ** 2) / (2 * sigma2))
        term3 = p0 * (1 + r ** 2 / (beta * sigmap)) ** (-beta / 2)
        return (term1 + term2 + term3) / (1 + b + p0)

    def get_psf_single_band(self, psf_params):
        assert (self.psf_slen % 2) == 1
        _psf_params = torch.exp(psf_params)
        return self.psf_fun(
            self.cached_radii_grid,
            _psf_params[0],
            _psf_params[1],
            _psf_params[2],
            _psf_params[3],
            _psf_params[4],
            _psf_params[5],
        )

    def get_psf(self):
        # TODO make the psf function vectorized ...
        psf = torch.tensor([])
        for i in range(self.n_bands):
            _psf = self.get_psf_single_band(self.params[i])
            _psf *= self.normalization_constant[i]
            psf = torch.cat((psf, _psf.unsqueeze(0)))

        assert (psf > 0).all()
        return psf

    def forward(self):
        psf = self.get_psf()
        norm = psf.sum(-1).sum(-1)
        psf *= (self.init_psf_sum / norm).unsqueeze(-1).unsqueeze(-1)
        l_pad = (self.image_slen - self.psf_slen) // 2
        return pad(psf, (l_pad,) * 4)
