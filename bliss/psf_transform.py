import torch
import torch.nn as nn
from torch.nn.functional import unfold, pad
from astropy.io import fits

from .models import decoder
from . import device

########################
# function for Power law PSF
########################
def get_psf_params(psfield_fit_file, bands):
    psfield = fits.open(psfield_fit_file)

    psf_params = torch.zeros(len(bands), 6)

    for i in range(len(bands)):
        band = bands[i]

        sigma1 = psfield[6].data["psf_sigma1"][0][band] ** 2
        sigma2 = psfield[6].data["psf_sigma2"][0][band] ** 2
        sigmap = psfield[6].data["psf_sigmap"][0][band] ** 2

        beta = psfield[6].data["psf_beta"][0][band]
        b = psfield[6].data["psf_b"][0][band]
        p0 = psfield[6].data["psf_p0"][0][band]

        # I think these parameters are constrained to be positive
        # take log; we will take exp later
        psf_params[i] = torch.log(torch.Tensor([sigma1, sigma2, sigmap, beta, b, p0]))

    return psf_params


def psf_fun(r, sigma1, sigma2, sigmap, beta, b, p0):
    term1 = torch.exp(-(r ** 2) / (2 * sigma1))
    term2 = b * torch.exp(-(r ** 2) / (2 * sigma2))
    term3 = p0 * (1 + r ** 2 / (beta * sigmap)) ** (-beta / 2)

    return (term1 + term2 + term3) / (1 + b + p0)


def get_psf(slen, psf_params, cached_radii_grid=None):
    assert (slen % 2) == 1

    if cached_radii_grid is None:
        grid = decoder.get_mgrid(slen) * (slen - 1) / 2
        radii_grid = (grid ** 2).sum(2).sqrt()
    else:
        radii_grid = cached_radii_grid

    _psf_params = torch.exp(psf_params)
    return psf_fun(
        radii_grid,
        _psf_params[0],
        _psf_params[1],
        _psf_params[2],
        _psf_params[3],
        _psf_params[4],
        _psf_params[5],
    )


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
        self.params = nn.Parameter(init_psf_params.clone())

        # get normalization_constant
        self.normalization_constant = torch.zeros(self.n_bands)
        for i in range(self.n_bands):
            self.normalization_constant[i] = (
                1
                / get_psf(
                    self.psf_slen, self.init_psf_params[i], self.cached_radii_grid
                ).sum()
            )

        # initial psf
        self.init_psf = self.get_psf()
        self.init_psf_sum = self.init_psf.sum(-1).sum(-1).detach()

    def get_psf(self):
        # TODO make the psf function vectorized ...
        for i in range(self.n_bands):
            _psf = (
                get_psf(self.psf_slen, self.params[i], self.cached_radii_grid)
                * self.normalization_constant[i]
            )

            if i == 0:
                psf = _psf.unsqueeze(0)
            else:
                psf = torch.cat((psf, _psf.unsqueeze(0)))

        assert (psf > 0).all()

        return psf

    def forward(self):
        psf = self.get_psf()
        psf = psf * (self.init_psf_sum / psf.sum(-1).sum(-1)).unsqueeze(-1).unsqueeze(
            -1
        )

        l_pad = (self.image_slen - self.psf_slen) // 2

        return pad(psf, (l_pad,) * 4)
