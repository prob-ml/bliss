import torch

import torch.nn as nn
from torch.nn.functional import unfold, softmax, pad

import image_utils
from utils import eval_normal_logprob
from simulated_datasets_lib import _get_mgrid
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_psf_params(psfield, band):
    sigma1 = psfield[6]['psf_sigma1'][0][0, band]
    sigma2 = psfield[6]['psf_sigma2'][0][0, band]
    sigmap = psfield[6]['psf_sigmap'][0][0, band]
    beta = psfield[6]['psf_beta'][0][0, band]
    b = psfield[6]['psf_b'][0][0, band]
    p0 = psfield[6]['psf_p0'][0][0, band]

    return torch.Tensor([sigma1, sigma2, sigmap, beta, b, p0])

def psf_fun(r, sigma1, sigma2, sigmap, beta, b, p0):
    term1 = torch.exp(-r**2 / (2 * sigma1**2))
    term2 = b * torch.exp(-r**2 / (2 * sigma2**2))
    term3 = p0 * (1 + r**2 / (beta * sigmap**2))**(-beta / 2)

    return (term1 + term2 + term3) / (1 + b + p0)

def get_psf(slen, psf_params, cached_radii_grid = None):
    assert (slen % 2) == 1

    if cached_radii_grid is None:
        grid = simulated_datasets_lib._get_mgrid(slen) * (slen - 1) / 2
        radii_grid = (grid**2).sum(2).sqrt()
    else:
        radii_grid = cached_radii_grid

    return psf_fun(radii_grid, psf_params[0], psf_params[1], psf_params[2],
                   psf_params[3], psf_params[4], psf_params[5])

class PowerLawPSF(nn.Module):
    def __init__(self, init_psf_params,
                    normalization_constant,
                    psf_slen = 25,
                    image_slen =  101):

        super(PowerLawPSF, self).__init__()

        assert len(init_psf_params.shape) == 2
        self.n_bands = init_psf_params.shape[0]
        assert len(normalization_constant) == self.n_bands

        self.init_psf_params = init_psf_params
        self.normalization_constant = normalization_constant

        self.psf_slen = psf_slen
        self.image_slen = image_slen

        grid = _get_mgrid(self.psf_slen) * (self.psf_slen - 1) / 2
        self.cached_radii_grid = (grid**2).sum(2).sqrt().to(device)

        # initial weights
        self.params = nn.Parameter(init_psf_params)

        # initial psf
        self.init_psf = self.get_psf()
        self.init_psf_sum = self.init_psf.sum(-1).sum(-1)

    def get_psf(self):
        # TODO make the psf function vectorized ...
        for i in range(self.n_bands):
            _psf = get_psf(self.psf_slen, self.params[i], self.cached_radii_grid) * \
                        self.normalization_constant[i]

            if i == 0:
                psf = _psf.unsqueeze(0)
            else:
                psf = torch.cat((psf, _psf.unsqueeze(0)))

        return psf

    def forward(self):
        psf = self.get_psf()
        psf = psf * (self.init_psf_sum / psf.sum(-1).sum(-1)).unsqueeze(-1).unsqueeze(-1)

        l_pad = (self.image_slen - self.psf_slen) // 2

        return pad(psf, (l_pad, ) * 4)
