from pathlib import Path
from typing import Optional

import numpy as np
import torch
from astropy.io import fits
from einops import rearrange, reduce
from torch import nn


def get_mgrid(slen: int):
    offset = (slen - 1) / 2
    # Currently type-checking with mypy doesn't work with np.mgrid
    # See https://github.com/python/mypy/issues/11185.
    x, y = np.mgrid[-offset : (offset + 1), -offset : (offset + 1)]  # type: ignore
    mgrid = torch.tensor(np.dstack((y, x))) / offset
    # mgrid is between -1 and 1
    # then scale slightly because of the way f.grid_sample
    # parameterizes the edges: (0, 0) is center of edge pixel
    return mgrid.float() * (slen - 1) / slen


class PSFDecoder(nn.Module):
    """Abstract decoder class to subclass whenever the decoded result will go through a PSF.

    PSF (point-spread function) use is common for decoding the final realistic astronomical
    image to account for sensor lens effects. PSF loading is suppported as a direct image (npy)
    or through attributes (npy or fits) file.

    For image loading, a psf_image_file for the PSF should be passed in along with the pixel_scale
    for the values in the saved pickled file. For parameter loading, a psf_params_file should
    be indicated, with the corresponding psf_slen and sdss_bands.

    Attributes:
        psf_image_file: Path to pickled .npy file image file
        psf_params_file: PSF parameters (saved either in a numpy file or fits file from SDSS)
        psf_slen: Side-length of the PSF.
        sdss_bands: Bands to retrieve from PSF.
        n_bands: Number of bands to retrieve from PSF.
    """

    def __init__(
        self,
        n_bands: int = 1,
        psf_image_file: Optional[str] = None,
        psf_params_file: Optional[str] = None,
        psf_slen: Optional[int] = None,
        sdss_bands: Optional[str] = None,
    ):
        super().__init__()

        self.n_bands = n_bands

        assert psf_image_file is not None or psf_params_file is not None
        if psf_image_file is not None:
            self.psf = np.load(psf_image_file)
        else:
            assert psf_params_file is not None and psf_slen is not None and sdss_bands is not None
            ext = Path(psf_params_file).suffix
            if ext == ".npy":
                psf_params = torch.from_numpy(np.load(psf_params_file))
                psf_params = psf_params[list(range(n_bands))]
            elif ext == ".fits":
                assert len(sdss_bands) == n_bands
                psf_params = self._get_fit_file_psf_params(psf_params_file, sdss_bands)
            else:
                raise NotImplementedError(
                    "Only .npy and .fits extensions are supported for PSF params files."
                )
            self.params = nn.Parameter(psf_params.clone(), requires_grad=True)
            self.psf_image = None
            self.psf_slen = psf_slen
            grid = get_mgrid(self.psf_slen) * (self.psf_slen - 1) / 2
            # extra factor to be consistent with old repo
            # but probably doesn't matter ...
            grid *= self.psf_slen / (self.psf_slen - 1)
            self.register_buffer("cached_radii_grid", (grid**2).sum(2).sqrt())

            # get psf normalization_constant
            self.normalization_constant = torch.zeros(self.n_bands)
            for i in range(self.n_bands):
                psf_i = self._get_psf_single_band(i)
                self.normalization_constant[i] = 1 / psf_i.sum()
            self.normalization_constant = self.normalization_constant.detach()
            self.psf = self.forward_adjusted_psf().detach().numpy()

    def forward(self, x):
        raise NotImplementedError("Please extend this class and implement forward()")

    def psf_forward(self):
        psf = self._get_psf()
        init_psf_sum = reduce(psf, "n m k -> n", "sum").detach()
        norm = reduce(psf, "n m k -> n", "sum")
        psf *= rearrange(init_psf_sum / norm, "n -> n 1 1")
        return psf

    @staticmethod
    def _get_fit_file_psf_params(psf_fit_file, bands=(2, 3)):
        data = fits.open(psf_fit_file, ignore_missing_end=True).pop(6).data
        psf_params = torch.zeros(len(bands), 6)
        for i, band in enumerate(bands):
            sigma1 = data["psf_sigma1"][0][band] ** 2
            sigma2 = data["psf_sigma2"][0][band] ** 2
            sigmap = data["psf_sigmap"][0][band] ** 2
            beta = data["psf_beta"][0][band]
            b = data["psf_b"][0][band]
            p0 = data["psf_p0"][0][band]

            psf_params[i] = torch.log(torch.tensor([sigma1, sigma2, sigmap, beta, b, p0]))

        return psf_params

    def _get_psf(self):
        psf_list = []
        for i in range(self.n_bands):
            band_psf = self._get_psf_single_band(i)
            band_psf *= self.normalization_constant[i]
            psf_list.append(band_psf.unsqueeze(0))
        psf = torch.cat(psf_list)

        assert (psf > 0).all()
        return psf

    @staticmethod
    def _psf_fun(r, sigma1, sigma2, sigmap, beta, b, p0):
        term1 = torch.exp(-(r**2) / (2 * sigma1))
        term2 = b * torch.exp(-(r**2) / (2 * sigma2))
        term3 = p0 * (1 + r**2 / (beta * sigmap)) ** (-beta / 2)
        return (term1 + term2 + term3) / (1 + b + p0)

    def _get_psf_single_band(self, band_idx):
        psf_params = torch.exp(self.params[band_idx])
        return self._psf_fun(
            self.cached_radii_grid,
            psf_params[0],
            psf_params[1],
            psf_params[2],
            psf_params[3],
            psf_params[4],
            psf_params[5],
        )

    def forward_adjusted_psf(self):
        # use power_law_psf and current psf parameters to forward and obtain fresh psf model.
        # first dimension of psf is number of bands
        # dimension of the psf/slen should be odd
        psf = self.psf_forward()
        psf_slen = psf.shape[2]
        assert len(psf.shape) == 3
        assert psf.shape[0] == self.n_bands
        assert psf.shape[1] == psf_slen
        assert (psf_slen % 2) == 1
        return psf
