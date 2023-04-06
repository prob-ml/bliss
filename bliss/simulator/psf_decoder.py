from pathlib import Path
from typing import Callable, Optional, Tuple

import galsim
import numpy as np
import torch
from astropy.io import fits
from einops import rearrange, reduce
from torch import Tensor, nn


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

    For parameter loading, a psf_params_file should be indicated, with the corresponding
    psf_slen and sdss_bands.

    Attributes:
        psf_params_file: PSF parameters (saved either in a numpy file or fits file from SDSS)
        psf_slen: Side-length of the PSF.
        sdss_bands: Bands to retrieve from PSF.
        n_bands: Number of bands to retrieve from PSF.
    """

    forward: Callable[..., Tensor]

    def __init__(
        self,
        n_bands: int = 1,
        pixel_scale: float = 0.393,
        psf_gauss_fwhm: Optional[float] = None,
        psf_params_file: Optional[str] = None,
        psf_slen: Optional[int] = None,
        sdss_bands: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.params = None  # psf params from fits file

        assert psf_params_file is not None or psf_gauss_fwhm is not None
        if psf_gauss_fwhm is not None:
            assert psf_slen is not None
            self.psf_galsim = galsim.Gaussian(fwhm=psf_gauss_fwhm, flux=1.0)
            self.psf = self.psf_galsim.drawImage(nx=psf_slen, ny=psf_slen, scale=pixel_scale).array
            self.psf = self.psf.reshape(1, psf_slen, psf_slen)

        else:
            assert psf_params_file is not None and psf_slen is not None and sdss_bands is not None
            assert Path(psf_params_file).suffix == ".fits"
            assert len(sdss_bands) == n_bands
            psf_params = self._get_fit_file_psf_params(psf_params_file, sdss_bands)
            self.params = nn.Parameter(psf_params.clone(), requires_grad=True)
            self.psf_slen = psf_slen
            grid = get_mgrid(self.psf_slen) * (self.psf_slen - 1) / 2
            # Bryan: extra factor to be consistent with old repo, probably unimportant...
            grid *= self.psf_slen / (self.psf_slen - 1)
            self.register_buffer("cached_radii_grid", (grid**2).sum(2).sqrt())

            self.psf = self.forward_psf_from_params().detach().numpy()
            psf_image = galsim.Image(self.psf[0], scale=pixel_scale)
            self.psf_galsim = galsim.InterpolatedImage(psf_image).withFlux(1.0)

    def forward(self, x):  # type: ignore
        raise NotImplementedError("Please extend this class and implement forward()")

    def forward_psf_from_params(self):
        assert self.params is not None, "Can only be used when `psf_params_file` is given."
        # get psf in each band
        psf_list = []
        for i in range(self.n_bands):
            band_psf = self._psf_fun(self.cached_radii_grid, *self.params[i])
            psf_list.append(band_psf.unsqueeze(0))
        psf = torch.cat(psf_list)
        assert (psf > 0).all()

        # ensure it's normalized
        norm = reduce(psf, "b m k -> b", "sum")
        psf *= rearrange(1 / norm, "b -> b 1 1")

        # check format
        n_bands, psf_slen, _ = psf.shape
        assert n_bands == self.n_bands and (psf_slen % 2) == 1 and psf_slen == psf.shape[2]
        return psf

    @staticmethod
    def _get_fit_file_psf_params(psf_fit_file: str, bands: Tuple[int, ...]):
        data = fits.open(psf_fit_file, ignore_missing_end=True).pop(6).data
        psf_params = torch.zeros(len(bands), 6)
        for i, band in enumerate(bands):
            sigma1 = data["psf_sigma1"][0][band] ** 2
            sigma2 = data["psf_sigma2"][0][band] ** 2
            sigmap = data["psf_sigmap"][0][band] ** 2
            beta = data["psf_beta"][0][band]
            b = data["psf_b"][0][band]
            p0 = data["psf_p0"][0][band]

            psf_params[i] = torch.tensor([sigma1, sigma2, sigmap, beta, b, p0])

        return psf_params

    @staticmethod
    def _psf_fun(r, sigma1, sigma2, sigmap, beta, b, p0):
        term1 = torch.exp(-(r**2) / (2 * sigma1))
        term2 = b * torch.exp(-(r**2) / (2 * sigma2))
        term3 = p0 * (1 + r**2 / (beta * sigmap)) ** (-beta / 2)
        return (term1 + term2 + term3) / (1 + b + p0)
