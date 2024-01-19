from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor, nn

from bliss.datasets.lsst import get_default_lsst_psf


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
    image to account for atmospheric and camera lens effects.
    PSF loading is suppported as a direct image (npy) or through attributes (npy or fits) file.

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
        psf_slen: int,
        n_bands: int = 1,
        pixel_scale: float = 0.2,  # lsst
        atmospheric_model: Optional[str] = "Kolmogorov",
    ):
        super().__init__()
        self.n_bands = n_bands
        self.psf_galsim = get_default_lsst_psf(atmospheric_model=atmospheric_model)

        psf = self.psf_galsim.drawImage(nx=psf_slen, ny=psf_slen, scale=pixel_scale).array
        self.psf = psf.reshape(1, psf_slen, psf_slen)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
