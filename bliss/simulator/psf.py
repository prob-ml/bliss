from typing import TypedDict

import galsim
import numpy as np
import torch
from einops import rearrange, reduce

PSFConfig = TypedDict(
    "PSFConfig",
    {
        "pixel_scale": float,
        "psf_slen": int,
    },
)


class ImagePSF:
    def __init__(self, bands, pixel_scale, psf_slen):
        self.bands = bands
        self.pixel_scale = pixel_scale
        self.psf_slen = psf_slen

    def _get_psf(self, params):
        """Construct PSF image from parameters. This is the main entry point for generating the psf.

        Args:
            params: list of psf parameters, loaded from _get_fit_file_psf_params

        Returns:
            images (List[InterpolatedImage]): list of psf transformations for each band
        """
        # get psf in each band
        psf_list = []
        for b in self.bands:
            grid = self._get_mgrid() * (self.psf_slen - 1) / 2
            radii_grid = (grid**2).sum(2).sqrt()
            band_psf = self._psf_fun(radii_grid, *params[b])
            psf_list.append(band_psf.unsqueeze(0))
        psf = torch.cat(psf_list)
        assert (psf > 0).all()

        # ensure it's normalized
        norm = reduce(psf, "b m k -> b", "sum")
        psf *= rearrange(1 / norm, "b -> b 1 1")

        # check format
        n_bands, psf_slen, _ = psf.shape
        assert n_bands == len(self.bands) and (psf_slen % 2) == 1 and psf_slen == psf.shape[2]

        # convert to image
        images = []
        for b in self.bands:
            psf_image = galsim.Image(psf.detach().numpy()[b], scale=self.pixel_scale)
            images.append(galsim.InterpolatedImage(psf_image).withFlux(1.0))

        return images

    def _get_mgrid(self):
        """Construct the base grid for the PSF."""
        offset = (self.psf_slen - 1) / 2
        x, y = np.mgrid[-offset : (offset + 1), -offset : (offset + 1)]
        mgrid = torch.tensor(np.dstack((y, x))) / offset
        return mgrid.float()

    def _psf_fun(self, r, **params):
        """Generate the PSF from the parameters.

        Args:
            r: radius
            **params: psf parameters (e.g., (sigma1, sigma2, sigmap, beta, b, p0) for SDSS)

        Raises:
            NotImplementedError: if the psf function is not implemented
        """
        raise NotImplementedError
