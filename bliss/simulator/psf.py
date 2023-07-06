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


class ImagePointSpreadFunction:
    def __init__(self, bands, pixel_scale, psf_slen):
        self.n_bands = len(bands)
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
        for i in range(self.n_bands):
            grid = self._get_mgrid() * (self.psf_slen - 1) / 2
            radii_grid = (grid**2).sum(2).sqrt()
            band_psf = self._psf_fun(radii_grid, *params[i])
            psf_list.append(band_psf.unsqueeze(0))
        psf = torch.cat(psf_list)
        assert (psf > 0).all()

        # ensure it's normalized
        norm = reduce(psf, "b m k -> b", "sum")
        psf *= rearrange(1 / norm, "b -> b 1 1")

        # check format
        n_bands, psf_slen, _ = psf.shape
        assert n_bands == self.n_bands and (psf_slen % 2) == 1 and psf_slen == psf.shape[2]

        # convert to image
        images = []
        for i in range(self.n_bands):
            psf_image = galsim.Image(psf.detach().numpy()[i], scale=self.pixel_scale)
            images.append(galsim.InterpolatedImage(psf_image).withFlux(1.0))

        return images

    def _get_mgrid(self):
        """Construct the base grid for the PSF."""
        offset = (self.psf_slen - 1) / 2
        x, y = np.mgrid[-offset : (offset + 1), -offset : (offset + 1)]
        mgrid = torch.tensor(np.dstack((y, x))) / offset
        return mgrid.float()

    @staticmethod
    def _psf_fun(r, sigma1, sigma2, sigmap, beta, b, p0):
        """Generate the PSF from the parameters using the power-law model.

        See https://data.sdss.org/datamodel/files/PHOTO_REDUX/RERUN/RUN/objcs/CAMCOL/psField.html
        for details on the parameters and the equation used.

        Args:
            r: radius
            sigma1: Inner gaussian sigma for the composite fit
            sigma2: Outer gaussian sigma for the composite fit
            sigmap: Width parameter for power law (pixels)
            beta: Slope of power law.
            b: Ratio of the outer PSF to the inner PSF at the origin
            p0: The value of the power law at the origin.

        Returns:
            The psf function evaluated at r.
        """

        term1 = torch.exp(-(r**2) / (2 * sigma1))
        term2 = b * torch.exp(-(r**2) / (2 * sigma2))
        term3 = p0 * (1 + r**2 / (beta * sigmap)) ** (-beta / 2)
        return (term1 + term2 + term3) / (1 + b + p0)
