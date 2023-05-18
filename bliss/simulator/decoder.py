from pathlib import Path
from typing import Tuple

import galsim
import numpy as np
import torch
from astropy.io import fits
from einops import rearrange, reduce
from torch import Tensor, nn

from bliss.catalog import TileCatalog


class ImageDecoder(nn.Module):
    def __init__(
        self,
        n_bands: int,
        pixel_scale: float,
        psf_params_file: str,
        psf_slen: int,
        sdss_bands: Tuple[int, ...],
    ) -> None:
        super().__init__()
        assert n_bands == 1, "Only 1 band is supported"
        assert Path(psf_params_file).suffix == ".fits"
        assert len(sdss_bands) == n_bands

        self.n_bands = n_bands
        self.pixel_scale = pixel_scale
        self.psf_slen = psf_slen

        # load raw params from file
        params = self._get_fit_file_psf_params(psf_params_file, sdss_bands)
        self.register_buffer("params", params)  # don't need to update params, so use buffer

        # generate grid for psf
        grid = self._get_mgrid() * (self.psf_slen - 1) / 2
        self.register_buffer("cached_radii_grid", (grid**2).sum(2).sqrt())

        self.psf_galsim = self._get_psf()

    def _get_psf(self):
        """Construct PSF image from parameters. This is the main entry point for generating the psf.

        Returns:
            galsim_psf (InterpolatedImage): the psf transformation to be used by GalSim
        """
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

        # convert to image
        psf_image = galsim.Image(psf.detach().numpy()[0], scale=self.pixel_scale)
        return galsim.InterpolatedImage(psf_image).withFlux(1.0)

    def _get_mgrid(self):
        """Construct the base grid for the PSF."""
        offset = (self.psf_slen - 1) / 2
        x, y = np.mgrid[-offset : (offset + 1), -offset : (offset + 1)]  # type: ignore
        mgrid = torch.tensor(np.dstack((y, x))) / offset
        return mgrid.float()

    @staticmethod
    def _get_fit_file_psf_params(psf_fit_file: str, bands: Tuple[int, ...]):
        """Load psf parameters from fits file.

        See https://data.sdss.org/datamodel/files/PHOTO_REDUX/RERUN/RUN/objcs/CAMCOL/psField.html
        for details on the parameters.

        Args:
            psf_fit_file (str): file to load from
            bands (Tuple[int, ...]): SDSS bands to load

        Returns:
            psf_params: tensor of parameters for each band
        """
        # HDU 6 contains the PSF header (after primary and eigenimages)
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

    def render_galaxy(self, galaxy_params: Tensor):
        galaxy_params = galaxy_params.cpu().detach()
        total_flux, disk_frac, beta_radians, disk_q, a_d, bulge_q, a_b = galaxy_params
        bulge_frac = 1 - disk_frac

        disk_flux = total_flux * disk_frac
        bulge_flux = total_flux * bulge_frac

        components = []
        if disk_flux > 0:
            b_d = a_d * disk_q
            disk_hlr_arcsecs = np.sqrt(a_d * b_d)
            disk = galsim.Exponential(flux=disk_flux, half_light_radius=disk_hlr_arcsecs).shear(
                q=disk_q,
                beta=beta_radians * galsim.radians,
            )
            components.append(disk)
        if bulge_flux > 0:
            b_b = bulge_q * a_b
            bulge_hlr_arcsecs = np.sqrt(a_b * b_b)
            bulge = galsim.DeVaucouleurs(flux=bulge_flux, half_light_radius=bulge_hlr_arcsecs)
            sheared_bulge = bulge.shear(q=bulge_q, beta=beta_radians * galsim.radians)
            components.append(sheared_bulge)
        galaxy = galsim.Add(components)
        return galsim.Convolution(galaxy, self.psf_galsim)

    def render_images(self, tile_cat: TileCatalog):
        batch_size, n_tiles_h, n_tiles_w = tile_cat.n_sources.shape
        slen_h = tile_cat.tile_slen * n_tiles_h
        slen_w = tile_cat.tile_slen * n_tiles_w
        images = np.zeros((batch_size, self.n_bands, slen_h, slen_w), dtype=np.float32)

        full_cat = tile_cat.to_full_params()

        for b in range(batch_size):
            n_sources = int(full_cat.n_sources[b].item())
            gs_img = galsim.Image(array=images[b, 0], scale=self.pixel_scale)
            for s in range(n_sources):
                if full_cat["galaxy_bools"][b][s] == 1:
                    galsim_obj = self.render_galaxy(full_cat["galaxy_params"][b][s])
                elif full_cat["star_bools"][b][s] == 1:
                    galsim_obj = self.psf_galsim.withFlux(full_cat["star_fluxes"][b][s].item())
                else:
                    raise AssertionError("Every source is a star or galaxy")

                plocs = full_cat.plocs[b][s]
                offset = np.array([plocs[1] - (slen_w / 2), plocs[0] - (slen_h / 2)])
                # essentially all the runtime of the simulator is incurred by this call to drawImage
                galsim_obj.drawImage(offset=offset, method="auto", add_to_image=True, image=gs_img)

        return torch.from_numpy(images)
