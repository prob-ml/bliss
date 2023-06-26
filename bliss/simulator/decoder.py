from pathlib import Path
from typing import Tuple

import galsim
import numpy as np
import torch
from astropy.io import fits
from einops import rearrange, reduce
from omegaconf import DictConfig
from torch import nn

from bliss.catalog import SourceType, TileCatalog
from bliss.surveys.sdss import SDSSDownloader


class ImageDecoder(nn.Module):
    def __init__(
        self,
        pixel_scale: float,
        psf_slen: int,
        sdss_fields: DictConfig,
    ) -> None:
        super().__init__()

        self.n_bands = len(sdss_fields["bands"])
        self.pixel_scale = pixel_scale
        self.psf_slen = psf_slen
        self.psf_galsim = {}  # Dictionary indexed by (run, camcol, field) tuple
        self.psf_params = {}

        sdss_dir = sdss_fields["dir"]
        sdss_bands = sdss_fields["bands"]

        for field_params in sdss_fields["field_list"]:
            run = field_params["run"]
            camcol = field_params["camcol"]
            fields = field_params["fields"]

            for field in fields:
                # load raw params from file
                field_dir = f"{sdss_dir}/{run}/{camcol}/{field}"
                filename = f"{field_dir}/psField-{run:06}-{camcol}-{field:04}.fits"
                if not Path(filename).exists():
                    SDSSDownloader(run, camcol, field, download_dir=sdss_dir).download_psfield()
                psf_params = self._get_fit_file_psf_params(filename, sdss_bands)

                # load psf image from params
                self.psf_galsim[(run, camcol, field)] = self._get_psf(psf_params)
                self.psf_params[(run, camcol, field)] = psf_params

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
        msg = (
            f"{psf_fit_file} does not exist. "
            + "Make sure data files are available for fields specified in config."
        )
        assert Path(psf_fit_file).exists(), msg
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

    def render_star(self, psf, band, source_params):
        """Render a star with given params and PSF.

        Args:
            source_params (Tensor): Tensor containing the parameters for a particular source
                (see prior.py for details about these parameters)
            psf (List): a list of PSFs for each band
            band (int): band

        Returns:
            GSObject: a galsim representation of the rendered star convolved with the PSF
        """
        return psf[band].withFlux(source_params["star_fluxes"][band].item())

    def render_galaxy(self, psf, band, source_params):
        """Render a galaxy with given params and PSF.

        Args:
            source_params (Tensor): Tensor containing the parameters for a particular source
                (see prior.py for details about these parameters)
            psf (List): a list of PSFs for each band
            band (int): band

        Returns:
            GSObject: a galsim representation of the rendered galaxy convolved with the PSF
        """
        galaxy_params = source_params["galaxy_params"]
        total_flux = galaxy_params[band]

        # the remaining parameters are the non-flux parameters for this galaxy
        nonflux_params = galaxy_params[self.n_bands :]
        disk_frac, beta_radians, disk_q, a_d, bulge_q, a_b = nonflux_params

        disk_flux = total_flux * disk_frac
        bulge_frac = 1 - disk_frac
        bulge_flux = total_flux * bulge_frac

        components = []
        if disk_flux > 0:
            b_d = a_d * disk_q
            disk_hlr_arcsecs = np.sqrt(a_d * b_d)
            disk = galsim.Exponential(flux=disk_flux, half_light_radius=disk_hlr_arcsecs)
            sheared_disk = disk.shear(q=disk_q, beta=beta_radians * galsim.radians)
            components.append(sheared_disk)
        if bulge_flux > 0:
            b_b = bulge_q * a_b
            bulge_hlr_arcsecs = np.sqrt(a_b * b_b)
            bulge = galsim.DeVaucouleurs(flux=bulge_flux, half_light_radius=bulge_hlr_arcsecs)
            sheared_bulge = bulge.shear(q=bulge_q, beta=beta_radians * galsim.radians)
            components.append(sheared_bulge)
        galaxy = galsim.Add(components)
        return galsim.Convolution(galaxy, psf[band])

    @property
    def source_renderers(self):
        return {
            SourceType.STAR: self.render_star,
            SourceType.GALAXY: self.render_galaxy,
        }

    def render_images(self, tile_cat: TileCatalog, rcf):
        """Render images from a tile catalog.

        Args:
            tile_cat (TileCatalog): the tile catalog to create images from
            rcf (ndarray): array containing the row/camcol/field of PSFs to use

        Raises:
            AssertionError: rcf must contain batch_size values

        Returns:
            Tuple[Tensor, List, Tensor]: tensor of images, list of PSFs, tensor of PSF params
        """
        batch_size, n_tiles_h, n_tiles_w = tile_cat.n_sources.shape
        assert rcf.shape[0] == batch_size

        slen_h = tile_cat.tile_slen * n_tiles_h
        slen_w = tile_cat.tile_slen * n_tiles_w
        images = np.zeros((batch_size, self.n_bands, slen_h, slen_w), dtype=np.float32)

        full_cat = tile_cat.to_full_params()

        # use the PSF from specified row/camcol/field
        psfs = [self.psf_galsim[tuple(rcf[b])] for b in range(batch_size)]
        param_list = [self.psf_params[tuple(rcf[b])] for b in range(batch_size)]
        psf_params = torch.stack(param_list, dim=0)

        for b in range(batch_size):
            n_sources = int(full_cat.n_sources[b].item())
            psf = psfs[b]
            for band in range(self.n_bands):
                gs_img = galsim.Image(array=images[b, band], scale=self.pixel_scale)
                for s in range(n_sources):
                    source_params = full_cat.one_source(b, s)
                    source_type = source_params["source_type"].item()
                    renderer = self.source_renderers[source_type]  # NOTE: SDSS-Specific!
                    galsim_obj = renderer(psf, band, source_params)
                    plocs0, plocs1 = source_params["plocs"]
                    offset = np.array([plocs1 - (slen_w / 2), plocs0 - (slen_h / 2)])

                    # essentially all the runtime of the simulator is incurred by this call
                    # to drawImage
                    galsim_obj.drawImage(
                        offset=offset, method="auto", add_to_image=True, image=gs_img
                    )

        # clamping here helps with an strange issue caused by galsim rendering
        images = torch.from_numpy(images).clamp(1e-8)

        return images, psfs, psf_params
