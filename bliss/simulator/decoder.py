from typing import Tuple

import galsim
import numpy as np
import torch
from torch import nn

from bliss.catalog import SourceType, TileCatalog


class ImageDecoder(nn.Module):
    def __init__(
        self,
        psf,
        bands: Tuple[int, ...] = (0, 1, 2, 3, 4),
    ) -> None:
        super().__init__()

        self.n_bands = len(bands)
        self.psf_galsim = psf.psf_galsim  # Dictionary indexed by image_id
        self.psf_params = psf.psf_params  # Dictionary indexed by image_id
        self.pixel_scale = psf.pixel_scale

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
            psf (List): a list of PSFs for each band
            band (int): band
            source_params (Tensor): Tensor containing the parameters for a particular source
                (see prior.py for details about these parameters)

        Returns:
            GSObject: a galsim representation of the rendered galaxy convolved with the PSF
        """
        galaxy_fluxes = source_params["galaxy_fluxes"]
        galaxy_params = source_params["galaxy_params"]

        total_flux = galaxy_fluxes[band]
        disk_frac, beta_radians, disk_q, a_d, bulge_q, a_b = galaxy_params

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

    def render_images(self, tile_cat: TileCatalog, image_ids):
        """Render images from a tile catalog.

        Args:
            tile_cat (TileCatalog): the tile catalog to create images from
            image_ids (ndarray): array containing the image_ids of PSFs to use

        Raises:
            AssertionError: image_ids must contain `batch_size` values

        Returns:
            Tuple[Tensor, List, Tensor]: tensor of images, list of PSFs, tensor of PSF params
        """
        batch_size, n_tiles_h, n_tiles_w = tile_cat.n_sources.shape
        assert image_ids.shape[0] == batch_size

        slen_h = tile_cat.tile_slen * n_tiles_h
        slen_w = tile_cat.tile_slen * n_tiles_w
        images = np.zeros((batch_size, self.n_bands, slen_h, slen_w), dtype=np.float32)

        full_cat = tile_cat.to_full_params()

        # use the PSF from specified row/camcol/field
        psfs = [self.psf_galsim[tuple(image_ids[b])] for b in range(batch_size)]
        param_list = [self.psf_params[tuple(image_ids[b])] for b in range(batch_size)]
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
