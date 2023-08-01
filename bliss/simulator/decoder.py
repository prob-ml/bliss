from typing import Tuple

import galsim
import numpy as np
import torch
from astropy.wcs import WCS
from torch import nn

from bliss.catalog import SourceType, TileCatalog


class ImageDecoder(nn.Module):
    def __init__(
        self, psf, bands: Tuple[int, ...], nmgy_to_nelec_dict: dict, pixel_shift: int, ref_band: int
    ) -> None:
        """Construct a decoder for a set of images.

        Args:
            psf: PSF object
            bands: bands to use for constructing the decoder, passed from Survey
            nmgy_to_nelec_dict: dicitonary specifying elec count conversions by imageid
            ref_band: reference band for pixel alignment
            pixel_shift: int indicating parameters for a Unif() to model pixel shifting
        """

        super().__init__()

        self.n_bands = len(bands)
        self.psf_galsim = psf.psf_galsim  # Dictionary indexed by image_id
        self.psf_params = psf.psf_params  # Dictionary indexed by image_id
        self.psf_draw_method = getattr(psf, "psf_draw_method", "auto")
        self.pixel_scale = psf.pixel_scale
        self.ref_band = ref_band
        self.shift = pixel_shift
        self.nmgy_to_nelec_dict = nmgy_to_nelec_dict

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

    def pixel_shift(self):
        """Generate random pixel shift and corresponding WCS list to undo shifts."""
        shift = np.random.uniform(-self.shift, self.shift, (self.n_bands, 2))
        shift[self.ref_band] = np.array([0.0, 0.0])
        wcs_base = WCS()
        base = np.array([5.0, 5.0])
        wcs_base.wcs.crpix = base
        wcs = []
        for i in range(self.n_bands):
            if i == self.ref_band:
                wcs.append(wcs_base.low_level_wcs)
                continue
            bnd_wcs = WCS()
            bnd_wcs.wcs.crpix = base + shift[i]
            wcs.append(bnd_wcs.low_level_wcs)
        return shift, wcs

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
        assert len(image_ids) == batch_size

        slen_h = tile_cat.tile_slen * n_tiles_h
        slen_w = tile_cat.tile_slen * n_tiles_w
        images = np.zeros((batch_size, self.n_bands, slen_h, slen_w), dtype=np.float32)

        # use the PSF from specified image_id
        psfs = [self.psf_galsim[image_ids[b]] for b in range(batch_size)]
        param_list = [self.psf_params[image_ids[b]] for b in range(batch_size)]
        psf_params = torch.stack(param_list, dim=0)

        # use the specified nmgy_to_nelec ratios indexed by image_id
        nmgy_to_nelec_rats = [self.nmgy_to_nelec_dict[image_ids[b]] for b in range(batch_size)]

        for b in range(batch_size):
            # Convert to electron counts
            tile_cat["star_fluxes"][b] *= nmgy_to_nelec_rats[b]
            tile_cat["galaxy_fluxes"][b] *= nmgy_to_nelec_rats[b]

        full_cat = tile_cat.to_full_params()
        wcs_batch = []

        for b in range(batch_size):
            n_sources = int(full_cat.n_sources[b].item())
            psf = psfs[b]
            shift, wcs = self.pixel_shift()
            wcs_batch.append(wcs)
            for band in range(self.n_bands):
                gs_img = galsim.Image(array=images[b, band], scale=self.pixel_scale)
                for s in range(n_sources):
                    source_params = full_cat.one_source(b, s)
                    source_type = source_params["source_type"].item()
                    renderer = self.source_renderers[source_type]  # NOTE: SDSS-Specific!
                    galsim_obj = renderer(psf, band, source_params)
                    plocs0, plocs1 = source_params["plocs"]
                    offset = np.array([plocs1 - (slen_w / 2), plocs0 - (slen_h / 2)])
                    offset += shift[band]

                    # essentially all the runtime of the simulator is incurred by this call
                    # to drawImage
                    galsim_obj.drawImage(
                        offset=offset,
                        method=self.psf_draw_method,
                        add_to_image=True,
                        image=gs_img,
                    )

        # clamping here helps with an strange issue caused by galsim rendering
        images = torch.from_numpy(images).clamp(1e-8)

        return images, psfs, psf_params, wcs_batch
