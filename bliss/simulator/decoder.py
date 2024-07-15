import copy
from typing import Tuple

import galsim
import numpy as np
import torch
from astropy.wcs import WCS
from einops import rearrange
from torch import nn

from bliss.align import align
from bliss.catalog import SourceType


class ImageDecoder(nn.Module):
    def __init__(
        self,
        psf,
        bands: Tuple[int, ...],
        background,
        flux_calibration_dict: dict,
        ref_band: int,
    ) -> None:
        """Construct a decoder for a set of images.

        Args:
            psf: PSF object
            bands: bands to use for constructing the decoder, passed from Survey
            background: sky backgrounds for each image and each band
            flux_calibration_dict: dictionary specifying elec count conversions by imageid
            ref_band: reference band for pixel alignment
        """

        super().__init__()

        self.n_bands = len(bands)

        self.psf_galsim = psf.psf_galsim  # Dictionary indexed by image_id
        self.psf_params = psf.psf_params  # Dictionary indexed by image_id
        self.psf_draw_method = getattr(psf, "psf_draw_method", "auto")

        self.background = background
        self.background.requires_grad_(False)

        self.pixel_scale = psf.pixel_scale
        self.ref_band = ref_band

        # why is this dict stored as a class attribute, rather than passed to render_images?
        # ditto for the psf parameters above.
        self.flux_calibration_dict = flux_calibration_dict

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

    def pixel_shifts(self, coadd_depth: int, n_bands: int, ref_depth: int = 0, no_dither=False):
        """Generate random pixel shifts and corresponding WCS list.
        This function generates `n_shifts` random pixel shifts `shifts` and corresponding WCS list
        `wcs` to undo these shifts, relative to `wcs[ref_band]`.

        Args:
            coadd_depth (int): number of images per band that will be co-added
            ref_depth (int): depth of the reference band
            n_bands (int): number of bands
            no_dither (bool): if True, set all shifts to zero (primarily intended for testing)

        Returns:
            shifts (np.ndarray): array of pixel shifts
            wcs (List[WCS]): list of WCS objects
        """
        shifts = np.random.uniform(-0.5, 0.5, (coadd_depth, n_bands, 2))
        shifts[ref_depth, self.ref_band] = np.array([0.0, 0.0])
        if no_dither:
            shifts.fill(0.0)

        wcs_base = WCS()
        base = np.array([5.0, 5.0])
        wcs_base.wcs.crpix = base

        wcs_list = [[] for _ in range(coadd_depth)]
        for d in range(coadd_depth):
            for b in range(n_bands):
                if d == ref_depth and b == self.ref_band:
                    wcs_list[d].append(wcs_base.low_level_wcs)
                    continue
                bnd_wcs = WCS()
                bnd_wcs.wcs.crpix = base + shifts[d, b]
                wcs_list[d].append(bnd_wcs.low_level_wcs)
        return shifts, wcs_list

    def coadd_images(self, images):
        batch_size = images.shape[0]
        assert self.coadd_depth > 1, "Coadd depth must be > 1 to use coaddition."
        coadded_images = np.zeros((batch_size, *images.shape[-3:]))
        for b in range(batch_size):
            coadded_images[b] = self.survey.coadd_images(images[b])
        return torch.from_numpy(coadded_images).float()

    def align_images(self, images, wcs_batch):
        """Align images to the reference depth and band."""
        batch_size = images.shape[0]
        for b in range(batch_size):
            aligned_image = align(
                images[b].numpy(),
                wcs_list=wcs_batch[b],
                ref_depth=0,
                ref_band=self.ref_band,
            )
            images[b] = torch.from_numpy(aligned_image)
        return images

    def draw_sources_on_band_image(
        self, band_img, n_sources, full_cat, batch, psf, band, image_dims, shift
    ):
        slen_h, slen_w = image_dims
        for s in range(n_sources):
            source_params = full_cat.one_source(batch, s)
            source_type = source_params["source_type"].item()
            renderer = self.source_renderers[source_type]
            galsim_obj = renderer(psf, band, source_params)
            plocs0, plocs1 = source_params["plocs"]
            offset = np.array([plocs1 - (slen_w / 2), plocs0 - (slen_h / 2)])
            offset += shift

            # essentially all the runtime of the simulator is incurred by this call
            # to drawImage
            galsim_obj.drawImage(
                offset=offset,
                method=self.psf_draw_method,
                add_to_image=True,
                image=band_img,
            )

    def render_images(
        self, tile_cat, image_ids, image_id_indices, coadd_depth=1, add_dither=True, add_noise=True
    ):
        """Render images from a tile catalog."""
        tile_cat = copy.deepcopy(tile_cat)  # make a copy to avoid modifying input
        batch_size, n_tiles_h, n_tiles_w = tile_cat["n_sources"].shape
        assert len(image_ids) == batch_size

        slen_h = tile_cat.tile_slen * n_tiles_h
        slen_w = tile_cat.tile_slen * n_tiles_w
        images_shape = (batch_size * coadd_depth, self.n_bands, slen_h, slen_w)
        background = self.background.sample(images_shape, image_id_indices=image_id_indices)
        background = rearrange(
            background, "b (cd bands) h w -> b cd bands h w", cd=coadd_depth, bands=self.n_bands
        )
        images = background.clone()
        images_np = images.numpy()  # memory is shared between images and images_np

        # use the PSF from specified image_id
        psfs = [self.psf_galsim[image_ids[b]] for b in range(batch_size)]
        param_list = [self.psf_params[image_ids[b]] for b in range(batch_size)]
        psf_params = torch.stack(param_list, dim=0)

        # use the specified flux_calibration ratios indexed by image_id
        flux_calibration_rats = [
            self.flux_calibration_dict[image_ids[b]] for b in range(batch_size)
        ]

        for i in range(batch_size):
            # Convert from (linear) physical units to electron counts
            tile_cat["star_fluxes"][i] *= flux_calibration_rats[i]
            if "galaxy_fluxes" in tile_cat:
                tile_cat["galaxy_fluxes"][i] *= flux_calibration_rats[i]  # noqa: WPS529

        full_cat = tile_cat.to_full_catalog()

        # generate random WCS shifts as manual image dithering via unaligning WCS
        wcs_batch = []

        # this loop is painfully slow and somewhat messy; we should interact with
        # galsim in a more efficient way
        for i in range(batch_size):
            n_sources = int(full_cat["n_sources"][i].item())
            psf = psfs[i]
            for d in range(coadd_depth):
                depth_band_shifts, depth_band_wcs_list = self.pixel_shifts(
                    coadd_depth,
                    self.n_bands,
                    no_dither=(not add_dither),
                )
                wcs_batch.append(depth_band_wcs_list)
                for band in range(self.n_bands):
                    band_img = galsim.Image(array=images_np[i, d, band], scale=self.pixel_scale)
                    self.draw_sources_on_band_image(
                        band_img,
                        n_sources,
                        full_cat,
                        i,
                        psf,
                        band,
                        image_dims=(slen_h, slen_w),
                        shift=depth_band_shifts[d, band],
                    )

                    if add_noise:
                        poisson_noise = galsim.PoissonNoise(sky_level=0.0)  # noqa: WPS220
                        band_img.addNoise(poisson_noise)  # noqa: WPS220

                    # we're producing sky subtracted images
                    band_img -= background[i, d, band].numpy()

                    # convert electron counts to physical units
                    band_img /= flux_calibration_rats[i][band]

        images = self.align_images(images, wcs_batch)

        if coadd_depth > 1:
            images = self.coadd_images(images)
        else:
            images = images.squeeze(1)

        return images, psf_params
