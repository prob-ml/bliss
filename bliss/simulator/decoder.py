import galsim
import numpy as np
import torch
from astropy.wcs import WCS
from einops import rearrange
from torch import nn

from bliss.align import align
from bliss.catalog import SourceType, TileCatalog
from bliss.surveys.survey import Survey


class Decoder(nn.Module):
    def __init__(
        self,
        tile_slen: int,
        survey: Survey,
        with_dither: bool = True,
        with_noise: bool = True,
    ) -> None:
        """Construct a decoder for a set of images.

        Args:
            tile_slen: side length in pixels of a tile
            survey: survey to mimic (psf, background, calibration, etc.)
            with_dither: if True, apply random pixel shifts to the images and align them
            with_noise: if True, add Poisson noise to the image pixels
        """

        super().__init__()

        self.tile_slen = tile_slen
        self.survey = survey
        self.with_dither = with_dither
        self.with_noise = with_noise

        survey.prepare_data()

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

    def pixel_shifts(self):
        """Generate random pixel shifts and corresponding WCS list.
        This function generates `n_shifts` random pixel shifts `shifts` and corresponding WCS list
        `wcs` to undo these shifts, relative to `wcs[ref_band]`.

        Returns:
            shifts (np.ndarray): array of pixel shifts
            wcs (List[WCS]): list of WCS objects
        """
        n_bands = len(self.survey.BANDS)
        shifts = np.random.uniform(-0.5, 0.5, (n_bands, 2))
        shifts[self.survey.align_to_band] = np.array([0.0, 0.0])

        wcs_base = WCS()
        base = np.array([5.0, 5.0])  # dither this too for coadds?
        wcs_base.wcs.crpix = base

        wcs_list = []
        for b in range(n_bands):
            if b == self.survey.align_to_band:
                wcs_list.append(wcs_base.low_level_wcs)
                continue
            bnd_wcs = WCS()
            bnd_wcs.wcs.crpix = base + shifts[b]
            wcs_list.append(bnd_wcs.low_level_wcs)
        return shifts, wcs_list

    def coadd_images(self, images):
        batch_size = images.shape[0]
        assert self.coadd_depth > 1, "Coadd depth must be > 1 to use coaddition."
        coadded_images = np.zeros((batch_size, *images.shape[-3:]))
        for b in range(batch_size):
            coadded_images[b] = self.survey.coadd_images(images[b])
        return torch.from_numpy(coadded_images).float()

    def render_image(self, tile_cat):
        """Render a single image from a tile catalog."""
        batch_size, n_tiles_h, n_tiles_w = tile_cat["n_sources"].shape
        assert batch_size == 1

        slen_h = self.tile_slen * n_tiles_h
        slen_w = self.tile_slen * n_tiles_w

        image = np.zeros((len(self.survey.BANDS), slen_h, slen_w), dtype=np.float32)

        image_idx = np.random.randint(len(self.survey), dtype=int)
        frame = self.survey[image_idx]

        # sample background from a random position in the frame
        height, width = frame["background"].shape[-2:]
        h_diff, w_diff = height - slen_h, width - slen_w
        h = 0 if h_diff == 0 else np.random.randint(h_diff)
        w = 0 if w_diff == 0 else np.random.randint(w_diff)
        background = frame["background"][:, h : (h + slen_h), w : (w + slen_w)]
        image += background

        full_cat = tile_cat.to_full_catalog(self.tile_slen)
        n_sources = int(full_cat["n_sources"][0].item())

        # calibration: convert from (linear) physical units to electron counts
        # use the specified flux_calibration ratios indexed by image_id
        avg_nelec_conv = np.mean(frame["flux_calibration"], axis=-1)
        if n_sources > 0:
            full_cat["star_fluxes"] *= rearrange(avg_nelec_conv, "bands -> 1 1 bands")
            if "galaxy_fluxes" in tile_cat:
                full_cat["galaxy_fluxes"] *= avg_nelec_conv

        # generate random WCS shifts as manual image dithering via unaligning WCS
        if self.with_dither:
            pixel_shifts, wcs_list = self.pixel_shifts()

        for band, _band_letter in enumerate(self.survey.BANDS):
            band_img = galsim.Image(array=image[band], scale=self.survey.psf.pixel_scale)

            for s in range(n_sources):
                source_params = full_cat.one_source(0, s)
                source_type = source_params["source_type"].item()
                renderer = self.source_renderers[source_type]
                galsim_obj = renderer(frame["psf_galsim"], band, source_params)
                plocs0, plocs1 = source_params["plocs"]
                offset = np.array([plocs1 - (slen_w / 2), plocs0 - (slen_h / 2)])
                if self.with_dither:
                    offset += pixel_shifts[band]

                # essentially all the runtime of the simulator is incurred by this call
                # to drawImage
                galsim_obj.drawImage(
                    offset=offset,
                    method=getattr(self.survey.psf, "psf_draw_method", "auto"),
                    add_to_image=True,
                    image=band_img,
                )

            if self.with_noise:
                poisson_noise = galsim.PoissonNoise(sky_level=0.0)
                band_img.addNoise(poisson_noise)

        # we're producing sky subtracted images
        image -= background

        # convert electron counts to physical units (now what we've subtracted the background)
        image /= rearrange(avg_nelec_conv, "bands -> bands 1 1")

        if self.with_dither:
            image = align(image, [wcs_list], self.survey.align_to_band)

        return torch.from_numpy(image), frame["psf_params"]

    def render_images(self, tile_cat):
        """Render images from a tile catalog."""
        batch_size = tile_cat["n_sources"].shape[0]

        images = []
        psf_params = []
        for i in range(batch_size):
            d = {k: v[i : (i + 1)] for k, v in tile_cat.items()}
            tc_one = TileCatalog(d)
            image, psf_param = self.render_image(tc_one)
            images.append(image)
            psf_params.append(psf_param)

        return torch.stack(images, dim=0), torch.stack(psf_params, dim=0)
