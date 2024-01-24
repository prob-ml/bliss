from typing import Optional

import galsim
import numpy as np
import torch
from torch import Tensor

from bliss.catalog import FullCatalog, TileCatalog
from bliss.models.psf_decoder import PSFDecoder


class SingleGalsimGalaxyDecoder(PSFDecoder):
    def __init__(
        self,
        slen: int,
        n_bands: int,
        pixel_scale: float,
        psf_slen: Optional[int] = None,
    ) -> None:
        super().__init__(
            psf_slen=psf_slen,
            n_bands=n_bands,
            pixel_scale=pixel_scale,
        )
        assert len(self.psf.shape) == 3 and self.psf.shape[0] == 1

        assert n_bands == 1, "Only 1 band is supported"
        self.slen = slen
        self.n_bands = 1
        self.pixel_scale = pixel_scale

    def __call__(self, z: Tensor, offset: Optional[Tensor] = None) -> Tensor:
        if z.shape[0] == 0:
            return torch.zeros(0, 1, self.slen, self.slen, device=z.device)

        if z.shape == (7,):
            assert offset is None or offset.shape == (2,)
            return self.render_galaxy(z, self.slen, offset)

        images = []
        for ii, latent in enumerate(z):
            off = offset if not offset else offset[ii]
            assert off is None or off.shape == (2,)
            image = self.render_galaxy(latent, self.slen, off)
            images.append(image)
        return torch.stack(images, dim=0).to(z.device)

    def _render_galaxy_np(
        self,
        galaxy_params: Tensor,
        psf: galsim.GSObject,
        slen: int,
        offset: Optional[Tensor] = None,
    ) -> Tensor:
        assert offset is None or offset.shape == (2,)
        if isinstance(galaxy_params, Tensor):
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
            bulge = galsim.DeVaucouleurs(
                flux=bulge_flux, half_light_radius=bulge_hlr_arcsecs
            ).shear(q=bulge_q, beta=beta_radians * galsim.radians)
            components.append(bulge)
        galaxy = galsim.Add(components)
        gal_conv = galsim.Convolution(galaxy, psf)
        offset = offset if offset is None else offset.numpy()
        return gal_conv.drawImage(nx=slen, ny=slen, scale=self.pixel_scale, offset=offset).array

    def render_galaxy(
        self,
        galaxy_params: Tensor,
        slen: int,
        offset: Optional[Tensor] = None,
    ) -> Tensor:
        image = self._render_galaxy_np(galaxy_params, self.psf_galsim, slen, offset)
        return torch.from_numpy(image).reshape(1, slen, slen)


class FullCatalogDecoder:
    def __init__(
        self, single_galaxy_decoder: SingleGalsimGalaxyDecoder, slen: int, bp: int
    ) -> None:
        self.single_galaxy_decoder = single_galaxy_decoder
        self.slen = slen
        self.bp = bp
        assert self.slen + 2 * self.bp >= self.single_galaxy_decoder.slen
        self.pixel_scale = self.single_galaxy_decoder.pixel_scale

    def __call__(self, full_cat: FullCatalog):
        return self.render_catalog(full_cat)

    def _render_star(self, flux: float, slen: int, offset: Optional[Tensor] = None) -> Tensor:
        assert offset is None or offset.shape == (2,)
        star = self.single_galaxy_decoder.psf_galsim.withFlux(flux)  # creates a copy
        offset = offset if offset is None else offset.numpy()
        image = star.drawImage(nx=slen, ny=slen, scale=self.pixel_scale, offset=offset)
        return torch.from_numpy(image.array).reshape(1, slen, slen)

    def render_catalog(self, full_cat: FullCatalog):
        size = self.slen + 2 * self.bp
        full_plocs = full_cat.plocs
        b, max_n_sources, _ = full_plocs.shape
        assert b == 1, "Only one batch supported for now."
        assert self.single_galaxy_decoder.n_bands == 1, "Only 1 band supported for now"

        image = torch.zeros(1, size, size)
        noiseless_centered = torch.zeros(max_n_sources, 1, size, size)
        noiseless_uncentered = torch.zeros(max_n_sources, 1, size, size)

        n_sources = int(full_cat.n_sources[0].item())
        galaxy_params = full_cat["galaxy_params"][0]
        star_fluxes = full_cat["star_fluxes"][0]
        galaxy_bools = full_cat["galaxy_bools"][0]
        star_bools = full_cat["star_bools"][0]
        plocs = full_plocs[0]
        for ii in range(n_sources):
            offset_x = plocs[ii][1] + self.bp - size / 2
            offset_y = plocs[ii][0] + self.bp - size / 2
            offset = torch.tensor([offset_x, offset_y])
            if galaxy_bools[ii] == 1:
                centered = self.single_galaxy_decoder.render_galaxy(galaxy_params[ii], size)
                uncentered = self.single_galaxy_decoder.render_galaxy(
                    galaxy_params[ii], size, offset
                )
            elif star_bools[ii] == 1:
                centered = self._render_star(star_fluxes[ii][0].item(), size)
                uncentered = self._render_star(star_fluxes[ii][0].item(), size, offset)
            else:
                continue
            noiseless_centered[ii] = centered
            noiseless_uncentered[ii] = uncentered
            image += uncentered

        return image, noiseless_centered, noiseless_uncentered

    def forward_tile(self, tile_cat: TileCatalog):
        full_cat = tile_cat.to_full_params()
        return self(full_cat)
