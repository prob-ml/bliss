from typing import Optional, Tuple

import galsim
import numpy as np
import torch
from einops import rearrange
from torch import Tensor, nn

from bliss.simulator.decoder import TileRenderer, fit_source_to_ptile
from bliss.simulator.galsim_decoder import SingleGalsimGalaxyDecoder


class SingleLensedGalsimGalaxyDecoder(SingleGalsimGalaxyDecoder):
    def __init__(
        self,
        slen: int,
        n_bands: int,
        pixel_scale: float,
        psf_params_file: Optional[str] = None,
        psf_slen: Optional[int] = None,
        sdss_bands: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__(
            slen=slen,
            n_bands=n_bands,
            pixel_scale=pixel_scale,
            psf_params_file=psf_params_file,
            psf_slen=psf_slen,
            sdss_bands=sdss_bands,
        )

    def __call__(
        self,
        z_lens: Tensor,
        offset: Optional[Tensor] = None,
    ) -> Tensor:
        if z_lens.shape[0] == 0:
            return torch.zeros(0, 1, self.slen, self.slen, device=z_lens.device)

        if z_lens.shape == (12,):
            assert offset is None or offset.shape == (2,)
            return self.render_lensed_galaxy(z_lens, self.psf_galsim, self.slen, offset)

        images = []
        for ii, lens_params in enumerate(z_lens):
            off = offset if not offset else offset[ii]
            assert off is None or off.shape == (2,)
            image = self.render_lensed_galaxy(lens_params, self.psf_galsim, self.slen, off)
            images.append(image)
        return torch.stack(images, dim=0).to(z_lens.device)

    def sie_deflection(self, x, y, lens_params):
        """Get deflection for grid_sample (in pixels) due to a gravitational lens.

        Adopted from: Adam S. Bolton, U of Utah, 2009

        Args:
            x: images of x coordinates
            y: images of y coordinates
            lens_params: vector of parameters with 5 elements, defined as follows:
                par[0]: lens strength, or 'Einstein radius'
                par[1]: x-center
                par[2]: y-center
                par[3]: e1 ellipticity
                par[4]: e2 ellipticity

        Returns:
            Tuple (xg, yg) of gradients at the positions (x, y)
        """
        b, center_x, center_y, e1, e2 = lens_params.cpu().numpy()
        ell = np.sqrt(e1**2 + e2**2)
        q = (1 - ell) / (1 + ell)
        phirad = np.arctan(e2 / e1)

        # Go into shifted coordinats of the potential:
        xsie = (x - center_x) * np.cos(phirad) + (y - center_y) * np.sin(phirad)
        ysie = (y - center_y) * np.cos(phirad) - (x - center_x) * np.sin(phirad)

        # Compute potential gradient in the transformed system:
        r_ell = np.sqrt(q * xsie**2 + ysie**2 / q)
        qfact = np.sqrt(1.0 / q - q)

        # (r_ell == 0) terms prevent divide-by-zero problems
        eps = 0.001
        if qfact >= eps:
            xtg = (b / qfact) * np.arctan(qfact * xsie / (r_ell + (r_ell == 0)))
            ytg = (b / qfact) * np.arctanh(qfact * ysie / (r_ell + (r_ell == 0)))
        else:
            xtg = b * xsie / (r_ell + (r_ell == 0))
            ytg = b * ysie / (r_ell + (r_ell == 0))

        # Transform back to un-rotated system:
        xg = xtg * np.cos(phirad) - ytg * np.sin(phirad)
        yg = ytg * np.cos(phirad) + xtg * np.sin(phirad)
        return (xg, yg)

    def bilinear_interpolate_numpy(self, im, x, y):
        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, im.shape[1] - 1)
        x1 = np.clip(x1, 0, im.shape[1] - 1)
        y0 = np.clip(y0, 0, im.shape[0] - 1)
        y1 = np.clip(y1, 0, im.shape[0] - 1)

        i_a = im[y0, x0]
        i_b = im[y1, x0]
        i_c = im[y0, x1]
        i_d = im[y1, x1]

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        return (i_a.T * wa).T + (i_b.T * wb).T + (i_c.T * wc).T + (i_d.T * wd).T

    def lens_galsim(self, unlensed_image, lens_params):
        nx, ny = unlensed_image.shape
        x_range = [-nx // 2, nx // 2]
        y_range = [-ny // 2, ny // 2]
        x = (x_range[1] - x_range[0]) * np.outer(np.ones(ny), np.arange(nx)) / float(
            nx - 1
        ) + x_range[0]
        y = (y_range[1] - y_range[0]) * np.outer(np.arange(ny), np.ones(nx)) / float(
            ny - 1
        ) + y_range[0]

        (xg, yg) = self.sie_deflection(x, y, lens_params)
        lensed_image = self.bilinear_interpolate_numpy(
            unlensed_image, (x - xg) + nx // 2, (y - yg) + ny // 2
        )
        return lensed_image.astype(unlensed_image.dtype)

    def render_lensed_galaxy(
        self,
        lens_params: Tensor,
        psf: galsim.GSObject,
        slen: int,
        offset: Optional[Tensor] = None,
    ) -> Tensor:
        lensed_galaxy_params, pure_lens_params = lens_params[:7], lens_params[7:]
        unlensed_src = self._render_galaxy_np(lensed_galaxy_params, psf, slen, offset)
        lensed_src = self.lens_galsim(unlensed_src, pure_lens_params)
        return torch.from_numpy(lensed_src).reshape(1, slen, slen)


class LensedGalaxyTileDecoder(nn.Module):
    def __init__(self, tile_slen, ptile_slen, n_bands, lens_model: SingleLensedGalsimGalaxyDecoder):
        super().__init__()
        self.n_bands = n_bands
        self.tiler = TileRenderer(tile_slen, ptile_slen)
        self.ptile_slen = ptile_slen
        self.lens_decoder = lens_model

    def forward(self, lens_params, lensed_galaxy_bools):
        """Renders galaxy tile from locations and galaxy parameters."""
        # max_sources obtained from locs, allows for more flexibility when rendering.
        n_ptiles = lensed_galaxy_bools.shape[0]
        max_sources = lensed_galaxy_bools.shape[1]

        n_lens_params = lens_params.shape[-1]
        lens_params = lens_params.view(n_ptiles, max_sources, n_lens_params)
        assert lens_params.shape[0] == lensed_galaxy_bools.shape[0] == n_ptiles
        assert lens_params.shape[1] == lensed_galaxy_bools.shape[1] == max_sources
        assert lensed_galaxy_bools.shape[2] == 1

        single_lensed_galaxies = self._render_single_lensed_galaxies(
            lens_params, lensed_galaxy_bools
        )
        single_lensed_galaxies *= lensed_galaxy_bools.unsqueeze(-1).unsqueeze(-1)
        return single_lensed_galaxies

    def _render_single_lensed_galaxies(self, lens_params, lensed_galaxy_bools):
        # flatten parameters
        n_galaxy_params = lens_params.shape[-1]
        z_lens = lens_params.view(-1, n_galaxy_params)
        b_lens = lensed_galaxy_bools.flatten()

        # allocate memory
        slen = self.ptile_slen + ((self.ptile_slen % 2) == 0) * 1
        lensed_gal = torch.zeros(
            z_lens.shape[0], self.n_bands, slen, slen, device=lens_params.device
        )

        # forward only galaxies that are on!
        # no background
        lensed_gal_on = self.lens_decoder(z_lens[b_lens == 1])

        # size the galaxy (either trims or crops to the size of ptile)
        lensed_gal_on = self.size_lens(lensed_gal_on)

        # set galaxies
        lensed_gal[b_lens == 1] = lensed_gal_on

        batchsize = lens_params.shape[0]
        gal_shape = (batchsize, -1, self.n_bands, lensed_gal.shape[-1], lensed_gal.shape[-1])
        return lensed_gal.view(gal_shape)

    def size_lens(self, galaxy: Tensor):
        n_galaxies, n_bands, h, w = galaxy.shape
        assert h == w
        assert (h % 2) == 1, "dimension of galaxy image should be odd"
        assert n_bands == self.n_bands
        galaxy = rearrange(galaxy, "n c h w -> (n c) h w")
        sized_galaxy = fit_source_to_ptile(galaxy, self.ptile_slen)
        outsize = sized_galaxy.shape[-1]
        return sized_galaxy.view(n_galaxies, self.n_bands, outsize, outsize)

    def _sample_lens_params(self, lensed_galaxy_bools):
        """Sample latent galaxy params from GalaxyPrior object."""
        base_shape = list(lensed_galaxy_bools.shape)[:-1]
        device = lensed_galaxy_bools.device
        lens_params = self._sample_param_from_dist(base_shape, 5, torch.rand, device)
        if self.prob_lensed_galaxy > 0.0:
            # latents are: theta_E, center_x/y, e_1/2
            base_radii = self._sample_param_from_dist(base_shape, 1, torch.rand, device)
            base_centers = self._sample_param_from_dist(base_shape, 2, torch.randn, device)
            base_qs = self._sample_param_from_dist(base_shape, 1, torch.rand, device)
            base_betas = self._sample_param_from_dist(base_shape, 1, torch.rand, device)

            lens_params[..., 0:1] = base_radii * 25.0 + 5.0
            lens_params[..., 1:3] = base_centers * 1.0

            # ellipticities must satisfy some angle relationships
            beta_radians = (base_betas - 0.5) * (np.pi / 2)  # [-pi / 4, pi / 4]
            ell_factors = (1 - base_qs) / (1 + base_qs)
            lens_params[..., 3:4] = ell_factors * torch.cos(beta_radians)
            lens_params[..., 4:5] = ell_factors * torch.sin(beta_radians)
        return lens_params * lensed_galaxy_bools
