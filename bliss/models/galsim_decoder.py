from typing import Dict, Optional, Tuple

import galsim
import numpy as np
import torch
from torch import Tensor

from bliss.catalog import FullCatalog, TileCatalog
from bliss.models.psf_decoder import PSFDecoder


class SingleGalsimGalaxyPrior:
    dim_latents = 7

    def __init__(
        self,
        flux_sample: str,
        min_flux: float,
        max_flux: float,
        a_sample: str,
        alpha: Optional[float] = None,
        min_a_d: Optional[float] = None,
        max_a_d: Optional[float] = None,
        min_a_b: Optional[float] = None,
        max_a_b: Optional[float] = None,
        a_concentration: Optional[float] = None,
        a_loc: Optional[float] = None,
        a_scale: Optional[float] = None,
        a_bulge_disk_ratio: Optional[float] = None,
    ) -> None:
        self.flux_sample = flux_sample
        self.min_flux = min_flux
        self.max_flux = max_flux
        self.alpha = alpha
        if self.flux_sample == "pareto":
            assert self.alpha is not None

        self.a_sample = a_sample
        self.min_a_d = min_a_d
        self.max_a_d = max_a_d
        self.min_a_b = min_a_b
        self.max_a_b = max_a_b

        self.a_concentration = a_concentration
        self.a_loc = a_loc
        self.a_scale = a_scale
        self.a_bulge_disk_ratio = a_bulge_disk_ratio

        if self.a_sample == "uniform":
            assert self.min_a_d is not None
            assert self.max_a_d is not None
            assert self.min_a_b is not None
            assert self.max_a_b is not None
        elif self.a_sample == "gamma":
            assert self.a_concentration is not None
            assert self.a_loc is not None
            assert self.a_scale is not None
            assert self.a_bulge_disk_ratio is not None
        else:
            raise NotImplementedError()

    def sample(self, total_latent, device="cpu"):
        # create galaxy as mixture of Exponential + DeVacauleurs
        if self.flux_sample == "uniform":
            total_flux = _uniform(self.min_flux, self.max_flux, n_samples=total_latent)
        elif self.flux_sample == "log_uniform":
            log_flux = _uniform(
                torch.log10(self.min_flux), torch.log10(self.max_flux), n_samples=total_latent
            )
            total_flux = 10**log_flux
        elif self.flux_sample == "pareto":
            total_flux = _draw_pareto(
                self.alpha, self.min_flux, self.max_flux, n_samples=total_latent
            )
        else:
            raise NotImplementedError()
        disk_frac = _uniform(0, 1, n_samples=total_latent)
        beta_radians = _uniform(0, 2 * np.pi, n_samples=total_latent)
        disk_q = _uniform(0, 1, n_samples=total_latent)
        bulge_q = _uniform(0, 1, n_samples=total_latent)
        if self.a_sample == "uniform":
            disk_a = _uniform(self.min_a_d, self.max_a_d, n_samples=total_latent)
            bulge_a = _uniform(self.min_a_b, self.max_a_b, n_samples=total_latent)
        elif self.a_sample == "gamma":
            disk_a = _gamma(self.a_concentration, self.a_loc, self.a_scale, n_samples=total_latent)
            bulge_a = _gamma(
                self.a_concentration,
                self.a_loc / self.a_bulge_disk_ratio,
                self.a_scale / self.a_bulge_disk_ratio,
                n_samples=total_latent,
            )
        else:
            raise NotImplementedError()
        return torch.stack(
            [total_flux, disk_frac, beta_radians, disk_q, disk_a, bulge_q, bulge_a], dim=1
        ).to(device)


class SingleGalsimGalaxyDecoder(PSFDecoder):
    def __init__(
        self,
        slen: int,
        n_bands: int,
        pixel_scale: float,
        psf_params_file: Optional[str] = None,
        psf_slen: Optional[int] = None,
        sdss_bands: Optional[Tuple[int, ...]] = None,
        psf_gauss_fwhm: Optional[float] = None,
    ) -> None:
        super().__init__(
            psf_gauss_fwhm=psf_gauss_fwhm,
            psf_params_file=psf_params_file,
            psf_slen=psf_slen,
            sdss_bands=sdss_bands,
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


class UniformGalsimPrior:
    def __init__(
        self,
        single_galaxy_prior: SingleGalsimGalaxyPrior,
        max_n_sources: int,
        mean_sources: float,
        max_shift: float,
        galaxy_prob: float,
    ):
        self.single_galaxy_prior = single_galaxy_prior
        self.max_shift = max_shift  # between 0 and 0.5, from center.
        self.max_n_sources = max_n_sources
        self.mean_sources = mean_sources
        self.dim_latents = self.single_galaxy_prior.dim_latents
        self.galaxy_prob = galaxy_prob
        assert 0 <= self.max_shift <= 0.5

    def sample(self) -> Dict[str, Tensor]:
        """Returns a single batch of source parameters."""
        n_sources = _sample_poisson_n_sources(self.mean_sources, self.max_n_sources)

        params = torch.zeros(self.max_n_sources, self.dim_latents)
        params[:n_sources, :] = self.single_galaxy_prior.sample(n_sources)

        locs = torch.zeros(self.max_n_sources, 2)
        locs[:n_sources, 0] = _uniform(-self.max_shift, self.max_shift, n_sources) + 0.5
        locs[:n_sources, 1] = _uniform(-self.max_shift, self.max_shift, n_sources) + 0.5

        galaxy_bools = torch.zeros(self.max_n_sources, 1)
        galaxy_bools[:n_sources, :] = _bernoulli(self.galaxy_prob, n_sources)[:, None]
        star_bools = torch.zeros(self.max_n_sources, 1)
        star_bools[:n_sources, :] = 1 - galaxy_bools[:n_sources, :]

        star_fluxes = torch.zeros(self.max_n_sources, 1)
        star_log_fluxes = torch.zeros(self.max_n_sources, 1)
        star_fluxes[:n_sources, 0] = params[:n_sources, 0]
        star_log_fluxes[:n_sources, 0] = torch.log(star_fluxes[:n_sources, 0])

        return {
            "n_sources": torch.tensor(n_sources),
            "locs": locs,
            "galaxy_bools": galaxy_bools,
            "star_bools": star_bools,
            "galaxy_params": params * galaxy_bools,
            "star_fluxes": star_fluxes * star_bools,
            "star_log_fluxes": star_log_fluxes * star_bools,
        }


class UniformBrightestCenterGalsimGalaxy(UniformGalsimPrior):
    def sample(self) -> Dict[str, Tensor]:
        """Returns a single batch of source parameters where brightest galaxy is centered."""
        sample = super().sample()
        n_sources = sample.pop("n_sources")
        flux = sample["galaxy_params"][:, 0].reshape(-1)
        idx_order = torch.argsort(-flux)
        reordered_sample = {k: v[idx_order] for k, v in sample.items()}
        reordered_sample["locs"][0, :] = 0.5 + _uniform(0.015, 0.03, 2)  # slight off-center.
        return {"n_sources": n_sources, **reordered_sample}


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


def _sample_poisson_n_sources(mean_sources, max_n_sources) -> int:
    n_sources = torch.distributions.Poisson(mean_sources).sample([1])
    return int(torch.clamp(n_sources, max=torch.tensor(max_n_sources)))


def _uniform(a, b, n_samples=1) -> Tensor:
    # uses pytorch to return a single float ~ U(a, b)
    return (a - b) * torch.rand(n_samples) + b


def _draw_pareto(alpha, min_x, max_x, n_samples=1) -> Tensor:
    # draw pareto conditioned on being less than f_max
    assert alpha is not None
    u_max = 1 - (min_x / max_x) ** alpha
    uniform_samples = torch.rand(n_samples) * u_max
    return min_x / (1.0 - uniform_samples) ** (1 / alpha)


def _gamma(concentration, loc, scale, n_samples=1):
    x = torch.distributions.Gamma(concentration, rate=1.0).sample((n_samples,))
    return x * scale + loc


def _bernoulli(prob, n_samples=1) -> Tensor:
    prob_list = [float(prob) for _ in range(n_samples)]
    return torch.bernoulli(torch.tensor(prob_list))
