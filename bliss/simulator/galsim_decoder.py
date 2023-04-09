from typing import Dict, Optional, Tuple

import galsim
import numpy as np
import torch
from torch import Tensor

from bliss.catalog import FullCatalog, TileCatalog
from bliss.simulator.psf_decoder import PSFDecoder


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

        # TODO: SingleGalsimGalaxyPrior should take an arbitrary distribution object as an arg,
        # not the parameters of either a gamma or a uniform distribution
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


class DefaultGalsimPrior:
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
