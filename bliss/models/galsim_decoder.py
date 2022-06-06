from typing import Dict, Optional

import galsim
import numpy as np
import torch
from torch import Tensor

from bliss.catalog import FullCatalog
from bliss.datasets.galsim_galaxies import load_psf_from_file


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

    def sample(self, total_latent):
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
        )


class SingleGalsimGalaxyDecoder:
    def __init__(
        self,
        slen,
        n_bands,
        pixel_scale,
        psf_image_file: str,
    ) -> None:

        self.slen = slen
        assert n_bands == 1, "Only 1 band is supported"
        self.n_bands = 1
        self.pixel_scale = pixel_scale
        self.psf = load_psf_from_file(psf_image_file, self.pixel_scale)

    def __call__(self, z: Tensor, offset: Optional[Tensor] = None) -> Tensor:
        if z.shape[0] == 0:
            return torch.zeros(0, 1, self.slen, self.slen, device=z.device)

        if z.shape == (7,) and offset.shape == (2,):
            return self.render_galaxy(z, self.psf, offset)

        images = []
        for (latent, off) in zip(z, offset):
            image = self.render_galaxy(latent, self.psf, off)
            images.append(image)
        return torch.stack(images, dim=0).to(z.device)

    def render_galaxy(
        self, galaxy_params: Tensor, psf: galsim.GSObject, offset: Optional[Tensor] = None
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
        # convolve with PSF
        gal_conv = galsim.Convolution(galaxy, psf)
        offset = offset if offset is None else offset.numpy()
        image = gal_conv.drawImage(
            nx=self.slen, ny=self.slen, method="auto", scale=self.pixel_scale, offset=offset
        )
        return torch.from_numpy(image.array).reshape(1, self.slen, self.slen)


# TODO: Separate prior can enforce centered galaxy brightest example.
class UniformGalsimGalaxiesPrior:
    def __init__(
        self,
        single_galaxy_prior: SingleGalsimGalaxyPrior,
        max_n_sources: int,
        max_shift: float,
    ):
        self.single_galaxy_prior = single_galaxy_prior
        self.max_shift = max_shift  # between 0 and 1
        self.max_n_sources = max_n_sources
        self.dim_latents = self.single_galaxy_prior.dim_latents
        assert 0 <= self.max_shift <= 1

    # assumption of TileCatalog is that all given params are inside border.
    # TODO: could create dataset that returns params in tiles directly -> automatic batches and
    # parallelization. Still need FullCatalog (batch_size=1) to get TileCatalog
    # and then do **tile_catalog.to_dict() I think.
    def sample(self) -> Dict[str, Tensor]:
        """Returns a single batch of source parameters."""
        n_sources = _sample_n_sources(self.max_n_sources)

        params = torch.zeros(self.max_n_sources, self.dim_latents)
        params[:n_sources, :] = self.single_galaxy_prior.sample(n_sources)

        locs = torch.zeros(self.max_n_sources, 2)
        locs[:n_sources, 0] = _uniform(0, self.max_shift, n_sources)
        locs[:n_sources, 1] = _uniform(0, self.max_shift, n_sources)

        # for now, galaxies only
        galaxy_bools = torch.ones(self.max_n_sources, 1)
        star_bools = torch.zeros(self.max_n_sources, 1)

        return {
            "n_sources": n_sources,
            "galaxy_params": params,
            "locs": locs,
            "galaxy_bools": galaxy_bools,
            "star_bools": star_bools,
        }


class GalsimGalaxiesDecoder:
    def __init__(
        self, slen: int, bp: int, single_galaxy_decoder: SingleGalsimGalaxyDecoder
    ) -> None:
        self.slen = slen
        self.bp = bp
        self.pixel_scale = single_galaxy_decoder.pixel_scale
        self.decoder = single_galaxy_decoder

    def __call__(self, full_cat: FullCatalog):
        size = self.slen + 2 * self.bp
        full_plocs = full_cat.plocs
        b, max_n_sources, _ = full_plocs.shape

        individual_noiseless_centered = torch.zeros(b, max_n_sources, 1, size, size)
        individual_noiseless_uncentered = torch.zeros(b, max_n_sources, 1, size, size)
        images = torch.zeros(b, 1, size, size)

        for ii in range(b):
            n_sources = full_cat.n_sources[b].item()
            galaxy_params = full_cat["galaxy_params"][b]
            plocs = full_plocs[b]
            for ii in range(1, n_sources):
                offset_x = plocs[ii][1] + self.bp - size / 2
                offset_y = plocs[ii][0] + self.bp - size / 2
                offset = (offset_x, offset_y)
                centered_galaxy_image = self.decoder(galaxy_params[ii])
                uncentered_galaxy_image = self.decoder(galaxy_params[ii], offset)
                individual_noiseless_centered[b][ii] = centered_galaxy_image
                individual_noiseless_uncentered[b][ii] = uncentered_galaxy_image
                images[b] += uncentered_galaxy_image

        return images, individual_noiseless_centered, individual_noiseless_uncentered


def _sample_n_sources(max_n_sources) -> int:
    return int(torch.randint(1, max_n_sources + 1, (1,)).int().item())


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
