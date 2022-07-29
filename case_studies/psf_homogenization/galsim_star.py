from pathlib import Path
from typing import Dict, Optional

import galsim
import numpy as np
import torch
from torch import Tensor

from bliss.catalog import FullCatalog, TileCatalog
from bliss.models.galsim_decoder import SingleGalsimGalaxyDecoder, SingleGalsimGalaxyPrior


class SingleGalsimStarPrior:
    dim_latents = 1

    def __init__(
        self,
        flux_sample: str,
        min_flux: float,
        max_flux: float,
        alpha: Optional[float] = None,
    ) -> None:
        self.flux_sample = flux_sample
        self.min_flux = min_flux
        self.max_flux = max_flux
        self.alpha = alpha
        if self.flux_sample == "pareto":
            assert self.alpha is not None

    def sample(self, total_latent, device="cpu"):
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
        return torch.stack([total_flux], dim=1).to(device)


class SingleGalsimStarDecoder:
    def __init__(
        self,
        slen: int,
        n_bands: int,
        pixel_scale: float,
        psf_image_file: str,
    ) -> None:
        assert n_bands == 1
        self.slen = slen
        self.n_bands = n_bands
        self.pixel_scale = pixel_scale
        self.psf = load_psf_from_file(psf_image_file, self.pixel_scale)

    def __call__(self, z: Tensor, offset: Optional[Tensor] = None) -> Tensor:
        if z.shape[0] == 0:
            return torch.zeros(0, 1, self.slen, self.slen, device=z.device)

        if z.shape == (1,):  # equal to dim_latents???
            assert offset is None or offset.shape == (2,)
            return self.render_star(z, self.psf, self.slen, offset)

        images = []
        for ii, latent in enumerate(z):
            off = offset if not offset else offset[ii]
            assert off is None or off.shape == (2,)
            image = self.render_star(latent, self.psf, self.slen, offset)
            images.append(image)
        return torch.stack(images, dim=0).to(z.device)

    def render_star(
        self,
        star_params: Tensor,
        psf: galsim.GSObject,
        slen: int,
        offset: Optional[Tensor] = None,
    ) -> Tensor:
        assert offset is None or offset.shape == (2,)
        if isinstance(star_params, Tensor):
            star_params = star_params.cpu().detach()
        total_flux = star_params

        star_withflux = psf.withFlux(total_flux)
        offset = offset if offset is None else offset.numpy()
        image = star_withflux.drawImage(
            nx=slen, ny=slen, method="auto", scale=self.pixel_scale, offset=offset
        )
        return torch.from_numpy(image.array).reshape(1, slen, slen)


class UniformGalsimPrior:
    def __init__(
        self,
        single_galaxy_prior: SingleGalsimGalaxyPrior,
        single_star_prior: SingleGalsimStarPrior,
        max_n_sources: int,
        max_shift: float,
        galaxy_prob: float,
    ):
        self.single_galaxy_prior = single_galaxy_prior
        self.single_star_prior = single_star_prior
        self.max_shift = max_shift
        self.max_n_sources = max_n_sources
        self.galaxy_prob = galaxy_prob
        self.galaxy_dim_latents = self.single_galaxy_prior.dim_latents
        self.star_dim_latents = self.single_star_prior.dim_latents
        assert 0 <= self.max_shift <= 0.5

    def sample(self) -> Dict[str, Tensor]:
        n_sources = _sample_n_sources(self.max_n_sources)

        galaxy_params = torch.zeros(self.max_n_sources, self.galaxy_dim_latents)
        star_params = torch.zeros(self.max_n_sources, self.star_dim_latents)
        galaxy_params[:n_sources, :] = self.single_galaxy_prior.sample(n_sources)
        star_params[:n_sources, :] = self.single_star_prior.sample(n_sources)

        locs = torch.zeros(self.max_n_sources, 2)
        locs[:n_sources, 0] = _uniform(-self.max_shift, self.max_shift, n_sources) + 0.5
        locs[:n_sources, 1] = _uniform(-self.max_shift, self.max_shift, n_sources) + 0.5

        galaxy_bools = torch.zeros(self.max_n_sources, 1)
        galaxy_bools[:n_sources, :] = _bernoulli(self.galaxy_prob, n_sources)[:, None]
        star_bools = torch.zeros(self.max_n_sources, 1)
        star_bools[:n_sources, :] = 1 - galaxy_bools[:n_sources, :]

        return {
            "n_sources": torch.tensor(n_sources),
            "galaxy_params": galaxy_params,
            "star_params": star_params,
            "locs": locs,
            "galaxy_bools": galaxy_bools,
            "star_bools": star_bools,
        }


class FullCatelogDecoderSG:
    def __init__(
        self,
        single_galaxy_decoder: SingleGalsimGalaxyDecoder,
        single_star_decoder: SingleGalsimStarDecoder,
        slen: int,
        bp: int,
    ) -> None:
        self.single_galaxy_decoder = single_galaxy_decoder
        self.single_star_decoder = single_star_decoder
        self.slen = slen
        self.bp = bp
        assert self.slen + 2 * self.bp >= self.single_galaxy_decoder.slen
        assert self.slen + 2 * self.bp >= self.single_star_decoder.slen

    def __call__(self, full_cat: FullCatalog):
        return self.render_catalog(full_cat, self.single_galaxy_decoder.psf)

    def render_catalog(self, full_cat: FullCatalog, psf: galsim.GSObject):
        size = self.slen + 2 * self.bp
        full_plocs = full_cat.plocs
        b, max_n_sources, _ = full_plocs.shape
        assert b == 1
        assert self.single_galaxy_decoder.n_bands == 1
        assert self.single_star_decoder.n_bands == 1

        image = torch.zeros(1, size, size)
        noiseless_centered = torch.zeros(max_n_sources, 1, size, size)
        noiseless_uncentered = torch.zeros(max_n_sources, 1, size, size)

        n_sources = int(full_cat.n_sources[0].item())
        galaxy_params = full_cat["galaxy_params"][0]
        star_params = full_cat["star_params"][0]
        galaxy_bools = full_cat["galaxy_bools"][0]
        star_bools = full_cat["star_bools"][0]
        plocs = full_plocs[0]
        for ii in range(n_sources):
            offset_x = plocs[ii][1] + self.bp - size / 2
            offset_y = plocs[ii][0] + self.bp - size / 2
            offset = torch.tensor([offset_x, offset_y])
            if galaxy_bools[ii] == 1:
                centered = self.single_galaxy_decoder.render_galaxy(galaxy_params[ii], psf, size)
                uncentered = self.single_galaxy_decoder.render_galaxy(
                    galaxy_params[ii], psf, size, offset
                )
            elif star_bools[ii] == 1:
                centered = self.single_star_decoder.render_star(star_params[ii], psf, size)
                uncentered = self.single_star_decoder.render_star(
                    star_params[ii], psf, size, offset
                )
            else:
                continue

            noiseless_centered[ii] = centered
            noiseless_uncentered[ii] = uncentered
            image += uncentered

        return image, noiseless_centered, noiseless_uncentered

    def forward_tile(self, tile_cat: TileCatalog):
        full_cat = tile_cat.to_full_params()
        return self(full_cat)


def load_psf_from_file(psf_image_file: str, pixel_scale: float) -> galsim.GSObject:
    """Return normalized PSF galsim.GSObject from numpy psf_file."""
    assert Path(psf_image_file).suffix == ".npy"
    psf_image = np.load(psf_image_file)
    assert len(psf_image.shape) == 3 and psf_image.shape[0] == 1
    psf_image = galsim.Image(psf_image[0], scale=pixel_scale)
    return galsim.InterpolatedImage(psf_image).withFlux(1.0)


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


def _bernoulli(prob, n_samples=1) -> Tensor:
    # return Bernoulli(prob)
    return torch.bernoulli(torch.tensor([float(prob)] * n_samples))
