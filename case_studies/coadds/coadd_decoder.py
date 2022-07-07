# galsim_decoder changes for coadds
import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path
from typing import Dict, Optional
import galsim
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
from hydra import compose, initialize
from hydra.utils import instantiate
from einops import rearrange, reduce
from bliss.catalog import FullCatalog
from bliss.catalog import TileCatalog, get_is_on_from_n_sources
from bliss.models.galaxy_net import OneCenteredGalaxyAE
from bliss.encoder import Encoder
from bliss.datasets.galsim_galaxies import SingleGalsimGalaxies
from bliss.models.decoder import GalaxyTileDecoder
from bliss.datasets.galsim_galaxies import GalsimBlends
from bliss.models.galsim_decoder import SingleGalsimGalaxyPrior, UniformGalsimGalaxiesPrior, FullCatalogDecoder
from bliss.catalog import FullCatalog, TileCatalog
from bliss.models.decoder import get_mgrid
from bliss.models.galsim_decoder import SingleGalsimGalaxyDecoder, load_psf_from_file

class CoaddSingleGalaxyDecoder(SingleGalsimGalaxyDecoder):
    def __init__(
        self,
        slen: int,
        n_bands: int,
        pixel_scale: float,
        psf_image_file: str,
    ) -> None:
        assert n_bands == 1, "Only 1 band is supported"
        self.slen = slen
        self.n_bands = 1
        self.pixel_scale = pixel_scale
        self.psf = load_psf_from_file(psf_image_file, self.pixel_scale)
    
    def render_galaxy(
        self,
        galaxy_params: Tensor,
        psf: galsim.GSObject,
        slen: int,
        offset: Optional[Tensor] = None,
        dithers: Optional[Tensor] = None,
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
        shift = torch.add(torch.Tensor(dither), torch.Tensor(offset))
        images = []
        for i in shift:
            image = gal_conv.drawImage(
                nx=slen, ny=slen, method="auto", scale=self.pixel_scale, offset=shift[i]
            )
            images.append(image)
        return torch.from_numpy(images.array).reshape(len(dithers), 1, slen, slen)

class FullCatalogDecoder:
    def __init__(
        self, single_galaxy_decoder: SingleGalsimGalaxyDecoder, slen: int, bp: int
    ) -> None:
        self.single_decoder = single_galaxy_decoder
        self.slen = slen
        self.bp = bp
        assert self.slen + 2 * self.bp >= self.single_decoder.slen

    def __call__(self, full_cat: FullCatalog):
        return self.render_catalog(full_cat, self.single_decoder.psf)

    def render_catalog(self, full_cat: FullCatalog, psf: galsim.GSObject, dithers: Optional[Tensor]):
        size = self.slen + 2 * self.bp
        full_plocs = full_cat.plocs
        b, max_n_sources, _ = full_plocs.shape
        assert b == 1, "Only one batch supported for now."
        assert self.single_decoder.n_bands == 1, "Only 1 band supported for now"

        image = torch.zeros(1, size, size)
        noiseless_centered = torch.zeros(max_n_sources, 1, size, size)
        noiseless_uncentered = torch.zeros(max_n_sources, 1, size, size)

        n_sources = int(full_cat.n_sources[0].item())
        galaxy_params = full_cat["galaxy_params"][0]
        plocs = full_plocs[0]
        for ii in range(n_sources):
            offset_x = plocs[ii][1] + self.bp - size / 2
            offset_y = plocs[ii][0] + self.bp - size / 2
            offset = torch.tensor([offset_x, offset_y])
            centered = self.single_decoder.render_galaxy(galaxy_params[ii], psf, size)
            uncentered = self.single_decoder.render_galaxy(galaxy_params[ii], psf, size, offset, dithers)
            noiseless_centered[ii] = centered
            noiseless_uncentered[ii] = uncentered
            image += uncentered
        return image, noiseless_centered, noiseless_uncentered

class PsfSampler:
    def __init__(
        self,
        psf_rmin: float = 0.7,
        psf_rmax: float = 0.9,
    ) -> None:
        self.rmin = psf_rmin
        self.rmax = psf_rmax

    def sample(self) -> galsim.GSObject:
        # sample psf from galsim Gaussian distribution
        if self.rmin == self.rmax:
            fwhm = self.rmin
        elif self.rmin > self.rmax:
            raise ValueError("invalid argument!!!")
        else:
            fwhm = torch.distributions.uniform.Uniform(self.rmin, self.rmax).sample([1]).item()
        
        return galsim.Gaussian(fwhm=fwhm)

class CoaddGalsimBlends(GalsimBlends):
    """Dataset of coadd galsim blends."""

    def __init__(self,
        prior: UniformGalsimGalaxiesPrior,
        decoder: FullCatalogDecoder,
        background: ConstantBackground,
        tile_slen: int,
        max_sources_per_tile: int,
        num_workers: int,
        batch_size: int,
        n_batches: int,
        psf_sampler: PsfSampler,
        fix_validation_set: bool = False,
        valid_n_batches: Optional[int] = None,
    ):
        super().__init__(
            prior, 
            decoder,
            background,
            tile_slen,
            max_sources_per_tile,
            num_workers,
            batch_size,
            n_batches,
            fix_validation_set,
            valid_n_batches,
        )
        self.slen = self.decoder.slen
        self.pixel_scale = self.decoder.single_decoder.pixel_scale
        self.psf = psf_sampler
        

    def _get_images(self, full_cat, dithers):
        psf_obj = self.psf.sample()
        noiseless, noiseless_centered, noiseless_uncentered = FullCatalogDecoder.render_catalog(
            full_cat, psf_obj, dithers
        )

        # get background and noisy image.
        background = self.background.sample((1, *noiseless.shape)).squeeze(0)
        noisy_image = _add_noise_and_background(noiseless, background)

        return noisy_image, noiseless, noiseless_centered, noiseless_uncentered, background



