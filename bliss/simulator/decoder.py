from pathlib import Path
from typing import Callable, Optional, Tuple

import galsim
import numpy as np
import torch
from astropy.io import fits
from einops import rearrange, reduce
from torch import Tensor, nn

from bliss.catalog import TileCatalog


class PSFDecoder(nn.Module):
    """Abstract decoder class to subclass whenever the decoded result will go through a PSF.

    PSF (point-spread function) use is common for decoding the final realistic astronomical
    image to account for sensor lens effects. PSF loading is suppported as a direct image (npy)
    or through attributes (npy or fits) file.

    For parameter loading, a psf_params_file should be indicated, with the corresponding
    psf_slen and sdss_bands.

    Attributes:
        psf_params_file: PSF parameters (saved either in a numpy file or fits file from SDSS)
        psf_slen: Side-length of the PSF.
        sdss_bands: Bands to retrieve from PSF.
        n_bands: Number of bands to retrieve from PSF.
    """

    forward: Callable[..., Tensor]

    def __init__(
        self,
        n_bands: int,
        pixel_scale: float,
        psf_params_file: str,
        psf_slen: int,
        sdss_bands: Tuple[int, ...],
    ):
        super().__init__()
        self.n_bands = n_bands
        self.params = None  # psf params from fits file

        assert Path(psf_params_file).suffix == ".fits"
        assert len(sdss_bands) == n_bands
        psf_params = self._get_fit_file_psf_params(psf_params_file, sdss_bands)
        self.params = nn.Parameter(psf_params.clone(), requires_grad=True)
        self.psf_slen = psf_slen
        grid = self._get_mgrid(self.psf_slen) * (self.psf_slen - 1) / 2
        # Bryan: extra factor to be consistent with old repo, probably unimportant...
        grid *= self.psf_slen / (self.psf_slen - 1)
        self.register_buffer("cached_radii_grid", (grid**2).sum(2).sqrt())

        self.psf = self.forward_psf_from_params().detach().numpy()
        psf_image = galsim.Image(self.psf[0], scale=pixel_scale)
        self.psf_galsim = galsim.InterpolatedImage(psf_image).withFlux(1.0)

    def forward(self, x):  # type: ignore
        raise NotImplementedError("Please extend this class and implement forward()")

    @staticmethod
    def _get_mgrid(slen: int):
        offset = (slen - 1) / 2
        # Currently type-checking with mypy doesn't work with np.mgrid
        # See https://github.com/python/mypy/issues/11185.
        x, y = np.mgrid[-offset : (offset + 1), -offset : (offset + 1)]  # type: ignore
        mgrid = torch.tensor(np.dstack((y, x))) / offset
        # mgrid is between -1 and 1
        # then scale slightly because of the way f.grid_sample
        # parameterizes the edges: (0, 0) is center of edge pixel
        return mgrid.float() * (slen - 1) / slen

    def forward_psf_from_params(self):
        assert self.params is not None, "Can only be used when `psf_params_file` is given."
        # get psf in each band
        psf_list = []
        for i in range(self.n_bands):
            band_psf = self._psf_fun(self.cached_radii_grid, *self.params[i])
            psf_list.append(band_psf.unsqueeze(0))
        psf = torch.cat(psf_list)
        assert (psf > 0).all()

        # ensure it's normalized
        norm = reduce(psf, "b m k -> b", "sum")
        psf *= rearrange(1 / norm, "b -> b 1 1")

        # check format
        n_bands, psf_slen, _ = psf.shape
        assert n_bands == self.n_bands and (psf_slen % 2) == 1 and psf_slen == psf.shape[2]
        return psf

    @staticmethod
    def _get_fit_file_psf_params(psf_fit_file: str, bands: Tuple[int, ...]):
        data = fits.open(psf_fit_file, ignore_missing_end=True).pop(6).data
        psf_params = torch.zeros(len(bands), 6)
        for i, band in enumerate(bands):
            sigma1 = data["psf_sigma1"][0][band] ** 2
            sigma2 = data["psf_sigma2"][0][band] ** 2
            sigmap = data["psf_sigmap"][0][band] ** 2
            beta = data["psf_beta"][0][band]
            b = data["psf_b"][0][band]
            p0 = data["psf_p0"][0][band]

            psf_params[i] = torch.tensor([sigma1, sigma2, sigmap, beta, b, p0])

        return psf_params

    @staticmethod
    def _psf_fun(r, sigma1, sigma2, sigmap, beta, b, p0):
        term1 = torch.exp(-(r**2) / (2 * sigma1))
        term2 = b * torch.exp(-(r**2) / (2 * sigma2))
        term3 = p0 * (1 + r**2 / (beta * sigmap)) ** (-beta / 2)
        return (term1 + term2 + term3) / (1 + b + p0)


class GalaxyDecoder(PSFDecoder):
    def __init__(
        self,
        slen: int,
        ptile_slen: int,
        n_bands: int,
        pixel_scale: float,
        psf_params_file: str,
        psf_slen: int,
        sdss_bands: Tuple[int, ...],
    ) -> None:
        super().__init__(
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
        self.ptile_slen = ptile_slen

    def _render_galaxy_np(
        self,
        galaxy_params: Tensor,
        psf: galsim.GSObject,
        slen: int,
        offset: Optional[Tensor] = None,
    ) -> Tensor:
        assert offset is None or offset.shape == (2,)
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


class ImageDecoder:
    def __init__(self, galaxy_decoder: GalaxyDecoder) -> None:
        self.galaxy_decoder = galaxy_decoder
        self.pixel_scale = self.galaxy_decoder.pixel_scale

    def _render_star(self, flux: float, slen: int, offset: Optional[Tensor] = None) -> Tensor:
        assert offset is None or offset.shape == (2,)
        star = self.galaxy_decoder.psf_galsim.withFlux(flux)  # creates a copy
        offset = offset if offset is None else offset.numpy()
        image = star.drawImage(nx=slen, ny=slen, scale=self.pixel_scale, offset=offset)
        return torch.from_numpy(image.array).reshape(1, slen, slen)

    def render_images(self, tile_cat: TileCatalog):
        batch_size, n_tiles_h, n_tiles_w = tile_cat.n_sources.shape
        assert n_tiles_h == n_tiles_w
        slen_h = tile_cat.tile_slen * n_tiles_h

        full_cat = tile_cat.to_full_params()
        assert self.galaxy_decoder.n_bands == 1, "only 1 band supported for now"

        images = torch.zeros(batch_size, self.galaxy_decoder.n_bands, slen_h, slen_h)

        for b in range(batch_size):
            n_sources = int(full_cat.n_sources[b].item())
            for s in range(n_sources):
                offset_yx = full_cat.plocs[b][s] - (slen_h / 2)
                # I don't think we should have to do this swap, though it does give us consistency
                # btw tile_cat and images...is plocs backwards?
                offset_xy = torch.tensor([offset_yx[1], offset_yx[0]])

                if full_cat["galaxy_bools"][b][s] == 1:
                    gp = full_cat["galaxy_params"][b][s]
                    images[b] += self.galaxy_decoder.render_galaxy(gp, slen_h, offset_xy)
                elif full_cat["star_bools"][b][s] == 1:
                    sp = full_cat["star_fluxes"][b][s].item()
                    images[b] += self._render_star(sp, slen_h, offset_xy)

        return images
