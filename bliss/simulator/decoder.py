from pathlib import Path
from typing import Callable, Optional, Tuple

import galsim
import numpy as np
import torch
from astropy.io import fits
from einops import rearrange, reduce
from torch import Tensor, nn

from bliss.catalog import FullCatalog, TileCatalog


def fit_source_to_ptile(source: Tensor, ptile_slen: int):
    if ptile_slen >= source.shape[-1]:
        fitted_source = expand_source(source, ptile_slen)
    else:
        fitted_source = trim_source(source, ptile_slen)
    return fitted_source


def expand_source(source: Tensor, ptile_slen: int):
    """Pad the source with zeros so that it is size ptile_slen."""
    assert len(source.shape) == 3
    slen = ptile_slen + ((ptile_slen % 2) == 0) * 1
    source_slen = source.shape[2]
    assert source_slen <= slen, "Should be using trim source."

    source_expanded = torch.zeros(source.shape[0], slen, slen, device=source.device)
    offset = int((slen - source_slen) / 2)
    source_expanded[:, offset : (offset + source_slen), offset : (offset + source_slen)] = source

    return source_expanded


def trim_source(source: Tensor, ptile_slen: int):
    """Crop the source to length ptile_slen x ptile_slen, centered at the middle."""
    assert len(source.shape) == 3

    # if self.ptile_slen is even, we still make source dimension odd.
    # otherwise, the source won't have a peak in the center pixel.
    local_slen = ptile_slen + ((ptile_slen % 2) == 0) * 1

    source_slen = source.shape[2]
    source_center = (source_slen - 1) / 2

    assert source_slen >= local_slen

    r = np.floor(local_slen / 2)
    l_indx = int(source_center - r)
    u_indx = int(source_center + r + 1)

    return source[:, l_indx:u_indx, l_indx:u_indx]


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
        n_bands: int = 1,
        pixel_scale: float = 0.393,
        psf_gauss_fwhm: Optional[float] = None,
        psf_params_file: Optional[str] = None,
        psf_slen: Optional[int] = None,
        sdss_bands: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.params = None  # psf params from fits file

        assert psf_params_file is not None or psf_gauss_fwhm is not None
        if psf_gauss_fwhm is not None:
            assert psf_slen is not None
            self.psf_galsim = galsim.Gaussian(fwhm=psf_gauss_fwhm, flux=1.0)
            self.psf = self.psf_galsim.drawImage(nx=psf_slen, ny=psf_slen, scale=pixel_scale).array
            self.psf = self.psf.reshape(1, psf_slen, psf_slen)
        else:
            assert psf_params_file is not None and psf_slen is not None and sdss_bands is not None
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
        self.ptile_slen = ptile_slen

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

    def forward(self, galaxy_params: Tensor, galaxy_bools: Tensor):
        """Renders galaxy tile from locations and galaxy parameters."""
        # max_sources obtained from locs, allows for more flexibility when rendering.
        n_ptiles, max_sources, _ = galaxy_bools.shape
        assert galaxy_bools.shape[2] == 1
        n_galaxy_params = galaxy_params.shape[-1]
        galaxy_params = galaxy_params.view(n_ptiles, max_sources, n_galaxy_params)
        single_galaxies = self._render_single_galaxies(galaxy_params, galaxy_bools)
        single_galaxies *= galaxy_bools.unsqueeze(-1).unsqueeze(-1)
        return single_galaxies

    def _render_single_galaxies(self, galaxy_params, galaxy_bools):
        # flatten parameters
        n_galaxy_params = galaxy_params.shape[-1]
        z = galaxy_params.view(-1, n_galaxy_params)
        b = galaxy_bools.flatten()

        # allocate memory
        slen = self.ptile_slen + ((self.ptile_slen % 2) == 0) * 1
        gal = torch.zeros(z.shape[0], self.n_bands, slen, slen, device=galaxy_params.device)

        # forward only galaxies that are on!
        # no background
        gal_on = self(z[b == 1])

        # size the galaxy (either trims or crops to the size of ptile)
        gal_on = self.size_galaxy(gal_on)

        # set galaxies
        gal[b == 1] = gal_on

        batchsize = galaxy_params.shape[0]
        gal_shape = (batchsize, -1, self.n_bands, gal.shape[-1], gal.shape[-1])
        return gal.view(gal_shape)

    def size_galaxy(self, galaxy: Tensor):
        n_galaxies, n_bands, h, w = galaxy.shape
        assert h == w
        assert (h % 2) == 1, "dimension of galaxy image should be odd"
        assert n_bands == self.n_bands
        galaxy = rearrange(galaxy, "n c h w -> (n c) h w")
        sized_galaxy = fit_source_to_ptile(galaxy, self.ptile_slen)
        outsize = sized_galaxy.shape[-1]
        return sized_galaxy.view(n_galaxies, self.n_bands, outsize, outsize)


class ImageDecoder:
    def __init__(self, galaxy_decoder: GalaxyDecoder, slen: int, bp: int, tile_slen: int) -> None:
        self.galaxy_decoder = galaxy_decoder
        self.slen = slen
        self.bp = bp
        self.tile_slen = tile_slen
        assert self.slen + 2 * self.bp >= self.galaxy_decoder.slen
        self.pixel_scale = self.galaxy_decoder.pixel_scale

    def _render_star(self, flux: float, slen: int, offset: Optional[Tensor] = None) -> Tensor:
        assert offset is None or offset.shape == (2,)
        star = self.galaxy_decoder.psf_galsim.withFlux(flux)  # creates a copy
        offset = offset if offset is None else offset.numpy()
        image = star.drawImage(nx=slen, ny=slen, scale=self.pixel_scale, offset=offset)
        return torch.from_numpy(image.array).reshape(1, slen, slen)

    def _add_source(self, b, s, full_cat, images):
        bp_slen_bp = self.slen + 2 * self.bp
        offset_yx = full_cat.plocs[b][s] + self.bp - bp_slen_bp / 2
        offset_xy = torch.tensor([offset_yx[1], offset_yx[0]])

        if full_cat["galaxy_bools"][b][s] == 1:
            gp = full_cat["galaxy_params"][b][s]
            images[b] += self.galaxy_decoder.render_galaxy(gp, bp_slen_bp, offset_xy)
        elif full_cat["star_bools"][b][s] == 1:
            sp = full_cat["star_fluxes"][b][s].item()
            images[b] += self._render_star(sp, bp_slen_bp, offset_xy)

    def render_images_fullcat(self, full_cat: FullCatalog):
        bp_slen_bp = self.slen + 2 * self.bp
        batch_size, _max_n_sources, _ = full_cat.plocs.shape
        assert self.galaxy_decoder.n_bands == 1, "Only 1 band supported for now"

        images = torch.zeros(batch_size, self.galaxy_decoder.n_bands, bp_slen_bp, bp_slen_bp)

        for b in range(batch_size):
            n_sources = int(full_cat.n_sources[b].item())
            for s in range(n_sources):
                self._add_source(b, s, full_cat, images)

        return images

    def render_images(self, tile_cat: TileCatalog):
        full_cat = tile_cat.to_full_params()
        return self.render_images_fullcat(full_cat)
