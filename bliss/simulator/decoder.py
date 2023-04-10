from pathlib import Path
from typing import Callable, Optional, Tuple

import galsim
import numpy as np
import pytorch_lightning as pl
import torch
from astropy.io import fits
from einops import rearrange, reduce
from torch import Tensor, nn
from torch.nn import functional as F

from bliss.catalog import TileCatalog, get_is_on_from_n_sources


def get_mgrid(slen: int):
    offset = (slen - 1) / 2
    # Currently type-checking with mypy doesn't work with np.mgrid
    # See https://github.com/python/mypy/issues/11185.
    x, y = np.mgrid[-offset : (offset + 1), -offset : (offset + 1)]  # type: ignore
    mgrid = torch.tensor(np.dstack((y, x))) / offset
    # mgrid is between -1 and 1
    # then scale slightly because of the way f.grid_sample
    # parameterizes the edges: (0, 0) is center of edge pixel
    return mgrid.float() * (slen - 1) / slen


class TileRenderer(nn.Module):
    """This class creates an image tile from multiple sources."""

    def __init__(self, tile_slen: int, ptile_slen: int):
        super().__init__()
        self.tile_slen = tile_slen
        self.ptile_slen = ptile_slen

        # caching the underlying
        # coordinates on which we simulate source
        # grid: between -1 and 1,
        # then scale slightly because of the way f.grid_sample
        # parameterizes the edges: (0, 0) is center of edge pixel
        self.register_buffer("cached_grid", get_mgrid(self.ptile_slen), persistent=False)
        self.register_buffer("swap", torch.tensor([1, 0]), persistent=False)

    def forward(self, locs: Tensor, sources: Tensor):
        """Renders tile from locations and sources.

        Arguments:
            locs: is (n_ptiles x max_num_stars x 2)
            sources: is (n_ptiles x max_num_stars x n_bands x stampsize x stampsize)

        Returns:
            ptile = (n_ptiles x n_bands x slen x slen)
        """
        max_sources = locs.shape[1]
        ptile_shape = (
            sources.size(0),
            sources.size(2),
            self.ptile_slen,
            self.ptile_slen,
        )
        ptile = torch.zeros(ptile_shape, device=locs.device)

        for n in range(max_sources):
            one_star = self._render_one_source(locs[:, n, :], sources[:, n])
            ptile += one_star

        return ptile

    def _render_one_source(self, locs: Tensor, source: Tensor):
        """Renders one source at a location from shape.

        Arguments:
            locs: is n_ptiles x len((x,y))
            source: is a (n_ptiles, n_bands, slen, slen) tensor, which could either be a
                        `expanded_psf` (psf repeated multiple times) for the case of of stars.
                        Or multiple galaxies in the case of galaxies.

        Returns:
            Tensor with shape = (n_ptiles x n_bands x slen x slen)
        """
        assert isinstance(self.swap, Tensor)
        assert isinstance(self.cached_grid, Tensor)
        assert source.shape[2] == source.shape[3]
        assert locs.shape[1] == 2

        # scale so that they land in the tile within the padded tile
        padding = (self.ptile_slen - self.tile_slen) / 2
        locs = locs * (self.tile_slen / self.ptile_slen) + (padding / self.ptile_slen)
        # scale locs so they take values between -1 and 1 for grid sample
        locs = (locs - 0.5) * 2
        local_grid = rearrange(
            self.cached_grid, "s1 s2 xy -> 1 s1 s2 xy", s1=self.ptile_slen, s2=self.ptile_slen, xy=2
        )

        locs_swapped = locs.index_select(1, self.swap)
        locs_swapped = rearrange(locs_swapped, "np xy -> np 1 1 xy")

        grid_loc = local_grid - locs_swapped
        return F.grid_sample(source, grid_loc, align_corners=True)


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
    assert len(source.shape) == 3

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
            grid = get_mgrid(self.psf_slen) * (self.psf_slen - 1) / 2
            # Bryan: extra factor to be consistent with old repo, probably unimportant...
            grid *= self.psf_slen / (self.psf_slen - 1)
            self.register_buffer("cached_radii_grid", (grid**2).sum(2).sqrt())

            self.psf = self.forward_psf_from_params().detach().numpy()
            psf_image = galsim.Image(self.psf[0], scale=pixel_scale)
            self.psf_galsim = galsim.InterpolatedImage(psf_image).withFlux(1.0)

    def forward(self, x):  # type: ignore
        raise NotImplementedError("Please extend this class and implement forward()")

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


class StarPSFDecoder(PSFDecoder):
    def forward(self, fluxes: Tensor, star_bools: Tensor, ptile_slen: int):
        """Renders star tile from fluxes."""
        # fluxes: Is (n_ptiles x max_stars x n_bands)
        # star_bools: Is (n_ptiles x max_stars x 1)
        # max_sources obtained from locs, allows for more flexibility when rendering.

        n_ptiles, max_sources, n_bands = fluxes.shape

        psf = self.forward_psf_from_params()
        psf = fit_source_to_ptile(psf, ptile_slen)
        n_bands, _, _ = psf.shape
        assert fluxes.shape[0] == star_bools.shape[0] == n_ptiles
        assert fluxes.shape[1] == star_bools.shape[1] == max_sources
        assert fluxes.shape[2] == self.n_bands == n_bands
        assert star_bools.shape[2] == 1

        # all stars are just the PSF so we copy it.
        expanded_psf = psf.expand(n_ptiles, max_sources, self.n_bands, -1, -1)
        sources = expanded_psf * rearrange(fluxes, "np ms nb -> np ms nb 1 1")
        sources *= rearrange(star_bools, "np ms 1 -> np ms 1 1 1")

        return sources


class GalsimGalaxyDecoder(PSFDecoder):
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


class ImageDecoder(pl.LightningModule):
    """Decodes latent variables into reconstructed astronomical image.

    Attributes:
        n_bands: Number of bands (colors) in the image
        tile_slen: Side-length of each tile.
        ptile_slen: Padded side-length of each tile (for reconstructing image).
        border_padding: Size of border around the final image where sources will not be present.
        star_tile_decoder: Module which renders stars on individual tiles.
        galaxy_tile_decoder: Module which renders galaxies on individual tiles.
    """

    def __init__(
        self,
        n_bands: int,
        tile_slen: int,
        ptile_slen: int,
        psf_slen: int,
        sdss_bands: Tuple[int, ...],
        psf_params_file: str,
        border_padding: Optional[int] = None,
        galaxy_decoder: Optional[GalsimGalaxyDecoder] = None,
    ):
        """Initializes ImageDecoder.

        Args:
            n_bands: Number of bands (colors) in the image
            tile_slen: Side-length of each tile.
            ptile_slen: Padded side-length of each tile (for reconstructing image).
            psf_slen: Side-length of reconstruced star image from PSF.
            sdss_bands: Bands to retrieve from PSF.
            psf_params_file: Path where point-spread-function (PSF) data is located.
            border_padding: Size of border around the final image where sources will not be present.
            galaxy_decoder: Specifies how galaxy shapes are decoded from latent representation.
        """
        super().__init__()
        self.n_bands = n_bands
        assert tile_slen <= ptile_slen
        self.tile_slen = tile_slen
        self.ptile_slen = ptile_slen
        self.border_padding = self._validate_border_padding(border_padding)
        self.tiler = TileRenderer(tile_slen, ptile_slen)

        self.star_tile_decoder = StarPSFDecoder(
            n_bands=self.n_bands,
            psf_slen=psf_slen,
            sdss_bands=tuple(sdss_bands),
            psf_params_file=psf_params_file,
        )

        self.galaxy_decoder = galaxy_decoder

    def render_images(self, tile_catalog: TileCatalog) -> Tensor:
        """Renders tile catalog latent variables into a full astronomical image.

        Args:
            tile_catalog: Tile catalog of astronomical image, comprising tensors
                of size (batch_size x n_tiles_h x n_tiles_w x max_sources). The only
                exception is 'n_sources`, which is size batch_size x n_tiles_h x n_tiles_w.

        Returns:
            The **full** image in shape (batch_size x n_bands x slen x slen).
        """
        image_ptiles = self._render_ptiles(tile_catalog)
        return self._reconstruct_image_from_ptiles(image_ptiles)

    def _render_ptiles(self, tile_catalog: TileCatalog) -> Tensor:
        # n_sources: is (batch_size x n_tiles_h x n_tiles_w)
        # locs: is (batch_size x n_tiles_h x n_tiles_w x max_sources x 2)
        # galaxy_bools: Is (batch_size x n_tiles_h x n_tiles_w x max_sources)
        # galaxy_params : is (batch_size x n_tiles_h x n_tiles_w x max_sources x latent_dim)
        # fluxes: Is (batch_size x n_tiles_h x n_tiles_w x max_sources x 2)

        # returns the ptiles with shape =
        # (batch_size x n_tiles_h x n_tiles_w x n_bands x ptile_slen x ptile_slen)

        batch_size, n_tiles_h, n_tiles_w, max_sources, _ = tile_catalog.locs.shape

        # view parameters being explicit about shapes
        n_sources = rearrange(tile_catalog.n_sources, "b nth ntw -> (b nth ntw)")
        assert (n_sources <= max_sources).all()
        locs = rearrange(tile_catalog.locs, "b nth ntw s xy -> (b nth ntw) s xy", xy=2)
        galaxy_bools = rearrange(tile_catalog["galaxy_bools"], "b nth ntw s 1 -> (b nth ntw) s 1")
        star_fluxes = rearrange(
            tile_catalog["star_fluxes"], "b nth ntw s band -> (b nth ntw) s band"
        )

        # draw stars and galaxies
        is_on_array = get_is_on_from_n_sources(n_sources, max_sources)
        is_on_array = rearrange(is_on_array, "b_nth_ntw s -> b_nth_ntw s 1")
        star_bools = (1 - galaxy_bools) * is_on_array

        # final shapes of images.
        img_shape = (
            batch_size,
            n_tiles_h,
            n_tiles_w,
            self.n_bands,
            self.ptile_slen,
            self.ptile_slen,
        )

        # draw stars and galaxies
        centered_stars = self.star_tile_decoder.forward(star_fluxes, star_bools, self.ptile_slen)
        stars = self.tiler.forward(locs, centered_stars)
        galaxies = torch.zeros(img_shape, device=locs.device)

        if self.galaxy_decoder is not None:
            centered_galaxies = self.galaxy_decoder.forward(
                tile_catalog["galaxy_params"], galaxy_bools
            )
            galaxies = self.tiler.forward(locs, centered_galaxies)

        return galaxies.view(img_shape) + stars.view(img_shape)

    def _reconstruct_image_from_ptiles(self, image_ptiles: Tensor) -> Tensor:
        """Reconstruct an image from a tensor of padded tiles.

        Given a tensor of padded tiles and the size of the original tiles, this function
        combines them into a full image with overlap.
        For now, the reconstructed image is assumed to be square. However, this function
        can easily be refactored to allow for different numbers of horizontal or vertical
        tiles.

        Args:
            image_ptiles: Tensor of size
                (batch_size x n_tiles_h x n_tiles_w x n_bands x ptile_slen x ptile_slen)

        Returns:
            Reconstructed image of size (batch_size x n_bands x height x width)
        """
        _, n_tiles_h, n_tiles_w, _, ptile_slen, _ = image_ptiles.shape
        image_ptiles_prefold = rearrange(image_ptiles, "b nth ntw c h w -> b (c h w) (nth ntw)")
        kernel_size = (ptile_slen, ptile_slen)
        stride = (self.tile_slen, self.tile_slen)
        n_tiles_hw = (n_tiles_h, n_tiles_w)

        output_size_list = []
        for i in (0, 1):
            output_size_list.append(kernel_size[i] + (n_tiles_hw[i] - 1) * stride[i])
        output_size = tuple(output_size_list)

        folded_image = F.fold(image_ptiles_prefold, output_size, kernel_size, stride=stride)

        # In default settings of ImageDecoder, no borders are cropped from
        # output image. However, we may want to crop
        max_padding = (ptile_slen - self.tile_slen) / 2
        assert max_padding % 1 == 0
        max_padding = int(max_padding)
        crop_idx = max_padding - self.border_padding
        return folded_image[:, :, crop_idx : (-crop_idx or None), crop_idx : (-crop_idx or None)]

    def _validate_border_padding(self, border_padding):
        # Border Padding
        # Images are first rendered on *padded* tiles (aka ptiles).
        # The padded tile consists of the tile and neighboring tiles
        # The width of the padding is given by ptile_slen.
        # border_padding is the amount of padding we leave in the final image. Useful for
        # avoiding sources getting too close to the edges.
        if border_padding is None:
            # default value matches encoder default.
            border_padding = (self.ptile_slen - self.tile_slen) / 2

        n_tiles_of_padding = (self.ptile_slen / self.tile_slen - 1) / 2
        ptile_padding = n_tiles_of_padding * self.tile_slen
        assert border_padding % 1 == 0, "amount of border padding must be an integer"
        assert n_tiles_of_padding % 1 == 0, "n_tiles_of_padding must be an integer"
        assert border_padding <= ptile_padding, "Too much border, increase ptile_slen"
        return int(border_padding)
