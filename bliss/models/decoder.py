from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from astropy.io import fits
from einops import rearrange, reduce
from torch import Tensor, nn
from torch.nn import functional as F

from bliss.models import galaxy_net
from bliss.models.location_encoder import get_is_on_from_n_sources


class ImageDecoder(pl.LightningModule):
    """Decodes latent variances into reconstructed astronomical image.

    Attributes:
        n_bands: Number of bands (colors) in the image
        slen: Side-length of astronomical image (image is assumed to be square).
        tile_slen: Side-length of each tile.
        ptile_slen: Padded side-length of each tile (for reconstructing image).
        border_padding: Size of border around the final image where sources will not be present.
        star_tile_decoder: Module which renders stars on individual tiles.
        galaxy_tile_decoder: Module which renders galaxies on individual tiles.
    """

    def __init__(
        self,
        n_bands: int = 1,
        slen: int = 50,
        tile_slen: int = 2,
        ptile_slen: int = 10,
        border_padding: int = None,
        galaxy_ae: Optional[galaxy_net.OneCenteredGalaxyAE] = None,
        galaxy_ae_ckpt: str = None,
        psf_params_file: str = None,
        psf_slen: int = 25,
        sdss_bands: Tuple[int, ...] = (2,),
    ):
        """Initializes ImageDecoder.

        Args:
            n_bands: Number of bands (colors) in the image
            slen: Side-length of astronomical image (image is assumed to be square).
            tile_slen: Side-length of each tile.
            ptile_slen: Padded side-length of each tile (for reconstructing image).
            border_padding: Size of border around the final image where sources will not be present.
            galaxy_ae: An autoencoder object for images of single galaxies.
            galaxy_ae_ckpt: Path where state_dict of trained galaxy autoencoder is located.
            psf_params_file: Path where point-spread-function (PSF) data is located.
            psf_slen: Side-length of reconstruced star image from PSF.
            sdss_bands: Bands to retrieve from PSF.
        """
        super().__init__()
        self.n_bands = n_bands
        assert slen % 1 == 0, "slen must be an integer."
        assert slen % tile_slen == 0, "slen must be divisible by tile_slen"
        assert tile_slen <= ptile_slen
        self.slen = int(slen)
        self.tile_slen = tile_slen
        self.ptile_slen = ptile_slen
        self.border_padding = self._validate_border_padding(border_padding)

        self.star_tile_decoder = StarTileDecoder(
            tile_slen,
            ptile_slen,
            self.n_bands,
            psf_slen,
            tuple(sdss_bands),
            psf_params_file=psf_params_file,
        )

        if galaxy_ae is not None:
            assert galaxy_ae_ckpt is not None
            galaxy_ae.load_state_dict(torch.load(galaxy_ae_ckpt, map_location=torch.device("cpu")))
            galaxy_ae.eval().requires_grad_(False)
            galaxy_decoder = galaxy_ae.get_decoder()
            self.galaxy_tile_decoder = GalaxyTileDecoder(
                tile_slen,
                ptile_slen,
                self.n_bands,
                galaxy_decoder,
            )
        else:
            self.galaxy_tile_decoder = None

    @property
    def galaxy_decoder(self):
        if self.galaxy_tile_decoder is None:
            return None
        return self.galaxy_tile_decoder.galaxy_decoder

    @property
    def n_tiles_per_image(self):
        n_tiles_per_image = (self.slen / self.tile_slen) ** 2
        return int(n_tiles_per_image)

    def render_images(
        self,
        n_sources: Tensor,
        locs: Tensor,
        galaxy_bools: Tensor,
        galaxy_params: Tensor,
        fluxes: Tensor,
    ) -> Tensor:
        """Renders catalog latent variables into a full astronomical image.

        Args:
            n_sources: Number of sources in each tile (batch_size x n_tiles_per_image)
            locs: Locations of sources in each tile
                (batch_size x n_tiles_h x n_tiles_w x max_sources x 2)
            galaxy_bools: Whether each source is a galaxy in each tile
                (batch_size x n_tiles_h x n_tiles_w x max_sources x 1)
            galaxy_params : Parameters of each galaxy in each tile
                (batch_size x n_tiles_h x n_tiles_w x max_sources x latent_dim)
            fluxes: Flux of each source in each time
                (batch_size x n_tiles_h x n_tiles_w x max_sources x n_bands)

        Returns:
            A tuple of the **full** image in shape (batch_size x n_bands x slen x slen) and
            its variance (same size).
        """
        assert n_sources.shape[0] == locs.shape[0]
        assert n_sources.shape[1] == locs.shape[1]
        assert galaxy_bools.shape[-1] == 1

        image_ptiles = self._render_ptiles(n_sources, locs, galaxy_bools, galaxy_params, fluxes)
        return reconstruct_image_from_ptiles(image_ptiles, self.tile_slen, self.border_padding)

    def forward(self):
        """Decodes latent representation into an image."""
        return self.star_tile_decoder.psf_forward()

    def _render_ptiles(self, n_sources, locs, galaxy_bools, galaxy_params, fluxes):
        # n_sources: is (batch_size x n_tiles_h x n_tiles_w)
        # locs: is (batch_size x n_tiles_h x n_tiles_w x max_sources x 2)
        # galaxy_bools: Is (batch_size x n_tiles_h x n_tiles_w x max_sources)
        # galaxy_params : is (batch_size x n_tiles_h x n_tiles_w x max_sources x latent_dim)
        # fluxes: Is (batch_size x n_tiles_h x n_tiles_w x max_sources x 2)

        # returns the ptiles with shape =
        # (batch_size x n_tiles_h x n_tiles_w x n_bands x ptile_slen x ptile_slen)

        batch_size, n_tiles_h, n_tiles_w, max_sources, _ = locs.shape
        assert (n_sources <= max_sources).all()
        batch_size = n_sources.shape[0]

        # view parameters being explicit about shapes
        n_sources = rearrange(n_sources, "b nth ntw -> (b nth ntw)")
        locs = rearrange(locs, "b nth ntw s xy -> (b nth ntw) s xy", xy=2)
        galaxy_bools = rearrange(galaxy_bools, "b nth ntw s 1 -> (b nth ntw) s 1")
        fluxes = rearrange(fluxes, "b nth ntw s band -> (b nth ntw) s band")

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
        stars = self.star_tile_decoder(locs, fluxes, star_bools)
        galaxies = torch.zeros(img_shape, device=locs.device)
        if self.galaxy_tile_decoder is not None:
            galaxies = self.galaxy_tile_decoder(locs, galaxy_params, galaxy_bools)

        return galaxies.view(img_shape) + stars.view(img_shape)

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


def reconstruct_image_from_ptiles(
    image_ptiles: Tensor, tile_slen: int, border_padding: int
) -> Tensor:
    """Reconstruct an image from a tensor of padded tiles.

    Given a tensor of padded tiles and the size of the original tiles, this function
    combines them into a full image with overlap.

    For now, the reconstructed image is assumed to be square. However, this function
    can easily be refactored to allow for different numbers of horizontal or vertical
    tiles.

    Args:
        image_ptiles: Tensor of size
            (batch_size x n_tiles_h x n_tiles_w x n_bands x ptile_slen x ptile_slen)
        tile_slen:
            Size of the original (non-overlapping) tiles.
        border_padding:
            Amount of border padding to keep beyond the original tiles.

    Returns:
        Reconstructed image of size (batch_size x n_bands x height x width)
    """
    _, n_tiles_h, n_tiles_w, _, ptile_slen, _ = image_ptiles.shape
    image_ptiles_prefold = rearrange(image_ptiles, "b nth ntw c h w -> b (c h w) (nth ntw)")
    kernel_size = (ptile_slen, ptile_slen)
    stride = (tile_slen, tile_slen)
    n_tiles_hw = (n_tiles_h, n_tiles_w)

    output_size = []
    for i in (0, 1):
        output_size.append(kernel_size[i] + (n_tiles_hw[i] - 1) * stride[i])
    output_size = tuple(output_size)

    folded_image = F.fold(image_ptiles_prefold, output_size, kernel_size, stride=stride)

    # In default settings of ImageDecoder, no borders are cropped from
    # output image. However, we may want to crop
    max_padding = (ptile_slen - tile_slen) / 2
    assert max_padding % 1 == 0
    max_padding = int(max_padding)
    crop_idx = max_padding - border_padding
    return folded_image[:, :, crop_idx : (-crop_idx or None), crop_idx : (-crop_idx or None)]


class Tiler(nn.Module):
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

    def fit_source_to_ptile(self, source: Tensor):
        if self.ptile_slen >= source.shape[-1]:
            fitted_source = self._expand_source(source)
        else:
            fitted_source = self._trim_source(source)
        return fitted_source

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

    def _expand_source(self, source: Tensor):
        """Pad the source with zeros so that it is size ptile_slen."""
        assert len(source.shape) == 3

        slen = self.ptile_slen + ((self.ptile_slen % 2) == 0) * 1
        assert len(source.shape) == 3

        source_slen = source.shape[2]

        assert source_slen <= slen, "Should be using trim source."

        source_expanded = torch.zeros(source.shape[0], slen, slen, device=source.device)
        offset = int((slen - source_slen) / 2)

        source_expanded[
            :, offset : (offset + source_slen), offset : (offset + source_slen)
        ] = source

        return source_expanded

    def _trim_source(self, source: Tensor):
        """Crop the source to length ptile_slen x ptile_slen, centered at the middle."""
        assert len(source.shape) == 3

        # if self.ptile_slen is even, we still make source dimension odd.
        # otherwise, the source won't have a peak in the center pixel.
        local_slen = self.ptile_slen + ((self.ptile_slen % 2) == 0) * 1

        source_slen = source.shape[2]
        source_center = (source_slen - 1) / 2

        assert source_slen >= local_slen

        r = np.floor(local_slen / 2)
        l_indx = int(source_center - r)
        u_indx = int(source_center + r + 1)

        return source[:, l_indx:u_indx, l_indx:u_indx]


def get_mgrid(slen: int):
    offset = (slen - 1) / 2
    x, y = np.mgrid[-offset : (offset + 1), -offset : (offset + 1)]
    mgrid = torch.tensor(np.dstack((y, x))) / offset
    # mgrid is between -1 and 1
    # then scale slightly because of the way f.grid_sample
    # parameterizes the edges: (0, 0) is center of edge pixel
    return mgrid.float() * (slen - 1) / slen


class StarTileDecoder(nn.Module):
    def __init__(
        self, tile_slen, ptile_slen, n_bands, psf_slen, sdss_bands=(2,), psf_params_file=None
    ):
        super().__init__()
        self.tiler = Tiler(tile_slen, ptile_slen)
        self.n_bands = n_bands
        self.psf_slen = psf_slen

        ext = Path(psf_params_file).suffix
        if ext == ".npy":
            psf_params = torch.from_numpy(np.load(psf_params_file))
            psf_params = psf_params[list(range(n_bands))]
        elif ext == ".fits":
            assert len(sdss_bands) == n_bands
            psf_params = self.get_fit_file_psf_params(psf_params_file, sdss_bands)
        else:
            raise NotImplementedError(
                "Only .npy and .fits extensions are supported for PSF params files."
            )
        self.params = nn.Parameter(psf_params.clone(), requires_grad=True)
        self.psf_image = None
        grid = get_mgrid(self.psf_slen) * (self.psf_slen - 1) / 2
        # extra factor to be consistent with old repo
        # but probably doesn't matter ...
        grid *= self.psf_slen / (self.psf_slen - 1)
        self.register_buffer("cached_radii_grid", (grid ** 2).sum(2).sqrt())

        # get psf normalization_constant
        self.normalization_constant = torch.zeros(self.n_bands)
        for i in range(self.n_bands):
            psf_i = self._get_psf_single_band(i)
            self.normalization_constant[i] = 1 / psf_i.sum()
        self.normalization_constant = self.normalization_constant.detach()

    def forward(self, locs, fluxes, star_bools):
        """Renders star tile from locations and fluxes."""
        # locs: is (n_ptiles x max_num_stars x 2)
        # fluxes: Is (n_ptiles x max_stars x n_bands)
        # star_bools: Is (n_ptiles x max_stars x 1)
        # max_sources obtained from locs, allows for more flexibility when rendering.

        psf = self._adjust_psf()
        n_ptiles = locs.shape[0]
        max_sources = locs.shape[1]

        assert len(psf.shape) == 3  # the shape is (n_bands, ptile_slen, ptile_slen)
        assert psf.shape[0] == self.n_bands
        assert fluxes.shape[0] == star_bools.shape[0] == n_ptiles
        assert fluxes.shape[1] == star_bools.shape[1] == max_sources
        assert fluxes.shape[2] == psf.shape[0] == self.n_bands
        assert star_bools.shape[2] == 1

        # all stars are just the PSF so we copy it.
        expanded_psf = psf.expand(n_ptiles, max_sources, self.n_bands, -1, -1)
        sources = expanded_psf * rearrange(fluxes, "np ms nb -> np ms nb 1 1")
        sources *= rearrange(star_bools, "np ms 1 -> np ms 1 1 1")

        return self.tiler(locs, sources)

    def psf_forward(self):
        psf = self._get_psf()
        init_psf_sum = reduce(psf, "n m k -> n", "sum").detach()
        norm = reduce(psf, "n m k -> n", "sum")
        psf *= rearrange(init_psf_sum / norm, "n -> n 1 1")
        return psf

    @staticmethod
    def get_fit_file_psf_params(psf_fit_file, bands=(2, 3)):
        data = fits.open(psf_fit_file, ignore_missing_end=True).pop(6).data
        psf_params = torch.zeros(len(bands), 6)
        for i, band in enumerate(bands):
            sigma1 = data["psf_sigma1"][0][band] ** 2
            sigma2 = data["psf_sigma2"][0][band] ** 2
            sigmap = data["psf_sigmap"][0][band] ** 2
            beta = data["psf_beta"][0][band]
            b = data["psf_b"][0][band]
            p0 = data["psf_p0"][0][band]

            psf_params[i] = torch.log(torch.tensor([sigma1, sigma2, sigmap, beta, b, p0]))

        return psf_params

    def _get_psf(self):
        psf_list = []
        for i in range(self.n_bands):
            band_psf = self._get_psf_single_band(i)
            band_psf *= self.normalization_constant[i]
            psf_list.append(band_psf.unsqueeze(0))
        psf = torch.cat(psf_list)

        assert (psf > 0).all()
        return psf

    @staticmethod
    def _psf_fun(r, sigma1, sigma2, sigmap, beta, b, p0):
        term1 = torch.exp(-(r ** 2) / (2 * sigma1))
        term2 = b * torch.exp(-(r ** 2) / (2 * sigma2))
        term3 = p0 * (1 + r ** 2 / (beta * sigmap)) ** (-beta / 2)
        return (term1 + term2 + term3) / (1 + b + p0)

    def _get_psf_single_band(self, band_idx):
        psf_params = torch.exp(self.params[band_idx])
        return self._psf_fun(
            self.cached_radii_grid,
            psf_params[0],
            psf_params[1],
            psf_params[2],
            psf_params[3],
            psf_params[4],
            psf_params[5],
        )

    def _adjust_psf(self):
        # use power_law_psf and current psf parameters to forward and obtain fresh psf model.
        # first dimension of psf is number of bands
        # dimension of the psf/slen should be odd
        psf = self.psf_forward()
        psf_slen = psf.shape[2]
        assert len(psf.shape) == 3
        assert psf.shape[0] == self.n_bands
        assert psf.shape[1] == psf_slen
        assert (psf_slen % 2) == 1

        return self.tiler.fit_source_to_ptile(psf)


class GalaxyTileDecoder(nn.Module):
    def __init__(
        self, tile_slen, ptile_slen, n_bands, galaxy_decoder: galaxy_net.CenteredGalaxyDecoder
    ):
        super().__init__()
        self.n_bands = n_bands
        self.tiler = Tiler(tile_slen, ptile_slen)
        self.ptile_slen = ptile_slen
        self.galaxy_decoder = galaxy_decoder

    def forward(self, locs, galaxy_params, galaxy_bools):
        """Renders galaxy tile from locations and galaxy parameters."""
        # max_sources obtained from locs, allows for more flexibility when rendering.
        n_ptiles = locs.shape[0]
        max_sources = locs.shape[1]
        n_galaxy_params = galaxy_params.shape[-1]

        galaxy_params = galaxy_params.view(n_ptiles, max_sources, n_galaxy_params)
        assert galaxy_params.shape[0] == galaxy_bools.shape[0] == n_ptiles
        assert galaxy_params.shape[1] == galaxy_bools.shape[1] == max_sources
        assert galaxy_bools.shape[2] == 1

        single_galaxies = self._render_single_galaxies(galaxy_params, galaxy_bools)

        return self.tiler(
            locs,
            single_galaxies * galaxy_bools.unsqueeze(-1).unsqueeze(-1),
        )

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
        gal_on = self.galaxy_decoder(z[b == 1])

        # size the galaxy (either trims or crops to the size of ptile)
        gal_on = self._size_galaxy(gal_on)

        # set galaxies
        gal[b == 1] = gal_on

        batchsize = galaxy_params.shape[0]
        gal_shape = (batchsize, -1, self.n_bands, gal.shape[-1], gal.shape[-1])
        return gal.view(gal_shape)

    def _size_galaxy(self, galaxy):
        # galaxy should be shape n_galaxies x n_bands x galaxy_slen x galaxy_slen
        assert len(galaxy.shape) == 4
        assert galaxy.shape[2] == galaxy.shape[3]
        assert (galaxy.shape[3] % 2) == 1, "dimension of galaxy image should be odd"
        assert galaxy.shape[1] == self.n_bands

        n_galaxies = galaxy.shape[0]
        galaxy_slen = galaxy.shape[3]
        galaxy = galaxy.view(n_galaxies * self.n_bands, galaxy_slen, galaxy_slen)

        sized_galaxy = self.tiler.fit_source_to_ptile(galaxy)

        outsize = sized_galaxy.shape[-1]
        return sized_galaxy.view(n_galaxies, self.n_bands, outsize, outsize)
