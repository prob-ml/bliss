from typing import Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange, reduce
from torch import Tensor, nn
from torch.nn import functional as F

from bliss.catalog import TileCatalog, get_is_on_from_n_sources
from bliss.models.galaxy_net import OneCenteredGalaxyAE
from bliss.models.galsim_decoder import SingleGalsimGalaxyDecoder, SingleLensedGalsimGalaxyDecoder
from bliss.models.psf_decoder import PSFDecoder, get_mgrid
from bliss.reporting import get_single_galaxy_ellipticities

GalaxyModel = Union[OneCenteredGalaxyAE, SingleGalsimGalaxyDecoder]


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
        galaxy_model: Optional[GalaxyModel] = None,
        lens_model: Optional[SingleLensedGalsimGalaxyDecoder] = None,
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
            galaxy_model: Specifies how galaxy shapes are decoded from latent representation.
            lens_model: Specifies how lensed galaxy shapes are decoded from latent representation.
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

        if galaxy_model is None:
            self.galaxy_tile_decoder: Optional[GalaxyDecoder] = None
        else:
            self.galaxy_tile_decoder = GalaxyDecoder(
                self.tile_slen,
                self.ptile_slen,
                self.n_bands,
                galaxy_model,
            )

        if lens_model is None:
            self.lensed_galaxy_tile_decoder: Optional[LensedGalaxyTileDecoder] = None
        else:
            self.lensed_galaxy_tile_decoder = LensedGalaxyTileDecoder(
                self.tile_slen,
                self.ptile_slen,
                self.n_bands,
                lens_model,
            )

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

    def render_large_scene(
        self, tile_catalog: TileCatalog, batch_size: Optional[int] = None
    ) -> Tensor:
        if batch_size is None:
            batch_size = 75**2 + 500 * 2

        _, n_tiles_h, n_tiles_w, _, _ = tile_catalog.locs.shape
        n_rows_per_batch = batch_size // n_tiles_w
        h = tile_catalog.locs.shape[1] * tile_catalog.tile_slen + 2 * self.border_padding
        w = tile_catalog.locs.shape[2] * tile_catalog.tile_slen + 2 * self.border_padding
        scene = torch.zeros(1, 1, h, w)
        for row in range(0, n_tiles_h, n_rows_per_batch):
            end_row = row + n_rows_per_batch
            start_h = row * tile_catalog.tile_slen
            end_h = end_row * tile_catalog.tile_slen + 2 * self.border_padding
            tile_cat_row = tile_catalog.crop((row, end_row), (0, None))
            img_row = self.render_images(tile_cat_row)
            scene[:, :, start_h:end_h] += img_row.cpu()
        return scene

    def get_galaxy_fluxes(self, galaxy_bools: Tensor, galaxy_params_in: Tensor):
        assert self.galaxy_tile_decoder is not None
        galaxy_bools_flat = rearrange(galaxy_bools, "b nth ntw s d -> (b nth ntw s) d")
        galaxy_params = rearrange(galaxy_params_in, "b nth ntw s d -> (b nth ntw s) d")
        galaxy_shapes = self.galaxy_tile_decoder.galaxy_decoder(
            galaxy_params[galaxy_bools_flat.squeeze(-1) > 0.5]
        )
        galaxy_fluxes = reduce(galaxy_shapes, "n 1 h w -> n", "sum")
        assert torch.all(galaxy_fluxes >= 0.0)
        galaxy_fluxes_all = torch.zeros_like(
            galaxy_bools_flat.reshape(-1), dtype=galaxy_fluxes.dtype
        )
        galaxy_fluxes_all[galaxy_bools_flat.squeeze(-1) > 0.5] = galaxy_fluxes
        galaxy_fluxes = rearrange(
            galaxy_fluxes_all,
            "(b nth ntw s) -> b nth ntw s 1",
            b=galaxy_params_in.shape[0],
            nth=galaxy_params_in.shape[1],
            ntw=galaxy_params_in.shape[2],
            s=galaxy_params_in.shape[3],
        )
        galaxy_fluxes *= galaxy_bools
        return galaxy_fluxes

    def get_galaxy_ellips(
        self, galaxy_bools: Tensor, galaxy_params_in: Tensor, scale: float = 0.393
    ) -> Tensor:
        assert self.galaxy_tile_decoder is not None
        b, nth, ntw, s, _ = galaxy_bools.shape
        b_flat = b * nth * ntw * s
        slen = self.ptile_slen + ((self.ptile_slen % 2) == 0) * 1

        unfit_psf = self.star_tile_decoder.forward_psf_from_params()
        psf = fit_source_to_ptile(unfit_psf, self.ptile_slen)
        psf_image = rearrange(psf, "1 h w -> h w", h=slen, w=slen)

        galaxy_bools_flat = rearrange(galaxy_bools, "b nth ntw s d -> (b nth ntw s) d")
        galaxy_params = rearrange(galaxy_params_in, "b nth ntw s d -> (b nth ntw s) d")
        galaxy_shapes = self.galaxy_tile_decoder.galaxy_decoder(
            galaxy_params[galaxy_bools_flat.squeeze(-1) > 0.5]
        )
        galaxy_shapes = self.galaxy_tile_decoder.size_galaxy(galaxy_shapes)
        single_galaxies = rearrange(galaxy_shapes, "n 1 h w -> n h w", h=slen, w=slen)
        ellips = get_single_galaxy_ellipticities(single_galaxies, psf_image, scale)

        ellips_all = torch.zeros(b_flat, 2, dtype=ellips.dtype, device=ellips.device)
        ellips_all[galaxy_bools_flat.squeeze(-1) > 0.5] = ellips
        ellips = rearrange(
            ellips_all, "(b nth ntw s) g -> b nth ntw s g", b=b, nth=nth, ntw=ntw, s=s, g=2
        )
        ellips *= galaxy_bools
        return ellips

    def forward(self):
        """Decodes latent representation into an image."""
        raise NotImplementedError("Please use `render_images` instead.")

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
        lensed_galaxies = torch.zeros(img_shape, device=locs.device)

        if self.galaxy_tile_decoder is not None:
            centered_galaxies = self.galaxy_tile_decoder.forward(
                tile_catalog["galaxy_params"], galaxy_bools
            )
            galaxies = self.tiler.forward(locs, centered_galaxies)

        if self.lensed_galaxy_tile_decoder is not None:
            lensed_galaxy_bools = rearrange(
                tile_catalog["lensed_galaxy_bools"], "b nth ntw s 1 -> (b nth ntw) s 1"
            )
            lensed_galaxy_bools *= galaxy_bools * is_on_array
            centered_lensed_galaxies = self.lensed_galaxy_tile_decoder.forward(
                tile_catalog["lens_params"], lensed_galaxy_bools
            )
            lensed_galaxies = self.tiler.forward(locs, centered_lensed_galaxies)

        return lensed_galaxies.view(img_shape) + galaxies.view(img_shape) + stars.view(img_shape)

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


class StarPSFDecoder(PSFDecoder):
    def forward(self, fluxes: Tensor, star_bools: Tensor, ptile_slen: int):
        """Renders star tile from locations and fluxes."""
        # locs: is (n_ptiles x max_num_stars x 2)
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


class GalaxyDecoder(nn.Module):
    def __init__(self, tile_slen, ptile_slen, n_bands, galaxy_model: GalaxyModel):
        super().__init__()
        self.n_bands = n_bands
        self.tiler = TileRenderer(tile_slen, ptile_slen)
        self.ptile_slen = ptile_slen

        if isinstance(galaxy_model, OneCenteredGalaxyAE):
            galaxy_decoder = galaxy_model.get_decoder()
        elif isinstance(galaxy_model, SingleGalsimGalaxyDecoder):
            galaxy_decoder = galaxy_model
        else:
            raise TypeError("galaxy_model is not a valid type.")
        self.galaxy_decoder = galaxy_decoder

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
        gal_on = self.galaxy_decoder(z[b == 1])

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
