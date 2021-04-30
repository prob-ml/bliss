import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Poisson
import pytorch_lightning as pl

from .encoder import get_is_on_from_n_sources, get_mgrid
from . import galaxy_net


class ImageDecoder(pl.LightningModule):
    # pylint: disable=too-many-statements
    def __init__(
        self,
        n_bands=1,
        slen=50,
        tile_slen=2,
        ptile_slen=10,
        border_padding=None,
        max_sources=2,
        mean_sources=0.4,
        min_sources=0,
        f_min=1e4,
        f_max=1e6,
        alpha=0.5,
        prob_galaxy=0.0,
        n_galaxy_params=8,
        gal_slen=53,
        autoencoder_ckpt=None,
        latents_file=None,
        psf_slen=25,
        psf_params_file="psf_params.npy",
        background_values=(686.0, 1123.0),
        loc_min=0.0,
        loc_max=1.0,
    ):
        super().__init__()
        ## Set class attributes
        self.n_bands = n_bands
        # side-length in pixels of an image (image is assumed to be square)
        assert slen % 1 == 0, "slen must be an integer."
        self.slen = int(slen)
        assert self.slen % tile_slen == 0, "slen must be divisible by tile_slen"
        # side-length of an image tile.
        # latent variables (locations, fluxes, etc) are drawn per-tile
        assert tile_slen <= ptile_slen
        self.tile_slen = tile_slen
        self.ptile_slen = ptile_slen
        # per-tile prior parameters on number (and type) of sources
        assert max_sources > 0, "No sources will be drawn."
        self.max_sources = max_sources
        self.mean_sources = mean_sources
        self.min_sources = min_sources
        # per-tile constraints on the location of sources
        self.loc_min = loc_min
        self.loc_max = loc_max
        # prior parameters on fluxes
        self.f_min = f_min
        self.f_max = f_max
        self.alpha = alpha  # pareto parameter.
        # Galaxy decoder
        self.prob_galaxy = float(prob_galaxy)
        self.n_galaxy_params = n_galaxy_params
        self.gal_slen = gal_slen
        self.autoencoder_ckpt = autoencoder_ckpt
        self.latents_file = latents_file
        # Star Decoder
        self.psf_slen = psf_slen
        self.psf_params_file = psf_params_file
        # number of tiles per image
        n_tiles_per_image = (self.slen / self.tile_slen) ** 2
        self.n_tiles_per_image = int(n_tiles_per_image)

        ## Border Padding
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
        self.border_padding = int(border_padding)

        ## Background
        assert len(background_values) == n_bands
        self.background_values = background_values

        ## Submodule for managing tiles (no learned parameters)
        self.tiler = Tiler(tile_slen, ptile_slen)

        ## Submodule for rendering stars on a tile
        self.star_tile_decoder = StarTileDecoder(
            self.tiler,
            self.n_bands,
            self.psf_params_file,
            self.psf_slen,
        )

        ## Submodule for rendering galaxies on a tile
        if prob_galaxy > 0.0:
            assert self.autoencoder_ckpt is not None and self.latents_file is not None
            self.galaxy_tile_decoder = GalaxyTileDecoder(
                self.n_bands,
                self.tile_slen,
                self.ptile_slen,
                self.gal_slen,
                self.n_galaxy_params,
                self.autoencoder_ckpt,
            )
            # load dataset of encoded simulated galaxies.
            self.register_buffer("latents", torch.load(latents_file))
        else:
            self.galaxy_tile_decoder = None
            self.register_buffer("latents", torch.zeros(1, 8))

        # background
        assert len(background_values) == n_bands
        self.background_values = background_values

    def forward(self):
        return self.star_tile_decoder.psf_forward()

    def sample_prior(self, batch_size=1):
        n_sources = self._sample_n_sources(batch_size)
        is_on_array = get_is_on_from_n_sources(n_sources, self.max_sources)
        locs = self._sample_locs(is_on_array, batch_size)

        _, _, galaxy_bool, star_bool = self._sample_n_galaxies_and_stars(n_sources, is_on_array)
        galaxy_params = self._sample_galaxy_params(galaxy_bool)
        fluxes = self._sample_fluxes(n_sources, star_bool, batch_size)
        log_fluxes = self._get_log_fluxes(fluxes)

        # per tile quantities.
        return {
            "n_sources": n_sources,
            "locs": locs,
            "galaxy_bool": galaxy_bool,
            "galaxy_params": galaxy_params,
            "fluxes": fluxes,
            "log_fluxes": log_fluxes,
        }

    def render_images(self, n_sources, locs, galaxy_bool, galaxy_params, fluxes, add_noise=True):
        # returns the **full** image in shape (batch_size x n_bands x slen x slen)

        # n_sources: is (batch_size x n_tiles_per_image)
        # locs: is (batch_size x n_tiles_per_image x max_sources x 2)
        # galaxy_bool: Is (batch_size x n_tiles_per_image x max_sources x 1)
        # galaxy_params : is (batch_size x n_tiles_per_image x max_sources x latent_dim)
        # fluxes: Is (batch_size x n_tiles_per_image x max_sources x n_bands)

        assert n_sources.shape[0] == locs.shape[0]
        assert n_sources.shape[1] == locs.shape[1]
        assert galaxy_bool.shape[-1] == 1

        # first render the padded tiles
        image_ptiles, var_ptiles = self._render_ptiles(
            n_sources, locs, galaxy_bool, galaxy_params, fluxes
        )

        # render the image from padded tiles
        images = self._construct_full_image_from_ptiles(
            image_ptiles, self.tile_slen, self.border_padding
        )
        var_images = self._construct_full_image_from_ptiles(
            var_ptiles, self.tile_slen, self.border_padding
        )

        # add background and noise
        background = self.get_background(images.shape[-1])
        images += background.unsqueeze(0)
        var_images += background.unsqueeze(0)
        if add_noise:
            images = self._apply_noise(images)

        return images, var_images

    @property
    def galaxy_decoder(self):
        if self.galaxy_tile_decoder is None:
            return None
        return self.galaxy_tile_decoder.galaxy_decoder

    def get_background(self, slen):
        background_shape = (self.n_bands, slen, slen)
        background = torch.zeros(*background_shape, device=self.device)
        for i in range(self.n_bands):
            background[i] = self.background_values[i]

        return background

    def _sample_n_sources(self, batch_size):
        # returns number of sources for each batch x tile
        # output dimension is batch_size x n_tiles_per_image

        # always poisson distributed.
        p = torch.full((1,), self.mean_sources, device=self.device, dtype=torch.float)
        m = Poisson(p)
        n_sources = m.sample([batch_size, self.n_tiles_per_image])

        # long() here is necessary because used for indexing and one_hot encoding.
        n_sources = n_sources.clamp(max=self.max_sources, min=self.min_sources)
        n_sources = n_sources.long().view(batch_size, self.n_tiles_per_image)
        return n_sources

    def _sample_locs(self, is_on_array, batch_size):
        # output dimension is batch_size x n_tiles_per_image x max_sources x 2

        # 2 = (x,y)
        shape = (
            batch_size,
            self.n_tiles_per_image,
            self.max_sources,
            2,
        )
        locs = torch.rand(*shape, device=is_on_array.device)
        locs *= self.loc_max - self.loc_min
        locs += self.loc_min
        locs *= is_on_array.unsqueeze(-1)

        return locs

    def _sample_n_galaxies_and_stars(self, n_sources, is_on_array):
        # the counts returned (n_galaxies, n_stars) are of shape (batch_size x n_tiles_per_image)
        # the booleans returned (galaxy_bool, star_bool) are of shape
        # (batch_size x n_tiles_per_image x max_sources x 1)
        # this last dimension is so it is parallel to other galaxy/star params.

        batch_size = n_sources.size(0)
        uniform = torch.rand(
            batch_size,
            self.n_tiles_per_image,
            self.max_sources,
            1,
            device=is_on_array.device,
        )
        galaxy_bool = uniform < self.prob_galaxy
        galaxy_bool = (galaxy_bool * is_on_array.unsqueeze(-1)).float()
        star_bool = (1 - galaxy_bool) * is_on_array.unsqueeze(-1)
        n_galaxies = galaxy_bool.sum((-2, -1))
        n_stars = star_bool.sum((-2, -1))
        assert torch.all(n_stars <= n_sources) and torch.all(n_galaxies <= n_sources)

        return n_galaxies, n_stars, galaxy_bool, star_bool

    def _pareto_cdf(self, x):
        return 1 - (self.f_min / x) ** self.alpha

    def _draw_pareto_maxed(self, shape):
        # draw pareto conditioned on being less than f_max

        u_max = self._pareto_cdf(self.f_max)
        uniform_samples = torch.rand(*shape, device=self.device) * u_max
        return self.f_min / (1.0 - uniform_samples) ** (1 / self.alpha)

    def _sample_fluxes(self, n_stars, star_bool, batch_size):
        """
        Returns:
            fluxes, tensor shape (batch_size x self.n_tiles_per_image x self.max_sources x n_bands)
        """
        assert n_stars.shape[0] == batch_size

        shape = (batch_size, self.n_tiles_per_image, self.max_sources, 1)
        base_fluxes = self._draw_pareto_maxed(shape)

        if self.n_bands > 1:
            shape = (
                batch_size,
                self.n_tiles_per_image,
                self.max_sources,
                self.n_bands - 1,
            )
            colors = torch.randn(*shape, device=base_fluxes.device)
            _fluxes = 10 ** (colors / 2.5) * base_fluxes
            fluxes = torch.cat((base_fluxes, _fluxes), dim=3)
            fluxes *= star_bool.float()
        else:
            fluxes = base_fluxes * star_bool.float()

        return fluxes

    def _sample_galaxy_params(self, galaxy_bool):
        # galaxy latent variables are obtaind from previously encoded variables from
        # large dataset of simulated galaxies stored in `self.latents`
        # NOTE: These latent variables DO NOT follow a specific distribution.

        assert galaxy_bool.shape[1:] == (self.n_tiles_per_image, self.max_sources, 1)
        batch_size = galaxy_bool.size(0)
        total_latent = batch_size * self.n_tiles_per_image * self.max_sources
        shape = (
            batch_size,
            self.n_tiles_per_image,
            self.max_sources,
            self.n_galaxy_params,
        )

        # first get random subset of indices to extract from self.latents
        indices = torch.randint(0, len(self.latents), (total_latent,), device=galaxy_bool.device)
        galaxy_params = self.latents[indices].reshape(shape)
        return galaxy_params * galaxy_bool

    @staticmethod
    def _get_log_fluxes(fluxes):
        log_fluxes = torch.where(
            fluxes > 0, fluxes, torch.ones_like(fluxes)
        )  # prevent log(0) errors.
        log_fluxes = torch.log(log_fluxes)

        return log_fluxes

    @staticmethod
    def _apply_noise(images_mean):
        # add noise to images.

        if torch.any(images_mean <= 0):
            warnings.warn("image mean less than 0")
            images_mean = images_mean.clamp(min=1.0)

        _images = torch.sqrt(images_mean) * torch.randn_like(images_mean)
        images = _images + images_mean

        return images

    def _render_ptiles(self, n_sources, locs, galaxy_bool, galaxy_params, fluxes):
        # n_sources: is (batch_size x n_tiles_per_image)
        # locs: is (batch_size x n_tiles_per_image x max_sources x 2)
        # galaxy_bool: Is (batch_size x n_tiles_per_image x max_sources)
        # galaxy_params : is (batch_size x n_tiles_per_image x max_sources x latent_dim)
        # fluxes: Is (batch_size x n_tiles_per_image x max_sources x 2)

        # returns the ptiles with shape =
        # (batch_size x n_tiles_per_image x n_bands x ptile_slen x ptile_slen)
        n_tiles_per_image = n_sources.shape[1]
        max_sources = locs.shape[2]
        assert (n_sources <= max_sources).all()
        batch_size = n_sources.shape[0]
        n_ptiles = batch_size * n_tiles_per_image

        # view parameters being explicit about shapes
        _n_sources = n_sources.view(n_ptiles)
        _locs = locs.view(n_ptiles, max_sources, 2)
        _galaxy_bool = galaxy_bool.view(n_ptiles, max_sources, 1)
        _fluxes = fluxes.view(n_ptiles, max_sources, self.n_bands)

        # draw stars and galaxies
        _is_on_array = get_is_on_from_n_sources(_n_sources, max_sources)
        _is_on_array = _is_on_array.view(n_ptiles, max_sources, 1)
        _star_bool = (1 - _galaxy_bool) * _is_on_array
        _star_bool = _star_bool.view(n_ptiles, max_sources, 1)

        # final shapes of images.
        img_shape = (
            batch_size,
            n_tiles_per_image,
            self.n_bands,
            self.ptile_slen,
            self.ptile_slen,
        )

        # draw stars and galaxies
        stars = self.star_tile_decoder(_locs, _fluxes, _star_bool)
        galaxies = torch.zeros(img_shape, device=locs.device)
        var_images = torch.zeros(img_shape, device=locs.device)
        if self.galaxy_tile_decoder is not None:
            galaxies, var_images = self.galaxy_tile_decoder(_locs, galaxy_params, _galaxy_bool)

        images = galaxies.view(img_shape) + stars.view(img_shape)
        var_images = var_images.view(img_shape)

        return images, var_images

    @staticmethod
    def _construct_full_image_from_ptiles(image_ptiles, tile_slen, border_padding):
        # image_tiles is (batch_size, n_tiles_per_image, n_bands, ptile_slen x ptile_slen)
        batch_size = image_ptiles.shape[0]
        n_tiles_per_image = image_ptiles.shape[1]
        n_bands = image_ptiles.shape[2]
        ptile_slen = image_ptiles.shape[3]
        assert image_ptiles.shape[4] == ptile_slen

        n_tiles1 = np.sqrt(n_tiles_per_image)
        # check it is an integer
        assert n_tiles1 % 1 == 0
        n_tiles1 = int(n_tiles1)

        # the number of tiles in ptile row
        # i.e the slen of a ptile but in units of tile_slen
        n_tiles1_in_ptile = ptile_slen / tile_slen
        n_tiles_of_padding = (n_tiles1_in_ptile - 1) / 2
        assert (
            n_tiles1_in_ptile % 1 == 0
        ), "tile_slen and ptile_slen are not compatible. check tile_slen argument"
        assert (
            n_tiles_of_padding % 1 == 0
        ), "tile_slen and ptile_slen are not compatible. check tile_slen argument"
        n_tiles1_in_ptile = int(n_tiles1_in_ptile)
        n_tiles_of_padding = int(n_tiles_of_padding)

        image_tiles_4d = image_ptiles.view(
            batch_size, n_tiles1, n_tiles1, n_bands, ptile_slen, ptile_slen
        )

        # zero pad tiles, so that the number of tiles in a row (and column)
        # are divisible by n_tiles1_in_ptile
        n_tiles_pad = n_tiles1_in_ptile - (n_tiles1 % n_tiles1_in_ptile)
        zero_pads1 = torch.zeros(
            batch_size,
            n_tiles_pad,
            n_tiles1,
            n_bands,
            ptile_slen,
            ptile_slen,
            device=image_ptiles.device,
        )
        zero_pads2 = torch.zeros(
            batch_size,
            n_tiles1 + n_tiles_pad,
            n_tiles_pad,
            n_bands,
            ptile_slen,
            ptile_slen,
            device=image_ptiles.device,
        )
        image_tiles_4d = torch.cat((image_tiles_4d, zero_pads1), dim=1)
        image_tiles_4d = torch.cat((image_tiles_4d, zero_pads2), dim=2)

        # construct the full image
        n_tiles = n_tiles1 + n_tiles_pad
        canvas = torch.zeros(
            batch_size,
            n_bands,
            (n_tiles + n_tiles1_in_ptile - 1) * tile_slen,
            (n_tiles + n_tiles1_in_ptile - 1) * tile_slen,
            device=image_ptiles.device,
        )

        # loop through all tiles in a ptile
        for i in range(n_tiles1_in_ptile):
            for j in range(n_tiles1_in_ptile):
                indx_vec1 = torch.arange(
                    start=i,
                    end=n_tiles,
                    step=n_tiles1_in_ptile,
                    device=image_ptiles.device,
                )
                indx_vec2 = torch.arange(
                    start=j,
                    end=n_tiles,
                    step=n_tiles1_in_ptile,
                    device=image_ptiles.device,
                )

                canvas_len = len(indx_vec1) * ptile_slen

                image_tile_rows = image_tiles_4d[:, indx_vec1]
                image_tile_cols = image_tile_rows[:, :, indx_vec2]

                # get canvas
                canvas[
                    :,
                    :,
                    (i * tile_slen) : (i * tile_slen + canvas_len),
                    (j * tile_slen) : (j * tile_slen + canvas_len),
                ] += image_tile_cols.permute(0, 3, 1, 4, 2, 5).reshape(
                    batch_size, n_bands, canvas_len, canvas_len
                )

        # trim to original image size
        x0 = n_tiles_of_padding * tile_slen - border_padding
        x1 = (n_tiles1 + n_tiles_of_padding) * tile_slen + border_padding
        return canvas[:, :, x0:x1, x0:x1]


class Tiler(nn.Module):
    """
    This class creates an image tile from multiple sources.
    """

    def __init__(self, tile_slen, ptile_slen):
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

    def forward(self, locs, source):
        return self.render_one_source(locs, source)

    def render_one_source(self, locs, source):
        """
        :param locs: is n_ptiles x len((x,y))
        :param source: is a (n_ptiles, n_bands, slen, slen) tensor, which could either be a
                        `expanded_psf` (psf repeated multiple times) for the case of of stars.
                        Or multiple galaxies in the case of galaxies.
        :return: shape = (n_ptiles x n_bands x slen x slen)
        """
        n_ptiles = locs.shape[0]
        assert source.shape[2] == source.shape[3]
        assert locs.shape[1] == 2

        # scale so that they land in the tile within the padded tile
        padding = (self.ptile_slen - self.tile_slen) / 2
        locs = locs * (self.tile_slen / self.ptile_slen) + (padding / self.ptile_slen)
        # scale locs so they take values between -1 and 1 for grid sample
        locs = (locs - 0.5) * 2
        _grid = self.cached_grid.view(1, self.ptile_slen, self.ptile_slen, 2)

        locs_swapped = locs.index_select(1, self.swap)
        grid_loc = _grid - locs_swapped.view(n_ptiles, 1, 1, 2)

        source_rendered = F.grid_sample(source, grid_loc, align_corners=True)
        return source_rendered

    def render_tile(self, locs, sources):
        """
        :param locs: is (n_ptiles x max_num_stars x 2)
        :param sources: is (n_ptiles x max_num_stars x n_bands x stampsize x stampsize)

        :return: ptile = (n_ptiles x n_bands x slen x slen)
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
            one_star = self.render_one_source(locs[:, n, :], sources[:, n])
            ptile += one_star

        return ptile

    def fit_source_to_ptile(self, source):
        if self.ptile_slen >= source.shape[-1]:
            fitted_source = self._expand_source(source)
        else:
            fitted_source = self._trim_source(source)
        return fitted_source

    def _expand_source(self, source):
        """Pad the source with zeros so that it is size ptile_slen,"""
        assert len(source.shape) == 3

        _slen = self.ptile_slen + ((self.ptile_slen % 2) == 0) * 1
        assert len(source.shape) == 3

        source_slen = source.shape[2]

        assert source_slen <= _slen, "Should be using trim source."

        source_expanded = torch.zeros(source.shape[0], _slen, _slen, device=source.device)
        offset = int((_slen - source_slen) / 2)

        source_expanded[
            :, offset : (offset + source_slen), offset : (offset + source_slen)
        ] = source

        return source_expanded

    def _trim_source(self, source):
        """Crop the source to length ptile_slen x ptile_slen, centered at the middle."""
        assert len(source.shape) == 3

        # if self.ptile_slen is even, we still make source dimension odd.
        # otherwise, the source won't have a peak in the center pixel.
        _slen = self.ptile_slen + ((self.ptile_slen % 2) == 0) * 1

        source_slen = source.shape[2]
        source_center = (source_slen - 1) / 2

        assert source_slen >= _slen

        r = np.floor(_slen / 2)
        l_indx = int(source_center - r)
        u_indx = int(source_center + r + 1)

        return source[:, l_indx:u_indx, l_indx:u_indx]


class StarTileDecoder(nn.Module):
    def __init__(
        self,
        tiler,
        n_bands,
        psf_params_file,
        psf_slen,
    ):
        super().__init__()
        self.tiler = tiler
        self.n_bands = n_bands

        ext = Path(psf_params_file).suffix
        if ext == ".npy":
            psf_params = torch.from_numpy(np.load(psf_params_file))
            psf_params = psf_params[list(range(n_bands))]
        elif ext == ".fits":
            assert n_bands == 2, "only 2 band fit files are supported."
            bands = (2, 3)
            psf_params = self.get_fit_file_psf_params(psf_params_file, bands)
        else:
            raise NotImplementedError(
                "Only .npy and .fits extensions are supported for PSF params files."
            )
        self.params = nn.Parameter(psf_params.clone(), requires_grad=True)
        self.psf_slen = psf_slen
        grid = get_mgrid(self.psf_slen) * (self.psf_slen - 1) / 2
        # extra factor to be consistent with old repo
        # but probably doesn't matter ...
        grid *= self.psf_slen / (self.psf_slen - 1)
        self.register_buffer("cached_radii_grid", (grid ** 2).sum(2).sqrt())

        # get psf normalization_constant
        self.normalization_constant = torch.zeros(self.n_bands)
        for i in range(self.n_bands):
            psf_i = self._get_psf_single_band(psf_params[i])
            self.normalization_constant[i] = 1 / psf_i.sum()
        self.normalization_constant = self.normalization_constant.detach()

    def forward(self, locs, fluxes, star_bool):
        # locs: is (n_ptiles x max_num_stars x 2)
        # fluxes: Is (n_ptiles x max_stars x n_bands)
        # star_bool: Is (n_ptiles x max_stars x 1)
        # max_sources obtained from locs, allows for more flexibility when rendering.

        psf = self._adjust_psf()
        n_ptiles = locs.shape[0]
        max_sources = locs.shape[1]

        assert len(psf.shape) == 3  # the shape is (n_bands, ptile_slen, ptile_slen)
        assert psf.shape[0] == self.n_bands
        assert fluxes.shape[0] == star_bool.shape[0] == n_ptiles
        assert fluxes.shape[1] == star_bool.shape[1] == max_sources
        assert fluxes.shape[2] == psf.shape[0] == self.n_bands
        assert star_bool.shape[2] == 1

        # all stars are just the PSF so we copy it.
        expanded_psf = psf.expand(n_ptiles, max_sources, self.n_bands, -1, -1)
        sources = expanded_psf * fluxes.unsqueeze(-1).unsqueeze(-1)
        sources *= star_bool.unsqueeze(-1).unsqueeze(-1)

        return self.tiler.render_tile(locs, sources)

    def psf_forward(self):
        psf = self._get_psf()
        init_psf_sum = psf.sum(-1).sum(-1).detach()
        norm = psf.sum(-1).sum(-1)
        psf *= (init_psf_sum / norm).unsqueeze(-1).unsqueeze(-1)
        return psf

    @staticmethod
    def get_fit_file_psf_params(psf_fit_file, bands=(2, 3)):
        psfield = fits.open(psf_fit_file, ignore_missing_end=True)
        psf_params = torch.zeros(len(bands), 6)
        for i, band in enumerate(bands):
            sigma1 = psfield[6].data["psf_sigma1"][0][band] ** 2
            sigma2 = psfield[6].data["psf_sigma2"][0][band] ** 2
            sigmap = psfield[6].data["psf_sigmap"][0][band] ** 2

            beta = psfield[6].data["psf_beta"][0][band]
            b = psfield[6].data["psf_b"][0][band]
            p0 = psfield[6].data["psf_p0"][0][band]

            psf_params[i] = torch.log(torch.tensor([sigma1, sigma2, sigmap, beta, b, p0]))

        return psf_params

    def _get_psf(self):
        psf_list = []
        for i in range(self.n_bands):
            _psf = self._get_psf_single_band(self.params[i])
            _psf *= self.normalization_constant[i]
            psf_list.append(_psf.unsqueeze(0))
        psf = torch.cat(psf_list)

        assert (psf > 0).all()
        return psf

    @staticmethod
    def _psf_fun(r, sigma1, sigma2, sigmap, beta, b, p0):
        term1 = torch.exp(-(r ** 2) / (2 * sigma1))
        term2 = b * torch.exp(-(r ** 2) / (2 * sigma2))
        term3 = p0 * (1 + r ** 2 / (beta * sigmap)) ** (-beta / 2)
        return (term1 + term2 + term3) / (1 + b + p0)

    def _get_psf_single_band(self, psf_params):
        _psf_params = torch.exp(psf_params)
        return self._psf_fun(
            self.cached_radii_grid,
            _psf_params[0],
            _psf_params[1],
            _psf_params[2],
            _psf_params[3],
            _psf_params[4],
            _psf_params[5],
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
        self,
        n_bands,
        tile_slen,
        ptile_slen,
        gal_slen,
        n_galaxy_params,
        autoencoder_ckpt,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.tiler = Tiler(tile_slen, ptile_slen)
        self.ptile_slen = ptile_slen

        # load decoder after loading autoencoder from checkpoint.
        autoencoder = galaxy_net.OneCenteredGalaxyAE.load_from_checkpoint(autoencoder_ckpt)
        assert gal_slen == autoencoder.hparams.slen
        assert n_galaxy_params == autoencoder.hparams.latent_dim
        dec = autoencoder.dec
        dec.eval().requires_grad_(False)
        self.galaxy_decoder = dec

        self.gal_slen = gal_slen
        self.n_galaxy_params = n_galaxy_params

    def forward(self, locs, galaxy_params, galaxy_bool):
        # max_sources obtained from locs, allows for more flexibility when rendering.
        n_ptiles = locs.shape[0]
        max_sources = locs.shape[1]

        galaxy_params = galaxy_params.view(n_ptiles, max_sources, self.n_galaxy_params)
        assert galaxy_params.shape[0] == galaxy_bool.shape[0] == n_ptiles
        assert galaxy_params.shape[1] == galaxy_bool.shape[1] == max_sources
        assert galaxy_params.shape[2] == self.n_galaxy_params
        assert galaxy_bool.shape[2] == 1

        single_galaxies, single_vars = self._render_single_galaxies(galaxy_params, galaxy_bool)

        ptile = self.tiler.render_tile(
            locs, single_galaxies * galaxy_bool.unsqueeze(-1).unsqueeze(-1)
        )
        var_ptile = self.tiler.render_tile(
            locs, single_vars * galaxy_bool.unsqueeze(-1).unsqueeze(-1)
        )

        return ptile, var_ptile

    def _render_single_galaxies(self, galaxy_params, galaxy_bool):

        # flatten parameters
        z = galaxy_params.view(-1, self.n_galaxy_params)
        b = galaxy_bool.flatten()

        # allocate memory
        _slen = self.ptile_slen + ((self.ptile_slen % 2) == 0) * 1
        gal = torch.zeros(z.shape[0], self.n_bands, _slen, _slen, device=galaxy_params.device)
        var = torch.zeros(z.shape[0], self.n_bands, _slen, _slen, device=galaxy_params.device)

        # forward only galaxies that are on!
        # no background
        gal_on = self.galaxy_decoder(z[b == 1])
        var_on = gal_on  # poisson approximation, mean = var.

        # size the galaxy (either trims or crops to the size of ptile)
        gal_on = self._size_galaxy(gal_on)
        var_on = self._size_galaxy(var_on)

        # set galaxies
        gal[b == 1] = gal_on
        var[b == 1] = var_on

        batchsize = galaxy_params.shape[0]
        gal_shape = (batchsize, -1, self.n_bands, gal.shape[-1], gal.shape[-1])
        single_galaxies = gal.view(gal_shape)
        single_vars = var.view(gal_shape)

        return single_galaxies, single_vars

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
