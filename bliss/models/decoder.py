import numpy as np
import warnings
from pathlib import Path
from astropy.io import fits

import torch
import torch.nn as nn
from torch.nn.functional import pad
import torch.nn.functional as F
from torch.distributions import Poisson, Normal

from .. import device
from .encoder import get_is_on_from_n_sources
from . import galaxy_net


def get_mgrid(slen):
    offset = (slen - 1) / 2
    x, y = np.mgrid[-offset : (offset + 1), -offset : (offset + 1)]
    mgrid = torch.tensor(np.dstack((y, x))) / offset
    # mgrid is between -1 and 1
    # then scale slightly because of the way f.grid_sample
    # parameterizes the edges: (0, 0) is center of edge pixel
    return mgrid.float().to(device) * (slen - 1) / slen


def get_fit_file_psf_params(psf_fit_file, bands=(2, 3)):
    psfield = fits.open(psf_fit_file)
    psf_params = torch.zeros(len(bands), 6)
    for i in range(len(bands)):
        band = bands[i]

        sigma1 = psfield[6].data["psf_sigma1"][0][band] ** 2
        sigma2 = psfield[6].data["psf_sigma2"][0][band] ** 2
        sigmap = psfield[6].data["psf_sigmap"][0][band] ** 2

        beta = psfield[6].data["psf_beta"][0][band]
        b = psfield[6].data["psf_b"][0][band]
        p0 = psfield[6].data["psf_p0"][0][band]

        psf_params[i] = torch.log(torch.tensor([sigma1, sigma2, sigmap, beta, b, p0]))

    return psf_params.to(device)


def get_star_bool(n_sources, galaxy_bool):
    assert n_sources.shape[0] == galaxy_bool.shape[0]
    assert galaxy_bool.shape[-1] == 1
    max_sources = galaxy_bool.shape[-2]
    assert n_sources.le(max_sources).all()
    is_on_array = get_is_on_from_n_sources(n_sources, max_sources)
    is_on_array = is_on_array.reshape(*galaxy_bool.shape)
    star_bool = (1 - galaxy_bool) * is_on_array
    return star_bool


class ImageDecoder(nn.Module):
    def __init__(
        self,
        n_bands=1,
        slen=50,
        tile_slen=2,
        ptile_padding=2,
        prob_galaxy=0.0,
        n_galaxy_params=8,
        max_sources=2,
        mean_sources=0.4,
        min_sources=0,
        f_min=1e4,
        f_max=1e6,
        alpha=0.5,
        gal_slen=51,
        psf_slen=25,
        decoder_file=None,
        psf_params_file="psf_params.npy",
        background_values=(686.0, 1123.0),
        loc_min=0.0,
        loc_max=1.0,
        add_noise=True,
    ):
        super(ImageDecoder, self).__init__()

        # side-length in pixels of an image (image is assumed to be square)
        self.slen = slen

        # side-length of an image tile.
        # latent variables (locations, fluxes, etc) are drawn per-tile
        self.tile_slen = tile_slen
        assert (self.slen % self.tile_slen) == 0, (
            "We assume that the tiles cover the image."
            + " So slen must be divisible by tile_slen"
        )

        # Images are first rendered on *padded* tiles (aka ptiles).
        # The padded tile consists of the tile and neighboring tiles
        # The width of the padding (in number of tiles) is given by ptile padding.
        # e.g. if ptile_padding is 1, then each ptile consists of 9 tiles:
        # the center tile, surrounded one tile on each side.
        self.ptile_padding = ptile_padding
        self.ptile_slen = int((self.ptile_padding * 2 + 1) * tile_slen)

        # number of tiles per image
        n_tiles_per_image = (self.slen / self.tile_slen) ** 2
        self.n_tiles_per_image = int(n_tiles_per_image)

        self.n_bands = n_bands  # number of bands

        # per-tile prior parameters on number (and type) of sources
        self.max_sources = max_sources
        self.mean_sources = mean_sources
        self.min_sources = min_sources
        self.prob_galaxy = float(prob_galaxy)

        # per-tile constraints on the location of sources
        self.loc_min = loc_min
        self.loc_max = loc_max

        self.add_noise = add_noise

        # prior parameters on fluxes
        self.f_min = f_min
        self.f_max = f_max
        self.alpha = alpha  # pareto parameter.

        # caching the underlying
        # coordinates on which we simulate source
        # grid: between -1 and 1,
        # then scale slightly because of the way f.grid_sample
        # parameterizes the edges: (0, 0) is center of edge pixel
        self.cached_grid = get_mgrid(self.ptile_slen)

        # misc
        self.swap = torch.tensor([1, 0], device=device)

        # background
        assert len(background_values) == n_bands
        background_shape = (self.n_bands, self.slen, self.slen)
        self.background = torch.zeros(background_shape, device=device)
        for i in range(n_bands):
            self.background[i] = background_values[i]

        # galaxy decoder
        self.gal_slen = gal_slen
        self.n_galaxy_params = n_galaxy_params
        self.galaxy_decoder = None
        assert self.prob_galaxy > 0.0 or decoder_file is None
        if decoder_file is not None:
            dec = galaxy_net.CenteredGalaxyDecoder(gal_slen, n_galaxy_params, n_bands)
            dec = dec.to(device)
            dec.load_state_dict(torch.load(decoder_file, map_location=device))
            dec.eval().requires_grad_(False)
            self.galaxy_decoder = dec

        # load psf params + grid
        ext = Path(psf_params_file).suffix
        if ext == ".npy":
            psf_params = torch.from_numpy(np.load(psf_params_file)).to(device)
            psf_params = psf_params[list(range(n_bands))]
        elif ext == ".fit":
            assert n_bands == 2, "only 2 band fit files are supported."
            bands = (2, 3)
            psf_params = get_fit_file_psf_params(psf_params_file, bands)
        else:
            raise NotImplementedError(
                "Only .npy and .fit extensions are supported for PSF params files."
            )

        self.params = nn.Parameter(psf_params.clone(), requires_grad=True)
        self.psf_slen = psf_slen
        grid = get_mgrid(self.psf_slen) * (self.psf_slen - 1) / 2
        self.register_buffer("cached_radii_grid", (grid ** 2).sum(2).sqrt())

        # get normalization_constant
        self.normalization_constant = torch.zeros(self.n_bands)
        for i in range(self.n_bands):
            psf_i = self._get_psf_single_band(psf_params[i])
            self.normalization_constant[i] = 1 / psf_i.sum()
        self.normalization_constant = self.normalization_constant.detach()

    def forward(self):
        psf = self._get_psf()
        init_psf_sum = psf.sum(-1).sum(-1).detach()
        norm = psf.sum(-1).sum(-1)
        psf *= (init_psf_sum / norm).unsqueeze(-1).unsqueeze(-1)
        l_pad = (self.slen + ((self.slen % 2) == 0) * 1 - self.psf_slen) // 2

        # add padding so psf has length of image_slen
        return pad(psf, [l_pad] * 4)

    def _get_psf(self):
        # TODO make the psf function vectorized ...
        psf = torch.tensor([])
        for i in range(self.n_bands):
            _psf = self._get_psf_single_band(self.params[i])
            _psf *= self.normalization_constant[i]
            psf = torch.cat((psf.type_as(_psf), _psf.unsqueeze(0)))

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

    def _sample_n_sources(self, batch_size):
        # returns number of sources for each batch x tile
        # output dimension is batch_size x n_tiles_per_image

        # always poisson distributed.
        p = torch.full((1,), self.mean_sources, device=device).float()
        m = Poisson(p)
        n_sources = m.sample([batch_size, self.n_tiles_per_image])

        # long() here is necessary because used for indexing and one_hot encoding.
        n_sources = n_sources.clamp(max=self.max_sources, min=self.min_sources)
        n_sources = n_sources.long().reshape(batch_size, self.n_tiles_per_image)
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
        locs = torch.rand(*shape, device=device)
        locs *= self.loc_max - self.loc_min
        locs += self.loc_min
        locs *= is_on_array.unsqueeze(-1)

        return locs

    def _sample_n_galaxies_and_stars(self, n_sources, is_on_array):
        # the counts returned (n_galaxies, n_stars) are of shape (batch_size x n_tiles_per_image)
        # the booleans returned (galaxy_bool, star_bool) are of shape
        # (batch_size x n_tiles_per_image x max_detections x 1)
        # this last dimension is so it is parallel to other galaxy/star params.

        batch_size = n_sources.size(0)
        uniform = torch.rand(
            batch_size, self.n_tiles_per_image, self.max_sources, 1, device=device
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
        uniform_samples = torch.rand(*shape, device=device) * u_max
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
            colors = torch.randn(*shape, device=device) 
            _fluxes = 10 ** (colors / 2.5) * base_fluxes
            fluxes = torch.cat((base_fluxes, _fluxes), dim=3)
            fluxes *= star_bool.float()
        else:
            fluxes = base_fluxes * star_bool.float()

        return fluxes

    def _sample_galaxy_params(self, n_galaxies, galaxy_bool):
        # galaxy params are just Normal(0,1) variables.

        assert len(n_galaxies.shape) == 2
        batch_size = n_galaxies.size(0)
        mean = torch.zeros(1, dtype=torch.float, device=device)
        std = torch.ones(1, dtype=torch.float, device=device)
        p_z = Normal(mean, std)
        shape = (
            batch_size,
            self.n_tiles_per_image,
            self.max_sources,
            self.n_galaxy_params,
        )
        sample_shape = torch.tensor(shape)
        galaxy_params = p_z.rsample(sample_shape)
        galaxy_params = galaxy_params.reshape(*shape)

        # zero out excess according to galaxy_bool.
        galaxy_params = galaxy_params * galaxy_bool
        return galaxy_params

    def sample_prior(self, batch_size=1):
        n_sources = self._sample_n_sources(batch_size)
        is_on_array = get_is_on_from_n_sources(n_sources, self.max_sources)
        locs = self._sample_locs(is_on_array, batch_size)

        n_galaxies, n_stars, galaxy_bool, star_bool = self._sample_n_galaxies_and_stars(
            n_sources, is_on_array
        )
        galaxy_params = self._sample_galaxy_params(n_galaxies, galaxy_bool)

        fluxes = self._sample_fluxes(n_sources, star_bool, batch_size)
        log_fluxes = self._get_log_fluxes(fluxes)

        # per tile quantities.
        return {
            "n_sources": n_sources,
            "n_galaxies": n_galaxies,
            "n_stars": n_stars,
            "locs": locs,
            "galaxy_params": galaxy_params,
            "fluxes": fluxes,
            "log_fluxes": log_fluxes,
            "galaxy_bool": galaxy_bool,
        }

    @staticmethod
    def _get_log_fluxes(fluxes):
        log_fluxes = torch.where(
            fluxes > 0, fluxes, torch.ones(*fluxes.shape).to(device)
        )  # prevent log(0) errors.
        log_fluxes = torch.log(log_fluxes)

        return log_fluxes

    @staticmethod
    def _apply_noise(images_mean):
        # add noise to images.

        if torch.any(images_mean <= 0):
            warnings.warn("image mean less than 0")
            images_mean = images_mean.clamp(min=1.0)

        _images = torch.sqrt(images_mean)
        images = _images * torch.randn_like(images_mean)
        images = images + images_mean

        return images

    def _trim_source(self, source):
        """Crop the source to length ptile_slen x ptile_slen,
        centered at the middle.
        """
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

    def _expand_source(self, source):
        """Pad the source with zeros so that it is size ptile_slen,"""
        assert len(source.shape) == 3

        _slen = self.ptile_slen + ((self.ptile_slen % 2) == 0) * 1
        assert len(source.shape) == 3

        source_slen = source.shape[2]

        assert source_slen <= _slen, "Should be using trim source."

        source_expanded = torch.zeros(source.shape[0], _slen, _slen, device=device)
        offset = int((_slen - source_slen) / 2)

        source_expanded[
            :, offset : (offset + source_slen), offset : (offset + source_slen)
        ] = source

        return source_expanded

    def _adjust_psf(self):
        # use power_law_psf and current psf parameters to forward and obtain fresh psf model.
        # first dimension of psf is number of bands
        # dimension of the psf/slen should be odd
        psf = self.forward()
        psf_slen = psf.shape[2]
        assert len(psf.shape) == 3
        assert psf.shape[1] == psf_slen
        assert (psf_slen % 2) == 1
        assert self.background.shape[0] == psf.shape[0] == self.n_bands

        if self.ptile_slen >= psf.shape[-1]:
            return self._expand_source(psf)
        else:
            return self._trim_source(psf)

    def _size_galaxy(self, galaxy):
        # galaxy should be shape n_galaxies x n_bands x galaxy_slen x galaxy_slen
        assert len(galaxy.shape) == 4
        assert galaxy.shape[2] == galaxy.shape[3]
        assert (galaxy.shape[3] % 2) == 1, "dimension of galaxy image should be odd"
        assert self.background.shape[0] == galaxy.shape[1] == self.n_bands

        n_galaxies = galaxy.shape[0]
        galaxy_slen = galaxy.shape[3]
        galaxy = galaxy.reshape(n_galaxies * self.n_bands, galaxy_slen, galaxy_slen)

        if self.ptile_slen >= galaxy.shape[-1]:
            sized_galaxy = self._expand_source(galaxy)
        else:
            sized_galaxy = self._trim_source(galaxy)

        outsize = sized_galaxy.shape[-1]
        return sized_galaxy.reshape(n_galaxies, self.n_bands, outsize, outsize)

    def _render_one_source(self, locs, source):
        """
        :param locs: is n_ptiles x len((x,y))
        :param source: is a (n_ptiles, n_bands, slen, slen) tensor, which could either be a
                        `expanded_psf` (psf repeated multiple times) for the case of of stars.
                        Or multiple galaxies in the case of galaxies.
        :return: shape = (n_ptiles x n_bands x slen x slen)
        """
        n_ptiles = locs.shape[0]
        assert source.shape[0] == n_ptiles
        assert source.shape[1] == self.n_bands
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

    def _render_multiple_stars_on_ptile(self, locs, fluxes, star_bool):
        # locs: is (n_ptiles x max_num_stars x 2)
        # fluxes: Is (n_ptiles x n_bands x max_stars)
        # star_bool: Is (n_ptiles x max_stars x 1)
        # max_sources obtained from locs, allows for more flexibility when rendering.

        psf = self._adjust_psf()
        n_ptiles = locs.shape[0]
        max_sources = locs.shape[1]
        ptile_shape = (n_ptiles, self.n_bands, self.ptile_slen, self.ptile_slen)
        ptile = torch.zeros(ptile_shape, device=device)

        assert len(psf.shape) == 3  # the shape is (n_bands, ptile_slen, ptile_slen)
        assert psf.shape[0] == self.n_bands
        assert fluxes.shape[0] == star_bool.shape[0] == n_ptiles
        assert fluxes.shape[1] == star_bool.shape[1] == max_sources
        assert fluxes.shape[2] == psf.shape[0] == self.n_bands
        assert star_bool.shape[2] == 1

        # all stars are just the PSF so we copy it.
        expanded_psf = psf.expand(n_ptiles, self.n_bands, -1, -1)

        # this loop plots each of the ith star in each of the (n_ptiles) images.
        for n in range(max_sources):
            star_bool_n = star_bool[:, n]
            locs_n = locs[:, n, :]
            fluxes_n = fluxes[:, n, :] * star_bool_n.reshape(-1, 1)
            fluxes_n = fluxes_n.view(n_ptiles, self.n_bands, 1, 1)
            one_star = self._render_one_source(locs_n, expanded_psf)
            ptile += one_star * fluxes_n

        return ptile

    def _render_single_galaxies(self, galaxy_params, galaxy_bool):

        # flatten parameters
        z = galaxy_params.reshape(-1, self.n_galaxy_params)
        b = galaxy_bool.flatten()

        # allocate memory
        gal = torch.zeros(
            z.shape[0], self.n_bands, self.gal_slen, self.gal_slen, device=device
        )

        # forward only galaxies that are on!
        gal_on, _ = self.galaxy_decoder.forward(z[b == 1])
        gal[b == 1] = gal_on

        # reshape
        n_ptiles = galaxy_params.shape[0]
        gal_shape = (n_ptiles, -1, self.n_bands, gal.shape[-1], gal.shape[-1])
        single_galaxies = gal.reshape(gal_shape)

        return single_galaxies

    def _render_multiple_galaxies_on_ptile(self, locs, galaxy_params, galaxy_bool):
        # max_sources obtained from locs, allows for more flexibility when rendering.
        n_ptiles = locs.shape[0]
        max_sources = locs.shape[1]
        ptile_shape = (n_ptiles, self.n_bands, self.ptile_slen, self.ptile_slen)
        ptile = torch.zeros(ptile_shape, device=device)

        assert self.galaxy_decoder is not None
        assert galaxy_params.shape[0] == galaxy_bool.shape[0] == n_ptiles
        assert galaxy_params.shape[1] == galaxy_bool.shape[1] == max_sources
        assert galaxy_params.shape[2] == self.n_galaxy_params
        assert galaxy_bool.shape[2] == 1

        single_galaxies = self._render_single_galaxies(galaxy_params, galaxy_bool)
        for n in range(max_sources):
            galaxy_bool_n = galaxy_bool[:, n]
            locs_n = locs[:, n, :]
            _galaxy = single_galaxies[:, n, :, :, :]
            galaxy = _galaxy * galaxy_bool_n.reshape(-1, 1, 1, 1)
            one_galaxy = self._render_one_source(locs_n, galaxy)
            ptile += one_galaxy

        return ptile

    def _render_ptiles(self, n_sources, locs, galaxy_bool, galaxy_params, fluxes):
        # n_sources: is (batch_size x n_tiles_per_image)
        # locs: is (batch_size x n_tiles_per_image x max_sources x 2)
        # galaxy_bool: Is (batch_size x n_tiles_per_image x max_sources)
        # galaxy_params : is (batch_size x n_tiles_per_image x max_sources x latent_dim)
        # fluxes: Is (batch_size x n_tiles_per_image x max_sources x 2)

        # returns the ptiles in
        # shape = (batch_size x n_tiles_per_image x n_bands x ptile_slen x ptile_slen)

        assert (n_sources <= self.max_sources).all()
        batch_size = n_sources.shape[0]
        n_ptiles = batch_size * self.n_tiles_per_image

        # reshape parameters being explicit about shapes
        _n_sources = n_sources.reshape(n_ptiles)
        _locs = locs.reshape(n_ptiles, self.max_sources, 2)
        _galaxy_bool = galaxy_bool.reshape(n_ptiles, self.max_sources, 1)
        _galaxy_params = galaxy_params.reshape(
            n_ptiles, self.max_sources, self.n_galaxy_params
        )
        _fluxes = fluxes.reshape(n_ptiles, self.max_sources, self.n_bands)

        # draw stars and galaxies
        _is_on_array = get_is_on_from_n_sources(_n_sources, self.max_sources)
        _is_on_array = _is_on_array.reshape(n_ptiles, self.max_sources, 1)
        _star_bool = (1 - _galaxy_bool) * _is_on_array
        _star_bool = _star_bool.reshape(n_ptiles, self.max_sources, 1)

        # draw stars and galaxies
        stars = self._render_multiple_stars_on_ptile(_locs, _fluxes, _star_bool)
        galaxies = 0.0
        if self.galaxy_decoder is not None:
            galaxies = self._render_multiple_galaxies_on_ptile(
                _locs, _galaxy_params, _galaxy_bool
            )
        images = galaxies + stars

        return images.reshape(
            batch_size,
            self.n_tiles_per_image,
            self.n_bands,
            self.ptile_slen,
            self.ptile_slen,
        )

    def render_images(self, n_sources, locs, galaxy_bool, galaxy_params, fluxes):
        # constructs the full slen x slen image

        # n_sources: is (batch_size x n_tiles_per_image)
        # locs: is (batch_size x n_tiles_per_image x max_sources x 2)
        # galaxy_bool: Is (batch_size x n_tiles_per_image x max_sources x 1)
        # galaxy_params : is (batch_size x n_tiles_per_image x max_sources x latent_dim)
        # fluxes: Is (batch_size x n_tiles_per_image x max_sources x n_bands)

        # returns the **full** image in shape (batch_size x n_bands x slen x slen)

        assert n_sources.shape[0] == locs.shape[0]
        assert n_sources.shape[1] == locs.shape[1]
        assert galaxy_bool.shape[-1] == 1

        # first render the padded tiles
        image_ptiles = self._render_ptiles(
            n_sources, locs, galaxy_bool, galaxy_params, fluxes
        )

        # render the image from padded tiles
        images = self._construct_full_image_from_ptiles(image_ptiles, self.tile_slen)

        # add background and noise
        images += self.background.unsqueeze(0)
        if self.add_noise:
            images = self._apply_noise(images)

        return images

    @staticmethod
    def _construct_full_image_from_ptiles(image_ptiles, tile_slen):
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
        ptile_padding = (n_tiles1_in_ptile - 1) / 2
        assert (
            n_tiles1_in_ptile % 1 == 0
        ), "tile_slen and ptile_slen are not compatible. check tile_slen argument"
        assert (
            ptile_padding % 1 == 0
        ), "tile_slen and ptile_slen are not compatible. check tile_slen argument"
        n_tiles1_in_ptile = int(n_tiles1_in_ptile)
        ptile_padding = int(ptile_padding)

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
            device=device,
        )
        zero_pads2 = torch.zeros(
            batch_size,
            n_tiles1 + n_tiles_pad,
            n_tiles_pad,
            n_bands,
            ptile_slen,
            ptile_slen,
            device=device,
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
            device=device,
        )

        # loop through all tiles in a ptile
        for i in range(n_tiles1_in_ptile):
            for j in range(n_tiles1_in_ptile):
                indx_vec1 = torch.arange(
                    start=i, end=n_tiles, step=n_tiles1_in_ptile, device=device
                )
                indx_vec2 = torch.arange(
                    start=j, end=n_tiles, step=n_tiles1_in_ptile, device=device
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
        return canvas[
            :,
            :,
            (ptile_padding * tile_slen) : ((n_tiles1 + ptile_padding) * tile_slen),
            (ptile_padding * tile_slen) : ((n_tiles1 + ptile_padding) * tile_slen),
        ]
