import numpy as np
import warnings

import torch
import torch.nn.functional as F
from torch.distributions import Poisson

from .. import device
from .encoder import get_is_on_from_n_sources


def get_mgrid(slen):
    offset = (slen - 1) / 2
    x, y = np.mgrid[-offset : (offset + 1), -offset : (offset + 1)]
    mgrid = torch.tensor(np.dstack((y, x))) / offset
    return mgrid.type(torch.FloatTensor).to(device)


class ImageDecoder(object):
    def __init__(
        self,
        galaxy_decoder,
        psf,
        background,
        n_bands=1,
        slen=100,
        prob_galaxy=0.5,
        max_sources=20,
        mean_sources=10,
        min_sources=0,
        loc_min=0.0,
        loc_max=1.0,
        f_min=1e3,
        f_max=1e6,
        alpha=0.5,
        add_noise=True,
    ):

        self.slen = slen
        self.n_bands = n_bands
        self.background = background.to(device)

        assert len(background.shape) == 3
        assert self.background.shape[0] == self.n_bands
        assert self.background.shape[1] == self.background.shape[2] == self.slen

        self.max_sources = max_sources
        self.mean_sources = mean_sources
        self.min_sources = min_sources
        self.prob_galaxy = float(prob_galaxy)
        self.all_stars = self.prob_galaxy == 0.0

        self.add_noise = add_noise

        if self.all_stars:
            self.galaxy_decoder = None
            self.latent_dim = 1
        else:
            self.galaxy_decoder = galaxy_decoder
            self.galaxy_slen = self.galaxy_decoder.slen
            self.latent_dim = self.galaxy_decoder.latent_dim
            assert self.galaxy_decoder.n_bands == self.n_bands

        # prior parameters
        self.f_min = f_min
        self.f_max = f_max
        self.alpha = alpha  # pareto parameter.

        self.loc_min = loc_min
        self.loc_max = loc_max
        assert 0.0 <= self.loc_min <= self.loc_max <= 1.0

        self.cached_grid = get_mgrid(self.slen)
        self.psf = psf

    def _trim_psf(self):
        """Crop the psf to length slen x slen,
        centered at the middle.
        """

        # if self.slen is even, we still make psf dimension odd.
        # otherwise, the psf won't have a peak in the center pixel.
        _slen = self.slen + ((self.slen % 2) == 0) * 1

        psf_slen = self._psf.shape[2]
        psf_center = (psf_slen - 1) / 2

        assert psf_slen >= _slen

        r = np.floor(_slen / 2)
        l_indx = int(psf_center - r)
        u_indx = int(psf_center + r + 1)

        self._psf = self._psf[:, l_indx:u_indx, l_indx:u_indx]

    def _expand_psf(self):
        """Pad the psf with zeros so that it is size slen,
        """

        _slen = self.slen + ((self.slen % 2) == 0) * 1
        n_bands = self._psf.shape[0]
        psf_slen = self._psf.shape[2]

        assert psf_slen <= _slen, "Should be using trim psf."

        psf_expanded = torch.zeros(n_bands, _slen, _slen, device=device)
        offset = int((_slen - psf_slen) / 2)

        psf_expanded[
            :, offset : (offset + psf_slen), offset : (offset + psf_slen)
        ] = self._psf

        self._psf = psf_expanded

    @property
    def psf(self):
        return self._psf

    @psf.setter
    def psf(self, psf):
        # first dimension of psf is number of bands
        # dimension of the psf/slen should be odd
        psf_slen = psf.shape[2]
        assert len(psf.shape) == 3
        assert psf.shape[1] == psf_slen
        assert (psf_slen % 2) == 1
        assert self.background.shape[0] == psf.shape[0] == self.n_bands

        self._psf = psf
        if self.slen >= psf.shape[-1]:
            self._expand_psf()
        else:
            self._trim_psf()

    def _sample_n_sources(self, batch_size):
        # always poisson distributed.

        m = Poisson(
            torch.full((1,), self.mean_sources, dtype=torch.float, device=device)
        )
        n_sources = m.sample([batch_size])

        # long() here is necessary because used for indexing and one_hot encoding.
        n_sources = n_sources.clamp(max=self.max_sources, min=self.min_sources)
        n_sources = n_sources.long().squeeze(1)
        return n_sources

    def _sample_locs(self, is_on_array, batch_size):

        # 2 = (x,y)
        locs = torch.rand(batch_size, self.max_sources, 2, device=device)
        locs *= is_on_array.unsqueeze(2)
        locs = locs * self.loc_max + self.loc_min

        return locs

    def _sample_n_galaxies_and_stars(self, n_sources, is_on_array):
        batch_size = n_sources.size(0)
        uniform = torch.rand(batch_size, self.max_sources, device=device)
        galaxy_bool = uniform < self.prob_galaxy
        galaxy_bool = (galaxy_bool * is_on_array).float()
        star_bool = (1 - galaxy_bool) * is_on_array
        n_galaxies = galaxy_bool.sum(1)
        n_stars = star_bool.sum(1)

        return n_galaxies, n_stars, galaxy_bool, star_bool

    def _pareto_cdf(self, x):
        return 1 - (self.f_min / x) ** self.alpha

    def _draw_pareto_maxed(self, shape):
        # draw pareto conditioned on being less than f_max

        u_max = self._pareto_cdf(self.f_max)
        uniform_samples = torch.rand(*shape, device=device) * u_max
        return self.f_min / (1.0 - uniform_samples) ** (1 / self.alpha)

    @staticmethod
    def _get_log_fluxes(fluxes):
        """
        To obtain fluxes from log_fluxes.

        >> is_on_array = get_is_on_from_n_stars(n_stars, max_sources)
        >> fluxes = np.exp(log_fluxes) * is_on_array
        """
        log_fluxes = torch.where(
            fluxes > 0, fluxes, torch.ones(*fluxes.shape).to(device)
        )  # prevent log(0) errors.
        log_fluxes = torch.log(log_fluxes)

        return log_fluxes

    def _sample_fluxes(self, n_stars, star_bool, batch_size):
        """

        :return: fluxes, a shape (batch_size x max_sources x n_bands) tensor
        """
        assert n_stars.shape[0] == batch_size

        shape = (batch_size, self.max_sources)
        base_fluxes = self._draw_pareto_maxed(shape)

        if self.n_bands > 1:
            colors = (
                torch.rand(
                    batch_size, self.max_sources, self.n_bands - 1, device=device
                )
                * 0.15
                + 0.3
            )
            _fluxes = 10 ** (colors / 2.5) * base_fluxes.unsqueeze(2)

            fluxes = torch.cat((base_fluxes.unsqueeze(2), _fluxes), dim=2)
            fluxes *= star_bool.unsqueeze(2)
        else:
            fluxes = (base_fluxes * star_bool.float()).unsqueeze(2)

        return fluxes

    def _sample_galaxy_params_and_single_images(self, n_galaxies, galaxy_bool):
        assert len(n_galaxies.shape) == 1

        batch_size = n_galaxies.size(0)
        n_samples = batch_size * self.max_sources

        # z has shape = (n_samples, latent_dim)
        # galaxies has shape = (n_samples, n_bands, slen, slen)
        with torch.no_grad():
            z, galaxies = self.galaxy_decoder.get_sample(n_samples)

        galaxy_params = z.reshape(batch_size, -1, self.latent_dim)
        single_galaxies = galaxies.reshape(
            batch_size, -1, self.n_bands, self.galaxy_slen, self.galaxy_slen
        )

        # zero out excess according to n_galaxies.
        galaxy_params *= galaxy_bool.unsqueeze(2)
        single_galaxies *= galaxy_bool.unsqueeze(2).unsqueeze(3).unsqueeze(4)

        return galaxy_params, single_galaxies

    def sample_parameters(self, batch_size=1):
        n_sources = self._sample_n_sources(batch_size)
        is_on_array = get_is_on_from_n_sources(n_sources, self.max_sources)
        locs = self._sample_locs(is_on_array, batch_size)

        n_galaxies, n_stars, galaxy_bool, star_bool = self._sample_n_galaxies_and_stars(
            n_sources, is_on_array
        )
        assert torch.all(n_stars <= n_sources) and torch.all(n_galaxies <= n_sources)

        galaxy_params = torch.zeros(
            batch_size, self.max_sources, self.latent_dim, device=device
        )
        single_galaxies = None
        if not self.all_stars:
            (
                galaxy_params,
                single_galaxies,
            ) = self._sample_galaxy_params_and_single_images(n_sources, galaxy_bool)

        fluxes = self._sample_fluxes(n_sources, star_bool, batch_size)
        log_fluxes = self._get_log_fluxes(fluxes)

        return (
            n_sources,
            n_galaxies,
            n_stars,
            locs,
            galaxy_params,
            single_galaxies,
            fluxes,
            log_fluxes,
            galaxy_bool,
            star_bool,
        )

    @staticmethod
    def _apply_noise(images_mean):
        # add noise to images.

        if torch.any(images_mean <= 0):
            warnings.warn("image mean less than 0")
            images_mean = images_mean.clamp(min=1.0)

        images = (
            torch.sqrt(images_mean) * torch.randn(*images_mean.shape, device=device)
            + images_mean
        )

        return images

    def _render_one_source(self, locs, source):
        """
        :param locs: is batch_size x len((x,y))
        :param source: is a (batch_size, n_bands, slen, slen) tensor, which could either be a
                        `expanded_psf` (psf repeated multiple times) for the case of of stars.
                        Or multiple galaxies in the case of galaxies.
        :return: shape = (batch_size x n_bands x slen x slen)
        """

        batch_size = locs.shape[0]
        assert locs.shape[1] == 2
        assert source.shape[0] == batch_size

        # scale locs so they take values between -1 and 1 for grid sample
        locs = (locs - 0.5) * 2

        _grid = self.cached_grid.view(1, self.slen, self.slen, 2)
        grid_loc = _grid - locs[:, [1, 0]].view(batch_size, 1, 1, 2)
        source_rendered = F.grid_sample(source, grid_loc, align_corners=True)
        return source_rendered

    def render_multiple_stars(self, n_sources, locs, fluxes):
        """

        Args:
            n_sources: has shape = (batch_size)
            locs: is (batch_size x max_num_stars x len(x_loc, y_loc))
            fluxes: Is (batch_size x n_bands x max_stars)

        Returns:
        """

        batch_size = locs.shape[0]
        n_bands = self.psf.shape[0]
        scene = torch.zeros(batch_size, n_bands, self.slen, self.slen, device=device)

        assert len(self.psf.shape) == 3  # the shape is (n_bands, slen, slen)
        assert fluxes.shape[0] == locs.shape[0]
        assert fluxes.shape[1] == locs.shape[1]
        assert fluxes.shape[2] == n_bands

        # all stars are just the PSF so we copy it.
        expanded_psf = self.psf.expand(batch_size, n_bands, -1, -1)

        # this loop plots each of the ith star in each of the (batch_size) images.
        max_n = locs.shape[1]
        for n in range(max_n):
            is_on_n = (n < n_sources).float()
            locs_n = locs[:, n, :] * is_on_n.unsqueeze(1)
            fluxes_n = fluxes[:, n, :]  # shape = (batch_size x n_bands)
            one_star = self._render_one_source(locs_n, expanded_psf)
            scene += one_star * (is_on_n.unsqueeze(1) * fluxes_n).view(
                batch_size, n_bands, 1, 1
            )

        return scene

    def render_multiple_galaxies(self, n_sources, locs, single_galaxies):
        batch_size = locs.shape[0]
        n_bands = single_galaxies.shape[2]

        assert single_galaxies.shape[0] == batch_size
        assert single_galaxies.shape[1] == locs.shape[1]  # max_galaxies

        scene = torch.zeros(batch_size, n_bands, self.slen, self.slen, device=device)
        max_n = locs.shape[1]
        for n in range(max_n):
            is_on_n = (n < n_sources).float()
            locs_n = locs[:, n, :] * is_on_n.unsqueeze(1)

            # shape = (batch_size x n_bands x slen x slen)
            galaxy = single_galaxies[:, n, :, :, :]
            one_galaxy = self._render_one_source(locs_n, galaxy)
            scene += one_galaxy

        return scene

    def generate_images(
        self, n_sources, galaxy_locs, star_locs, single_galaxies, fluxes
    ):

        galaxies = 0.0
        if not self.all_stars:
            # need n_sources because *_locs are not necessarily ordered.
            galaxies = self.render_multiple_galaxies(
                galaxy_locs, n_sources, single_galaxies,
            )
        stars = self.render_multiple_stars(star_locs, n_sources, fluxes,)

        # shape = (n_images x n_bands x slen x slen)
        images = galaxies + stars

        # add background and noise
        images = images + self.background.unsqueeze(0)
        if self.add_noise:
            images = self._apply_noise(images)

        return images
