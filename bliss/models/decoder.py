import numpy as np
import warnings

import torch
import torch.nn.functional as F
from torch.distributions import Poisson, Categorical
from torch.utils.data import IterableDataset

from .. import device
from .. import psf_transform
from . import galaxy_net
from .encoder import get_is_on_from_n_sources


def _pareto_cdf(x, f_min, alpha):
    return 1 - (f_min / x) ** alpha


def _draw_pareto_maxed(f_min, f_max, alpha, shape):
    # draw pareto conditioned on being less than f_max

    u_max = _pareto_cdf(f_max, f_min, alpha)
    uniform_samples = torch.rand(*shape, device=device) * u_max

    return f_min / (1.0 - uniform_samples) ** (1 / alpha)


def _check_psf(psf, slen):
    # first dimension of psf is number of bands
    assert len(psf.shape) == 3
    n_bands = psf.shape[0]

    # dimension of the psf should be odd
    psf_slen = psf.shape[2]
    assert psf.shape[1] == psf_slen
    assert (psf_slen % 2) == 1
    # same for slen
    assert (slen % 2) == 1


def _trim_psf(psf, slen):
    """
    Crop the psf to length slen x slen,
    centered at the middle
    :param psf:
    :param slen:
    :return:
    """

    _check_psf(psf, slen)
    psf_slen = psf.shape[2]
    psf_center = (psf_slen - 1) / 2

    assert psf_slen >= slen

    r = np.floor(slen / 2)
    l_indx = int(psf_center - r)
    u_indx = int(psf_center + r + 1)

    return psf[:, l_indx:u_indx, l_indx:u_indx]


def _expand_psf(psf, slen):
    """
    Pad the psf with zeros so that it is size slen,
    :param psf:
    :param slen:
    :return:
    """

    _check_psf(psf, slen)
    n_bands = psf.shape[0]
    psf_slen = psf.shape[2]

    assert psf_slen <= slen, "Should be using trim psf."

    psf_expanded = torch.zeros(n_bands, slen, slen, device=device)

    offset = int((slen - psf_slen) / 2)

    psf_expanded[:, offset : (offset + psf_slen), offset : (offset + psf_slen)] = psf

    return psf_expanded


def _sample_n_sources(
    mean_sources, min_sources, max_sources, batch_size=1, draw_poisson=True
):
    """
    Return tensor of size batch_size.
    :return: A tensor with shape = (batch_size)
    """
    if draw_poisson:
        m = Poisson(torch.full((1,), mean_sources, dtype=torch.float, device=device))
        n_sources = m.sample([batch_size])
    else:
        categorical_param = torch.full(
            (1,), max_sources - min_sources, dtype=torch.float, device=device
        )
        m = Categorical(categorical_param)
        n_sources = m.sample([batch_size]) + min_sources

    # long() here is necessary because used for indexing and one_hot encoding.
    return n_sources.clamp(max=max_sources, min=min_sources).long().squeeze(1)


def _sample_locs(
    max_sources, is_on_array, batch_size, locs_min=0.0, locs_max=1.0,
):
    assert 0 <= locs_min <= locs_max <= 1.0

    # 2 = (x,y)
    locs = torch.rand(batch_size, max_sources, 2, device=device)
    locs *= is_on_array.unsqueeze(2)
    locs = locs * locs_max + locs_min

    return locs


def _render_one_source(slen, locs, source, cached_grid=None):
    """
    :param slen:
    :param locs: is batch_size x len((x,y))
    :param source: is a (batch_size, n_bands, slen, slen) tensor, which could either be a
                    `expanded_psf` (psf repeated multiple times) for the case of of stars.
                    Or multiple galaxies in the case of galaxies.
    :param cached_grid:
    :return: shape = (batch_size x n_bands x slen x slen)
    """

    batch_size = locs.shape[0]
    assert locs.shape[1] == 2

    if cached_grid is None:
        grid = get_mgrid(slen)
    else:
        assert cached_grid.shape[0] == slen
        assert cached_grid.shape[1] == slen
        grid = cached_grid

    # scale locs so they take values between -1 and 1 for grid sample
    locs = (locs - 0.5) * 2
    grid_loc = grid.view(1, slen, slen, 2) - locs[:, [1, 0]].view(batch_size, 1, 1, 2)

    assert (
        source.shape[0] == batch_size
    ), "PSF should be expanded, check if shape is correct."
    source_rendered = F.grid_sample(source, grid_loc, align_corners=True)
    return source_rendered


def _get_grid(slen, cached_grid=None):
    if cached_grid is None:
        grid = get_mgrid(slen)
    else:
        assert cached_grid.shape[0] == slen
        assert cached_grid.shape[1] == slen
        grid = cached_grid

    return grid


def _check_sources_and_locs(locs, n_sources, batch_size):
    assert len(locs.shape) == 3, "Using batch_size as the first dimension."
    assert locs.shape[2] == 2
    assert len(n_sources) == batch_size
    assert len(n_sources.shape) == 1


def get_mgrid(slen):
    offset = (slen - 1) / 2
    x, y = np.mgrid[-offset : (offset + 1), -offset : (offset + 1)]
    mgrid = torch.tensor(np.dstack((y, x))) / offset
    return mgrid.type(torch.FloatTensor).to(device)


def get_galaxy_decoder(decoder_state_file, slen=51, n_bands=1, latent_dim=8):
    dec = galaxy_net.CenteredGalaxyDecoder(slen, latent_dim, n_bands).to(device)
    dec.load_state_dict(torch.load(decoder_state_file, map_location=device))
    dec.eval()
    return dec


def get_background(background_file, n_bands, slen):
    # for numpy background that are not necessarily of the correct size.
    background = np.load(background_file)
    background = torch.from_numpy(background).float()

    assert n_bands == background.shape[0]

    # TODO: way to vectorize this?
    # now convert background to size of scenes
    values = background.mean((1, 2))  # shape = (n_bands)
    background = torch.zeros(n_bands, slen, slen)
    for i, value in enumerate(values):
        background[i, ...] = value

    return background


def get_fitted_powerlaw_psf(psf_file):
    psf_params = torch.from_numpy(np.load(psf_file)).to(device)
    power_law_psf = psf_transform.PowerLawPSF(psf_params)
    psf = power_law_psf.forward().detach()
    assert psf.size(0) == 2 and psf.size(1) == psf.size(2) == 101
    return psf


def render_multiple_stars(slen, locs, n_sources, psf, fluxes, cached_grid=None):
    """

    Args:
        slen:
        locs: is (batch_size x max_num_stars x len(x_loc, y_loc))
        n_sources: has shape = (batch_size)
        psf: A psf/star with shape (n_bands x slen x slen) tensor
        fluxes: Is (batch_size x n_bands x max_stars)
        cached_grid: Grid where the stars should be rendered with shape (slen x slen)

    Returns:
    """

    batch_size = locs.shape[0]
    _check_sources_and_locs(locs, n_sources, batch_size)
    grid = _get_grid(slen, cached_grid)

    n_bands = psf.shape[0]
    scene = torch.zeros(batch_size, n_bands, slen, slen, device=device)

    assert len(psf.shape) == 3  # the shape is (n_bands, slen, slen)
    assert fluxes is not None
    assert fluxes.shape[0] == locs.shape[0]
    assert fluxes.shape[1] == locs.shape[1]
    assert fluxes.shape[2] == n_bands

    expanded_psf = psf.expand(
        batch_size, n_bands, -1, -1
    )  # all stars are just the PSF so we copy it.

    # this loop plots each of the ith star in each of the (batch_size) images.
    max_n = locs.shape[1]
    for n in range(max_n):
        is_on_n = (n < n_sources).float()
        locs_n = locs[:, n, :] * is_on_n.unsqueeze(1)
        fluxes_n = fluxes[:, n, :]  # shape = (batch_size x n_bands)
        one_star = _render_one_source(slen, locs_n, expanded_psf, cached_grid=grid)
        scene += one_star * (is_on_n.unsqueeze(1) * fluxes_n).view(
            batch_size, n_bands, 1, 1
        )

    return scene


def render_multiple_galaxies(slen, locs, n_sources, single_galaxies, cached_grid=None):
    batch_size = locs.shape[0]
    n_bands = single_galaxies.shape[2]

    assert single_galaxies.shape[0] == batch_size
    assert single_galaxies.shape[1] == locs.shape[1]  # max_galaxies

    _check_sources_and_locs(locs, n_sources, batch_size)
    grid = _get_grid(slen, cached_grid)

    scene = torch.zeros(batch_size, n_bands, slen, slen, device=device)
    max_n = locs.shape[1]
    for n in range(max_n):
        is_on_n = (n < n_sources).float()
        locs_n = locs[:, n, :] * is_on_n.unsqueeze(1)
        galaxy = single_galaxies[
            :, n, :, :, :
        ]  # shape = (batch_size x n_bands x slen x slen)

        # shape= (batch_size, n_bands, slen, slen)
        one_galaxy = _render_one_source(slen, locs_n, galaxy, cached_grid=grid)
        scene += one_galaxy

    return scene


class SourceSimulator(object):
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
        use_pareto=True,
        transpose_psf=False,
        add_noise=True,
        draw_poisson=True,
    ):

        self.slen = slen
        self.n_bands = n_bands
        self.background = background.to(device)

        assert len(background.shape) == 3
        assert background.shape[0] == self.n_bands
        assert background.shape[1] == background.shape[2] == self.slen

        self.max_sources = max_sources
        self.mean_sources = mean_sources
        self.min_sources = min_sources
        self.prob_galaxy = float(prob_galaxy)
        self.all_stars = self.prob_galaxy == 0.0

        self.draw_poisson = draw_poisson
        self.add_noise = add_noise
        self.cached_grid = get_mgrid(self.slen)

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
        self.use_pareto = use_pareto

        self.loc_min = loc_min
        self.loc_max = loc_max
        assert 0.0 <= self.loc_min <= self.loc_max <= 1.0

        # psf
        self.transpose_psf = transpose_psf
        self.psf = psf.to(device)
        self.psf_og = self.psf.clone()
        # get psf shape to match image shape
        # if slen is even, we still make psf dimension odd.
        # otherwise, the psf won't have a peak in the center pixel.
        _slen = self.slen + ((self.slen % 2) == 0) * 1
        if self.slen >= self.psf_og.shape[-1]:
            self.psf = _expand_psf(self.psf_og, _slen)
        else:
            self.psf = _trim_psf(self.psf_og, _slen)

        if self.transpose_psf:
            self.psf = self.psf.transpose(1, 2)

        assert len(self.psf.shape) == 3
        assert self.background.shape[0] == self.psf.shape[0] == self.n_bands
        assert self.background.shape[1] == self.slen
        assert self.background.shape[2] == self.slen

    def _sample_n_sources(self, batch_size):
        # sample number of sources
        n_sources = _sample_n_sources(
            self.mean_sources,
            self.min_sources,
            self.max_sources,
            batch_size,
            draw_poisson=self.draw_poisson,
        )

        # multiply by zero where they are no sources (all 1s up front each row)
        is_on_array = get_is_on_from_n_sources(n_sources, self.max_sources)

        return n_sources, is_on_array

    def _sample_n_galaxies_and_stars(self, n_sources, is_on_array):
        batch_size = n_sources.size(0)

        # n_galaxies shouldn't exceed n_sources.
        uniform = torch.rand(batch_size, self.max_sources, device=device)
        galaxy_bool = uniform < self.prob_galaxy
        galaxy_bool = (galaxy_bool * is_on_array).float()
        star_bool = (1 - galaxy_bool) * is_on_array
        n_galaxies = galaxy_bool.sum(1)
        n_stars = star_bool.sum(1)

        return n_galaxies, n_stars, galaxy_bool, star_bool

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

        if self.use_pareto:
            base_fluxes = _draw_pareto_maxed(
                self.f_min,
                self.f_max,
                alpha=self.alpha,
                shape=(batch_size, self.max_sources),
            )
        else:  # use uniform in range (f_min, f_max)
            uniform_base = torch.rand(batch_size, self.max_sources, device=device)
            base_fluxes = uniform_base * (self.f_max - self.f_min) + self.f_min

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
        n_sources, is_on_array = self._sample_n_sources(batch_size)

        locs = _sample_locs(
            self.max_sources,
            is_on_array,
            batch_size,
            locs_min=self.loc_min,
            locs_max=self.loc_max,
        )

        n_galaxies, n_stars, galaxy_bool, star_bool = self._sample_n_galaxies_and_stars(
            n_sources, is_on_array
        )
        assert torch.all(n_stars <= n_sources) and torch.all(n_galaxies <= n_sources)

        if self.all_stars:
            galaxy_params = torch.zeros(
                batch_size, self.max_sources, self.latent_dim, device=device
            )
            single_galaxies = None
        else:
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

    def _prepare_images(self, images):
        """Apply background and noise if requested.
        """

        images = images + self.background.unsqueeze(0)

        if self.add_noise:
            images = self._apply_noise(images)

        return images

    def _draw_image_from_params(
        self, n_sources, galaxy_locs, star_locs, single_galaxies, fluxes
    ):

        if self.all_stars:
            galaxies = 0.0
        else:
            # need n_sources because *_locs are not necessarily ordered.
            galaxies = render_multiple_galaxies(
                self.slen,
                galaxy_locs,
                n_sources,
                single_galaxies,
                cached_grid=self.cached_grid,
            )
        stars = render_multiple_stars(
            self.slen,
            star_locs,
            n_sources,
            self.psf,
            fluxes,
            cached_grid=self.cached_grid,
        )

        # shape = (n_images x n_bands x slen x slen)
        return galaxies + stars

    def generate_images(
        self, n_sources, galaxy_locs, star_locs, single_galaxies, fluxes
    ):
        images = self._draw_image_from_params(
            n_sources, galaxy_locs, star_locs, single_galaxies, fluxes
        )
        return self._prepare_images(images)


class SimulatedDataset(IterableDataset):
    def __init__(
        self, n_batches: int, batch_size: int, simulator_args, simulator_kwargs
    ):
        super(SimulatedDataset, self).__init__()

        self.n_batches = n_batches
        self.batch_size = batch_size

        self.simulator = SourceSimulator(*simulator_args, **simulator_kwargs)
        self.slen = self.simulator.slen
        self.n_bands = self.simulator.n_bands

    def __iter__(self):
        return self.batch_generator()

    def batch_generator(self):
        for i in range(self.n_batches):
            yield self.get_batch()

    def get_batch(self):
        (
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
        ) = self.simulator.sample_parameters(batch_size=self.batch_size)

        galaxy_locs = locs * galaxy_bool.unsqueeze(2)
        star_locs = locs * star_bool.unsqueeze(2)
        images = self.simulator.generate_images(
            n_sources, galaxy_locs, star_locs, single_galaxies, fluxes
        )

        return {
            "n_sources": n_sources,
            "n_galaxies": n_galaxies,
            "n_stars": n_stars,
            "locs": locs,
            "galaxy_params": galaxy_params,
            "log_fluxes": log_fluxes,
            "galaxy_bool": galaxy_bool,
            "images": images,
            "background": self.simulator.background,
        }

    # TODO: finish adding these methods.
    # @classmethod
    # def from_args(cls, simulator_args, args):
    #     args_dict = vars(args)
    #     parameters = inspect.signature(SourceSimulator).parameters
    #     simulator_kwargs = {
    #         param: value for param, value in args_dict.items() if param in parameters
    #     }
    #     return cls(*simulator_args, **simulator_kwargs)

    # def setup_dataset(args, paths):
    #     decoder_file = paths["data"].joinpath(args.galaxy_decoder_file)
    #     background_file = paths["data"].joinpath(args.background_file)
    #     psf_file = paths["data"].joinpath(args.psf_file)
    #
    #     if args.verbose:
    #         print(
    #             f"files to be used:\n decoder: {decoder_file}\n background_file: {background_file}\n"
    #             f"psf_file: {psf_file}"
    #         )
    #
    #     galaxy_decoder = decoder.get_galaxy_decoder(decoder_file, n_bands=args.n_bands)
    #     psf = decoder.get_fitted_powerlaw_psf(psf_file)[None, 0]
    #     background = decoder.get_background(background_file, args.n_bands, args.slen)
    #
    #     simulator_args = (
    #         galaxy_decoder,
    #         psf,
    #         background,
    #     )
    #
    #     simulator_kwargs = dict(
    #         max_sources=args.max_sources, mean_sources=args.mean_sources, min_sources=0,
    #     )
    #
    #     dataset = decoder.SimulatedDataset(args.n_images, simulator_args, simulator_kwargs)
    #
    #     assert args.n_bands == 1, "Only 1 band is supported at the moment."
    #     assert (
    #         dataset.simulator.n_bands
    #         == psf.shape[0]
    #         == background.shape[0]
    #         == galaxy_decoder.n_bands
    #     ), "All bands should be consistent"
    #
    #     return dataset
