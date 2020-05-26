import numpy as np
import warnings

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.distributions import Poisson, Categorical

from .. import device


def _draw_pareto(f_min, alpha, shape):
    uniform_samples = torch.rand(*shape, device=device)
    return f_min / (1.0 - uniform_samples) ** (1 / alpha)


def _draw_pareto_maxed(f_min, f_max, alpha, shape):
    # draw pareto conditioned on being less than f_max

    pareto_samples = _draw_pareto(f_min, alpha, shape)

    while torch.any(pareto_samples > f_max):
        indx = pareto_samples > f_max
        pareto_samples[indx] = _draw_pareto(f_min, alpha, [torch.sum(indx).item()])

    return pareto_samples


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
    mean_sources, min_sources, max_sources, batchsize=1, draw_poisson=True
):
    """
    Return tensor of size batchsize.
    :param mean_sources:
    :param min_sources:
    :param max_sources:
    :param batchsize:
    :param draw_poisson:
    :return: A tensor with shape = (batchsize)
    """
    if draw_poisson:
        m = Poisson(torch.full((1,), mean_sources, dtype=torch.float, device=device))
        n_sources = m.sample([batchsize])
    else:
        categorical_param = torch.full(
            (1,), max_sources - min_sources, dtype=torch.float, device=device
        )
        m = Categorical(categorical_param)
        n_sources = m.sample([batchsize]) + min_sources

    return n_sources.clamp(max=max_sources, min=min_sources).int().squeeze(1)


def _sample_locs(max_sources, is_on_array, batchsize=1):
    # 2 = (x,y)
    # torch.rand returns numbers between (0,1)
    locs = (
        torch.rand(batchsize, max_sources, 2, device=device)
        * is_on_array.unsqueeze(2).float()
    )
    return locs


def _plot_one_source(slen, locs, source, cached_grid=None):
    """
    :param slen:
    :param locs: is batchsize x len((x,y))
    :param source: is a (batchsize, n_bands, slen, slen) tensor, which could either be a `expanded_psf`
                    (psf repeated multiple times) for the case of of stars. Or multiple galaxies in the case of
                    galaxies.
    :param cached_grid:
    :return: shape = (batchsize x n_bands x slen x slen)
    """

    batchsize = locs.shape[0]
    assert locs.shape[1] == 2

    if cached_grid is None:
        grid = get_mgrid(slen)
    else:
        assert cached_grid.shape[0] == slen
        assert cached_grid.shape[1] == slen
        grid = cached_grid

    # scale locs so they take values between -1 and 1 for grid sample
    locs = (locs - 0.5) * 2
    grid_loc = grid.view(1, slen, slen, 2) - locs[:, [1, 0]].view(batchsize, 1, 1, 2)

    assert (
        source.shape[0] == batchsize
    ), "PSF should be expanded, check if shape is correct."
    source_plotted = F.grid_sample(source, grid_loc, align_corners=True)
    return source_plotted


def _check_sources_and_locs(locs, n_sources, batchsize):
    assert len(locs.shape) == 3, "Using batchsize as the first dimension."
    assert locs.shape[2] == 2
    assert len(n_sources) == batchsize
    assert len(n_sources.shape) == 1
    assert max(n_sources) <= locs.shape[1]


def _get_grid(slen, cached_grid=None):

    if cached_grid is None:
        grid = get_mgrid(slen)
    else:
        assert cached_grid.shape[0] == slen
        assert cached_grid.shape[1] == slen
        grid = cached_grid

    return grid


def load_background(background_file, n_bands, slen):
    background = np.load(background_file)
    background = torch.from_numpy(background).float()

    assert n_bands == background.shape[0]

    # TODO: easier way of doing this?
    # now convert background to size of scenes
    values = background.mean((1, 2))  # shape = (n_bands)
    background = torch.zeros(n_bands, slen, slen)
    for i, value in enumerate(values):
        background[i, ...] = value

    return background


def get_mgrid(slen):
    offset = (slen - 1) / 2
    x, y = np.mgrid[-offset : (offset + 1), -offset : (offset + 1)]
    mgrid = torch.tensor(np.dstack((y, x))) / offset
    return mgrid.type(torch.FloatTensor).to(device)


def plot_multiple_stars(slen, locs, n_sources, psf, fluxes, cached_grid=None):
    """

    Args:
        slen:
        locs: is (batchsize x max_num_stars x len(x_loc, y_loc))
        n_sources: has shape = (batchsize)
        psf: A psf/star with shape (n_bands x slen x slen) tensor
        fluxes: Is (batchsize x n_bands x max_stars)
        cached_grid: Grid where the stars should be plotted with shape (slen x slen)

    Returns:
    """

    batchsize = locs.shape[0]
    _check_sources_and_locs(locs, n_sources, batchsize)
    grid = _get_grid(slen, cached_grid)

    n_bands = psf.shape[0]
    scene = torch.zeros(batchsize, n_bands, slen, slen, device=device)

    assert len(psf.shape) == 3  # the shape is (n_bands, slen, slen)
    assert fluxes is not None
    assert fluxes.shape[0] == locs.shape[0]
    assert fluxes.shape[1] == locs.shape[1]
    assert fluxes.shape[2] == n_bands

    expanded_psf = psf.expand(
        batchsize, n_bands, -1, -1
    )  # all stars are just the PSF so we copy it.

    # this loop plots each of the ith star in each of the (batchsize) images.
    for n in range(max(n_sources)):
        is_on_n = (n < n_sources).float()
        locs_n = locs[:, n, :] * is_on_n.unsqueeze(1)
        fluxes_n = fluxes[:, n, :]  # shape = (batchsize x n_bands)
        one_star = _plot_one_source(slen, locs_n, expanded_psf, cached_grid=grid)
        scene += one_star * (is_on_n.unsqueeze(1) * fluxes_n).view(
            batchsize, n_bands, 1, 1
        )

    return scene


def plot_multiple_galaxies(slen, locs, n_sources, single_galaxies, cached_grid=None):
    batchsize = locs.shape[0]
    n_bands = single_galaxies.shape[2]

    assert single_galaxies.shape[0] == batchsize
    assert single_galaxies.shape[1] == locs.shape[1]  # max_galaxies

    _check_sources_and_locs(locs, n_sources, batchsize)
    grid = _get_grid(slen, cached_grid)

    scene = torch.zeros(batchsize, n_bands, slen, slen, device=device)
    for n in range(max(n_sources)):
        is_on_n = (n < n_sources).float()
        locs_n = locs[:, n, :] * is_on_n.unsqueeze(1)
        galaxy = single_galaxies[
            :, n, :, :, :
        ]  # shape = (batchsize x n_bands x slen x slen)

        # shape= (batchsize, n_bands, slen, slen)
        one_galaxy = _plot_one_source(slen, locs_n, galaxy, cached_grid=grid)
        scene += one_galaxy

    return scene


def get_is_on_from_n_sources(n_sources, max_sources):
    """
    Return a boolean array of shape=(batchsize, max_sources) whose (k,l)th entry indicates
    whether there are more than l sources on the kth batch.
    :param n_sources:
    :param max_sources:
    :return:
    """
    assert len(n_sources.shape) == 1

    batchsize = len(n_sources)
    is_on_array = torch.zeros(batchsize, max_sources, device=device, dtype=torch.long)

    for i in range(max_sources):
        is_on_array[:, i] = n_sources > i

    return is_on_array


class SourceSimulator(object):
    def __init__(
        self,
        galaxy_decoder,
        psf,
        background,
        slen=100,
        n_bands=1,
        max_sources=20,
        mean_sources=10,
        min_sources=0,
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

        self.transpose_psf = transpose_psf

        self.draw_poisson = draw_poisson
        self.add_noise = add_noise
        self.cached_grid = get_mgrid(self.slen)

        assert self.psf is not None
        assert len(self.psf.shape) == 3
        assert self.background.shape[0] == self.psf.shape[0] == self.n_bands
        assert self.background.shape[1] == self.slen
        assert self.background.shape[2] == self.slen

        self.psf = psf.to(device)

        # prior parameters
        self.f_min = f_min
        self.f_max = f_max
        self.alpha = alpha  # pareto parameter.
        self.use_pareto = use_pareto

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

        self.galaxy_decoder = galaxy_decoder  # full path
        self.galaxy_slen = self.galaxy_decoder.slen
        self.latent_dim = self.galaxy_decoder.latent_dim

        assert self.galaxy_decoder.n_bands == self.n_bands

    def _sample_n_sources_and_locs(self, batchsize):
        # sample number of sources
        n_sources = _sample_n_sources(
            self.mean_sources,
            self.min_sources,
            self.max_sources,
            batchsize,
            draw_poisson=self.draw_poisson,
        )

        # multiply by zero where they are no sources (recall parameters have entry for up
        # to max_sources)
        is_on_array = get_is_on_from_n_sources(n_sources, self.max_sources)

        # sample locations
        locs = _sample_locs(self.max_sources, is_on_array, batchsize=batchsize)

        return n_sources, locs, is_on_array

    @staticmethod
    def _get_log_fluxes(fluxes):
        log_fluxes = torch.where(
            fluxes > 0, fluxes, torch.ones(*fluxes.shape).to(device)
        )  # prevent log(0) errors.
        log_fluxes = torch.log(log_fluxes)

        return log_fluxes

    def _sample_fluxes(self, n_stars, is_on_array, batchsize=1):
        """

        :return: fluxes, a shape (batchsize x max_sources x n_bands) tensor
        """
        assert n_stars.shape[0] == batchsize

        if self.use_pareto:
            base_fluxes = _draw_pareto_maxed(
                self.f_min,
                self.f_max,
                alpha=self.alpha,
                shape=(batchsize, self.max_sources),
            )
        else:  # use uniform in range (f_min, f_max)
            uniform_base = torch.rand(batchsize, self.max_sources, device=device)
            base_fluxes = uniform_base * (self.f_max - self.f_min) + self.f_min

        if self.n_bands > 1:
            colors = (
                torch.rand(batchsize, self.max_sources, self.n_bands - 1, device=device)
                * 0.15
                + 0.3
            )
            _fluxes = 10 ** (colors / 2.5) * base_fluxes.unsqueeze(2)

            fluxes = (
                torch.cat((base_fluxes.unsqueeze(2), _fluxes), dim=2)
                * is_on_array.unsqueeze(2).float()
            )
        else:
            fluxes = (base_fluxes * is_on_array.float()).unsqueeze(2)

        return fluxes

    # TODO: make parallelization better?
    def _sample_gal_params_and_single_images(self, n_galaxies, batchsize):
        assert len(n_galaxies.shape) == 1
        assert n_galaxies.shape[0] == batchsize

        galaxy_params = torch.zeros(
            batchsize, self.max_sources, self.latent_dim, device=device
        )
        single_galaxies = torch.zeros(
            batchsize,
            self.max_sources,
            self.n_bands,
            self.galaxy_slen,
            self.galaxy_slen,
            device=device,
        )

        # z has shape = (num_samples, latent_dim)
        # galaxies has shape = (num_samples, n_bands, slen, slen)
        num_samples = int(n_galaxies.sum().item())
        z, galaxies = self.galaxy_decoder.get_batch(num_samples)

        count = 0
        for batch_i, n_gal in enumerate(n_galaxies):
            n_gal = int(n_gal)
            galaxy_params[batch_i, 0:n_gal, :] = z[count : count + n_gal, :]
            single_galaxies[batch_i, 0:n_gal, :, :, :] = galaxies[
                count : count + n_gal, :, :, :
            ]
            count += n_gal

        return galaxy_params, single_galaxies

    def sample_parameters(self, batchsize=1):
        n_sources, locs, is_on_array = self._sample_n_sources_and_locs(batchsize)
        n_stars = self._sample_n_stars(n_sources)
        assert torch.all(n_stars <= n_sources)
        n_galaxies = n_sources - n_stars

        fluxes = self._sample_fluxes(n_stars, is_on_array, batchsize)
        gal_params, single_galaxies = self._sample_gal_params_and_single_images(
            n_sources, batchsize
        )

        log_fluxes = self._get_log_fluxes(fluxes)
        return n_sources, n_stars, locs, fluxes, log_fluxes, gal_params, single_galaxies

    # ToDo: Change so that it uses galsim (Poisson Noise?)
    @staticmethod
    def _apply_noise(images_mean):
        # add noise to images.

        if torch.any(images_mean <= 0):
            warnings.warn("image mean less than 0")
            images_mean = images_mean.clamp(min=1.0)

        images = (
            torch.sqrt(images_mean) * torch.rand(*images_mean.shape, device=device)
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

    # TODO: What to do for non-aligned multi-band images in the case of galaxies.
    def _draw_image_from_params(
        self, n_stars, n_galaxies, locs, fluxes, single_galaxies
    ):
        stars = plot_multiple_stars(
            self.slen, locs, n_stars, self.psf, fluxes, cached_grid=self.cached_grid
        )
        galaxies = plot_multiple_galaxies(
            self.slen, locs, n_galaxies, single_galaxies, cached_grid=self.cached_grid
        )

        # shape = (n_images x n_bands x slen x slen)
        return stars + galaxies

    def generate_images(self, n_stars, n_galaxies, locs, fluxes, single_galaxies):
        images = self._draw_image_from_params(
            n_stars, n_galaxies, locs, fluxes, single_galaxies
        )
        return self._prepare_images(images)


class SourceDataset(Dataset):
    def __init__(self, n_images, simulator_args, simulator_kwargs):
        """
        :param n_images: same as batchsize.
        """
        self.n_images = n_images  # = batchsize.
        self.simulator_args = simulator_args
        self.simulator_kwargs = simulator_kwargs
        self.simulator = SourceSimulator(*simulator_args, *simulator_kwargs)
        self.slen = self.simulator.slen
        self.n_bands = self.simulator.n_bands

    def __len__(self):
        return self.n_images

    def get_batch(self, batchsize=32):
        (
            n_sources,
            n_stars,
            locs,
            fluxes,
            log_fluxes,
            gal_params,
            single_galaxies,
        ) = self.simulator.sample_parameters(batchsize=batchsize)

        n_galaxies = n_sources - n_stars
        images = self.simulator.generate_images(
            n_stars, n_galaxies, locs, fluxes, single_galaxies
        )

        return {
            "images": images,
            "background": self.simulator.background,
            "locs": locs,
            "fluxes": fluxes,
            "log_fluxes": log_fluxes,
            "gal_params": gal_params,
            "n_sources": n_sources,
            "n_stars": n_stars,
            "n_galaxies": n_galaxies,
        }
