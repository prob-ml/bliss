import numpy as np
import torch
import torch.nn.functional as F
from gmodel.data.galaxy_datasets import DecoderSamples

from ..utils import const

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _draw_pareto(f_min, alpha, shape):
    uniform_samples = torch.rand(shape).to(device)

    return f_min / (1 - uniform_samples) ** (1 / alpha)


def _draw_pareto_maxed(f_min, f_max, alpha, shape):
    # draw pareto conditioned on being less than f_max

    pareto_samples = _draw_pareto(f_min, alpha, shape)

    while torch.any(pareto_samples > f_max):
        indx = pareto_samples > f_max
        pareto_samples[indx] = \
            _draw_pareto(f_min, alpha, torch.sum(indx))

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

    assert psf_slen <= slen

    psf_expanded = torch.zeros((n_bands, slen, slen))

    offset = int((slen - psf_slen) / 2)

    psf_expanded[:, offset:(offset + psf_slen), offset:(offset + psf_slen)] = psf

    return psf_expanded


def _sample_n_sources(mean_sources, min_sources, max_sources, batchsize, draw_poisson=True):
    """
    Return tensor of size batchsize.
    :param mean_sources:
    :param min_sources:
    :param max_sources:
    :param batchsize:
    :param draw_poisson:
    :return:
    """
    if draw_poisson:
        n_sources = np.random.poisson(mean_sources, batchsize)
    else:
        n_sources = np.random.choice(np.arange(min_sources, max_sources + 1),
                                     batchsize)

    return torch.Tensor(n_sources).clamp(max=max_sources,
                                         min=min_sources).type(torch.LongTensor).to(device)


def _sample_locs(batchsize, max_sources, is_on_array):
    # 2 = (x,y)
    # torch.rand returns numbers between (0,1)
    locs = torch.rand((batchsize, max_sources, 2)).to(device) * is_on_array.unsqueeze(2).float()
    return locs


def _plot_one_source(slen, locs, source, cached_grid=None, is_star=True):
    """

    :param slen:
    :param locs: is batchsize x len((x,y))
    :param source: is a (slen x slen) tensor in the case of a star, and a
                   (batchsize, n_bands, slen, slen) tensor in the case of a galaxy.
    :param is_star:
    :param cached_grid:
    :return:
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

    if is_star:  # psf is the star!

        psf = source
        assert len(psf.shape) == 3
        n_bands = psf.shape[0]
        expanded_psf = psf.expand(batchsize, n_bands, -1, -1)  # all stars are just the PSF so we copy it.
        source_plotted = F.grid_sample(expanded_psf, grid_loc, align_corners=True)

    else:  # plotting a galaxy.
        galaxy = source
        source_plotted = F.grid_sample(galaxy, grid_loc, align_corners=True)

    return source_plotted


def _plot_multiple_sources(slen, locs, n_sources, sources, fluxes=None, cached_grid=None, is_star=True):
    """

    :param slen:
    :param locs: is (batchsize x max_num_sources x (x_loc, y_loc))
    :param n_sources: has shape: (batchsize)
    :param sources: either a psf(star) with shape (n_bands x slen x slen) tensor, or a galaxy in which case it has
                   shape (batchsize x max_galaxies x n_bands x slen x slen)
    :param fluxes: is (batchsize x n_bands x max_stars)
    :param cached_grid: Grid where the stars should be plotted with shape (slen x slen)
    :param is_star: Whether we are plotting a galaxy or a star (default: star)
    :return:
    """

    batchsize = locs.shape[0]
    assert locs.shape[2] == 2
    assert len(n_sources) == batchsize
    assert len(n_sources.shape) == 1

    assert max(n_sources) <= locs.shape[1]

    if cached_grid is None:
        grid = get_mgrid(slen)
    else:
        assert cached_grid.shape[0] == slen
        assert cached_grid.shape[1] == slen
        grid = cached_grid

    if is_star:
        stars = 0.
        psf = sources
        n_bands = psf.shape[0]

        assert fluxes is not None
        assert fluxes.shape[0] == locs.shape[0]
        assert fluxes.shape[1] == locs.shape[1]
        assert fluxes.shape[2] == n_bands

        # this loop plots each of the ith star in each of the (batchsize) images.
        for n in range(max(n_sources)):
            is_on_n = (n < n_sources).float()
            locs_n = locs[:, n, :] * is_on_n.unsqueeze(1)
            fluxes_n = fluxes[:, n, :]
            one_star = _plot_one_source(slen, locs_n, psf, cached_grid=grid, is_star=True)
            stars += one_star * (is_on_n.unsqueeze(1) * fluxes_n).view(batchsize, n_bands, 1, 1)

            return stars

    else:  # is a galaxy.
        assert fluxes is None
        assert sources.shape[0] == locs.shape[0]  # batchsize
        assert sources.shape[1] == locs.shape[1]  # max_galaxies

        galaxies = 0.
        for n in range(max(n_sources)):
            is_on_n = (n < n_sources).float()
            locs_n = locs[:, n, :] * is_on_n.unsqueeze(1)
            source = sources[:, n, :, :, :]  # shape = (batchsize x n_bands x slen x slen)
            one_galaxy = _plot_one_source(slen, locs_n, source, cached_grid=grid, is_star=False)
            galaxies += one_galaxy

        return galaxies


def get_mgrid(slen):
    offset = (slen - 1) / 2
    x, y = np.mgrid[-offset:(offset + 1), -offset:(offset + 1)]
    return (torch.Tensor(np.dstack((y, x))) / offset).to(device)


class SourceDataset:
    def __init__(self, n_images,
                 slen,
                 background,
                 n_bands,
                 max_sources,
                 mean_sources,
                 min_sources,
                 psf=None,
                 transpose_psf=False,
                 add_noise=True,
                 draw_poisson=True,
                 is_star=True):
        """

        :param psf:
        :param n_images: same as batchsize.
        :param slen: side length of the image.
        :param max_sources: Default value 1500
        :param mean_sources: Default value 1200
        :param min_sources: Default value 0
        :param background:
        :param transpose_psf:
        :param add_noise:
        :param draw_poisson:
        """

        # image parameters
        self.slen = slen
        self.n_bands = n_bands
        self.background = background
        self.is_star = is_star
        self.n_images = n_images  # = batchsize.

        self.simulator = StarSimulator(self.slen, self.background, self.n_bands,
                                       psf=psf, transpose_psf=transpose_psf, is_star=self.is_star,
                                       add_noise=add_noise, draw_poisson=draw_poisson)

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # ToDo: compress parameters one dimension.
        n_sources, locs, params = self.simulator.sample_parameters(1)

        if self.is_star:
            fluxes = params
            gal_params, single_galaxies = None, None
        else:
            fluxes = None
            gal_params, single_galaxies = params

        images = self.simulator.draw_image_from_params(locs, n_sources, sources=single_galaxies,
                                                       fluxes=fluxes)

        return {'image': images,
                'background': self.background,
                'locs': locs,
                'params': fluxes if self.is_star else gal_params,
                'n_stars': n_sources


class SourceSimulator:

    def __init__(self, slen, background, n_bands, max_sources, mean_sources, min_sources,
                 psf=None, transpose_psf=False, is_star=True, add_noise=True):

        self.slen = slen  # side length of the image
        self.n_bands = n_bands
        self.background = background

        assert len(background.shape) == 3
        assert background.shape[0] == self.n_bands
        assert background.shape[1] == background.shape[2] == self.slen
        self.background = background[None, :, :, :]

        self.cached_grid = get_mgrid(self.slen).to(device)
        self.is_star = is_star

        self.max_sources = max_sources
        self.mean_sources = mean_sources
        self.min_sources = min_sources

        self.psf = psf
        self.transpose_psf = transpose_psf

        self.add_noise = add_noise
        self.cached_grid = get_mgrid(self.slen).to(device)

    def draw_image_from_params(self, locs, n_sources, sources=None, fluxes=None):
        """

        :param locs:
        :param fluxes:
        :param sources: can either be `self.psf` or the galaxies to be plotted in locs.
        :param n_sources:
        :return: `images`, torch.Tensor of shape (n_images x n_bands x slen x slen)

        NOTE: The different sources in `images` are already aligned between bands.
        """

        if self.is_star:
            assert fluxes is not None
            sources = self.psf
        else:
            assert sources is not None

        images_mean = \
            _plot_multiple_sources(self.slen, locs, n_sources, sources,
                                   fluxes=fluxes, cached_grid=self.cached_grid, is_star=self.is_star) + \
            self.background[None, :, :, :].to(device)

        if self.add_noise:
            if torch.any(images_mean <= 0):
                print('warning: image mean less than 0')
                images_mean = images_mean.clamp(min=1.0)

            images = (torch.sqrt(images_mean) * torch.randn(images_mean.shape).to(device)
                      + images_mean)
        else:
            images = images_mean

        return images

    def get_source_params(self, *args):
        """
        Return either fluxes or the (galaxy latent representation, images)
        """
        return torch.empty()

    def sample_parameters(self, batchsize):
        # sample number of sources
        n_sources = _sample_n_sources(self.mean_sources, self.min_sources, self.max_sources, batchsize,
                                      draw_poisson=self.draw_poisson)

        # multiply by zero where they are no stars (recall parameters have entry for up to max_stars)
        is_on_array = const.get_is_on_from_n_sources(n_sources, self.max_sources)

        # sample locations
        locs = _sample_locs(batchsize, self.max_sources, is_on_array)

        params = self.get_source_params(n_sources, is_on_array)  # either fluxes or galaxy parameters.

        return n_sources, locs, params


# ToDo: Decide if galaxies should be aligned across bands or not? (locs would change shape)
class GalaxyDataset(SourceDataset):
    def __init__(self, gal_decoder_file, *args, **kwargs):
        """
        :param decoder_file: Decoder file where decoder network trained on individual galaxy images is.
        """
        self.gal_decoder_file = gal_decoder_file
        super().__init__(*args, **kwargs)


class GalaxySimulator(SourceSimulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ds = DecoderSamples(self.slen, self.gal_decoder_file)
        self.latent_dim = self.ds.latent_dim
        assert self.ds.num_bands == self.n_bands

    def get_source_params(self, n_galaxy):
        galaxy_params = torch.zeros(self.max_sources, self.latent_dim)
        single_galaxies = torch.zeros(self.max_sources, self.n_bands, self.slen, self.slen)

        z, galaxies = self.ds.sample(n_galaxy)
        galaxy_params[0:n_galaxy, :] = z
        single_galaxies[0:n_galaxy, : , :, :] = galaxies

        return galaxy_params, single_galaxies


class StarsDataset(SourceDataset):

    def __init__(self, f_min, f_max, alpha, *args, **kwargs):
        """
        :param max_sources: Default value 1500
        :param mean_sources: Default value 1200
        :param min_sources: Default value 0
        """
        # prior parameters
        self.f_min = f_min
        self.f_max = f_max
        self.alpha = alpha  # pareto parameter.

        super().__init__(*args, **kwargs)


    @classmethod
    def load_dataset_from_params(cls, psf, data_params,
                                 n_images,
                                 background,
                                 transpose_psf=False,
                                 add_noise=True):
        # data parameters
        slen = data_params['slen']

        f_min = data_params['f_min']
        f_max = data_params['f_max']
        alpha = data_params['alpha']

        max_stars = data_params['max_stars']
        mean_stars = data_params['mean_stars']
        min_stars = data_params['min_stars']

        # draw data
        return cls(psf,
                   n_images,
                   slen=slen,
                   f_min=f_min,
                   f_max=f_max,
                   max_stars=max_stars,
                   mean_stars=mean_stars,
                   min_stars=min_stars,
                   alpha=alpha,
                   background=background,
                   transpose_psf=transpose_psf,
                   add_noise=add_noise)


class StarSimulator(SourceSimulator):
    def __init__(self, f_min, f_max, alpha, *args, **kwargs):
        """
        :param f_min:
        :param f_max:
        :param alpha:
        """
        super().__init__(*args, **kwargs)
        assert self.psf is not None
        assert self.is_star
        assert len(self.psf.shape) == 3
        assert len(self.background.shape) == 3
        assert self.background.shape[0] == self.psf.shape[0]
        assert self.background.shape[1] == self.slen
        assert self.background.shape[2] == self.slen

        self.f_min = f_min
        self.f_max = f_max
        self.alpha = alpha

        self.psf_og = self.psf.clone()  # .detach() ?
        # get psf shape to match image shape
        # if slen is even, we still make psf dimension odd.
        # otherwise, the psf won't have a peak in the center pixel.
        _slen = self.slen + ((self.slen % 2) == 0) * 1
        if self.slen >= self.psf_og.shape[-1]:
            self.psf = _expand_psf(self.psf_og, _slen).to(device)
        else:
            self.psf = _trim_psf(self.psf_og, _slen).to(device)

        if self.transpose_psf:
            self.psf = self.psf.transpose(1, 2)

    def get_source_params(self, batchsize, is_on_array):
        # sample fluxes
        base_fluxes = _draw_pareto_maxed(self.f_min, self.f_max, alpha=self.alpha,
                                         shape=(batchsize, self.max_sources))

        if self.n_bands > 1:
            colors = torch.randn(batchsize, self.max_sources, self.n_bands - 1).to(device) * 0.15 + 0.3

            _fluxes = 10 ** (colors / 2.5) * base_fluxes.unsqueeze(2)

            fluxes = torch.cat((base_fluxes.unsqueeze(2), _fluxes), dim=2) * is_on_array.unsqueeze(2).float()
        else:
            fluxes = (base_fluxes * is_on_array.float()).unsqueeze(2)

        return fluxes
