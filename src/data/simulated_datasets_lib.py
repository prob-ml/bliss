import numpy as np
import torch
import torch.nn.functional as F
from gmodel.data.galaxy_datasets import DecoderSamples

from ..utils import const
from ..utils.const import device


# ToDo: Get rid of `device`  on top of files,
# ToDo: push decoder to cuda so that images returned are produced in cuda and already in cuda.


def _draw_pareto(f_min, alpha, shape, cuda=torch.device("cpu")):
    uniform_samples = torch.rand(shape, device=cuda)
    return f_min / (1 - uniform_samples) ** (1 / alpha)


def _draw_pareto_maxed(f_min, f_max, alpha, shape, cuda=torch.device("cpu")):
    # draw pareto conditioned on being less than f_max

    pareto_samples = _draw_pareto(f_min, alpha, shape, cuda)

    while torch.any(pareto_samples > f_max):
        indx = pareto_samples > f_max
        pareto_samples[indx] = \
            _draw_pareto(f_min, alpha, torch.sum(indx), cuda)

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


def _sample_n_sources(mean_sources, min_sources, max_sources, batchsize=1, draw_poisson=True, cuda=torch.device("cpu")):
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

    return torch.Tensor(n_sources, device=cuda).clamp(max=max_sources,
                                                      min=min_sources).type(torch.LongTensor)


def _sample_locs(max_sources, is_on_array, batchsize=1, cuda=torch.device("cpu")):
    # 2 = (x,y)
    # torch.rand returns numbers between (0,1)
    locs = torch.rand((batchsize, max_sources, 2), device=cuda) * is_on_array.unsqueeze(2).float()
    return locs


def _plot_one_source(slen, locs, source, cached_grid=None, is_star=True):
    """
    :param slen:
    :param locs: is batchsize x len((x,y))
    :param source: is a (n_bands x slen x slen) tensor in the case of a star, and a
                   (batchsize, n_bands, slen, slen) tensor in the case of a galaxy.
    :param is_star:
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

    if is_star:  # psf is the star!

        psf = source
        assert len(psf.shape) == 3
        n_bands = psf.shape[0]
        expanded_psf = psf.expand(batchsize, n_bands, -1, -1)  # all stars are just the PSF so we copy it.
        source_plotted = F.grid_sample(expanded_psf, grid_loc, align_corners=True)

    else:  # plotting a galaxy.
        assert source.shape[0] == batchsize
        galaxy = source
        source_plotted = F.grid_sample(galaxy, grid_loc, align_corners=True)

    return source_plotted


def _plot_multiple_sources(slen, locs, n_sources, sources, fluxes=None, cached_grid=None, is_star=True,
                           cuda=torch.device("cpu")):
    """

    :param slen:
    :param locs: is (batchsize x max_num_sources x len(x_loc, y_loc))
    :param n_sources: has shape: (batchsize)
    :param sources: either a psf(star) with shape (n_bands x slen x slen) tensor, or a galaxy in which case it has
                   shape (batchsize x max_galaxies x n_bands x slen x slen)
    :param fluxes: is (batchsize x n_bands x max_stars)
    :param cached_grid: Grid where the stars should be plotted with shape (slen x slen)
    :param is_star: Whether we are plotting a galaxy or a star (default: star)
    :return:
    """

    batchsize = locs.shape[0]
    max_galaxies = locs.shape[1]

    assert len(locs.shape) == 3, "Using batchsize as the first dimension."
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
            fluxes_n = fluxes[:, n, :]  # shape = (batchsize x n_bands)
            one_star = _plot_one_source(slen, locs_n, psf, cached_grid=grid, is_star=True)
            stars += one_star * (is_on_n.unsqueeze(1) * fluxes_n).view(batchsize, n_bands, 1, 1)

        return stars

    else:  # is a galaxy.
        assert fluxes is None
        assert sources.shape[0] == locs.shape[0]  # batchsize
        assert sources.shape[1] == locs.shape[1]  # max_galaxies

        n_bands = sources.shape[2]

        galaxies = torch.zeros(batchsize, n_bands, slen, slen, device=cuda)
        for n in range(max(n_sources)):
            is_on_n = (n < n_sources).float()
            locs_n = locs[:, n, :] * is_on_n.unsqueeze(1)
            source = sources[:, n, :, :, :]  # shape = (batchsize x n_bands x slen x slen)

            # shape=(batchsize, n_bands, slen, slen)
            one_galaxy = _plot_one_source(slen, locs_n, source, cached_grid=grid, is_star=False)
            # galaxies += one_galaxy * is_on_n.unsqueeze(1).view(batchsize, n_bands, 1, 1)
            galaxies += one_galaxy
        return galaxies


def get_mgrid(slen, cuda=torch.device("cpu")):
    offset = (slen - 1) / 2
    x, y = np.mgrid[-offset:(offset + 1), -offset:(offset + 1)]
    return torch.Tensor(np.dstack((y, x)), device=cuda) / offset


class SourceSimulator:

    def __init__(self, slen, n_bands, background,
                 max_sources, mean_sources, min_sources,
                 psf=None, transpose_psf=False, add_noise=True, draw_poisson=True,
                 is_star=True, cuda=None):
        """
        :param slen: side length of the image.
        :param n_bands: number of bands of each image.
        :param background: shape = (n_bands, slen, slen)
        :param max_sources: Default value 1500
        :param mean_sources: Default value 1200
        :param min_sources: Default value 0
        :param transpose_psf:
        :param add_noise:
        :param draw_poisson:
        """
        assert cuda is not None, "Should be gpu or cpu."

        self.slen = slen  # side length of the image
        self.n_bands = n_bands
        self.background = background.cuda()
        self.cuda = cuda

        assert len(background.shape) == 3
        assert background.shape[0] == self.n_bands
        assert background.shape[1] == background.shape[2] == self.slen

        self.is_star = is_star

        self.max_sources = max_sources
        self.mean_sources = mean_sources
        self.min_sources = min_sources

        self.psf = psf
        self.transpose_psf = transpose_psf

        self.draw_poisson = draw_poisson
        self.add_noise = add_noise
        self.cached_grid = get_mgrid(self.slen, cuda=self.cuda)

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
        else:  # galaxies
            assert fluxes is None and sources is not None

        images_mean = \
            _plot_multiple_sources(self.slen, locs, n_sources, sources,
                                   fluxes=fluxes, cached_grid=self.cached_grid, is_star=self.is_star) + \
            self.background.unsqueeze(0).to(device)

        # ToDo: Change so that it uses galsim (Poisson Noise?) for now.
        if self.add_noise:
            if torch.any(images_mean <= 0):
                print('warning: image mean less than 0')
                images_mean = images_mean.clamp(min=1.0)

            images = (torch.sqrt(images_mean) * torch.randn(images_mean.shape).to(device)
                      + images_mean)
        else:
            images = images_mean

        return images

    def get_source_params(self, n_sources, is_on_array=None, batchsize=1):
        """
        Return either fluxes or the (galaxy latent representation, images)
        """
        return torch.empty()

    def sample_parameters(self, batchsize=1):
        # sample number of sources
        n_sources = _sample_n_sources(self.mean_sources, self.min_sources, self.max_sources, batchsize,
                                      draw_poisson=self.draw_poisson)

        # multiply by zero where they are no stars (recall parameters have entry for up to max_stars)
        is_on_array = const.get_is_on_from_n_sources(n_sources, self.max_sources)

        # sample locations
        locs = _sample_locs(self.max_sources, is_on_array, batchsize=batchsize)

        # either fluxes or galaxy parameters.
        params = self.get_source_params(n_sources, is_on_array=is_on_array, batchsize=batchsize)

        return n_sources, locs, params


class SourceDataset:
    simulator_cls = SourceSimulator

    def __init__(self, n_images, simulator_args, simulator_kwargs,
                 is_star=True, use_cuda=True):
        """
        :param n_images: same as batchsize.
        """
        assert torch.cuda.is_available() or not use_cuda, "No GPU is available"
        self.cuda = torch.device("cuda") if use_cuda else torch.device("cpu")  # use context manager to specify device.

        self.n_images = n_images  # = batchsize.
        self.is_star = is_star

        self.simulator_args = simulator_args

        self.simulator_kwargs = simulator_kwargs
        self.simulator_kwargs.update(dict(cuda=self.cuda))
        assert 'is_star' in simulator_kwargs and simulator_kwargs['is_star'] == self.is_star

        self.simulator = self.simulator_cls(*simulator_args, **simulator_kwargs)

        # image parameters
        self.slen = self.simulator.slen
        self.n_bands = self.simulator.n_bands
        self.background = self.simulator.background  # in cuda.

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        n_sources, locs, params = self.simulator.sample_parameters(batchsize=1)

        if self.is_star:
            fluxes = params
            gal_params, single_galaxies = None, None
        else:
            fluxes = None
            gal_params, single_galaxies = params

        images = self.simulator.draw_image_from_params(locs, n_sources, sources=single_galaxies,
                                                       fluxes=fluxes)

        return {'image': images.squeeze(),
                'background': self.background,
                'locs': locs.squeeze(),
                'params': fluxes.squeeze() if self.is_star else gal_params.squeeze(),
                'n_stars': n_sources.squeeze()
                }


class GalaxySimulator(SourceSimulator):
    def __init__(self, galaxy_slen, gal_decoder_file, *args, **kwargs):
        """
        :param decoder_file: Decoder file where decoder network trained on individual galaxy images is.
        """
        super(GalaxySimulator, self).__init__(*args, **kwargs)
        self.gal_decoder_path = const.models_path.joinpath(gal_decoder_file)
        self.ds = DecoderSamples(galaxy_slen, self.gal_decoder_path, num_bands=self.n_bands)
        self.galaxy_slen = galaxy_slen
        self.latent_dim = self.ds.latent_dim

        assert self.ds.num_bands == self.n_bands == self.background.shape[0]

    def get_source_params(self, n_galaxy, is_on_array=None, batchsize=1):
        assert len(n_galaxy.shape) == 1
        assert n_galaxy.shape[0] == batchsize

        galaxy_params = torch.zeros(batchsize, self.max_sources, self.latent_dim)
        single_galaxies = torch.zeros(batchsize, self.max_sources, self.n_bands, self.galaxy_slen, self.galaxy_slen)

        # z, shape = (num_samples, latent_dim)
        # galaxies, shape = (num_samples, n_bands, slen, slen)
        num_samples = int(n_galaxy.sum().item())
        z, galaxies = self.ds.sample(num_samples)
        count = 0
        for batch_i, n_gal in enumerate(n_galaxy):
            n_gal = int(n_gal)
            galaxy_params[batch_i, 0:n_gal, :] = z[count:count + n_gal, :]
            single_galaxies[batch_i, 0:n_gal, :, :, :] = galaxies[count:count + n_gal, :, :, :]
            count += n_gal

        return galaxy_params, single_galaxies


class GalaxyDataset(SourceDataset):
    simulator_cls = GalaxySimulator

    def __init__(self, n_images, simulator_args, simulator_kwargs):
        """
        """
        simulator_kwargs.update(dict(is_star=False))
        super(GalaxyDataset, self).__init__(n_images, simulator_args, simulator_kwargs,
                                            is_star=False, use_cuda=True)

    @classmethod
    def load_dataset_from_params(cls, n_images, data_params,
                                 add_noise=True, draw_poisson=True):
        # prepare background.
        background_path = const.data_path.joinpath(data_params['background_file'])
        slen = data_params['slen']
        n_bands = data_params['n_bands']

        background = np.load(background_path)
        tbackground = torch.zeros([n_bands, slen, slen])
        assert n_bands == background.shape[0]
        assert background.shape[1] == background.shape[2] == data_params['galaxy_slen']

        for n in range(n_bands):
            tbackground[n, :, :] = float(background[n][0][0])

        simulator_args = [
            data_params['galaxy_slen'],
            data_params['gal_decoder_file'],
            data_params['slen'],
            data_params['n_bands'],
            tbackground,
            data_params['max_galaxies'],
            data_params['mean_galaxies'],
            data_params['min_galaxies'],
        ]

        simulator_kwargs = dict(
            add_noise=add_noise,
            draw_poisson=draw_poisson,
            is_star=False
        )

        return cls(n_images, simulator_args, simulator_kwargs)


class StarSimulator(SourceSimulator):
    def __init__(self, f_min, f_max, alpha, *args, **kwargs):
        """
        :param f_min:
        :param f_max:
        :param alpha:
        :param max_sources: Default value 1500
        :param mean_sources: Default value 1200
        :param min_sources: Default value 0
        """
        super(StarSimulator, self).__init__(*args, **kwargs)
        assert self.psf is not None
        assert self.is_star
        assert len(self.psf.shape) == 3
        assert len(self.background.shape) == 3
        assert self.background.shape[0] == self.psf.shape[0]
        assert self.background.shape[1] == self.slen
        assert self.background.shape[2] == self.slen

        # prior parameters
        self.f_min = f_min
        self.f_max = f_max
        self.alpha = alpha  # pareto parameter.

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

    def get_source_params(self, n_stars, is_on_array=None, batchsize=1):
        """

        :return: fluxes, a shape (batchsize x max_sources x nbands) tensor
        """
        assert is_on_array is not None
        assert n_stars.shape[0] == batchsize
        base_fluxes = _draw_pareto_maxed(self.f_min, self.f_max, alpha=self.alpha,
                                         shape=(batchsize, self.max_sources))

        if self.n_bands > 1:
            colors = torch.randn(batchsize, self.max_sources, self.n_bands - 1).to(device) * 0.15 + 0.3

            _fluxes = 10 ** (colors / 2.5) * base_fluxes.unsqueeze(2)

            fluxes = torch.cat((base_fluxes.unsqueeze(2), _fluxes), dim=2) * is_on_array.unsqueeze(2).float()
        else:
            fluxes = (base_fluxes * is_on_array.float()).unsqueeze(2)

        return fluxes


class StarsDataset(SourceDataset):
    simulator_cls = StarSimulator

    def __init__(self, n_images, simulator_args, simulator_kwargs):
        """

        """
        simulator_kwargs.update(dict(is_star=True))
        super(StarsDataset, self).__init__(n_images, simulator_args, simulator_kwargs,
                                           is_star=True, use_cuda=True)

    @classmethod
    def load_dataset_from_params(cls, n_images, data_params,
                                 psf,
                                 background,
                                 transpose_psf=False,
                                 add_noise=True, draw_poisson=True):
        simulator_args = [
            data_params['f_min'],
            data_params['f_max'],
            data_params['alpha'],
            data_params['slen'],
            data_params['n_bands'],
            background,
            data_params['max_stars'],
            data_params['mean_stars'],
            data_params['min_stars'],
        ]

        simulator_kwargs = dict(
            psf=psf,
            transpose_psf=transpose_psf,
            add_noise=add_noise,
            draw_poisson=draw_poisson,
            is_star=True
        )

        return cls(n_images, simulator_args, simulator_kwargs)
