import numpy as np
import scipy.stats as stats

import torch
from torch.utils.data import Dataset, DataLoader, sampler

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, '../')
import sdss_psf
import utils

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def _trim_psf(psf, slen):
    # crop the psf to length slen x slen
    # centered at the middle

    assert len(psf.shape) == 3
    n_bands = psf.shape[0]

    # dimension of the psf should be odd
    psf_slen = psf.shape[2]
    assert psf.shape[1] == psf_slen
    assert (psf_slen % 2) == 1
    psf_center = (psf_slen - 1) / 2

    r = np.floor(slen / 2)
    l_indx = int(psf_center  - r)
    u_indx = int(psf_center + r + 1)

    return psf[:, l_indx:u_indx, l_indx:u_indx]

def _expand_psf(psf, slen):
    # first dimension of psf is number of bands
    assert len(psf.shape) == 3
    n_bands = psf.shape[0]

    psf_slen = psf.shape[2]
    assert psf.shape[1] == psf_slen
    # dimension of psf should be odd
    assert (psf_slen % 2) == 1
    # sim for slen
    assert (slen % 2) == 1

    psf_expanded = np.zeros((n_bands, slen, slen))

    offset = int((slen - psf_slen) / 2)

    psf_expanded[:, offset:(offset+psf_slen), offset:(offset+psf_slen)] = psf

    return psf_expanded

def _get_mgrid(slen):
    offset = (slen - 1) / 2
    x, y = np.mgrid[-offset:(offset + 1), -offset:(offset + 1)]
    # return torch.Tensor(np.dstack((x, y))) / offset
    return torch.Tensor(np.dstack((y, x))) / offset

def plot_one_star(slen, locs, psf, cached_grid = None):
    # locs is batchsize x x_loc x y_loc: takes values between 0 and 1
    # psf is a slen x slen tensor

    # assert torch.all(locs <= 1)
    # assert torch.all(locs >= 0)

    # slen = psf.shape[-1]
    # assert slen == psf.shape[-2]
    assert len(psf.shape) == 3
    n_bands = psf.shape[0]

    batchsize = locs.shape[0]
    assert locs.shape[1] == 2

    if cached_grid is None:
        grid = _get_mgrid(slen)
    else:
        assert cached_grid.shape[0] == slen
        assert cached_grid.shape[1] == slen
        grid = cached_grid

    # scale locs so they take values between -1 and 1 for grid sample
    locs = (locs - 0.5) * 2
    grid_loc = grid.view(1, slen, slen, 2) - locs[:, [1, 0]].view(batchsize, 1, 1, 2)

    star = F.grid_sample(psf.expand(batchsize, n_bands, -1, -1), grid_loc)

    # normalize so one star still sums to 1
    return star # / star.sum(3, keepdim=True).sum(2, keepdim=True)

def plot_multiple_stars(slen, locs, n_stars, fluxes, psf, cached_grid = None):
    # locs is batchsize x max_stars x x_loc x y_loc
    # fluxes is batchsize x n_bands x max_stars
    # n_stars is length batchsize
    # psf is a n_bands x slen x slen tensor

    n_bands = psf.shape[0]

    batchsize = locs.shape[0]
    max_stars = locs.shape[1]
    assert locs.shape[2] == 2

    assert fluxes.shape[0] == batchsize
    assert fluxes.shape[1] == max_stars
    assert fluxes.shape[2] == n_bands
    assert len(n_stars) == batchsize
    assert len(n_stars.shape) == 1

    if cached_grid is None:
        grid = _get_mgrid(slen)
    else:
        assert cached_grid.shape[0] == slen
        assert cached_grid.shape[1] == slen
        grid = cached_grid

    stars = 0. #torch.zeros((batchsize, 1, slen, slen)).to(device)

    for n in range(max(n_stars)):
        is_on_n = (n < n_stars).float()
        locs_n = locs[:, n, :] * is_on_n.unsqueeze(1)
        fluxes_n = fluxes[:, n, :]

        one_star = plot_one_star(slen, locs_n, psf, cached_grid = grid)

        stars += one_star * (is_on_n.unsqueeze(1) * fluxes_n).view(batchsize, n_bands, 1, 1)

    return stars

def _draw_pareto(f_min, alpha, shape):
    uniform_samples = torch.rand(shape).to(device)

    return f_min / (1 - uniform_samples)**(1 / alpha)

def _draw_pareto_maxed(f_min, f_max, alpha, shape):
    # draw pareto conditioned on being less than f_max

    pareto_samples = _draw_pareto(f_min, alpha, shape)

    while torch.any(pareto_samples > f_max):
        indx = pareto_samples > f_max
        pareto_samples[indx] = \
            _draw_pareto(f_min, alpha, torch.sum(indx))

    return pareto_samples

class StarSimulator:
    def __init__(self, psf, slen, sky_intensity, transpose_psf):
        assert len(psf.shape) == 3

        self.n_bands = psf.shape[0]
        self.psf_og = psf

        # side length of the image
        self.slen = slen

        # get psf shape to match image shape
        if (slen >= self.psf_og.shape[-1]):
            self.psf = torch.Tensor(_expand_psf(self.psf_og, slen)).to(device)
        else:
            self.psf = torch.Tensor(_trim_psf(self.psf_og, slen)).to(device)

        if transpose_psf:
            self.psf = self.psf.transpose(1, 2)

        # TODO:
        # should we then upsample??

        # normalize
        # self.psf = self.psf / torch.sum(self.psf)

        self.cached_grid = _get_mgrid(slen).to(device)

        assert len(sky_intensity) == self.n_bands
        assert len(sky_intensity.shape) == 1
        self.sky_intensity = sky_intensity.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    def draw_image_from_params(self, locs, fluxes, n_stars,
                                        add_noise = True):
        images_mean = \
            plot_multiple_stars(self.slen, locs, n_stars, fluxes,
                                    self.psf, self.cached_grid) + \
                self.sky_intensity

        # add noise
        if add_noise:
            images = torch.sqrt(images_mean) * torch.randn(images_mean.shape).to(device) + \
                                                            images_mean
        else:
            images = images_mean

        return images


class StarsDataset(Dataset):

    def __init__(self, psf, n_images,
                        slen,
                         max_stars,
                         mean_stars,
                         min_stars,
                         f_min,
                         f_max,
                         sky_intensity,
                         alpha,
                         transpose_psf,
                         add_noise = True):

        self.slen = slen
        self.n_bands = psf.shape[0]

        assert sky_intensity.shape == (self.n_bands, )
        self.sky_intensity = sky_intensity
        self.simulator = StarSimulator(psf, slen, sky_intensity, transpose_psf)

        # image parameters
        self.max_stars = max_stars
        self.mean_stars = mean_stars
        self.min_stars = min_stars
        self.add_noise = add_noise

        # TODO: make this an argument
        self.draw_poisson = False

        # prior parameters
        self.f_min = f_min
        self.f_max = f_max

        self.alpha = alpha

        # dataset parameters
        self.n_images = n_images

        # set data
        self.set_params_and_images()

        self.background = \
            torch.ones(self.n_images, self.n_bands, self.slen, self.slen).to(device) * \
                            self.sky_intensity.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)



    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):

        return {'image': self.images[idx],
                'background': self.background[idx],
                'locs': self.locs[idx],
                'fluxes': self.fluxes[idx],
                'n_stars': self.n_stars[idx]}

    def draw_batch_parameters(self, batchsize, return_images = True):
        # draw number of stars
        if self.draw_poisson:
            n_stars = np.random.poisson(self.mean_stars, batchsize)
        else:
            n_stars = np.random.choice(np.arange(self.min_stars, self.max_stars + 1),
                                        batchsize)

        n_stars = torch.Tensor(n_stars).clamp(max = self.max_stars,
                        min = self.min_stars).type(torch.LongTensor).to(device)
        is_on_array = utils.get_is_on_from_n_stars(n_stars, self.max_stars)

        # draw locations
        locs = torch.rand((batchsize, self.max_stars, 2)).to(device) * \
                is_on_array.unsqueeze(2).float()

        # draw fluxes
        base_fluxes = _draw_pareto_maxed(self.f_min, self.f_max, alpha = self.alpha,
                                shape = (batchsize, self.max_stars))
        # # just for fun: lets see what happens
        # base_log_fluxes = torch.rand(batchsize, self.max_stars).to(device) * (np.log(self.f_max) - np.log(self.f_min)) + np.log(self.f_min)
        # base_fluxes = torch.exp(base_log_fluxes)

        if self.n_bands > 1:
            colors = torch.randn(batchsize, self.max_stars, self.n_bands - 1).to(device) * 0.2 - 0.14

            _fluxes = 10**(- colors / 2.5) * base_fluxes.unsqueeze(2)

            fluxes = torch.cat((base_fluxes.unsqueeze(2), _fluxes), dim = 2) * \
                                is_on_array.unsqueeze(2).float()
        else:
            fluxes = (base_fluxes * is_on_array.float()).unsqueeze(2)

        if return_images:
            images = self.simulator.draw_image_from_params(locs, fluxes, n_stars,
                                                add_noise = self.add_noise)

            return locs, fluxes, n_stars, images
        else:
            return locs, fluxes, n_stars

    def set_params_and_images(self):
        self.locs, self.fluxes, self.n_stars, self.images = \
            self.draw_batch_parameters(self.n_images, return_images = True)


def load_dataset_from_params(psf, data_params,
                                n_images,
                                sky_intensity,
                                transpose_psf,
                                add_noise = True):
    # data parameters
    slen = data_params['slen']

    f_min = data_params['f_min']
    f_max = data_params['f_max']
    alpha = data_params['alpha']

    max_stars = data_params['max_stars']
    mean_stars = data_params['mean_stars']
    min_stars = data_params['min_stars']

    # draw data
    return StarsDataset(psf,
                            n_images,
                            slen = slen,
                            f_min=f_min,
                            f_max=f_max,
                            max_stars = max_stars,
                            mean_stars = mean_stars,
                            min_stars = min_stars,
                            alpha = alpha,
                            sky_intensity = sky_intensity,
                            transpose_psf = transpose_psf,
                            add_noise = add_noise)
