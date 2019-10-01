import numpy as np
import scipy.stats as stats

import torch
from torch.utils.data import Dataset, DataLoader, sampler

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, '../')
import sdss_psf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _trim_psf(psf, slen):
    # crop the psf to length slen x slen
    # centered at the middle

    # dimension of the psf should be odd
    psf_slen = psf.shape[0]
    assert psf.shape[1] == psf_slen
    assert (psf_slen % 2) == 1
    psf_center = (psf_slen - 1) / 2

    r = np.floor(slen / 2)
    l_indx = int(psf_center  - r)
    u_indx = int(psf_center + r + 1)

    return psf[l_indx:u_indx, l_indx:u_indx]

def _expand_psf(psf, slen):
    psf_slen = psf.shape[0]
    assert psf.shape[1] == psf_slen
    # dimension of psf should be odd
    assert (psf_slen % 2) == 1
    # sim for slen
    assert (slen % 2) == 1

    psf_expanded = np.zeros((slen, slen))

    offset = int((slen - psf_slen) / 2)

    psf_expanded[offset:(offset+psf_slen), offset:(offset+psf_slen)] = psf

    return psf_expanded
def _get_mgrid(slen):
    offset = (slen - 1) / 2
    x, y = np.mgrid[-offset:(offset + 1), -offset:(offset + 1)]
    return torch.Tensor(np.dstack((x, y))) / offset

def plot_one_star(slen, locs, psf, cached_grid = None):
    # locs is batchsize x x_loc x y_loc: takes values between 0 and 1
    # psf is a slen x slen tensor

    # assert torch.all(locs <= 1)
    # assert torch.all(locs >= 0)

    # slen = psf.shape[-1]
    # assert slen == psf.shape[-2]

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
    grid_loc = grid.view(1, slen, slen, 2) - locs.view(batchsize, 1, 1, 2)

    star = F.grid_sample(psf.expand(batchsize, 1, -1, -1), grid_loc)

    # normalize so one star still sums to 1
    return star # star.sum(3, keepdim=True).sum(2, keepdim=True)

def plot_multiple_stars(slen, locs, n_stars, fluxes, psf, cached_grid = None):
    # locs is batchsize x max_stars x x_loc x y_loc
    # fluxes is batchsize x max_stars
    # n_stars is length batchsize
    # psf is a slen x slen tensor

    batchsize = locs.shape[0]
    max_stars = locs.shape[1]
    assert locs.shape[2] == 2

    assert fluxes.shape[0] == batchsize
    assert fluxes.shape[1] == max_stars
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
        fluxes_n = fluxes[:, n]

        one_star = plot_one_star(slen, locs_n, psf, cached_grid = grid)

        stars += one_star * (is_on_n * fluxes_n).view(-1, 1, 1, 1)

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
    def __init__(self, psf_fit_file, slen, sky_intensity):
        self.psf_full = sdss_psf.psf_at_points(0, 0, psf_fit_file = psf_fit_file)

        self.slen = slen

        # get psf shape to match image shape
        if (slen >= self.psf_full.shape[0]):
            self.psf = torch.Tensor(_expand_psf(self.psf_full, slen)).to(device)
        else:
            self.psf = torch.Tensor(_trim_psf(self.psf_full, slen)).to(device)

        # TODO:
        # should we then upsample??

        # normalize
        # self.psf = self.psf / torch.sum(self.psf)

        self.cached_grid = _get_mgrid(slen).to(device)

        self.sky_intensity = sky_intensity


    def draw_image_from_params(self, locs, fluxes, n_stars,
                                        add_noise = True):
        images_mean = \
            plot_multiple_stars(self.slen, locs, n_stars, fluxes, self.psf, self.cached_grid) + \
                self.sky_intensity

        # add noise
        if add_noise:
            images = torch.sqrt(images_mean) * torch.randn(images_mean.shape).to(device) + \
                                                            images_mean
        else:
            images = images_mean

        return images


class StarsDataset(Dataset):

    def __init__(self, psf_fit_file, n_images,
                        slen = 21,
                         max_stars = 5,
                         min_stars = 0,
                         f_min = 700.0,
                         f_max = 1000.0,
                         sky_intensity = 686.0,
                         alpha = 4,
                         add_noise = True):

        self.slen = slen
        self.sky_intensity = sky_intensity
        self.simulator = StarSimulator(psf_fit_file, slen, sky_intensity)

        # image parameters
        self.max_stars = max_stars
        self.min_stars = min_stars
        self.add_noise = add_noise

        # prior parameters
        self.f_min = f_min
        self.f_max = f_max

        self.alpha = alpha

        # dataset parameters
        self.n_images = n_images

        # set data
        self.set_params_and_images()

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):

        locs = self.locs[idx:(idx+1)]
        n_stars = self.n_stars[idx:(idx+1)]
        fluxes = self.fluxes[idx:(idx+1)]
        images = self.images[idx:(idx+1)]

        return {'image': images[0, :, :, :],
                'background': torch.Tensor([[[self.sky_intensity]]]),
                'locs': locs.squeeze(dim = 0),
                'fluxes': fluxes.squeeze(dim = 0),
                'n_stars': n_stars.squeeze(dim = 0)}

    def draw_batch_parameters(self, batchsize, return_images = True):
        # draw number of stars
        n_stars = np.random.choice(np.arange(self.min_stars, self.max_stars + 1),
                                    batchsize)
        n_stars = torch.Tensor(n_stars).type(torch.LongTensor).to(device)

        # draw locations
        locs = torch.rand((batchsize, self.max_stars, 2)).to(device)
        # locs = _sort_locs(locs, is_on_array)

        # draw fluxes
        fluxes = _draw_pareto_maxed(self.f_min, self.f_max, alpha = self.alpha,
                                shape = (batchsize, self.max_stars))

        if return_images:
            images = self.simulator.draw_image_from_params(locs, fluxes, n_stars,
                                                add_noise = self.add_noise)

            return locs, fluxes, n_stars, images
        else:
            return locs, fluxes, n_stars

    def set_params_and_images(self):
        self.locs, self.fluxes, self.n_stars, self.images = \
            self.draw_batch_parameters(self.n_images, return_images = True)

def _sort_locs(locs, is_on_array):

    sort_perm = \
        torch.argsort(torch.sum(locs**2, dim = 2) * is_on_array,
                        descending=True)

    seq_tensor = torch.LongTensor([[i] for i in range(locs.shape[0])])
    return locs[seq_tensor, sort_perm] * is_on_array.unsqueeze(2)


def get_is_on_from_n_stars(n_stars, max_stars):
    batchsize = len(n_stars)
    is_on_array = torch.zeros((batchsize, max_stars)).to(device)
    for i in range(max_stars):
        is_on_array[:, i] = (n_stars > i).float()

    return is_on_array


def load_dataset_from_params(psf_fit_file, data_params, n_images,
                                add_noise = True):
    # data parameters
    slen = data_params['slen']

    f_min = data_params['f_min']
    f_max = data_params['f_max']
    alpha = data_params['alpha']

    max_stars = data_params['max_stars']
    min_stars = data_params['min_stars']

    sky_intensity = data_params['sky_intensity']

    # draw data
    return StarsDataset(psf_fit_file,
                            n_images,
                            slen = slen,
                            f_min=f_min,
                            f_max=f_max,
                            max_stars = max_stars,
                            min_stars = min_stars,
                            alpha = alpha,
                            sky_intensity = sky_intensity,
                            add_noise = add_noise)
