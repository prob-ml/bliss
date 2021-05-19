import numpy as np

import torch
from torch.nn.functional import grid_sample

device = 'cuda:4'

def _trim_psf(psf, slen):
    # crop the psf to length slen x slen
    # centered at the middle

    assert len(psf.shape) == 3
    n_bands = psf.shape[0]

    # dimension of the psf should be odd
    psf_slen = psf.shape[2]
    assert psf.shape[1] == psf_slen
    assert (psf_slen % 2) == 1
    assert (slen % 2) == 1
    psf_center = (psf_slen - 1) / 2

    assert psf_slen >= slen

    r = np.floor(slen / 2)
    l_indx = int(psf_center  - r)
    u_indx = int(psf_center + r + 1)

    return psf[:, l_indx:u_indx, l_indx:u_indx]

def _expand_psf(psf, slen):
    # pad the psf with zeros so that it is size slen

    # first dimension of psf is number of bands
    assert len(psf.shape) == 3
    n_bands = psf.shape[0]

    psf_slen = psf.shape[2]
    assert psf.shape[1] == psf_slen
    # dimension of psf should be odd
    assert (psf_slen % 2) == 1
    # sim for slen
    assert (slen % 2) == 1

    assert psf_slen <= slen

    psf_expanded = torch.zeros((n_bands, slen, slen))

    offset = int((slen - psf_slen) / 2)

    psf_expanded[:, offset:(offset+psf_slen), offset:(offset+psf_slen)] = psf

    return psf_expanded


def plot_one_star(slen, locs, psf, cached_grid = None):
    # locs is batchsize x 2: takes values between 0 and 1
    # psf is a slen x slen tensor

    # assert torch.all(locs <= 1)
    # assert torch.all(locs >= 0)

    # slen = psf.shape[-1]
    # assert slen == psf.shape[-2]
    assert len(psf.shape) == 4
    n_bands = psf.shape[1]

    batchsize = locs.shape[0]
    assert locs.shape[1] == 2
    assert psf.shape[0] == batchsize

    if cached_grid is None:
        grid = _get_mgrid(slen)
    else:
        assert cached_grid.shape[0] == slen
        assert cached_grid.shape[1] == slen
        grid = cached_grid

    # scale locs so they take values between -1 and 1 for grid sample
    locs = (locs - 0.5) * 2
    locs = locs.index_select(1,  torch.tensor([1, 0], device=device))
    grid_loc = grid.view(1, slen, slen, 2) - locs.view(batchsize, 1, 1, 2)

    star = grid_sample(psf, grid_loc, align_corners = True)

    # normalize so one star still sums to 1
    return star 

def plot_multiple_stars(slen, locs, n_stars, psf, cached_grid = None):
    # locs is batchsize x max_stars x x_loc x y_loc
    # fluxes is batchsize x n_bands x max_stars
    # n_stars is length batchsize
    # psf is a n_bands x slen x slen tensor

    n_bands = psf.shape[0]

    batchsize = locs.shape[0]
    max_stars = locs.shape[1]
    assert locs.shape[2] == 2

    assert fluxes.shape[0] == locs.shape[0]
    assert fluxes.shape[1] == locs.shape[1]
    assert fluxes.shape[2] == n_bands
    assert len(n_stars) == batchsize
    assert len(n_stars.shape) == 1

    assert max(n_stars) <= locs.shape[1]

    if cached_grid is None:
        grid = _get_mgrid(slen)
    else:
        assert cached_grid.shape[0] == slen
        assert cached_grid.shape[1] == slen
        grid = cached_grid

    stars = 0. 

    for n in range(max(n_stars)):
        is_on_n = (n < n_stars).float()
        locs_n = locs[:, n, :] * is_on_n.unsqueeze(1)
        sources_n = psf[:, n, :, :, :]

        one_star = plot_one_star(slen, locs_n, sources_n, cached_grid = grid)

        stars += one_star * is_on_n.view(batchsize, 1, 1, 1)

    return stars


