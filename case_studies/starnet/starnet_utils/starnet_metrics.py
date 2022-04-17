import torch
import numpy as np

import galsim
import numpy as np
import torch
import tqdm
from einops import rearrange, reduce
from scipy import optimize as sp_optim

def inner_join_locs(locs1, locs2):

    # locs1 and locs2 are arrays of locations,
    # each of shape (number of sources) x 2

    # permutes locs2 to find minimal error between locs1 and locs2.
    # The matching is done using scipy.optimize.linear_sum_assignment,
    # which implements the Hungarian algorithm.

    # if locs1 is less than locs2, not every locs2 is returned;
    # if locs2 is less than locs1, not every locs1 is returned.
    # Only those with a match is returned, hence the "inner_join."

    assert len(locs1.shape) == 2
    assert locs1.shape[1] == 2

    assert len(locs2.shape) == 2
    assert locs2.shape[1] == 2

    # mse of locs:
    # entry (i,j) is l1 distance between of ith loc in locs1
    # and to jth loc in locs2
    locs_err = (rearrange(locs1, "i j -> i 1 j") - rearrange(locs2, "i j -> 1 i j")).abs()
    locs_err = reduce(locs_err, "i j k -> i j", "sum")

    # find minimal permutation
    row_indx, col_indx = sp_optim.linear_sum_assignment(locs_err.detach().cpu())

    # return locations with their match
    return locs1[row_indx], locs2[col_indx], row_indx, col_indx


def inner_join_locs_and_fluxes(locs1, fluxes1, locs2, fluxes2):
    # permutes locations and fluxes using the Hungarian algorithm
    # on locations to find matches.
    # see `inner_join_locs`.

    # find matches based on locations,
    # and permute locations.
    locs1, locs2, row_indx, col_indx = inner_join_locs(locs1, locs2)

    # permute fluxes based on how we permuted locations
    fluxes1 = fluxes1[row_indx]
    fluxes2 = fluxes2[col_indx]

    return locs1, fluxes1, locs2, fluxes2


def get_l1_error(true_locs, true_mag, est_locs, est_mag):

    # returns a vector of absolute errors for locations and fluxes,
    # one for each star.

    # The number of estimated stars does not have to equal the number of true stars!

    # We only evaluate the error on the sources that have a match:
    # the returned vector of absolute erros have length min(num. true sources, num. est. sources).
    # See `inner_join_locs_and_fluxes` above.

    true_locs, true_mag, est_locs, est_mag = inner_join_locs_and_fluxes(
        true_locs, true_mag, est_locs, est_mag
    )

    # L1 error over locations
    locs_mae = (true_locs - est_locs).abs().sum(1)

    # flux error (for all bands)
    fluxes_mae = (true_mag - est_mag).abs().flatten()

    return locs_mae, fluxes_mae


def get_tpr_ppv(true_locs, true_mag, est_locs, est_mag, slack=1.0):

    # l-infty error in location,
    # matrix of true x est error
    locs_error = torch.abs(est_locs.unsqueeze(0) - true_locs.unsqueeze(1)).max(-1)[0]

    # worst error in either band
    mag_error = torch.abs(est_mag.unsqueeze(0) - true_mag.unsqueeze(1)).max(-1)[0]

    tpr_bool = torch.any((locs_error < slack) * (mag_error < slack), dim=1).float()
    ppv_bool = torch.any((locs_error < slack) * (mag_error < slack), dim=0).float()

    return tpr_bool.mean(), ppv_bool.mean()



def get_tpr_vec(
    true_locs, true_mags, est_locs, est_mags, mag_bins
):

    # convert to magnitude
    tpr_vec = np.zeros(len(mag_bins) - 1)
    counts_vec = np.zeros(len(mag_bins) - 1)

    for i in range(len(mag_bins) - 1):
        which_true = (true_mags > mag_bins[i]) & (true_mags < mag_bins[i + 1])
        which_true = which_true.squeeze()
        counts_vec[i] = torch.sum(which_true)
           
        if counts_vec[i] > 0: 
            tpr_vec[i] = get_tpr_ppv(
                true_locs[which_true],
                true_mags[which_true],
                est_locs,
                est_mags,
                slack = 0.5
            )[0]
        else: 
            tpr_vec[i] = 0

    return tpr_vec, mag_bins, counts_vec


def get_ppv_vec(
    true_locs, true_mags, est_locs, est_mags, mag_bins
):

    ppv_vec = np.zeros(len(mag_bins) - 1)
    counts_vec = np.zeros(len(mag_bins) - 1)

    for i in range(len(mag_bins) - 1):
        which_est = (est_mags > mag_bins[i]) & (est_mags < mag_bins[i + 1])
        which_est = which_est.squeeze()
        
        counts_vec[i] = torch.sum(which_est)
        
        if counts_vec[i] > 0: 
            ppv_vec[i] = get_tpr_ppv(
                        true_locs,
                        true_mags,
                        est_locs[which_est],
                        est_mags[which_est],
                        slack = 0.5
                    )[1]
        else: 
            ppv_vec[i] = 0
            
    return ppv_vec, mag_bins, counts_vec
