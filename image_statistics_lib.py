import torch
import numpy as np

def get_locs_error(locs, true_locs):
    # get matrix of Linf error in locations
    # truth x estimated
    return torch.abs(locs.unsqueeze(0) - true_locs.unsqueeze(1)).max(2)[0]

def get_fluxes_error(fluxes, true_fluxes):
    # get matrix of l1 error in log flux
    # truth x estimated
    return torch.abs(torch.log10(fluxes).unsqueeze(0) - \
                     torch.log10(true_fluxes).unsqueeze(1))

def get_summary_stats(est_locs, true_locs, slen, est_fluxes, true_fluxes):
    if (est_fluxes is None) or (true_fluxes is None):
        fluxes_error = 0.
    else:
        fluxes_error = get_fluxes_error(est_fluxes, true_fluxes)

    locs_error = get_locs_error(est_locs * (slen - 1), true_locs * (slen - 1))

    completeness_bool = torch.any((locs_error < 0.5) * (fluxes_error < 0.5), dim = 1).float()

    tpr_bool = torch.any((locs_error < 0.5) * (fluxes_error < 0.5), dim = 0).float()

    return completeness_bool.mean(), tpr_bool.mean(), completeness_bool, tpr_bool

def get_completeness_vec(est_locs, true_locs, slen, est_fluxes, true_fluxes):
    true_mag = torch.log10(true_fluxes)

    # round to nearest half integer
    max_mag = torch.ceil(true_mag.max() * 2) / 2
    min_mag = torch.floor(true_mag.min() * 2) / 2

    mag_vec = np.arange(min_mag, max_mag + 0.1, 0.5)

    completeness_vec = np.zeros(len(mag_vec) - 1)

    counts_vec = np.zeros(len(mag_vec) - 1)

    for i in range(len(mag_vec) - 1):
        which_true = (true_mag > mag_vec[i]) & (true_mag < mag_vec[i + 1])
        counts_vec[i] = torch.sum(which_true)

        completeness_vec[i] = \
            get_summary_stats(est_locs, true_locs[which_true], slen,
                            est_fluxes, true_fluxes[which_true])[0]

    return completeness_vec, mag_vec, counts_vec

def get_tpr_vec(est_locs, true_locs, slen, est_fluxes, true_fluxes):
    est_mag = torch.log10(est_fluxes)

    max_mag = torch.ceil(est_mag.max() * 2) / 2
    min_mag = torch.floor(est_mag.min() * 2) / 2

    mag_vec = np.arange(min_mag, max_mag + 0.1, 0.5)

    tpr_vec = np.zeros(len(mag_vec) - 1)
    counts_vec = np.zeros(len(mag_vec) - 1)

    for i in range(len(mag_vec) - 1):
        which_est = (est_mag > mag_vec[i]) & (est_mag < mag_vec[i + 1])

        counts_vec[i] = torch.sum(which_est)

        if torch.sum(which_est) == 0:
            continue

        tpr_vec[i] = \
            get_summary_stats(est_locs[which_est], true_locs, slen,
                            est_fluxes[which_est], true_fluxes)[1]

    return tpr_vec, mag_vec, counts_vec
