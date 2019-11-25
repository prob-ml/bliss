import torch
import numpy as np

def filter_params(locs, fluxes, slen, pad = 5):
    assert len(locs.shape) == 2
    assert len(fluxes.shape) == 1

    _locs = locs * (slen - 1)
    which_params = (_locs[:, 0] > pad) & (_locs[:, 0] < (slen - pad)) & \
                        (_locs[:, 1] > pad) & (_locs[:, 1] < (slen - pad))


    return locs[which_params], fluxes[which_params]

def get_locs_error(locs, true_locs):
    # get matrix of Linf error in locations
    # truth x estimated
    return torch.abs(locs.unsqueeze(0) - true_locs.unsqueeze(1)).max(2)[0]

def get_fluxes_error(fluxes, true_fluxes):
    # get matrix of l1 error in log flux
    # truth x estimated
    return torch.abs(torch.log10(fluxes).unsqueeze(0) - \
                     torch.log10(true_fluxes).unsqueeze(1))

def get_summary_stats(est_locs, true_locs, slen, est_fluxes, true_fluxes, pad = 5):

    est_locs, est_fluxes = filter_params(est_locs, est_fluxes, slen, pad)
    true_locs, true_fluxes = filter_params(true_locs, true_fluxes, slen, pad)

    if (est_fluxes is None) or (true_fluxes is None):
        fluxes_error = 0.
    else:
        fluxes_error = get_fluxes_error(est_fluxes, true_fluxes)

    locs_error = get_locs_error(est_locs * (slen - 1), true_locs * (slen - 1))

    completeness_bool = torch.any((locs_error < 0.5) * (fluxes_error < 0.5), dim = 1).float()

    tpr_bool = torch.any((locs_error < 0.5) * (fluxes_error < 0.5), dim = 0).float()

    return completeness_bool.mean(), tpr_bool.mean(), completeness_bool, tpr_bool

def get_completeness_vec(est_locs, true_locs, slen, est_fluxes, true_fluxes,
                            pad = 5, mag_vec = None):

    est_locs, est_fluxes = filter_params(est_locs, est_fluxes, slen, pad)
    true_locs, true_fluxes = filter_params(true_locs, true_fluxes, slen, pad)

    true_mag = torch.log10(true_fluxes)

    if mag_vec is None:
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

def get_tpr_vec(est_locs, true_locs, slen, est_fluxes, true_fluxes, pad = 5, mag_vec = None):

    est_locs, est_fluxes = filter_params(est_locs, est_fluxes, slen, pad)
    true_locs, true_fluxes = filter_params(true_locs, true_fluxes, slen, pad)

    est_mag = torch.log10(est_fluxes)

    if mag_vec is None:
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

def get_l1_error(est_locs, true_locs, slen, est_fluxes, true_fluxes, pad = 5):
    est_locs, est_fluxes = filter_params(est_locs, est_fluxes, slen, pad)

    true_locs, true_fluxes = filter_params(true_locs, true_fluxes, slen, pad)
    fluxes_error = get_fluxes_error(est_fluxes, true_fluxes)

    locs_error = get_locs_error(est_locs * (slen - 1), true_locs * (slen - 1))

    tpr_bool = torch.any((locs_error < 0.5) * (fluxes_error < 0.5), dim = 0).float()

    locs_matched_error = locs_error[:, tpr_bool == 1]
    fluxes_matched_error = fluxes_error[:, tpr_bool == 1]

    seq_tensor = torch.Tensor([i for i in range(fluxes_matched_error.shape[1])]).type(torch.long)

    locs_error, which_match = locs_matched_error.min(0)

    return locs_error, fluxes_matched_error[which_match, seq_tensor]
