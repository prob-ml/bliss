import torch
import numpy as np

from m2_data import convert_nmgy_to_mag

def filter_params(locs, fluxes, slen, pad = 5):
    assert len(locs.shape) == 2

    if fluxes is not None:
        assert len(fluxes.shape) == 1
        assert len(fluxes) == len(locs)

    _locs = locs * (slen - 1)
    which_params = (_locs[:, 0] > pad) & (_locs[:, 0] < (slen - pad)) & \
                        (_locs[:, 1] > pad) & (_locs[:, 1] < (slen - pad))

    if fluxes is not None:
        return locs[which_params], fluxes[which_params]
    else:
        return locs[which_params], None

def get_locs_error(locs, true_locs):
    # get matrix of Linf error in locations
    # truth x estimated
    return torch.abs(locs.unsqueeze(0) - true_locs.unsqueeze(1)).max(2)[0]

def get_fluxes_error(fluxes, true_fluxes):
    # get matrix of l1 error in log flux
    # truth x estimated
    return torch.abs(torch.log10(fluxes).unsqueeze(0) - \
                     torch.log10(true_fluxes).unsqueeze(1))

def get_mag_error(mags, true_mags):
    # get matrix of l1 error in magnitude
    # truth x estimated
    return torch.abs(mags.unsqueeze(0) - \
                     true_mags.unsqueeze(1))

def get_summary_stats(est_locs, true_locs, slen, est_fluxes, true_fluxes,
                        nelec_per_nmgy,
                        pad = 5, slack = 0.5):

    # remove border
    est_locs, est_fluxes = filter_params(est_locs, est_fluxes, slen, pad)
    true_locs, true_fluxes = filter_params(true_locs, true_fluxes, slen, pad)

    if (est_fluxes is None) or (true_fluxes is None):
        mag_error = 0.
    else:
        # convert to magnitude
        est_mags = convert_nmgy_to_mag(est_fluxes / nelec_per_nmgy)
        true_mags = convert_nmgy_to_mag(true_fluxes / nelec_per_nmgy)
        mag_error = get_mag_error(est_mags, true_mags)

    locs_error = get_locs_error(est_locs * (slen - 1), true_locs * (slen - 1))

    tpr_bool = torch.any((locs_error < slack) * (mag_error < slack), dim = 1).float()

    ppv_bool = torch.any((locs_error < slack) * (mag_error < slack), dim = 0).float()

    return tpr_bool.mean(), ppv_bool.mean(), tpr_bool, ppv_bool

def get_tpr_vec(est_locs, true_locs, slen, est_fluxes, true_fluxes,
                nelec_per_nmgy, mag_vec,
                pad = 5):
    
    est_locs, est_fluxes = filter_params(est_locs, est_fluxes, slen, pad)
    true_locs, true_fluxes = filter_params(true_locs, true_fluxes, slen, pad)

    # convert to magnitude
    true_mags = convert_nmgy_to_mag(true_fluxes / nelec_per_nmgy)

    tpr_vec = np.zeros(len(mag_vec) - 1)

    counts_vec = np.zeros(len(mag_vec) - 1)

    for i in range(len(mag_vec) - 1):
        which_true = (true_mags > mag_vec[i]) & (true_mags < mag_vec[i + 1])
        counts_vec[i] = torch.sum(which_true)

        tpr_vec[i] = \
            get_summary_stats(est_locs, true_locs[which_true], slen,
                            est_fluxes, true_fluxes[which_true],
                            nelec_per_nmgy, pad = pad)[0]

    return tpr_vec, mag_vec, counts_vec

def get_ppv_vec(est_locs, true_locs, slen, est_fluxes, true_fluxes,
                nelec_per_nmgy, mag_vec,
                pad = 5):

    est_locs, est_fluxes = filter_params(est_locs, est_fluxes, slen, pad)
    true_locs, true_fluxes = filter_params(true_locs, true_fluxes, slen, pad)

    est_mags = convert_nmgy_to_mag(est_fluxes / nelec_per_nmgy)

    ppv_vec = np.zeros(len(mag_vec) - 1)
    counts_vec = np.zeros(len(mag_vec) - 1)

    for i in range(len(mag_vec) - 1):
        which_est = (est_mags > mag_vec[i]) & (est_mags < mag_vec[i + 1])

        counts_vec[i] = torch.sum(which_est)

        if torch.sum(which_est) == 0:
            continue

        ppv_vec[i] = \
            get_summary_stats(est_locs[which_est], true_locs, slen,
                            est_fluxes[which_est], true_fluxes,
                            nelec_per_nmgy, pad = pad)[1]

    return ppv_vec, mag_vec, counts_vec

def get_l1_error(est_locs, true_locs, slen, est_fluxes, true_fluxes, pad = 5):
    est_locs, est_fluxes = filter_params(est_locs, est_fluxes, slen, pad)

    true_locs, true_fluxes = filter_params(true_locs, true_fluxes, slen, pad)
    fluxes_error = get_fluxes_error(est_fluxes, true_fluxes)

    locs_error = get_locs_error(est_locs * (slen - 1), true_locs * (slen - 1))

    ppv_bool = torch.any((locs_error < 0.5) * (fluxes_error < 0.5), dim = 0).float()

    locs_matched_error = locs_error[:, ppv_bool == 1]
    fluxes_matched_error = fluxes_error[:, ppv_bool == 1]

    seq_tensor = torch.Tensor([i for i in range(fluxes_matched_error.shape[1])]).type(torch.long)

    locs_error, which_match = locs_matched_error.min(0)

    return locs_error, fluxes_matched_error[which_match, seq_tensor]
