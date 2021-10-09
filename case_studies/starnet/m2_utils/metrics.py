import numpy as np
from bliss.metrics import get_tpr_ppv

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

        tpr_vec[i] = get_tpr_ppv(
            true_locs[which_true],
            true_mags[which_true],
            est_locs,
            est_mags,
            slack = 0.5
        )[0]

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
        
        ppv_vec[i] = get_tpr_ppv(
                    true_locs,
                    true_mags,
                    est_locs[which_est],
                    est_mags[which_est],
                    slack = 0.5
                )[1]
    
    return ppv_vec, mag_bins, counts_vec
