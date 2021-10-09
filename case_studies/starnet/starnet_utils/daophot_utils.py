import torch
import numpy as np

from utils import convert_mag_to_nmgy
def load_daophot_results(data_file, 
                            nelec_per_nmgy,
                            wcs, 
                            slen = 100,
                            x0 = 630,
                            x1 = 310): 

    daophot_file = np.loadtxt(data_file)
    
    # load desired quantities
    daophot_ra = daophot_file[:, 4]
    daophot_decl = daophot_file[:, 5]
    daophot_mags = daophot_file[:, 22]
    
    # get pixel coordinates
    pix_coords = wcs.wcs_world2pix(daophot_ra, daophot_decl, 0, ra_dec_order = True)
    
    # get locations inside our square
    which_locs = (pix_coords[1] > x0) & (pix_coords[1] < (x0 + slen - 1)) & \
                        (pix_coords[0] > x1) & (pix_coords[0] < (x1 + slen - 1))
    
    # scale between zero and ones
    daophot_locs0 = pix_coords[1][which_locs]
    daophot_locs1 = pix_coords[0][which_locs]
    daophot_locs = torch.Tensor(np.array([daophot_locs0, daophot_locs1]).transpose())
    
    # get fluxes
    daophot_fluxes = convert_mag_to_nmgy(daophot_mags[which_locs]) * \
                        nelec_per_nmgy 
    daophot_fluxes = torch.Tensor(daophot_fluxes).unsqueeze(1)
    
    return daophot_locs, daophot_fluxes


def align_daophot_locs(daophot_locs, daophot_fluxes, hubble_locs, hubble_fluxes, 
                       slen = 100, 
                       align_on_logflux = 4.5): 
    # take only bright stars
    log10_fluxes = torch.log10(daophot_fluxes).squeeze()
    log10_hubble_fluxes = torch.log10(hubble_fluxes).squeeze()
    which_est_brightest = torch.nonzero(log10_fluxes > align_on_logflux).squeeze()
    which_hubble_brightest = torch.nonzero(log10_hubble_fluxes > align_on_logflux).squeeze()
    
    _daophot_locs = daophot_locs[which_est_brightest]
    _hubble_locs = hubble_locs[which_hubble_brightest]
        
    # match daophot locations to hubble locations
    perm = _get_locs_error(_daophot_locs, _hubble_locs).argmin(0)
    
    # get error and estimate bias
    locs_err = (_daophot_locs - _hubble_locs[perm]) 
    bias_x1 = locs_err[:, 1].median() 
    bias_x0 = locs_err[:, 0].median() 
    
    # shift by bias
    daophot_locs[:, 0] -= bias_x0
    daophot_locs[:, 1] -= bias_x1
    
    return daophot_locs, daophot_fluxes


def _get_locs_error(locs, true_locs):
    # get matrix of Linf error in locations
    # truth x estimated
    return torch.abs(locs.unsqueeze(0) - true_locs.unsqueeze(1)).max(2)[0]

