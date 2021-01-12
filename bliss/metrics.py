import torch

from scipy import optimize as sp_optim

def inner_join_locs(locs1, locs2): 
    
    # locs1 and locs2 are arrays of locations, 
    # each of shape (number of sources) x 2
     
    # permutes locs2 to find minimal error between locs1 and locs2.  
    # The matching is done using scipy.optimize.linear_sum_assignment, 
    # which implements the Hungarian algorithm. 
    
    # if locs1 is less than locs2, not every locs2 is returned; 
    # if locs2 is less than locs1, not every locs1 is return. 
    # Only those with a match is returned, hence the "inner_join."
    
    assert len(locs1.shape) == 2
    assert locs1.shape[1] == 2
    
    assert len(locs2.shape) == 2
    assert locs2.shape[1] == 2
    
    ntrue = locs1.shape[0]
    nest = locs2.shape[0]

    # mse of locs: 
    # entry (i,j) is l1 distance between of ith loc in locs1
    # and to jth loc in locs2
    locs_err = (locs1.view(-1, 1, 2) - locs2.view(1, -1, 2)).abs().sum(2)
    
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
    locs1, locs2, row_indx, col_indx = \
        inner_join_locs(locs1, locs2)
    
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
            
    true_locs, true_mag, est_locs, est_mag = \
        inner_join_locs_and_fluxes(true_locs, true_mag, est_locs, est_mag)

    # L1 error over locations  
    locs_mae = (true_locs - est_locs).abs().sum(1)

    # flux error (for all bands)
    fluxes_mae = (true_mag - est_mag).abs().flatten()
    
    return locs_mae, fluxes_mae


def get_tpr_ppv(true_locs, true_mag, est_locs, est_mag,
                      slack = 1.0):
    
    
    # l-infty error in location, 
    # matrix of true x est error
    locs_error = torch.abs(est_locs.unsqueeze(0) - true_locs.unsqueeze(1)).max(-1)[0]
    
    # worst error in either band
    mag_error = torch.abs(est_mag.unsqueeze(0) - true_mag.unsqueeze(1)).max(-1)[0]
    
    tpr_bool = torch.any((locs_error < slack) * (mag_error < slack), dim=1).float()
    ppv_bool = torch.any((locs_error < slack) * (mag_error < slack), dim=0).float()
    
    return tpr_bool.mean(), ppv_bool.mean()


def eval_error_on_batch(true_params, est_params, slen): 
    
                
    # check batch sizes are equal 
    assert len(true_params['n_sources']) == len(est_params['n_sources'])

    batch_size = len(true_params['n_sources'])
    n_bands = true_params["fluxes"].shape[-1]
    
    tpr_vec = torch.zeros(batch_size)
    ppv_vec = torch.zeros(batch_size)
    
    locs_mae_vec = []
    fluxes_mae_vec = []
    
    # accuracy of counting number of sources
    count_bool =  true_params['n_sources'].eq(est_params['n_sources'])
    
    # accuracy of galaxy counts
    est_n_gal = est_params["galaxy_bool"].view(batch_size, -1).sum(-1)
    true_n_gal = true_params["galaxy_bool"].view(batch_size, -1).sum(-1)
    galaxy_counts_bool = est_n_gal.eq(true_n_gal)

    for i in range(batch_size):
        
        # get number of sources
        ntrue = int(true_params['n_sources'][i])
        nest = int(est_params['n_sources'][i])
        
        if (nest > 0) and (ntrue > 0): 

            # prepare locs and get them in units of pixels.
            true_locs = true_params['locs'][i, 0:ntrue].view(ntrue, 2) * slen
            est_locs = est_params['locs'][i, 0:nest].view(nest, 2) * slen        

            # prepare fluxes 
            true_fluxes = true_params['fluxes'][i, 0:ntrue].view(ntrue, n_bands)
            est_fluxes = est_params['fluxes'][i, 0:nest].view(nest, n_bands)
            
            # convert fluxes to magnitude (off by a constant, but
            # doesn't matter since we are looking at the diff)
            true_mag = torch.log10(true_fluxes) * 2.5
            est_mag = torch.log10(est_fluxes) * 2.5
            
            # get TPR and PPV
            tpr_i, ppv_i = get_tpr_ppv(true_locs, true_mag, est_locs, est_mag)
            
            tpr_vec[i] = tpr_i
            ppv_vec[i] = ppv_i
            
            # get l1 error in locations and fluxes
            locs_mae_i, fluxes_mae_i = get_l1_error(true_locs, true_mag,
                                                    est_locs, est_mag)
            
            for k in range(len(locs_mae_i)):
                locs_mae_vec.append(locs_mae_i[k].item())
                fluxes_mae_vec.append(fluxes_mae_i[k].item())
            
    locs_mae_vec = torch.Tensor(locs_mae_vec)
    fluxes_mae_vec = torch.Tensor(fluxes_mae_vec)
    
    return locs_mae_vec, fluxes_mae_vec, count_bool, galaxy_counts_bool, tpr_vec, ppv_vec



