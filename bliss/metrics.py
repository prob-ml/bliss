import torch

from scipy import optimize as sp_optim


def inner_join_locs(locs1, locs2): 
    
    # locs1 and locs2 are arrays of locations, 
    # each of shape (number of sources) x 2
    
    # locs1 and locs2 may contain unequal number of locations.
    
    # returns two arrays of locations, both equal shape, 
    # with shape = min(locs1.shape, locs2.shape) x 2
    # such that each location in locs1 has a match in locs2
    # and vice versa
    
    # the matching is done using scipy.optimize.linear_sum_assignment, 
    # which implements the Hungarian algorithm. 
    
    assert len(locs1.shape) == 2
    assert locs1.shape[1] == 2
    
    assert len(locs2.shape) == 2
    assert locs2.shape[1] == 2
    
    n1 = locs1.shape[0]
    n2 = locs2.shape[0]

    # mse of locs: 
    # entry (i,j) is l1 distance between of ith loc in locs1
    # and to jth loc in locs2
    locs_err = (locs1.view(-1, 1, 2) - locs2.view(1, -1, 2)).abs().sum(2)
    
    # find minimal permutation
    row_indx, col_indx = sp_optim.linear_sum_assignment(locs_err.detach().cpu())
    
    # return locations with their match
    return locs1[row_indx], locs2[col_indx], row_indx, col_indx

def inner_join_locs_and_fluxes(locs1, fluxes1, locs2, fluxes2): 
    
    # permutes locations and finds matches using 
    # inner_join_locs
    
    # then permutes the fluxes accordingly
    
    # permute locs
    locs1, locs2, row_indx, col_indx = \
        inner_join_locs(locs1, locs2)
    
    # permute fluxes
    fluxes1 = fluxes1[row_indx]
    fluxes2 = fluxes2[col_indx]
    
    return locs1, fluxes1, locs2, fluxes2

def eval_error_on_batch(true_params, est_params, slen): 
    
    # true_params and est_params are parameter dictionaries of 
    # true and estimated parameters, respectively, *on the full image*. 
    
    # returns a vector of absolute errors for locations and fluxes
    
    # the parameters are "inner_joined", meaning that for 
    # whichever of true_params or estimated_params has the fewer number 
    # of sources -- call it params1 -- 
    # we find a match for every source in params1 
    # in the other parameter list. 
                
    # check batch sizes are equal 
    assert len(true_params['n_sources']) == len(est_params['n_sources'])

    batch_size = len(true_params['n_sources'])
    n_bands = true_params["fluxes"].shape[-1]
    
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
        n1 = int(true_params['n_sources'][i])
        n2 = int(est_params['n_sources'][i])
        
        if (n2 > 0) and (n1 > 0): 

            # prepare locs and get them in units of pixels.
            locs1 = true_params['locs'][i, 0:n1].view(n1, 2) * slen
            locs2 = est_params['locs'][i, 0:n2].view(n2, 2) * slen        

            # prepare fluxes 
            fluxes1 = true_params['fluxes'][i, 0:n1].view(n1, n_bands)
            fluxes2 = est_params['fluxes'][i, 0:n2].view(n2, n_bands)
            
            # convert fluxes to magnitude
            mag1 = torch.log10(fluxes1) * 2.5
            mag2 = torch.log10(fluxes2) * 2.5
            
            # permute sources so each source has a match
            locs1, mag1, locs2, mag2 = \
                inner_join_locs_and_fluxes(locs1, mag1, locs2, mag2)
            
            # L1 error over locations  
            locs_mae_i = (locs1 - locs2).abs().sum(1)
            
            # flux error (for all bands)
            fluxes_mae_i = (mag1 - mag2).abs()

            for k in range(len(locs_mae_i)):
                locs_mae_vec.append(locs_mae_i[k].item())
                fluxes_mae_vec.append(fluxes_mae_i[k].item())
    
    locs_mae_vec = torch.Tensor(locs_mae_vec)
    fluxes_mae_vec = torch.Tensor(fluxes_mae_vec)
    
    return locs_mae_vec, fluxes_mae_vec, count_bool, galaxy_counts_bool