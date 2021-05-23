import torch 
from torch import nn

import numpy as np

# define parameters for the catalog
class CatalogParameters(nn.Module):
    def __init__(self, init_locs): 
        
        super(CatalogParameters, self).__init__()
                
        assert torch.all(init_locs < 1)
        assert torch.all(init_locs >= 0)
        
        self.init_loc_free = torch.logit(init_locs)
        
        self.loc_param_free = nn.Parameter(self.init_loc_free.clone())
        
        # initialize at zero
        init_params = torch.zeros(init_locs.shape[0],
                                  init_locs.shape[1], 
                                  1)
        # star parameters
        self.star_flux_param_free = nn.Parameter(init_params.clone())
        
        # galaxy parameters
        self.gal_flux_param_free = nn.Parameter(init_params.clone())
        self.theta_param_free = nn.Parameter(init_params.clone())
        self.ell_param_free = nn.Parameter(init_params.clone())
        self.rdev_param_free = nn.Parameter(init_params.clone())
        self.rexp_dev_free = nn.Parameter(init_params.clone())
        self.pdev_dev_free = nn.Parameter(init_params.clone())
        
    def get_locations(self): 
        # return locations
        # (constrained between 0, 1)
        return self.loc_param_free.sigmoid() 
    
    def get_star_params(self): 
        # return fluxes (constrained to be positive
        return self.star_flux_param_free.exp() 
    
    def get_galaxy_params(self): 
        # fluxes are positive
        flux = self.gal_flux_param_free.exp() 
        
        # between zero and 2pi
        theta = torch.sigmoid(self.theta_param_free) * 2 * np.pi
        
        # between zero and 1
        ell = torch.sigmoid(self.ell_param_free) 
        
        # radii are positive 
        r_dev = self.rdev_param_free.exp()
        r_exp = self.rexp_dev_free.exp()
        
        # between zero and 
        p_dev = torch.sigmoid(self.pdev_dev_free)
        
        return dict(flux = flux, 
                    theta = theta, 
                    ell = ell, 
                    r_dev = r_dev, 
                    r_exp = r_exp,
                    p_dev = p_dev)