import torch 
from torch import nn
from torch import optim

from torch.distributions import Normal

import numpy as np


# define parameters for the catalog
class CatalogParameters(nn.Module):
    def __init__(self, init_locs, init_fluxes): 
        
        super(CatalogParameters, self).__init__()
                
        assert torch.all(init_locs < 1)
        assert torch.all(init_locs >= 0)
        
        self.init_loc_free = torch.logit(init_locs).contiguous()
        
        self.loc_param_free = nn.Parameter(self.init_loc_free.clone())
        
        # initialize at zero
        init_params = torch.zeros(init_locs.shape[0],
                                  init_locs.shape[1], 
                                  1)
        # star parameters
        init_free_fluxes = torch.log(init_fluxes).contiguous()
        self.star_flux_param_free = nn.Parameter(init_free_fluxes.clone())
        
        # galaxy parameters
        self.gal_flux_param_free = nn.Parameter(init_free_fluxes.clone())
        self.theta_param_free = nn.Parameter(init_params.clone())
        self.ell_param_free = nn.Parameter(init_params.clone())
        self.rad_free = nn.Parameter(init_params.clone())
        
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
        rad = self.rad_free.exp()
        
        return dict(flux = flux, 
                    theta = theta, 
                    ell = ell, 
                    rad = rad)
    
def get_loss(image, 
             background,
             source_simulator,
             my_catalog,
             star_bool, 
             gal_bool): 
    
    assert image.shape == background.shape
    
    locations = my_catalog.get_locations()
    
    stars = source_simulator.render_stars(locations, 
                                          my_catalog.get_star_params(), 
                                          star_bool)
    
    galaxies = source_simulator.render_galaxies(locations, 
                                                my_catalog.get_galaxy_params(), 
                                                gal_bool)
    
    recon_mean = background + stars + galaxies
    
    normal = Normal(loc = recon_mean, scale = torch.sqrt(recon_mean))
    
    return - normal.log_prob(image).mean(), recon_mean


def estimate_catalog(image, 
                     background, 
                     source_simulator,
                     my_catalog, 
                     star_bool, 
                     gal_bool, 
                     lr = 1, 
                     max_outer_iter = 10,
                     max_inner_iter = 50,
                     tol = 1e-8,
                     print_every = False):

    optimizer = optim.LBFGS(my_catalog.parameters(), 
                            lr = lr, 
                            max_iter = max_inner_iter,
                            line_search_fn = 'strong_wolfe')
    
    def closure():
        optimizer.zero_grad()
        loss = get_loss(image, 
                        background, 
                        source_simulator,
                        my_catalog,
                        star_bool, 
                        gal_bool)[0]
        loss.backward()
        
        return loss
    
    init_loss = closure()
    
    old_loss = 1e16
    for i in range(max_outer_iter):
        loss = optimizer.step(closure)

        if print_every:
            print(loss)

        diff = (loss - old_loss).abs()
        if diff < (tol * init_loss.abs()):
            break

        old_loss = loss
        
