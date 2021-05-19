import torch
from torch.utils.data import Dataset

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Dirichlet

import numpy as np

device = 'cuda:4'


#################
# Functions to set up and transform (aka rotate or stretch)
# coordinate system 
#################
def _get_rotation_matrix(theta): 
    
    batchsize = len(theta)
    
    sine_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    rotation = torch.dstack((cos_theta, -sine_theta,
                             sine_theta, cos_theta)).view(batchsize, 2, 2) 
    
    return rotation.to(device)

def _get_strech_matrix(ell): 
    
    batchsize = len(ell)
    
    stretch = torch.zeros(batchsize, 2, 2, device = device)
    
    stretch[:, 0, 0] = 1
    stretch[:, 1, 1] = ell
    
    return stretch

def _transform_mgrid_to_radius_grid(mgrid, theta, ell): 
    
    rotation = _get_rotation_matrix(theta)
    stretch = _get_strech_matrix(ell)
    
    rotated_basis = torch.einsum('nij, njk -> nik', rotation, stretch)
    precision = torch.einsum('nij, nkj -> nik', rotated_basis, rotated_basis)
    
    r2_grid = torch.einsum('xyi, nij, xyj -> nxy', 
                           mgrid, precision, mgrid)
    
    return r2_grid

#################
# galaxy profiles
#################
def render_gaussian_galaxies(r2_grid, half_light_radii): 
    
    assert r2_grid.shape[0] == len(half_light_radii)
    batchsize = len(half_light_radii)
    
    scale = half_light_radii.view(batchsize, 1, 1)**2 / np.log(2)
    
    return torch.exp(-r2_grid / scale)

def render_exponential_galaxies(r2_grid, half_light_radii): 
    
    assert r2_grid.shape[0] == len(half_light_radii)
    batchsize = len(half_light_radii)
    
    scale = half_light_radii.view(batchsize, 1, 1) / np.log(2)
    r_grid = torch.sqrt(r2_grid)
    
    return torch.exp(- r_grid / scale)

def render_devaucouleurs_galaxies(r2_grid, half_light_radii): 
    
    assert r2_grid.shape[0] == len(half_light_radii)
    batchsize = len(half_light_radii)

    scale = half_light_radii.view(batchsize, 1, 1)**(0.25) / np.log(2)
    r_grid = torch.sqrt(r2_grid)
    
    return torch.exp(-r_grid**0.25 / scale)

###############
# function to render galaxies
###############
def render_centered_galaxy(flux, theta, ell, r_dev, r_exp, p_dev, 
                           galaxy_mgrid): 
    
    # number of galaxies
    n_galaxies = len(flux)
    
    # radial coordinate system
    r2_grid = _transform_mgrid_to_radius_grid(galaxy_mgrid, 
                                              theta = theta,  
                                              ell = ell)
    
    # exponential profile
    exp_profile = render_exponential_galaxies(r2_grid, 
                                              half_light_radii = r_exp) 
    
    # dev. profile 
    dev_profile = render_exponential_galaxies(r2_grid, 
                                              half_light_radii = r_dev)
    
    # mixture weight and fluxes
    p_dev = p_dev.view(n_galaxies, 1, 1)
    flux = flux.view(n_galaxies, 1, 1)
    
    centered_galaxy = exp_profile * (1 - p_dev) + exp_profile * p_dev
    centered_galaxy *= flux
    
    return centered_galaxy.unsqueeze(1)