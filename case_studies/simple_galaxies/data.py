import torch
from torch.utils.data import Dataset

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Dirichlet

import numpy as np

device = 'cuda:0'


#################
# Functions to set up and transform (aka rotate or stretch)
# coordinate system 
#################
def _get_mgrid(slen):
    offset = (slen - 1) / 2
    x, y = np.mgrid[-offset:(offset + 1), -offset:(offset + 1)]
    return (torch.Tensor(np.dstack((y, x)))).to(device)

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
    
    return torch.exp(-torch.sqrt(r2_grid) / scale)

def render_devaucouleurs_galaxies(r2_grid, half_light_radii): 
    
    assert r2_grid.shape[0] == len(half_light_radii)
    batchsize = len(half_light_radii)

    scale = (half_light_radii.view(batchsize, 1, 1)**(0.25) / np.log(2))**4
    
    return torch.exp(-torch.sqrt(r2_grid / scale)**(0.25))

#################
# function to sample unformly
#################
def _sample_uniform(unif_range, shape): 
    # a list of two numbers giving the range
    # from which to sample
    assert unif_range[0] <= unif_range[1]

    return torch.rand(shape, device = device) * (unif_range[1] - unif_range[0]) + unif_range[0]
    
class SimulatedGalaxies(Dataset):

    def __init__(self, 
                 slen = 51,
                 flux_range = [1000, 5000.],
                 ell_range = [0.5, 1], 
                 theta_range = [0, np.pi],
                 hlr_range = [1, 5],
                 background = 686., 
                 n_images = 50000):
        
        # range for fluxes
        self.flux_range = flux_range 
        
        # range for ellipticity 
        # (between 0 and 1)
        self.ell_range = ell_range
        
        # range for angle 
        # (between 0 and pi)
        self.theta_range = theta_range

        # range for half light radius
        # (in pixels)
        self.hlr_range = hlr_range
        
        # distribution on mixture weights of the profile
        self.dirichlet = Dirichlet(concentration = torch.ones(3, device = device))
        
        # sky background
        self.background = background
        
        # number of images in the dataset
        self.n_images = n_images
        
        # set grid
        self.slen = slen
        self.mgrid = _get_mgrid(slen)
        
        # create the dataset
        self._set_images()
        
    def _set_images(self): 

        # generate the images in this dataset
        
        # sample fluxes 
        fluxes = _sample_uniform(self.flux_range, self.n_images)
        
        # sample ellipticity
        ells = _sample_uniform(self.ell_range, self.n_images)

        # sample angle
        thetas = _sample_uniform(self.theta_range, self.n_images)
        
        # sample half-light radii
        # one for each profile
        hlr_range = _sample_uniform(self.hlr_range, (self.n_images, 3))

        # now get the coordinate system 
        r2_grid = _transform_mgrid_to_radius_grid(self.mgrid, 
                                                  thetas, 
                                                  ells)
        
        # render gaussians 
        gaussians = render_gaussian_galaxies(r2_grid, 
                                             hlr_range[:, 0])
        
        # render exponentials
        exponentials = render_exponential_galaxies(r2_grid, 
                                                   hlr_range[:, 1])
        
        # render devaucouleurs
        devaucouleurs = render_devaucouleurs_galaxies(r2_grid, 
                                                      hlr_range[:, 2])
        
        # sample mixture weights
        mixture_weights = self.dirichlet.sample((self.n_images, ))
        
        p0 = mixture_weights[:, 0].view(self.n_images, 1, 1)
        p1 = mixture_weights[:, 1].view(self.n_images, 1, 1)
        p2 = mixture_weights[:, 2].view(self.n_images, 1, 1)
        
        galaxies = gaussians * p0 + \
                        exponentials * p1 + \
                        devaucouleurs * p2
        
        galaxies *= fluxes.view(self.n_images, 1, 1)
        
        # our encoder expects one band. 
        # give it a band
        galaxies = galaxies.view(self.n_images, 
                                 1, 
                                 self.slen,
                                 self.slen)
            
         
        
        image_mean = galaxies + self.background
        
        # add noise
        random_noise = torch.randn(image_mean.shape, device = device)
        self.images = torch.sqrt(image_mean) * random_noise + image_mean
                
    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
                
        return self.images[idx]

