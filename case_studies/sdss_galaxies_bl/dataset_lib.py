import numpy as np
import torch
from torch.utils.data import Dataset

from bliss.models.decoder import Tiler

import galaxy_simulator_lib

device = 'cuda:4'

def _get_mgrid(slen, normalize = True):
    offset = (slen - 1) / 2
    x, y = np.mgrid[-offset:(offset + 1), -offset:(offset + 1)]
    
    mgrid = torch.Tensor(np.dstack((x, y))).to(device)
    
    if normalize: 
        mgrid = mgrid / offset
    
    return mgrid

def convert_mag_to_nmgy(mag):
    return 10**((22.5 - mag) / 2.5)

def _sample_uniform(unif_range, shape): 
    # a list of two numbers giving the range
    # from which to sample
    assert unif_range[0] <= unif_range[1]

    return torch.rand(shape, device = device) * (unif_range[1] - unif_range[0]) + unif_range[0]

def _convolve_w_psf(images, psf): 
    
    # first dimension is number of bands
    assert len(psf.shape) == 3 
    
    # need to flip based on how the pytorch convolution works ... 
    _psf = psf.flip(-1).flip(1).unsqueeze(0)
    padding = int((psf.shape[-1] - 1) / 2)
    images = \
        torch.nn.functional.conv2d(images, 
                                   _psf, 
                                   stride = 1,
                                   padding = padding)
    
    return images

class SimulatedImages(Dataset):

    def __init__(self, 
                 psf, 
                 slen = 200,
                 gal_slen = 51,
                 tile_slen = 4, 
                 ptile_slen = 52, 
                 mean_sources_per_tile = 0.05,
                 max_sources_per_tile = 2, 
                 flux_range = [1000, 5000.],
                 ell_range = [0.25, 1], 
                 theta_range = [0, 2 * np.pi],
                 hlr_range = [0, 3],
                 background = 686., 
                 batchsize = 64):
        
        # some constants
        self.batchsize = batchsize
        n_tiles_per_image = slen / tile_slen
        assert n_tiles_per_image % 1 == 0
        self.n_tiles_per_image = int(n_tiles_per_image)
        
        self.max_sources_per_tile = max_sources_per_tile
        
        self.n_sources = self.batchsize * self.n_tiles_per_image * \
                            self.max_sources_per_tile

        
        # first dimension is bands 
        self.nbands = psf.shape[0]
        if self.nbands > 1: 
            raise NotImplementedError()
        
        # the psf
        psf_slen = psf.shape[1]
        assert psf.shape[2] == psf.shape[1]
        self.psf = psf
        self.psf_expanded = psf.expand(self.n_sources, 
                                       self.nbands, 
                                       psf_slen, 
                                       psf_slen)
                
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
                
        # sky background
        self.background = background
        
        # grid for galaxy
        self.gal_slen = gal_slen
        assert (self.gal_slen % 2 == 1)
        self.galaxy_mgrid = _get_mgrid(self.gal_slen, 
                                       normalize = False)
        
        # the tiler
        self.tiler = Tiler(tile_slen, ptile_slen)
        
    def _sample_galaxies(self): 
        
        # sample fluxes 
        fluxes = _sample_uniform(self.flux_range, self.n_sources)
        
        # sample ellipticity
        ells = _sample_uniform(self.ell_range, self.n_sources)

        # sample angle
        thetas = _sample_uniform(self.theta_range, self.n_sources)
        
        # sample half-light radii
        # one for each profile
        r_dev = _sample_uniform(self.hlr_range, self.n_sources)
        r_exp = _sample_uniform(self.hlr_range, self.n_sources)

        # mixture weights 
        p_dev = _sample_uniform((0, 1), self.n_sources)
                
        # render gaussians 
        centered_galaxies = \
            galaxy_simulator_lib.render_centered_galaxy(flux = fluxes, 
                                                        theta = thetas, 
                                                        ell = ells, 
                                                        r_dev = r_dev, 
                                                        r_exp = r_exp,
                                                        p_dev = p_dev, 
                                                        galaxy_mgrid = self.galaxy_mgrid)
        
        centered_galaxies = _convolve_w_psf(centered_galaxies, self.psf)
        
        
        return centered_galaxies
    
    
    def _sample_stars(self): 
                
        fluxes = _sample_uniform(self.flux_range, (self.n_sources, self.nbands))
        
        stars = self.psf_expanded * fluxes.view(self.n_sources, 
                                               self.nbands, 
                                               1, 1)
        return stars, fluxes
        
    
    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
                
        return self.images[idx]

