import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from bliss.models.decoder import Tiler, get_is_on_from_n_sources

from torch.distributions import Poisson


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
    images = F.conv2d(images, 
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
                 prob_galaxy = 0.5, 
                 flux_range = [1000, 5000.],
                 ell_range = [0.25, 1], 
                 theta_range = [0, 2 * np.pi],
                 hlr_range = [0, 3],
                 background = 686., 
                 n_images = 64):
        
        # some constants
        self.n_images = n_images
        self.slen = slen 
        self.tile_slen = tile_slen 
        self.ptile_slen = ptile_slen
        
        n_tiles_per_image = (self.slen / self.tile_slen)**2
        assert n_tiles_per_image % 1 == 0
        self.n_tiles_per_image = int(n_tiles_per_image)
        
        self.max_sources_per_tile = max_sources_per_tile
        
        self.n_sources = self.n_images * self.n_tiles_per_image * \
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
        
        # probability of galaxy
        self.prob_galaxy = prob_galaxy
        
        # mean number of sources
        self.mean_sources_per_tile = mean_sources_per_tile
                
        # sky background
        self.background = background
        
        # grid for galaxy
        self.gal_slen = gal_slen
        assert (self.gal_slen % 2 == 1)
        self.galaxy_mgrid = _get_mgrid(self.gal_slen, 
                                       normalize = False)
        
        # the tiler
        self.tiler = Tiler(tile_slen, ptile_slen).to(device)
        
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
    
    def _sample_ptiles(self): 
        
        # sample number of sources 
        m = Poisson(self.mean_sources_per_tile)
        n_sources = m.sample([self.n_images, self.n_tiles_per_image])

        # long() here is necessary because used for indexing and one_hot encoding.
        n_sources = n_sources.clamp(max=self.max_sources_per_tile)
        n_sources = n_sources.long().view(self.n_images * self.n_tiles_per_image)
        
        is_on_array = get_is_on_from_n_sources(n_sources, 
                                               self.max_sources_per_tile).to(device)
        
        # sample galaxies
        u = _sample_uniform((0, 1), is_on_array.shape)
        galaxy_bool = is_on_array * (u < self.prob_galaxy)
        
        # sample locations 
        locs = _sample_uniform((0, 1), (self.n_sources, 2))
        
        # sample stars
        stars, fluxes = self._sample_stars()
        star_ptiles = self.tiler(locs, stars).view(-1, 
                                                   self.max_sources_per_tile, 
                                                   self.nbands, 
                                                   self.ptile_slen, 
                                                   self.ptile_slen)
        
        # sample galaxies 
        gals = self._sample_galaxies()
        gal_ptiles = self.tiler(locs, gals).view(-1, 
                                                 self.max_sources_per_tile, 
                                                 self.nbands, 
                                                 self.ptile_slen, 
                                                 self.ptile_slen)
        
        # combine 
        _is_on = is_on_array.view(-1, self.max_sources_per_tile, 1, 1, 1)
        _galaxy_bool = galaxy_bool.view(-1, self.max_sources_per_tile, 1, 1, 1)
        image_ptiles = ((gal_ptiles * _galaxy_bool + star_ptiles * (1 - _galaxy_bool)) * _is_on).sum(1)
        
        # reshape appropriately 
        image_ptiles = image_ptiles.view(self.n_images, 
                                         self.n_tiles_per_image, 
                                         self.nbands, 
                                         self.ptile_slen, 
                                         self.ptile_slen)
        
        _is_on2 = is_on_array.view(self.n_images, 
                                   self.n_tiles_per_image, 
                                   self.max_sources_per_tile, 
                                   1)
        
        locs = locs.view(self.n_images, 
                         self.n_tiles_per_image, 
                         self.max_sources_per_tile, 
                         2) * _is_on2
        
        fluxes = fluxes.view(self.n_images, 
                             self.n_tiles_per_image, 
                             self.max_sources_per_tile, 
                             self.nbands) * _is_on2
        
        n_sources = n_sources.view(self.n_images, self.n_tiles_per_image)
        
        galaxy_bool = galaxy_bool.view(self.n_images, 
                                       self.n_tiles_per_image, 
                                       self.max_sources_per_tile, 
                                       1)
                                   
        
        
        return image_ptiles, n_sources, locs, fluxes, galaxy_bool
        
        
    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
                
        return self.images[idx]

