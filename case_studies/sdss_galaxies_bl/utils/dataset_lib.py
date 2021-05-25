import numpy as np

import torch
from torch.utils.data import Dataset

from bliss.models.decoder import Tiler, get_is_on_from_n_sources, ImageDecoder

from torch.distributions import Poisson


from source_simulator_lib import SourceSimulator
from which_device import device


def _convert_mag_to_nmgy(mag):
    return 10**((22.5 - mag) / 2.5)

def _sample_uniform(unif_range, shape): 
    # a list of two numbers giving the range
    # from which to sample
    assert unif_range[0] <= unif_range[1]

    return torch.rand(shape, device = device) * (unif_range[1] - unif_range[0]) + unif_range[0]

class SimulatedImages(Dataset):

    def __init__(self, 
                 psf, 
                 slen = 200,
                 gal_slen = 51,
                 tile_slen = 4, 
                 ptile_slen = 52, 
                 mean_sources_per_tile = 1e-2,
                 max_sources_per_tile = 2, 
                 prob_galaxy = 0.5, 
                 lflux_range = [3, 5.],
                 ell_range = [0.4, 1], 
                 theta_range = [0, 2 * np.pi],
                 hlr_range = [0, 2.5],
                 background = 686., 
                 border_padding = 0, 
                 n_images = 64):
        
        # some constants
        self.n_images = n_images
        self.slen = slen 
        self.tile_slen = tile_slen
        self.ptile_slen = ptile_slen
        
        n_tiles_per_image = (self.slen / self.tile_slen)**2
        assert n_tiles_per_image % 1 == 0
        self.n_tiles_per_image = int(n_tiles_per_image)
        
        self.n_ptiles = self.n_images * self.n_tiles_per_image
        
        self.max_sources_per_tile = max_sources_per_tile
        
        self.n_sources = self.n_images * self.n_tiles_per_image * \
                            self.max_sources_per_tile
        
        self.border_padding = border_padding
        
        self.n_bands = psf.shape[0]
        if self.n_bands > 1: 
            raise NotImplementedError()
        
        # the source simulator 
        self.background = background
        self.source_simulator = SourceSimulator(psf,
                                                tile_slen = tile_slen,
                                                ptile_slen = ptile_slen, 
                                                gal_slen = gal_slen)

        # range for log fluxes
        self.lflux_range = lflux_range 
        
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

        
    def _sample_galaxy_params(self): 
        
        # sample fluxes 
        lflux = _sample_uniform(self.lflux_range, self.n_sources)
        flux = 10**lflux
        
        # sample ellipticity
        ell = _sample_uniform(self.ell_range, self.n_sources)

        # sample angle
        theta = _sample_uniform(self.theta_range, self.n_sources)
        
        # sample half-light radii
        rad = _sample_uniform(self.hlr_range, self.n_sources)
        
        galaxy_params = dict(flux = flux, 
                             theta = theta, 
                             ell = ell, 
                             rad = rad)
        
        # flatten so that its n_ptiles x max_sources x -1
        for key in galaxy_params:
            galaxy_params[key] = galaxy_params[key].view(self.n_ptiles, 
                                                         self.max_sources_per_tile, 
                                                         -1)

        
        return galaxy_params
    
    
    def _sample_star_fluxes(self): 
                
        lfluxes = _sample_uniform(self.lflux_range, (self.n_ptiles, 
                                                   self.max_sources_per_tile,
                                                   self.n_bands))
        
        return 10**lfluxes
    
    def _sample_ptiles(self): 
        
        # sample number of sources 
        m = Poisson(self.mean_sources_per_tile)
        n_sources = m.sample([self.n_ptiles])

        # long() here is necessary because used for indexing and one_hot encoding.
        n_sources = n_sources.clamp(max=self.max_sources_per_tile).long().to(device)
        
        is_on_array = get_is_on_from_n_sources(n_sources, 
                                               self.max_sources_per_tile).to(device)
        is_on_array = is_on_array.unsqueeze(-1)
        
        # sample galaxies
        u = _sample_uniform((0, 1), is_on_array.shape)
        galaxy_bool = is_on_array * (u < self.prob_galaxy)
        star_bool = is_on_array * (1 - galaxy_bool)
        
        # sample locations 
        locs = _sample_uniform((0, 1), (self.n_ptiles, 
                                        self.max_sources_per_tile, 
                                        2))
        
        # sample stars
        star_fluxes = self._sample_star_fluxes()
        
        stars = self.source_simulator.render_stars(locs, 
                                              star_fluxes,
                                              star_bool)
        
        # sample galaxies
        galaxy_params = self._sample_galaxy_params()
        galaxies = self.source_simulator.render_galaxies(locs, 
                                                         galaxy_params,
                                                         galaxy_bool)
    
        image_ptiles = stars + galaxies        
        
        return image_ptiles, locs, star_fluxes, n_sources, galaxy_bool
        
    
    def sample_batch(self): 
        image_ptiles, locs, star_fluxes, n_sources, galaxy_bool = \
            self._sample_ptiles()
        
        image_ptiles = image_ptiles.view(self.n_images, 
                                         self.n_tiles_per_image, 
                                         self.n_bands, 
                                         self.ptile_slen, 
                                         self.ptile_slen)
        
        images = ImageDecoder._construct_full_image_from_ptiles(image_ptiles, 
                                                  tile_slen = self.tile_slen, 
                                                  border_padding = self.border_padding)
        
        images += self.background
        
        images = images.clip(min = 1.)
        
        # add noise
        images += torch.randn(images.shape, device = device) * torch.sqrt(images)
        
        
        def _reshape_params(param): 
            return param.view(self.n_images, 
                              self.n_tiles_per_image,
                              self.max_sources_per_tile, 
                              -1)
        
        batch = dict(images = images, 
                     locs = _reshape_params(locs), 
                     log_fluxes = _reshape_params(torch.log(star_fluxes)), 
                     n_sources = n_sources.view(self.n_images, 
                                                self.n_tiles_per_image), 
                     galaxy_bool = _reshape_params(galaxy_bool))
                     
        return batch
                     
        
    
    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
                
        return self.images[idx]

