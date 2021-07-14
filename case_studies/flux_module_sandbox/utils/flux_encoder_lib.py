import numpy as np

import torch
from torch import nn

from einops import rearrange, repeat
import torch.nn.functional as F

from which_device import device

def _get_images_in_tiles(images, tile_slen, ptile_slen):
    
    # this tiling code adapted from the ImageEncoder method 
    # of the same name. Make this its own function then?
    
    # images should be batchsize x n_bands x slen x slen
    assert len(images.shape) == 4
    
    n_bands = images.shape[1]
    
    window = ptile_slen
    tiles = F.unfold(images, kernel_size=window, stride=tile_slen)
    
    # b: batch, c: channel, h: tile height, w: tile width, n: num of total tiles for each batch
    tiles = rearrange(tiles, "b (c h w) n -> (b n) c h w", c=n_bands, h=window, w=window)
    
    return tiles

def _trim_images(images, trim_slen): 
    
    slen = images.shape[-1]
    
    diff = slen - trim_slen
        
    indx0 = int(np.floor(diff / 2))
    indx1 = indx0 + trim_slen
        
    return images[:, :, indx0:indx1, indx0:indx1]

    

class MLPEncoder(nn.Module):
    def __init__(self, 
                 ptile_slen = 52, 
                 tile_slen = 4,
                 flux_tile_slen = 20, 
                 n_bands = 1,
                 max_sources = 1,
                 latent_dim = 64):

        super(MLPEncoder, self).__init__()

        # image / model parameters
        self.ptile_slen = ptile_slen
        self.tile_slen = tile_slen
        self.n_bands = n_bands
        
        self.max_sources = max_sources
        
        # output dimension 
        outdim = 2 * self.max_sources * self.n_bands
        
        # the size of the ptiles passed to this encoder
        self.flux_tile_slen = flux_tile_slen
        self.n_pixels = self.flux_tile_slen ** 2 * self.n_bands
        
        # the network
        self.fc1 = nn.Linear(self.n_pixels, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, outdim)
        
        self.relu = torch.nn.ReLU()
        self.softplus = torch.nn.Softplus()
        
    def forward(self, images): 
        
        batch_size = images.shape[0]
        
        # tile the image
        image_ptiles = self._get_ptiles_from_images(images)
        
        # pass through nn
        mean, sd = self._forward_ptiles(image_ptiles)
        
        # sample 
        z = torch.randn(mean.shape, device = device)
        samples = mean + z * sd
        
        # save everything in a dictionary
        out_dict = dict(mean = mean, 
                        sd = sd, 
                        samples = samples)
        
        # reshape
        n_tiles_per_image = int(image_ptiles.shape[0] / batch_size)
        
        for k in out_dict.keys(): 
            out_dict[k] = out_dict[k].view(batch_size,
                                           n_tiles_per_image,
                                           self.max_sources, 
                                           self.n_bands)
        
        return out_dict 
        
    
    def _forward_ptiles(self, image_ptiles):
        
        n = image_ptiles.shape[0]
                
        # pass through neural network
        h = image_ptiles.flatten(1, -1)
        
        h = self.fc1(h)
        h = self.relu(h)
        
        h = self.fc2(h)
        h = self.relu(h)
        
        h = self.fc3(h)
        
        
        indx0 = self.max_sources * self.n_bands
        indx1 = 2 * indx0
        
        mean = h[:, 0:indx0]
        sd = self.softplus(h[:, indx0:indx1]) + 1e-6
        
        # sd_scale = torch.sigmoid(h[:, 1])
        # sd = mean * sd_scale
    
        return mean, sd
    
    def _trim_ptiles(self, image_ptiles): 
        
        return _trim_images(image_ptiles, self.flux_tile_slen)

    
    def _get_ptiles_from_images(self, images): 
        image_ptiles = _get_images_in_tiles(images, 
                                            tile_slen = self.tile_slen, 
                                            ptile_slen = self.ptile_slen)
        
        return self._trim_ptiles(image_ptiles)
