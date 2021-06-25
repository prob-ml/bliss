import torch 
import numpy as np

from torch.utils.data import Dataset

from which_device import device

def trim_psf(psf, slen): 
    
    # trims the psf array
    assert len(psf.shape) == 2
    assert psf.shape[0] == psf.shape[1]
    
    # size of psf should be odd
    assert ((psf.shape[0] - 1) % 2) == 0
    
    psf_slen = psf.shape[-1]
    psf_center = (psf_slen - 1) / 2
    
    r = np.floor(slen / 2)
    l_indx = int(psf_center - r)
    u_indx = int(psf_center + r + 1)

    return psf[l_indx:u_indx, l_indx:u_indx]


class CenteredStarsData(Dataset):

    def __init__(self, 
                 psf, 
                 n_images = 60000, 
                 log10_flux_range = [3, 5.], 
                 background = 800):
        
        self.psf = psf
        self.slen = psf.shape[0]
        assert psf.shape[1] == psf.shape[0]
        
        self.n_images = n_images
        
        # range for log10(flux)
        self.lflux_max = log10_flux_range[1]
        self.lflux_min = log10_flux_range[0]
        assert self.lflux_max >= self.lflux_min
        
        # sky background 
        self.background = background
        
    def __len__(self): 
        return self.n_images
    
    def __getitem__(self, indx): 
        
        # uniform draw 
        u = torch.rand(size = (1,), device = device)
        
        # sample log flux 
        log10_flux = u * (self.lflux_max - self.lflux_min) + self.lflux_min
        
        # get flux 
        flux = 10**log10_flux
        
        # construct image 
        image = self.psf * flux + self.background
        image += torch.sqrt(image) * torch.randn(size = image.shape, device = device)
        
        # encoder expects a band: 
        image = image.view(1, self.slen, self.slen)
        
        return {'image': image, 
                'flux': flux}