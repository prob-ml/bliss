import torch
from torch.utils.data import Dataset

from torch.distributions.multivariate_normal import MultivariateNormal

import numpy as np

device = 'cuda:0'

def _get_mgrid(slen):
    offset = (slen - 1) / 2
    x, y = np.mgrid[-offset:(offset + 1), -offset:(offset + 1)]
    return (torch.Tensor(np.dstack((y, x)))).to(device)


def sample_uniform(unif_range, n_samples): 
    # a list of two numbers giving the range
    # from which to sample
    assert unif_range[0] <= unif_range[1]

    return torch.rand(n_samples, device = device) * (unif_range[1] - unif_range[0]) + unif_range[0]
    
class GaussianGalaxies(Dataset):

    def __init__(self, 
                 slen = 51,
                 f_range = [1000, 5000.],
                 e_range = [0.5, 1], 
                 d_range = [10, 30],
                 a_range = [0, np.pi],
                 background = 686., 
                 n_images = 50000):
        
        # range for fluxes
        self.f_range = f_range 
        
        # range for ellipticity 
        # (between 0 and 1)
        self.e_range = e_range
        
        # range for diameter
        # (in pixels)
        self.d_range = d_range
        
        # range for angle 
        # (between 0 and pi)
        self.a_range = a_range
        
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
        
        # sample diameter 
        d_major = sample_uniform(self.d_range, self.n_images)
        d_major = d_major / 6

        # sample ellipticity
        e = sample_uniform(self.e_range, self.n_images)

        # sample angle 
        angle = sample_uniform(self.a_range, self.n_images)

        # construct the covariance matrix 
        # get eigenvectors
        zeros = torch.zeros(self.n_images, device = device)
        eigvec = torch.dstack((d_major, zeros, zeros, d_major * e)).view(self.n_images, 2, 2) 

        # get rotation matrix
        sine_theta = torch.sin(angle)
        cos_theta = torch.cos(angle)
        rotation = torch.dstack((cos_theta, -sine_theta,
                                 sine_theta, cos_theta)).view(self.n_images, 2, 2) 

        eigen_rotated = torch.einsum('nij, njk -> nik', rotation, eigvec)
        cov = torch.einsum('nij, nkj -> nik', eigen_rotated, eigen_rotated)
        
        # get the normal pdf
        normal = MultivariateNormal(loc = torch.zeros(2, device = device), 
                                    covariance_matrix = cov[:, None, None, :, :])

        image_mean = normal.log_prob(self.mgrid[None]).exp()
        
        # get fluxes
        fluxes = sample_uniform(self.f_range, self.n_images).to(device)
        normalization = image_mean.view(self.n_images, -1).max(-1)[0]
        image_mean = image_mean / (normalization / fluxes)[:, None, None] + self.background
        
        # add noise
        random_noise = torch.randn(image_mean.shape, device = device)
        self.images = torch.sqrt(image_mean) * random_noise + image_mean
        
        self.images = self.images.unsqueeze(1)
        
    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
                
        return self.images[idx]

