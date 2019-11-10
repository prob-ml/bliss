import pathlib
import pickle
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import scipy.stats as stats
from torch.utils.data import Dataset


class Synthetic(Dataset):

    def __init__(self, slen, mean_galaxies=2, min_galaxies=0, max_galaxies=3,
                 num_images=1600, num_bands=5, padding=3, centered=False,
                 brightness=30000):
        """
        Questions: 
        - What is s_density?
        - What  is A**B where both are matrices? 
        """
        super(Synthetic, self).__init__()

        self.slen = slen #number of pixel dimensions. 
        self.mean_galaxies = mean_galaxies
        self.min_galaxies = min_galaxies
        self.max_galaxies = max_galaxies
        self.num_images = num_images
        self.num_bands = num_bands
        self.padding = padding
        self.centered = centered
        self.brightness = brightness

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        #right now this completely ignors the index. 

        sky = 700.0
        background_sample = np.random.randn(self.num_bands, self.slen, self.slen) * np.sqrt(sky) + sky
        background = np.full_like(background_sample, sky, dtype=np.float32)
        image = np.asarray(background_sample, dtype=np.float32)

        axis_lengths = np.array([[3.0, 0.0], [0.0, 7.0]])

        poisson_galaxies = np.random.poisson(self.mean_galaxies)
        num_galaxies = np.maximum(np.minimum(poisson_galaxies, self.max_galaxies), self.min_galaxies)

        for j in range(num_galaxies):
            angle = np.pi * np.random.rand()
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            covar = np.matmul(np.matmul(rotation.transpose(), axis_lengths), rotation) #rotate the initial axis lenghts
            loc = np.random.rand(2) - 0.5 #centered by default. 
            if not self.centered:
                loc *= self.slen - 2 * self.padding
            bvn = stats.multivariate_normal(loc, covar)

            offset = (self.slen - 1) / 2
            y, x = np.mgrid[-offset:(offset + 1), -offset:(offset + 1)]
            pos = np.dstack((y, x)) #all the positions of the different pixels in the considered grid. 
            s_density = bvn.pdf(pos)

            temperature = 1.0 + np.random.rand()
            brightness = self.brightness * temperature ** np.mgrid[1:(self.num_bands + 1)] 
            brightness = brightness.reshape(self.num_bands, 1, 1)
            s_density = s_density.reshape(1, self.slen, self.slen)
            s_intensity = s_density * brightness
            s_noise = np.random.randn(self.num_bands, self.slen, self.slen)
            s_contrib = s_intensity + np.sqrt(s_intensity) * s_noise
            image += s_contrib

        return {'image': image,
                'background': background,
                'num_galaxies': num_galaxies}
