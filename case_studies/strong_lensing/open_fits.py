 


from astropy.io import fits
import matplotlib.pyplot as plt

import numpy as np

# Set up matplotlib
import matplotlib.pyplot as plt

from astropy.io import fits
from matplotlib.colors import LogNorm

image_file = "output_yaml/galsim_realgalaxy_004.fits"
image_data = fits.getdata(image_file)
# print(image_data[0].shape)
c = plt.imshow(image_data)
plt.colorbar(c)
plt.savefig("output_yaml/galsim_realgalaxy_004.png")