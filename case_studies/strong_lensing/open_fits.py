from astropy.io import fits
import matplotlib.pyplot as plt

import numpy as np

# Set up matplotlib
import matplotlib.pyplot as plt

from astropy.io import fits
from matplotlib.colors import LogNorm
import sys

i = sys.argv[1]
output_dir = sys.argv[2]

image_file = f"{output_dir}/image{i}.fits"
image_data = fits.getdata(image_file)
c = plt.imshow(image_data)
plt.colorbar(c)
plt.savefig(f"{output_dir}/image{i}.png")
