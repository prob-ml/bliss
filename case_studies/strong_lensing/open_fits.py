import sys

from astropy.io import fits
from matplotlib import pyplot as plt

i = sys.argv[1]
OUTPUT_DIR = sys.argv[2]

image_file = f"{OUTPUT_DIR}/image{i}.fits"
image_data = fits.getdata(image_file)
c = plt.imshow(image_data)
plt.colorbar(c)
plt.savefig(f"{OUTPUT_DIR}/image{i}.png")
