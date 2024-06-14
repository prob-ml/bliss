from astropy.io import fits
import matplotlib.pyplot as plt

import numpy as np

# Set up matplotlib
import matplotlib.pyplot as plt

from astropy.io import fits
from matplotlib.colors import LogNorm
import sys

import glob


catalog_file = "data/real_galaxy_catalog_combined.fits"

# Open the FITS file
with fits.open(catalog_file, mode='update') as hdulist:
    # Access the data in the second HDU (index 1)
    catalog_data = hdulist[1].data
    
    # Print the original value of the specific field (assuming the 7th column/field, index 6)
    print("Original value:", catalog_data[0][6])
    
    # Update the field with the new value
    for row in catalog_data:
        row[6] = 'combined_images.fits'
    
    # Print the updated value to confirm the change
    print("Updated value:", catalog_data[0][6])
    
    # Save the changes to the FITS file
    hdulist.flush()  # This writes the changes back to the file

# Verify the change by reopening the file and checking the field value
with fits.open(catalog_file) as hdulist:
    updated_data = hdulist[1].data
    for row in updated_data[0:2]:
        print(row)

