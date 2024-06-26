import glob
import sys

from astropy.io import fits


def stack_fits():
    '''Stacks all galsim .fits files into one file.'''

    input_fits_files = glob.glob(sys.argv[1] + "/galsim_iter*.fits")

    # Create a list to hold the HDUs
    hdulist = []

    # Open the first FITS file and set it as the primary HDU
    primary_hdu = fits.open(input_fits_files[0])[0]  # Open the file and get the primary HDU
    hdulist.append(fits.PrimaryHDU(data=primary_hdu.data, header=primary_hdu.header))  # pylint: disable=E1101

    # Loop through the remaining input files and add them as ImageHDUs
    for file in input_fits_files[1:]:
        hdu = fits.open(file)[0]  # Open the file and get the primary HDU
        image_hdu = fits.ImageHDU(data=hdu.data, header=hdu.header)  # pylint: disable=E1101
        hdulist.append(image_hdu)

    # Write the combined HDUs to a new FITS file
    output_file = "data/combined_images.fits"
    hdul = fits.HDUList(hdulist)
    hdul.writeto(output_file, overwrite=True)

if __name__=="__main__":
    stack_fits()
