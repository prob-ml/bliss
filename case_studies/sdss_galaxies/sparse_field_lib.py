import numpy as np
import torch

import fitsio

from astropy.io import fits
from astropy.wcs import WCS

from bliss.datasets import sdss


def load_data(
    catalog_file="coadd_field_catalog_runjing_liu.fit",
    sdss_dir="../../data/sdss/",
    run=94,
    camcol=1,
    field=12,
    bands=(2,),
):

    assert len(bands) == 1, "Only 1 band is supported until we align them."
    n_bands = len(bands)

    band_letters = ["ugriz"[bands[i]] for i in range(n_bands)]

    ##################
    # load sdss data
    ##################
    sdss_data = sdss.SloanDigitalSkySurvey(
        sdss_dir=sdss_dir,
        run=run,
        camcol=camcol,
        fields=(field,),
        bands=bands,
        overwrite_cache=True,
        overwrite_fits_cache=True,
    )

    image = torch.Tensor(sdss_data[0]["image"])
    slen0 = image.shape[-2]
    slen1 = image.shape[-1]

    ##################
    # load coordinate files
    ##################
    frame_names = [
        "frame-{}-{:06d}-{:d}-{:04d}.fits".format(band_letters[i], run, camcol, field)
        for i in range(n_bands)
    ]

    wcs_list = []
    for i in range(n_bands):
        hdulist = fits.open(
            sdss_dir + str(run) + "/" + str(camcol) + "/" + str(field) + "/" + frame_names[i]
        )
        wcs_list.append(WCS(hdulist["primary"].header))

    min_coords = wcs_list[0].wcs_pix2world(np.array([[0, 0]]), 0)
    max_coords = wcs_list[0].wcs_pix2world(np.array([[slen1, slen0]]), 0)

    ##################
    # load catalog
    ##################
    fits_file = fitsio.FITS(catalog_file)[1]
    true_ra = fits_file["ra"][:]
    true_decl = fits_file["dec"][:]

    # make sure our catalog covers the whole image
    assert true_ra.min() < min_coords[0, 0]
    assert true_ra.max() > max_coords[0, 0]

    assert true_decl.min() < min_coords[0, 1]
    assert true_decl.max() > max_coords[0, 1]

    return image, fits_file, wcs_list, sdss_data
