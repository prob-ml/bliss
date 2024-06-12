import os
from urllib import request

import cluster_utils as utils
import numpy as np
from astropy import units
from numpy.core.defchararray import startswith
from pyvo.dal import sia

DES_DATAPATH = os.getcwd() + "/data/DES_images"
if not os.path.exists(DES_DATAPATH):
    os.makedirs(DES_DATAPATH)

DATASET = "des_dr1"
DEF_ACCESS_URL = "https://datalab.noirlab.edu/sia/" + DATASET
svc = sia.SIAService(DEF_ACCESS_URL)


def compute_fov(m500, z):
    m200 = utils.m500_to_m200(m500, z) * units.solMass
    r200 = utils.m200_to_r200(m200, z)
    da = utils.angular_diameter_distance(z)
    fov = (r200 / da) * (360 / (2 * np.pi))
    return 2 * fov.value


def download_image(m500, z, ra, dec, band):
    fov = compute_fov(m500, z)
    img_table = svc.search(
        (ra, dec), (fov / np.cos(dec * np.pi / 180), fov), verbosity=2
    ).to_table()
    sel = (
        (img_table["proctype"] == "Stack")
        & (img_table["prodtype"] == "image")
        & (startswith(img_table["obs_bandpass"].astype(str), band))
    )
    row = img_table[sel][0]
    url = row["access_url"]  # get the download URL
    filename = DES_DATAPATH + "/" + str(ra) + "_" + str(dec) + "_" + band + ".fits"
    request.urlretrieve(url, filename)
