# flake8: noqa
import os

import numpy as np
import requests
from astropy import units
from numpy.core.defchararray import startswith
from pyvo.dal import sia

from case_studies.galaxy_clustering import cluster_utils as utils

DES_DATAPATH = os.environ["BLISS_HOME"] + "/case_studies/galaxy_clustering/data/DES_images"
if not os.path.exists(DES_DATAPATH):
    os.makedirs(DES_DATAPATH)

DATASET = "des_dr2"
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
    dec_radian = dec * np.pi / 180
    img_table = svc.search((ra, dec), (fov / np.cos(dec_radian), fov), verbosity=2).to_table()
    sel = (
        (img_table["proctype"] == "Stack")
        & (img_table["prodtype"] == "image")
        & (startswith(img_table["obs_bandpass"].astype(str), band))
    )
    row = img_table[sel][0]
    url = row["access_url"]  # get the download URL
    filename = DES_DATAPATH + "/" + str(ra) + "_" + str(dec) + "_" + band + ".fits"
    if not url.lower().startswith("http"):
        raise ValueError("URL must start with http")
    response = requests.get(url, timeout=200)
    if response.status_code == 200:
        with open(filename, "wb") as file:
            file.write(response.content)
    else:
        raise ValueError("Failed to download file.")
