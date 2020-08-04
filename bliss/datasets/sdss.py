import pathlib
import os

import pickle
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import torch
from torch.utils.data import Dataset

from astropy.io import fits
from astropy.wcs import WCS


# Reconstruct the SDSS model PSF from KL basis functions.
#   hdu: the psField hdu for the band you are looking at.
#      eg, for r-band:
# 	     psfield = pyfits.open('psField-%06i-%i-%04i.fit' % (run,camcol,field))
#        bandnum = 'ugriz'.index('r')
# 	     hdu = psfield[bandnum+1]
#
#   x,y can be scalars or 1-d numpy arrays.
# Return value:
#    if x,y are scalars: a PSF image
#    if x,y are arrays:  a list of PSF images
def psf_at_points(x, y, psf_fit_file):
    psfield = fits.open(psf_fit_file)
    hdu = psfield[3]
    psf = hdu.data

    rtnscalar = np.isscalar(x) and np.isscalar(y)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    psfimgs = None
    (outh, outw) = (None, None)

    # From the IDL docs:
    # http://photo.astro.princeton.edu/photoop_doc.html#SDSS_PSF_RECON
    #   acoeff_k = SUM_i{ SUM_j{ (0.001*ROWC)^i * (0.001*COLC)^j * C_k_ij } }
    #   psfimage = SUM_k{ acoeff_k * RROWS_k }
    for k in range(len(psf)):
        nrb = psf[k]["nrow_b"]
        ncb = psf[k]["ncol_b"]

        c = psf[k]["c"].reshape(5, 5)
        c = c[:nrb, :ncb]

        (gridi, gridj) = np.meshgrid(range(nrb), range(ncb))

        if psfimgs is None:
            psfimgs = [np.zeros_like(hdu["rrows"][k][0]) for xy in np.broadcast(x, y)]
            (outh, outw) = (hdu["rnrow"][k][0], hdu["rncol"][k][0])

        for i, (xi, yi) in enumerate(np.broadcast(x, y)):
            acoeff_k = sum(((0.001 * xi) ** gridi * (0.001 * yi) ** gridj * c))
            psfimgs[i] += acoeff_k * hdu["rrows"][k][0]

    psfimgs = [img.reshape((outh, outw)) for img in psfimgs]

    if rtnscalar:
        return psfimgs[0]
    return psfimgs


class SloanDigitalSkySurvey(Dataset):

    # this is adapted from
    # https://github.com/jeff-regier/celeste_net/blob/935fbaa96d8da01dd7931600dee059bf6dd11292/datasets.py#L10
    # to run on a specified run, camcol, field, and band
    # returns one 1 x 1489 x 2048 image
    def __init__(
        self, sdssdir="../sdss_stage_dir/", run=3900, camcol=6, field=269, bands=[2]
    ):

        super(SloanDigitalSkySurvey, self).__init__()
        self.sdss_path = pathlib.Path(sdssdir)

        self.rcfgs = []

        self.bands = bands

        # meta data for the run + camcol
        pf_file = "photoField-{:06d}-{:d}.fits".format(run, camcol)
        camcol_path = self.sdss_path.joinpath(str(run), str(camcol))
        pf_path = camcol_path.joinpath(pf_file)
        pf_fits = fits.getdata(pf_path)

        fieldnums = pf_fits["FIELD"]
        fieldgains = pf_fits["GAIN"]

        # get desired field
        for i in range(len(fieldnums)):
            _field = fieldnums[i]
            gain = fieldgains[i]
            if _field == field:
                self.rcfgs.append((run, camcol, field, gain))

        self.items = [None] * len(self.rcfgs)

    def __len__(self):
        return len(self.rcfgs)

    def __getitem__(self, idx):
        if not self.items[idx]:
            self.items[idx] = self.get_from_disk(idx)
        return self.items[idx]

    def get_from_disk(self, idx):
        run, camcol, field, gain = self.rcfgs[idx]

        camcol_dir = self.sdss_path.joinpath(str(run), str(camcol))
        field_dir = camcol_dir.joinpath(str(field))

        image_list = []
        background_list = []
        nelec_per_nmgy_list = []
        calibration_list = []
        gain_list = []

        cache_path = field_dir.joinpath("cache.pkl")

        for b, bl in enumerate("ugriz"):
            if not (b in self.bands):
                continue

            frame_name = "frame-{}-{:06d}-{:d}-{:04d}.fits".format(
                bl, run, camcol, field
            )
            frame_path = str(field_dir.joinpath(frame_name))
            print("loading sdss image from", frame_path)
            frame = fits.open(frame_path)

            calibration = frame[1].read()
            nelec_per_nmgy = gain[b] / calibration

            (sky_small,) = frame[2]["ALLSKY"].read()
            (sky_x,) = frame[2]["XINTERP"].read()
            (sky_y,) = frame[2]["YINTERP"].read()

            small_rows = np.mgrid[0 : sky_small.shape[0]]
            small_cols = np.mgrid[0 : sky_small.shape[1]]
            sky_interp = RegularGridInterpolator(
                (small_rows, small_cols), sky_small, method="nearest"
            )

            sky_y = sky_y.clip(0, sky_small.shape[0] - 1)
            sky_x = sky_x.clip(0, sky_small.shape[1] - 1)
            large_points = np.stack(np.meshgrid(sky_y, sky_x)).transpose()
            large_sky = sky_interp(large_points)
            large_sky_nelec = large_sky * gain[b]

            pixels_ss_nmgy = frame[0].read()
            pixels_ss_nelec = pixels_ss_nmgy * nelec_per_nmgy
            pixels_nelec = pixels_ss_nelec + large_sky_nelec

            image_list.append(pixels_nelec)
            background_list.append(large_sky_nelec)

            gain_list.append(gain[b])
            nelec_per_nmgy_list.append(nelec_per_nmgy)
            calibration_list.append(calibration)

            frame.close()

        ret = {
            "image": np.stack(image_list),
            "background": np.stack(background_list),
            "nelec_per_nmgy": np.stack(nelec_per_nmgy_list),
            "gain": np.stack(gain_list),
            "calibration": np.stack(calibration_list),
        }
        pickle.dump(ret, field_dir.joinpath("cache.pkl").open("wb+"))

        return ret
