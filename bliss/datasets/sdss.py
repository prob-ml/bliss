import pathlib
import warnings

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from einops import rearrange
from scipy.interpolate import RegularGridInterpolator
from torch.utils.data import Dataset


def convert_mag_to_flux(mag, nelec_per_nmgy=987.31):
    # default corresponds to average value of columns for run 94, camcol 1, field 12
    return 10 ** ((22.5 - mag) / 2.5) * nelec_per_nmgy


def convert_flux_to_mag(flux, nelec_per_nmgy=987.31):
    # default corresponds to average value of columns for run 94, camcol 1, field 12
    return 22.5 - 2.5 * np.log10(flux / nelec_per_nmgy)


class SloanDigitalSkySurvey(Dataset):
    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        sdss_dir="data/sdss",
        run=3900,
        camcol=6,
        fields=(269,),
        bands=(0, 1, 2, 3, 4),
    ):
        super().__init__()

        self.sdss_path = pathlib.Path(sdss_dir)
        self.rcfgcs = []
        self.bands = bands
        pf_file = f"photoField-{run:06d}-{camcol:d}.fits"
        camcol_path = self.sdss_path.joinpath(str(run), str(camcol))
        pf_path = camcol_path.joinpath(pf_file)
        self.pf_fits = fits.getdata(pf_path)

        fieldnums = self.pf_fits["FIELD"]
        fieldgains = self.pf_fits["GAIN"]

        # get desired field
        for i, field in enumerate(fieldnums):
            gain = fieldgains[i]
            if (not fields) or field in fields:
                self.rcfgcs.append((run, camcol, field, gain))
        self.items = [None] * len(self.rcfgcs)

    def __len__(self):
        return len(self.rcfgcs)

    def __getitem__(self, idx):
        if not self.items[idx]:
            self.items[idx] = self.get_from_disk(idx)
        return self.items[idx]

    def get_from_disk(self, idx, verbose=False):
        # pylint: disable=too-many-statements
        if self.rcfgcs[idx] is None:
            return None
        run, camcol, field, gain = self.rcfgcs[idx]

        camcol_dir = self.sdss_path.joinpath(str(run), str(camcol))
        field_dir = camcol_dir.joinpath(str(field))

        image_list = []
        background_list = []
        nelec_per_nmgy_list = []
        calibration_list = []
        gain_list = []
        wcs_list = []

        for b, bl in enumerate("ugriz"):
            if b not in self.bands:
                continue

            frame_name = f"frame-{bl}-{run:06d}-{camcol:d}-{field:04d}.fits"
            frame_path = str(field_dir.joinpath(frame_name))
            if verbose:
                print("loading sdss image from", frame_path)
            frame = fits.open(frame_path)
            calibration = frame[1].data  # pylint: disable=maybe-no-member
            nelec_per_nmgy = gain[b] / calibration

            sky_small = frame[2].data["ALLSKY"][0]  # pylint: disable=maybe-no-member
            sky_x = frame[2].data["XINTERP"][0]  # pylint: disable=maybe-no-member
            sky_y = frame[2].data["YINTERP"][0]  # pylint: disable=maybe-no-member

            small_rows = np.mgrid[0 : sky_small.shape[0]]
            small_cols = np.mgrid[0 : sky_small.shape[1]]
            small_rcs = (small_rows, small_cols)
            sky_interp = RegularGridInterpolator(small_rcs, sky_small, method="nearest")

            sky_y = sky_y.clip(0, sky_small.shape[0] - 1)
            sky_x = sky_x.clip(0, sky_small.shape[1] - 1)
            large_points = rearrange(np.meshgrid(sky_y, sky_x), "n x y -> y x n")
            large_sky = sky_interp(large_points)
            large_sky_nelec = large_sky * gain[b]

            pixels_ss_nmgy = frame[0].data  # pylint: disable=maybe-no-member
            pixels_ss_nelec = pixels_ss_nmgy * nelec_per_nmgy
            pixels_nelec = pixels_ss_nelec + large_sky_nelec

            image_list.append(pixels_nelec)
            background_list.append(large_sky_nelec)

            gain_list.append(gain[b])
            nelec_per_nmgy_list.append(nelec_per_nmgy)
            calibration_list.append(calibration)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FITSFixedWarning)
                wcs = WCS(frame[0])
            wcs_list.append(wcs)

            frame.close()

        return {
            "image": np.stack(image_list),
            "field": field,
            "background": np.stack(background_list),
            "nelec_per_nmgy": np.stack(nelec_per_nmgy_list),
            "gain": np.stack(gain_list),
            "calibration": np.stack(calibration_list),
            "wcs": wcs_list,
        }
