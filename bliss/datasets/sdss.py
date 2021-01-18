import torch
import pathlib
import pickle
import warnings
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from torch.utils.data import Dataset
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning


class SloanDigitalSkySurvey(Dataset):
    def __init__(self, sdss_dir, run=3900, camcol=6, field=None, bands=[2]):
        super(SloanDigitalSkySurvey, self).__init__()

        self.sdss_path = pathlib.Path(sdss_dir)
        self.rcfgcs = []
        self.bands = bands

        # meta data for the run + camcol
        pf_file = "photoField-{:06d}-{:d}.fits".format(run, camcol)
        camcol_path = self.sdss_path.joinpath(str(run), str(camcol))
        pf_path = camcol_path.joinpath(pf_file)
        self.pf_fits = fits.getdata(pf_path)

        fieldnums = self.pf_fits["FIELD"]
        fieldgains = self.pf_fits["GAIN"]

        # get desired field
        for i in range(len(fieldnums)):
            _field = fieldnums[i]
            gain = fieldgains[i]
            if (not field) or _field == field:
                # load the catalog distributed with SDSS
                po_file = "photoObj-{:06d}-{:d}-{:04d}.fits".format(run, camcol, field)
                po_path = camcol_path.joinpath(str(field), po_file)
                po_fits = fits.getdata(po_path)

                self.rcfgcs.append((run, camcol, field, gain, po_fits))

        self.items = [None] * len(self.rcfgcs)

    def __len__(self):
        return len(self.rcfgcs)

    def __getitem__(self, idx):
        if not self.items[idx]:
            self.items[idx] = self.get_from_disk(idx)
        return self.items[idx]

    def fetch_bright_stars(self, po_fits, img, wcs):
        is_star = po_fits["objc_type"] == 6
        is_bright = po_fits["psfflux"].sum(axis=1) > 100
        is_thing = po_fits["thing_id"] != -1
        is_target = is_star & is_bright & is_thing
        ras = po_fits["ra"][is_target]
        decs = po_fits["dec"][is_target]

        band = 2
        stamps = []
        for (ra, dec, f) in zip(ras, decs, po_fits["thing_id"][is_target]):
            # pt = "time" in pixel coordinates
            pt, pr = wcs.wcs_world2pix(ra, dec, 0)
            pt, pr = int(pt + 0.5), int(pr + 0.5)
            stamp = img[(pr - 2) : (pr + 3), (pt - 2) : (pt + 3)]
            stamps.append(stamp)

        return np.asarray(stamps)

    def get_from_disk(self, idx):
        run, camcol, field, gain, po_fits = self.rcfgcs[idx]

        camcol_dir = self.sdss_path.joinpath(str(run), str(camcol))
        field_dir = camcol_dir.joinpath(str(field))

        image_list = []
        background_list = []
        nelec_per_nmgy_list = []
        calibration_list = []
        gain_list = []
        wcs_list = []

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

            calibration = frame[1].data
            nelec_per_nmgy = gain[b] / calibration

            (sky_small,) = frame[2].data["ALLSKY"]
            (sky_x,) = frame[2].data["XINTERP"]
            (sky_y,) = frame[2].data["YINTERP"]

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

            pixels_ss_nmgy = frame[0].data
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

        stamps = self.fetch_bright_stars(po_fits, image_list[2], wcs_list[2])

        ret = {
            "image": np.stack(image_list),
            "background": np.stack(background_list),
            "nelec_per_nmgy": np.stack(nelec_per_nmgy_list),
            "gain": np.stack(gain_list),
            "calibration": np.stack(calibration_list),
            "wcs": wcs_list,
            "bright_stars": stamps,
        }
        pickle.dump(ret, field_dir.joinpath("cache.pkl").open("wb+"))

        return ret


# functions to evaluate the SDSS detections against the
# hubble catalog
def convert_mag_to_nmgy(mag):
    return 10 ** ((22.5 - mag) / 2.5)


def get_locs_error(locs, true_locs):
    # get matrix of Linf error in locations
    # truth x estimated
    return torch.abs(locs.unsqueeze(0) - true_locs.unsqueeze(1)).max(2)[0]


def get_fluxes_error(fluxes, true_fluxes):
    # get matrix of l1 error in log flux
    # truth x estimated
    return torch.abs(
        torch.log10(fluxes).unsqueeze(0) - torch.log10(true_fluxes).unsqueeze(1)
    )


def get_mag_error(mags, true_mags):
    # get matrix of l1 error in magnitude
    # truth x estimated
    return torch.abs(mags.unsqueeze(0) - true_mags.unsqueeze(1))


def convert_nmgy_to_mag(nmgy):
    return 22.5 - 2.5 * torch.log10(nmgy)


def get_summary_stats(
    est_locs, true_locs, slen, est_fluxes, true_fluxes, nelec_per_nmgy, slack=0.5
):
    est_mags = convert_nmgy_to_mag(est_fluxes / nelec_per_nmgy)
    true_mags = convert_nmgy_to_mag(true_fluxes / nelec_per_nmgy)
    mag_error = get_mag_error(est_mags, true_mags)

    locs_error = get_locs_error(est_locs * (slen - 1), true_locs * (slen - 1))
    tpr_bool = torch.any((locs_error < slack) * (mag_error < slack), dim=1).float()
    ppv_bool = torch.any((locs_error < slack) * (mag_error < slack), dim=0).float()
    return tpr_bool.mean(), ppv_bool.mean(), tpr_bool, ppv_bool
