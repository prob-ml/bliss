import pathlib
import pickle
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from torch.utils.data import Dataset
from astropy.io import fits


class SloanDigitalSkySurvey(Dataset):
    def __init__(self, sdss_dir, run=3900, camcol=6, field=269, bands=[2]):
        super(SloanDigitalSkySurvey, self).__init__()

        self.sdss_path = pathlib.Path(sdss_dir)
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
