import pathlib
import pickle
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import scipy.stats as stats
from torch.utils.data import Dataset
import fitsio

class SloanDigitalSkySurvey(Dataset):

    # this is adapted from
    # https://github.com/jeff-regier/celeste_net/blob/935fbaa96d8da01dd7931600dee059bf6dd11292/datasets.py#L10
    # to run on a specified run, camcol, field, and band
    # returns one 1 x 1489 x 2048 image 
    def __init__(self, sdssdir = '../sdss_stage_dir/',
                 run = 3900, camcol = 6, field = 269, band = 2):

        super(SloanDigitalSkySurvey, self).__init__()
        self.sdss_path = pathlib.Path(sdssdir)

        self.rcfgs = []

        # meta data for the run + camcol
        pf_file = "photoField-{:06d}-{:d}.fits".format(run, camcol)
        camcol_path = self.sdss_path.joinpath(str(run), str(camcol))
        pf_path = camcol_path.joinpath(pf_file)

        print(pf_path)

        pf_fits = fitsio.read(pf_path)

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

        cache_path = field_dir.joinpath("cache.pkl")
        if cache_path.exists():
            return pickle.load(cache_path.open("rb"))

        for b, bl in enumerate("ugriz"):
            if b != 2:
                # taking only the green band
                continue

            frame_name = "frame-{}-{:06d}-{:d}-{:04d}.fits".format(bl, run, camcol, field)
            print("loading", frame_name)
            frame_path = str(field_dir.joinpath(frame_name))
            frame = fitsio.FITS(frame_path)

            calibration = frame[1].read()
            nelec_per_nmgy = gain[b] / calibration

            sky_small, = frame[2]["ALLSKY"].read()
            sky_x, = frame[2]["XINTERP"].read()
            sky_y, = frame[2]["YINTERP"].read()

            small_rows = np.mgrid[0:sky_small.shape[0]]
            small_cols = np.mgrid[0:sky_small.shape[1]]
            sky_interp = RegularGridInterpolator((small_rows, small_cols), sky_small, method="nearest")

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

            frame.close()

        ret = {'image': np.stack(image_list),
               'background': np.stack(background_list)}
        pickle.dump(ret, field_dir.joinpath("cache.pkl").open("wb+"))

        return ret
