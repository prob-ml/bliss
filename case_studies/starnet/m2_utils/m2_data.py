import pathlib
import os

import pickle
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import scipy.stats as stats
import torch
from torch.utils.data import Dataset

from astropy.io import fits
from astropy.wcs import WCS

import matplotlib.pyplot as plt

def _get_mgrid2(slen0, slen1):
    offset0 = (slen0 - 1) / 2
    offset1 = (slen1 - 1) / 2
    x, y = np.mgrid[-offset0:(offset0 + 1), -offset1:(offset1 + 1)]
    # return torch.Tensor(np.dstack((x, y))) / offset
    return torch.Tensor(np.dstack((y, x))) / torch.Tensor([[[offset1, offset0]]])

class SloanDigitalSkySurvey(Dataset):

    # this is adapted from
    # https://github.com/jeff-regier/celeste_net/blob/935fbaa96d8da01dd7931600dee059bf6dd11292/datasets.py#L10
    # to run on a specified run, camcol, field, and band
    # returns one 1 x 1489 x 2048 image
    def __init__(self, sdssdir = '../sdss_stage_dir/',
                 run = 3900, camcol = 6, field = 269, bands = [2]):

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
        # if cache_path.exists():
        #     print('loading cached sdss image from ', cache_path)
        #     return pickle.load(cache_path.open("rb"))

        for b, bl in enumerate("ugriz"):
            if not(b in self.bands):
                continue

            frame_name = "frame-{}-{:06d}-{:d}-{:04d}.fits".format(bl, run, camcol, field)
            frame_path = str(field_dir.joinpath(frame_name))
            print("loading sdss image from", frame_path)
            frame = fits.open(frame_path)

            calibration = frame[1].data
            nelec_per_nmgy = gain[b] / calibration

            (sky_small,) = frame[2].data["ALLSKY"]
            (sky_x,) = frame[2].data["XINTERP"]
            (sky_y,) = frame[2].data["YINTERP"]

            small_rows = np.mgrid[0:sky_small.shape[0]]
            small_cols = np.mgrid[0:sky_small.shape[1]]
            sky_interp = RegularGridInterpolator((small_rows, small_cols), sky_small, method="nearest")

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

        ret = {'image': np.stack(image_list),
               'background': np.stack(background_list),
               'nelec_per_nmgy': np.stack(nelec_per_nmgy_list),
               'gain': np.stack(gain_list),
               'calibration': np.stack(calibration_list)}
        pickle.dump(ret, field_dir.joinpath("cache.pkl").open("wb+"))

        return ret

def convert_mag_to_nmgy(mag):
    return 10**((22.5 - mag) / 2.5)

def convert_nmgy_to_mag(nmgy):
    return 22.5 - 2.5 * torch.log10(nmgy)

def load_m2_data(sdss_dir = '../../data/sdss/',
                    hubble_dir = './hubble_data/',
                    f_min = 1000.): 
    # returns the SDSS image of M2 in the r and i bands
    # along with the corresponding Hubble catalog
    
    #####################
    # Load SDSS data
    #####################
    run = 2583
    camcol = 2
    field = 136

    sdss_data = SloanDigitalSkySurvey(sdss_dir,
                                      run = run,
                                      camcol = camcol,
                                      field = field,
                                      # returns the r and i band
                                      bands = [2, 3])
    
    # the full SDSS image, ~1500 x 2000 pixels
    sdss_image = torch.Tensor(sdss_data[0]['image'])
    sdss_background = torch.Tensor(sdss_data[0]['background'])

    #####################
    # load hubble catalog
    #####################
    hubble_cat_file = hubble_dir + \
                        'hlsp_acsggct_hst_acs-wfc_ngc7089_r.rdviq.cal.adj.zpt'
    print('loading hubble data from ', hubble_cat_file)
    HTcat = np.loadtxt(hubble_cat_file, skiprows=True)

    # hubble magnitude
    hubble_rmag = HTcat[:,9]
    # right ascension and declination
    hubble_ra = HTcat[:,21]
    hubble_dc = HTcat[:,22]

    # convert hubble r.a and declination to pixel coordinates
    # (0, 0) is top left of sdss_image
    frame_name = "frame-{}-{:06d}-{:d}-{:04d}.fits".format('r', run, camcol, field)
    field_dir = pathlib.Path(sdss_dir).joinpath(str(run), str(camcol), str(field))
    frame_path = str(field_dir.joinpath(frame_name))
    print('getting sdss coordinates from: ', frame_path)
    hdulist = fits.open(str(frame_path))
    wcs = WCS(hdulist['primary'].header)
    # NOTE: pix_coordinates are (column x row), i.e. pix_coord[0] corresponds to a column
    pix_coordinates = \
        wcs.wcs_world2pix(hubble_ra, hubble_dc, 0, ra_dec_order = True)
    hubble_locs_x0 = pix_coordinates[1] # the row of pixel
    hubble_locs_x1 = pix_coordinates[0] # the column of pixel
    
    hubble_locs = np.stack([hubble_locs_x0, hubble_locs_x1]).transpose(1, 0)
    
    # convert hubble magnitude to n_electron count
    # only take r band
    nelec_per_nmgy = sdss_data[0]['nelec_per_nmgy'][0].squeeze()
    which_cols = np.floor(hubble_locs_x1 / len(nelec_per_nmgy)).astype(int)
    hubble_nmgy = convert_mag_to_nmgy(hubble_rmag)
       
    hubble_r_fluxes = hubble_nmgy * nelec_per_nmgy[which_cols]
    
    hubble_fluxes = np.stack([hubble_r_fluxes, 
                              hubble_r_fluxes]).transpose(1, 0)

    #####################
    # using hubble ground truth locations,
    # align i-band with r-band 
    #####################
    frame_name_i = "frame-{}-{:06d}-{:d}-{:04d}.fits".format('i',
                                            run, camcol, field)
    frame_path_i = str(field_dir.joinpath(frame_name_i))
    print('\n aligning images. \n Getting sdss coordinates from: ', frame_path_i)
    hdu = fits.open(str(frame_path_i))
    wcs_other = WCS(hdu['primary'].header)

    # get pixel coords
    pix_coordinates_other = wcs_other.wcs_world2pix(hubble_ra,
                                                    hubble_dc, 0, 
                                                    ra_dec_order = True)
    
    # estimate the amount to shift
    shift_x0 = np.median(hubble_locs_x0 - pix_coordinates_other[1]) / (sdss_image.shape[-2] - 1)
    shift_x1 = np.median(hubble_locs_x1 - pix_coordinates_other[0]) / (sdss_image.shape[-1] - 1)
    shift = torch.Tensor([[[[shift_x1, shift_x0 ]]]]) * 2
    
    # align image
    grid = _get_mgrid2(sdss_image.shape[-2],
                       sdss_image.shape[-1]).unsqueeze(0) - shift
    sdss_image[1] = \
        torch.nn.functional.grid_sample(sdss_image[1].unsqueeze(0).unsqueeze(0),
                                        grid, align_corners=True).squeeze()
    
    
    hubble_catalog = dict(locs = hubble_locs,
                          fluxes = hubble_fluxes)
    
    return sdss_image, hubble_catalog, sdss_background