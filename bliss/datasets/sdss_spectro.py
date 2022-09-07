import warnings
import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from astropy.io import fits
from einops import rearrange
from scipy.interpolate import RegularGridInterpolator
from torch import Tensor
from torch.utils.data import Dataset
from functools import reduce


def convert_mag_to_flux(mag: Tensor, nelec_per_nmgy=987.31) -> Tensor:
    # default corresponds to average value of columns for run 94, camcol 1, field 12
    return 10 ** ((22.5 - mag) / 2.5) * nelec_per_nmgy


def convert_flux_to_mag(flux: Tensor, nelec_per_nmgy=987.31) -> Tensor:
    # default corresponds to average value of columns for run 94, camcol 1, field 12
    return 22.5 - 2.5 * torch.log10(flux / nelec_per_nmgy)

class SloanDigitalSkySurveySpectro(Dataset):
    def __init__(
        self,
        spec_info_loc="data/sdss/specObj-dr16.fits",
        raw_data_dir="data/sdss"
        
    ):
        super().__init__()
        self.spec_info_loc = spec_info_loc
        self.raw_data_dir = raw_data_dir
        self.f = fits.open(self.spec_info_loc)
        self.filters = {
            'ZWARNING': 1,
            'PLATEQUALITY': 'bad',
            'SPECPRIMARY': 0,
        }
        self.z_max = 1.
        self.train_size = 10

    def _get_objs(self):
        to_intersect = [np.where(self.f[1].data[:][key] != value) for key,value in self.filters.items()]
        to_intersect.append(np.where(self.f[1].data[:]['Z'] <= self.z_max))
        to_intersect.append(np.where(self.f[1].data[:]['Z'] >= 0.))
        indices = reduce(np.intersect1d, to_intersect)[:self.train_size]

        return pd.DataFrame({
            'MJD': self.f[1].data[indices]['MJD'],
            'PLATE': self.f[1].data[indices]['PLATE'],
            'FIBERID': self.f[1].data[indices]['FIBERID'],
            'REDSHIFT': self.f[1].data[indices]['Z'],
        })

    def _make_download_script(self, objs):
        to_download = open('{}/sed_download.txt'.format(self.raw_data_dir), 'w')
        for i in range(len(objs)):
            plate = objs.loc[i,'PLATE']
            mjd = objs.loc[i,'MJD']
            fiberid = objs.loc[i,'FIBERID']
            file_name1 = "https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/lite/{0}/spec-{0}-{1}-{2}.fits".format(f"{plate:04d}", mjd, f"{fiberid:04d}")
            file_name2 = "https://data.sdss.org/sas/dr16/sdss/spectro/redux/103/spectra/lite/{0}/spec-{0}-{1}-{2}.fits".format(f"{plate:04d}", mjd, f"{fiberid:04d}")
            file_name3 = "https://data.sdss.org/sas/dr16/sdss/spectro/redux/104/spectra/lite/{0}/spec-{0}-{1}-{2}.fits".format(f"{plate:04d}", mjd, f"{fiberid:04d}")
            file_name4 = "https://data.sdss.org/sas/dr16/sdss/spectro/redux/v5_13_0/spectra/lite/{0}/spec-{0}-{1}-{2}.fits".format(f"{plate:04d}", mjd, f"{fiberid:04d}")
            to_download.write(file_name1 + '\n')
            to_download.write(file_name2 + '\n')
            to_download.write(file_name3 + '\n')
            to_download.write(file_name4 + '\n')
        to_download.close()

    def _download(self):
        objs = self._get_objs()
        self._make_download_script(objs)
        script_location = '{}/sed_download.txt'.format(self.raw_data_dir)
        raw_sed_location = '{}/raw_sed/'.format(self.raw_data_dir)
        os.system("wget -i {} -P {}".format(script_location, raw_sed_location))
        self.f.close()
        return



        