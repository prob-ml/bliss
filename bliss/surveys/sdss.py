import warnings
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from einops import rearrange
from scipy.interpolate import RegularGridInterpolator
from torch import Tensor
from torch.utils.data import Dataset

from bliss.catalog import FullCatalog


def convert_mag_to_flux(mag: Tensor, nelec_per_nmgy=987.31) -> Tensor:
    # default corresponds to average value of columns for run 94, camcol 1, field 12
    return 10 ** ((22.5 - mag) / 2.5) * nelec_per_nmgy


def convert_flux_to_mag(flux: Tensor, nelec_per_nmgy=987.31) -> Tensor:
    # default corresponds to average value of columns for run 94, camcol 1, field 12
    return 22.5 - 2.5 * torch.log10(flux / nelec_per_nmgy)


def column_to_tensor(table, colname):
    dtypes = {
        np.dtype(">i2"): int,
        np.dtype(">i4"): int,
        np.dtype(">i8"): int,
        np.dtype("bool"): bool,
        np.dtype(">f4"): np.float32,
        np.dtype(">f8"): np.float32,
        np.dtype("float32"): np.float32,
        np.dtype("float64"): np.dtype("float64"),
    }
    x = np.array(table[colname])
    dtype = dtypes[x.dtype]
    x = x.astype(dtype)
    return torch.from_numpy(x)


class SloanDigitalSkySurvey(Dataset):
    def __init__(
        self,
        sdss_dir="data/sdss",
        run=3900,
        camcol=6,
        fields=(269,),
        bands=(0, 1, 2, 3, 4),
    ):
        super().__init__()

        self.sdss_path = Path(sdss_dir)
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
        self.items = [None for _ in range(len(self.rcfgcs))]

    def __len__(self):
        return len(self.rcfgcs)

    def __getitem__(self, idx):
        if not self.items[idx]:
            self.items[idx] = self.get_from_disk(idx)
        return self.items[idx]

    def get_from_disk(self, idx):
        if self.rcfgcs[idx] is None:
            return None
        run, camcol, field, gain = self.rcfgcs[idx]

        camcol_dir = self.sdss_path.joinpath(str(run), str(camcol))
        field_dir = camcol_dir.joinpath(str(field))
        frame_list = []

        for b, bl in enumerate("ugriz"):
            if b in self.bands:
                frame = self.read_frame_for_band(bl, field_dir, run, camcol, field, gain[b])
                frame_list.append(frame)

        ret = {}
        for k in frame_list[0]:
            data_per_band = [frame[k] for frame in frame_list]
            if isinstance(data_per_band[0], np.ndarray):
                ret[k] = np.stack(data_per_band)
            else:
                ret[k] = data_per_band
        ret.update({"field": field})
        return ret

    def read_frame_for_band(self, bl, field_dir, run, camcol, field, gain):
        frame_name = f"frame-{bl}-{run:06d}-{camcol:d}-{field:04d}.fits"
        frame_path = str(field_dir.joinpath(frame_name))
        frame = fits.open(frame_path)
        calibration = frame[1].data  # pylint: disable=maybe-no-member
        nelec_per_nmgy = gain / calibration

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
        large_sky_nelec = large_sky * gain

        pixels_ss_nmgy = frame[0].data  # pylint: disable=maybe-no-member
        pixels_ss_nelec = pixels_ss_nmgy * nelec_per_nmgy
        pixels_nelec = pixels_ss_nelec + large_sky_nelec

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FITSFixedWarning)
            wcs = WCS(frame[0])

        frame.close()
        return {
            "image": pixels_nelec,
            "background": large_sky_nelec,
            "gain": np.array(gain),
            "nelec_per_nmgy_list": nelec_per_nmgy,
            "calibration": calibration,
            "wcs": wcs,
        }


class PhotoFullCatalog(FullCatalog):
    """Class for the SDSS PHOTO Catalog.

    Some resources:
    - https://www.sdss.org/dr12/algorithms/classify/
    - https://www.sdss.org/dr12/algorithms/resolve/
    """

    @classmethod
    def from_file(cls, sdss_path, run, camcol, field, band):
        sdss_path = Path(sdss_path)
        camcol_dir = sdss_path / str(run) / str(camcol) / str(field)
        po_path = camcol_dir / f"photoObj-{run:06d}-{camcol:d}-{field:04d}.fits"
        po_fits = fits.getdata(po_path)
        objc_type = column_to_tensor(po_fits, "objc_type")
        thing_id = column_to_tensor(po_fits, "thing_id")
        ras = column_to_tensor(po_fits, "ra")
        decs = column_to_tensor(po_fits, "dec")
        galaxy_bools = (objc_type == 3) & (thing_id != -1)
        star_bools = (objc_type == 6) & (thing_id != -1)
        star_fluxes = column_to_tensor(po_fits, "psfflux") * star_bools.reshape(-1, 1)
        star_mags = column_to_tensor(po_fits, "psfmag") * star_bools.reshape(-1, 1)
        galaxy_fluxes = column_to_tensor(po_fits, "cmodelflux") * galaxy_bools.reshape(-1, 1)
        galaxy_mags = column_to_tensor(po_fits, "cmodelmag") * galaxy_bools.reshape(-1, 1)
        fluxes = star_fluxes + galaxy_fluxes
        mags = star_mags + galaxy_mags
        keep = galaxy_bools | star_bools
        galaxy_bools = galaxy_bools[keep]
        star_bools = star_bools[keep]
        ras = ras[keep]
        decs = decs[keep]
        fluxes = fluxes[keep][:, band]
        mags = mags[keep][:, band]

        sdss = SloanDigitalSkySurvey(sdss_path, run, camcol, fields=(field,), bands=(band,))
        wcs: WCS = sdss[0]["wcs"][0]

        # get pixel coordinates
        pts = []
        prs = []
        for ra, dec in zip(ras, decs):
            pt, pr = wcs.wcs_world2pix(ra, dec, 0)
            pts.append(float(pt))
            prs.append(float(pr))
        pts = torch.tensor(pts) + 0.5  # For consistency with BLISS
        prs = torch.tensor(prs) + 0.5
        plocs = torch.stack((prs, pts), dim=-1)
        nobj = plocs.shape[0]

        d = {
            "plocs": plocs.reshape(1, nobj, 2),
            "n_sources": torch.tensor((nobj,)),
            "galaxy_bools": galaxy_bools.reshape(1, nobj, 1).float(),
            "star_bools": star_bools.reshape(1, nobj, 1).float(),
            "fluxes": fluxes.reshape(1, nobj, 1),
            "mags": mags.reshape(1, nobj, 1),
            "ra": ras.reshape(1, nobj, 1),
            "dec": decs.reshape(1, nobj, 1),
        }

        height = sdss[0]["image"].shape[1]
        width = sdss[0]["image"].shape[2]

        return cls(height, width, d)
