import bz2
import gzip
import warnings
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import requests
import torch
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from einops import rearrange
from scipy.interpolate import RegularGridInterpolator
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from bliss.catalog import FullCatalog, SourceType


class SloanDigitalSkySurvey(pl.LightningDataModule, Dataset):
    def __init__(
        self,
        sdss_dir="data/sdss",
        run=3900,
        camcol=6,
        fields=(269,),
        predict_device=None,
        predict_crop=None,
        # take a way to specify labels? e.g., from sdss PhotoFullCatalog or decals catalog
    ):
        super().__init__()

        self.sdss_path = Path(sdss_dir)
        self.rcfgcs = []

        self.run = run
        self.camcol = camcol
        self.fields = fields
        self.bands = [0, 1, 2, 3, 4]

        self.downloader = SDSSDownloader(run, camcol, fields[0], download_dir=sdss_dir)

        self.prepare_data()
        assert self.items is not None, "No data found even after prepare_data()."

        self.predict_device = predict_device
        self.predict_crop = predict_crop

    def prepare_data(self):
        pf_file = f"photoField-{self.run:06d}-{self.camcol:d}.fits"
        camcol_path = self.sdss_path.joinpath(str(self.run), str(self.camcol))
        pf_path = camcol_path.joinpath(pf_file)
        if not Path(pf_path).exists():
            self.downloader.download_pf()
        msg = (
            f"{pf_path} does not exist. "
            + "Make sure data files are available for specified fields."
        )
        assert Path(pf_path).exists(), msg
        self.pf_fits = fits.getdata(pf_path)

        fieldnums = self.pf_fits["FIELD"]
        fieldgains = self.pf_fits["GAIN"]

        # get desired field
        for i, field in enumerate(fieldnums):
            gain = fieldgains[i]
            if (not self.fields) or field in self.fields:
                self.rcfgcs.append((self.run, self.camcol, field, gain))
        self.items = [None for _ in range(len(self.rcfgcs))]

        for rcfgc in self.rcfgcs:
            _, _, field, _ = rcfgc
            field_dir = camcol_path.joinpath(str(field))
            self.downloader.field = field
            for b, bl in enumerate("ugriz"):
                if b not in self.bands:
                    continue
                frame_name = f"frame-{bl}-{self.run:06d}-{self.camcol:d}-{field:04d}.fits"
                frame_path = str(field_dir.joinpath(frame_name))
                if not Path(frame_path).exists():
                    self.downloader.download_frame(band=bl)
                assert Path(frame_path).exists(), f"{frame_path} does not exist."

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

    def predict_dataloader(self):
        img = prepare_image(self[0]["image"], device=self.predict_device)
        bg = prepare_image(self[0]["background"], device=self.predict_device)
        batch = prepare_batch(img, bg, self.predict_crop)
        return DataLoader([batch], batch_size=1)


class SDSSDownloader:
    """Class for downloading SDSS data."""

    URLBASE = "https://data.sdss.org/sas/dr12/boss"
    BANDS = ("u", "g", "r", "i", "z")

    def __init__(self, run, camcol, field, download_dir):
        self.run = run
        self.camcol = camcol
        self._field = field
        self.download_dir = download_dir

        self.run_stripped = str(run).lstrip("0")
        self.field_stripped = str(field).lstrip("0")
        self.run6 = f"{int(self.run_stripped):06d}"
        self.field4 = f"{int(self.field_stripped):04d}"
        self.subdir2 = f"{self.run_stripped}/{camcol}"
        self.subdir3 = f"{self.run_stripped}/{camcol}/{self.field_stripped}"

    def download_pf(self):
        download_file(
            f"{SDSSDownloader.URLBASE}/photoObj/301/{self.run_stripped}/"
            f"photoField-{self.run6}-{self.camcol}.fits",
            f"{self.download_dir}/{self.subdir2}/" f"photoField-{self.run6}-{self.camcol}.fits",
        )

    def download_po(self):
        download_file(
            f"{SDSSDownloader.URLBASE}/photoObj/301/{self.run_stripped}/{self.camcol}/"
            f"photoObj-{self.run6}-{self.camcol}-{self.field4}.fits",
            f"{self.download_dir}/{self.subdir3}/"
            f"photoObj-{self.run6}-{self.camcol}-{self.field4}.fits",
        )

    def download_frame(self, band="r"):
        download_file(
            f"{SDSSDownloader.URLBASE}/photo/redux/301/{self.run_stripped}/objcs/{self.camcol}/"
            f"fpM-{self.run6}-{band}{self.camcol}-{self.field4}.fit.gz",
            f"{self.download_dir}/{self.subdir3}/"
            f"fpM-{self.run6}-{band}{self.camcol}-{self.field4}.fits",
            gzip.decompress,
        )

        download_file(
            f"{SDSSDownloader.URLBASE}/photoObj/frames/301/{self.run_stripped}/{self.camcol}/"
            f"frame-{band}-{self.run6}-{self.camcol}-{self.field4}.fits.bz2",
            f"{self.download_dir}/{self.subdir3}/"
            f"frame-{band}-{self.run6}-{self.camcol}-{self.field4}.fits",
            bz2.decompress,
        )

    def download_psfield(self):
        download_file(
            f"{SDSSDownloader.URLBASE}/photo/redux/301/{self.run_stripped}/objcs/{self.camcol}/"
            f"psField-{self.run6}-{self.camcol}-{self.field4}.fit",
            f"{self.download_dir}/{self.subdir3}/"
            f"psField-{self.run6}-{self.camcol}-{self.field4}.fits",
        )

    def download_all(self):
        if not Path(self.download_dir).exists():
            # create download directory
            Path(self.download_dir).mkdir(parents=True, exist_ok=True)

        self.download_pf()
        self.download_po()

        for band in SDSSDownloader.BANDS:
            self.download_frame(band)

        self.download_psfield()

    @property
    def field(self):
        return self._field

    @field.setter
    def field(self, value):
        self._field = value
        self.field_stripped = str(self._field).lstrip("0")
        self.field4 = f"{int(self.field_stripped):04d}"
        self.subdir3 = f"{self.run_stripped}/{self.camcol}/{self.field_stripped}"


class PhotoFullCatalog(FullCatalog):
    """Class for the SDSS PHOTO Catalog.

    Some resources:
    - https://www.sdss.org/dr12/algorithms/classify/
    - https://www.sdss.org/dr12/algorithms/resolve/
    """

    @classmethod
    def from_file(cls, sdss_path, run, camcol, field, sdss_obj):
        sdss_path = Path(sdss_path)
        camcol_dir = sdss_path / str(run) / str(camcol) / str(field)
        po_path = camcol_dir / f"photoObj-{run:06d}-{camcol:d}-{field:04d}.fits"
        if not Path(po_path).exists():
            SDSSDownloader(run, camcol, field, download_dir=str(sdss_path)).download_po()
        assert Path(po_path).exists(), f"File {po_path} does not exist, even after download."
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
        fluxes = fluxes[keep]
        mags = mags[keep]
        n_bands = 5

        wcs: WCS = sdss_obj[0]["wcs"][2]  # r-band WCS

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

        assert torch.all(star_bools + galaxy_bools)
        source_type = SourceType.STAR * star_bools + SourceType.GALAXY * galaxy_bools

        d = {
            "plocs": plocs.reshape(1, nobj, 2),
            "n_sources": torch.tensor((nobj,)),
            "source_type": source_type.reshape(1, nobj, 1),
            "fluxes": fluxes.reshape(1, nobj, n_bands),
            "mags": mags.reshape(1, nobj, n_bands),
            "ra": ras.reshape(1, nobj, 1),
            "dec": decs.reshape(1, nobj, 1),
        }

        height = sdss_obj[0]["image"].shape[1]
        width = sdss_obj[0]["image"].shape[2]

        return cls(height, width, d)

    def restrict_by_ra_dec(self, ra_lim, dec_lim):
        """Helper function to restrict photo catalog to within RA and DEC limits."""
        ra = self["ra"].squeeze()
        dec = self["dec"].squeeze()

        keep = (ra > ra_lim[0]) & (ra < ra_lim[1]) & (dec >= dec_lim[0]) & (dec <= dec_lim[1])
        plocs = self.plocs[:, keep]
        n_sources = torch.tensor([plocs.size()[1]])

        d = {"plocs": plocs, "n_sources": n_sources}
        for key, val in self.items():
            d[key] = val[:, keep]

        return PhotoFullCatalog(
            plocs[0, :, 0].max() - plocs[0, :, 0].min(),  # new height
            plocs[0, :, 1].max() - plocs[0, :, 1].min(),  # new width
            d,
        )


# Data type conversion helpers
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


# Predict helpers
def prepare_image(x, device):
    x = torch.from_numpy(x).unsqueeze(0)
    x = x.to(device=device)
    # image dimensions must be a multiple of 16
    height = x.size(2) - (x.size(2) % 16)
    width = x.size(3) - (x.size(3) % 16)
    return x[:, :, :height, :width]


def crop_image(image, background, crop_params):
    """Crop the image (and background) to a subregion for prediction."""
    idx0 = crop_params.left_upper_corner[0]
    idx1 = crop_params.left_upper_corner[1]
    width = crop_params.width
    height = crop_params.height
    if ((idx0 + height) <= image.shape[2]) and ((idx1 + width) <= image.shape[3]):
        image = image[:, :, idx0 : idx0 + height, idx1 : idx1 + width]
        background = background[:, :, idx0 : idx0 + height, idx1 : idx1 + width]
    return image, background


def prepare_batch(images, background, crop_params):
    batch = {"images": images, "background": background}
    if crop_params.do_crop:
        batch["images"], batch["background"] = crop_image(images, background, crop_params)
    batch["images"] = batch["images"].squeeze(0)
    batch["background"] = batch["background"].squeeze(0)
    return batch


# Downloader helpers
def download_file(url, filename, preprocess_fn=lambda x: x):  # noqa: WPS404
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=10)
    with open(filename, "wb") as out_file:
        out_file.write(preprocess_fn(response.content))
