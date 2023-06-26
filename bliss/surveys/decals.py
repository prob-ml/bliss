from pathlib import Path
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from numpy.core.defchararray import startswith
from pyvo.dal import sia
from torch.utils.data import DataLoader, Dataset

from bliss.catalog import FullCatalog, SourceType
from bliss.surveys.sdss import column_to_tensor, prepare_batch, prepare_image
from bliss.utils.download_utils import download_file_to_dst


class DarkEnergyCameraLegacySurvey(pl.LightningDataModule, Dataset):
    PIX_SCALE = 0.262  # arcsec/pixel
    PIX_SIZE = 3600  # arcsec/degree

    def __init__(
        self,
        decals_dir="data/decals",
        ra=335,
        dec=0,
        width=2400,
        height=1489,
        bands=("g", "r", "i", "z"),
        predict_device=None,
        predict_crop=None,
    ):
        super().__init__()

        self.decals_path = Path(decals_dir)
        self.ra = ra
        self.dec = dec
        self.width = width
        self.height = height
        self.bands = bands

        self.downloader = DecalsDownloader(ra, dec, width, height, self.decals_path)

        self.items = []
        self.prepare_data()
        assert self.items is not None, "No data found even after prepare_data()."

        self.predict_device = predict_device
        self.predict_crop = predict_crop

    def prepare_data(self):
        for band in self.bands:
            self.brick_name = self.downloader.download_cutout(band)
        assert self.brick_name is not None, "No brick ID found even after prepare_data()."
        self.downloader.download_catalog(self.brick_name)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.get_from_disk(idx)

    def get_from_disk(self, idx):
        cutout_list = []
        for band in self.bands:
            cutout_list.append(self.read_cutout_for_band(band))

        ret = {}
        for k in cutout_list[0]:
            data_per_band = [cutout[k] for cutout in cutout_list]
            if isinstance(data_per_band[0], np.ndarray):
                ret[k] = np.stack(data_per_band)
            else:
                ret[k] = data_per_band
        return ret

    def read_cutout_for_band(self, band):
        cutout = fits.open(self.decals_path / f"cutout_{self.ra}_{self.dec}_{band}.fits")
        image = cutout[0].data  # pylint: disable=no-member
        hr = cutout[0].header  # pylint: disable=no-member
        wcs = WCS(hr)
        return {
            "image": image,
            "background": np.zeros_like(image),  # TODO: find a way to get background
            "wcs": wcs,
        }

    def get_brick_name(self) -> str:  # noqa: WPS615
        if self.brick_name is None:
            self.prepare_data()
        return self.brick_name  # type: ignore

    def predict_dataloader(self):
        img = prepare_image(self[0]["image"], device=self.predict_device)
        bg = prepare_image(self[0]["background"], device=self.predict_device)
        batch = prepare_batch(img, bg, self.predict_crop)
        return DataLoader([batch], batch_size=1)

    @staticmethod
    def pix_to_deg(pix: int) -> float:
        return pix * DarkEnergyCameraLegacySurvey.PIX_SCALE / DarkEnergyCameraLegacySurvey.PIX_SIZE


class DecalsDownloader:
    """Class for downloading DECaLS data."""

    URLBASE = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9"
    DEF_ACCESS_URL = "https://datalab.noirlab.edu/sia/ls_dr9"

    def __init__(self, ra, dec, width, height, download_dir):
        self.ra = ra
        self.dec = dec
        self.width = DarkEnergyCameraLegacySurvey.pix_to_deg(width)  # in degrees
        self.height = DarkEnergyCameraLegacySurvey.pix_to_deg(height)  # in degrees
        self.download_dir = download_dir

        self.svc = sia.SIAService(DecalsDownloader.DEF_ACCESS_URL)

    def download_cutout(self, band="r"):
        img_table = self.svc.search(
            pos=(self.ra, self.dec), size=(self.width, self.height), verbosity=2
        ).to_table()
        sel = (
            (img_table["proctype"] == "Stack")
            & (img_table["prodtype"] == "image")
            & (startswith(img_table["obs_bandpass"].astype(str), band))
        )
        row = img_table[sel]
        if row is None:
            raise ValueError(
                f"No image found for band {band} for region "
                f"(RA={self.ra}, Dec={self.dec}, width={self.width}, height={self.height})."
            )
        img_url = row[0]["access_url"]
        cutout_filename = self.download_dir / f"cutout_{self.ra}_{self.dec}_{band}.fits"
        download_file_to_dst(img_url, cutout_filename)
        cutout = fits.open(cutout_filename)
        return cutout[0].header["BRICK"]  # pylint: disable=no-member

    def download_catalog(self, brick_name):
        """Download tractor catalog for specified RA, Dec, width, height."""
        tractor_filename = self.download_dir / f"tractor-{brick_name}.fits"
        ra_int = int(float(str(self.ra).lstrip("0")))
        download_file_to_dst(
            f"{DecalsDownloader.URLBASE}/south/tractor/{ra_int}/tractor-{brick_name}.fits",
            tractor_filename,
        )

    @staticmethod
    def download_catalog_from_filename(tractor_filename: str):
        """Download tractor catalog given tractor-<brick_name>.fits filename."""
        brick_name = tractor_filename.split("-")[1].split(".")[0]
        ra_int = int(brick_name[:4])
        download_file_to_dst(
            f"{DecalsDownloader.URLBASE}/south/tractor/{ra_int}/{tractor_filename}",
            tractor_filename,
        )


class DecalsFullCatalog(FullCatalog):
    """Class for the Decals Sweep Tractor Catalog.

    Some resources:
    - https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/south/sweep/9.0/
    - https://www.legacysurvey.org/dr9/files/#sweep-catalogs-region-sweep
    - https://www.legacysurvey.org/dr5/description/#photometry
    - https://www.legacysurvey.org/dr9/bitmasks/
    """

    @staticmethod
    def _flux_to_mag(flux):
        return 22.5 - 2.5 * torch.log10(flux)

    @classmethod
    def from_file(
        cls,
        decals_cat_path: str,
        ra_lim: Tuple[int, int] = (-360, 360),
        dec_lim: Tuple[int, int] = (-90, 90),
        band: str = "r",
        wcs: WCS = None,
    ):
        """Loads DECaLS catalog from FITS file.

        Args:
            decals_cat_path (str): Path to .fits file.
            ra_lim (Tuple[int, int]): Range of RA values to keep.
                Defaults to (-360, 360).
            dec_lim (Tuple[int, int]): Range of DEC values to keep.
                Defaults to (-90, 90).
            band (str): Band to read from. Defaults to "r".
            wcs (WCS): An optional WCS object to use to convert the RA/DEC values to pixel
                coordinates. If not provided, the returned plocs will all be zero.

        Returns:
            A DecalsFullCatalog containing data from the provided file. Note that the
            coordinates in (RA, DEC) are not converted to plocs by default. For this,
            use get_plocs_from_ra_dec after loading the data.
        """
        catalog_path = Path(decals_cat_path)
        if not catalog_path.exists():
            DecalsDownloader.download_catalog_from_filename(catalog_path.name)
        table = Table.read(catalog_path, format="fits", unit_parse_strict="silent")
        table = {k.upper(): v for k, v in table.items()}  # uppercase keys
        band = band.capitalize()

        # filter out pixels that aren't in primary region, had issues with source fitting,
        # in SGA large galaxy, or in a globular cluster. In the future this should probably
        # be an input parameter.
        bitmask = 0b0011010000000001  # noqa: WPS339

        objid = column_to_tensor(table, "OBJID")
        objc_type = table["TYPE"].data.astype(str)
        bits = table["MASKBITS"].data.astype(int)
        is_galaxy = torch.from_numpy(
            (objc_type == "DEV")
            | (objc_type == "REX")
            | (objc_type == "EXP")
            | (objc_type == "SER")
        )
        is_star = torch.from_numpy(objc_type == "PSF")
        ra = column_to_tensor(table, "RA")
        dec = column_to_tensor(table, "DEC")
        flux = column_to_tensor(table, f"FLUX_{band}")
        mask = torch.from_numpy((bits & bitmask) == 0).bool()

        galaxy_bool = is_galaxy & mask & (flux > 0)
        star_bool = is_star & mask & (flux > 0)

        # filter on locations
        # first lims imposed on frame
        keep_coord = (ra > ra_lim[0]) & (ra < ra_lim[1]) & (dec > dec_lim[0]) & (dec < dec_lim[1])
        keep = (galaxy_bool | star_bool) & keep_coord

        # filter quantities
        objid = objid[keep]
        galaxy_bool = galaxy_bool[keep]
        star_bool = star_bool[keep]
        ra = ra[keep]
        dec = dec[keep]
        flux = flux[keep]
        mag = cls._flux_to_mag(flux)
        nobj = objid.shape[0]

        assert torch.all(star_bool + galaxy_bool)
        source_type = SourceType.STAR * star_bool + SourceType.GALAXY * galaxy_bool

        d = {
            "objid": objid.reshape(1, nobj, 1),
            "ra": ra.reshape(1, nobj, 1),
            "dec": dec.reshape(1, nobj, 1),
            "plocs": torch.zeros((1, nobj, 2)),  # compatibility with FullCatalog
            "n_sources": torch.tensor((nobj,)),
            "source_type": source_type.reshape(1, nobj, 1),
            "fluxes": flux.reshape(1, nobj, 1),
            "mags": mag.reshape(1, nobj, 1),
        }

        height = dec_lim[1] - dec_lim[0]
        width = ra_lim[1] - ra_lim[0]
        cat = cls(height, width, d)

        # if WCS provided, convert RA/DEC to plocs
        if wcs is not None:
            plocs = cat.get_plocs_from_ra_dec(wcs)
            cat.plocs = plocs
            cat.height, cat.width = wcs.array_shape

        return cat

    def get_plocs_from_ra_dec(self, wcs: WCS):
        """Converts RA/DEC coordinates into pixel coordinates.

        Args:
            wcs (WCS): WCS object to use for transformation.

        Returns:
            A 1xNx2 tensor containing the locations of the light sources in pixel coordinates. This
            function does not write self.plocs, so you should do that manually if necessary.
        """
        ra = self["ra"].numpy().squeeze()
        dec = self["dec"].numpy().squeeze()

        pt, pr = wcs.all_world2pix(ra, dec, 0)  # convert to pixel coordinates
        pt = torch.tensor(pt)
        pr = torch.tensor(pr)
        plocs = torch.stack((pr, pt), dim=-1)
        return plocs.reshape(1, plocs.size()[0], 2) + 0.5  # BLISS consistency
