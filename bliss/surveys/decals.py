import warnings
from pathlib import Path
from typing import List
from urllib.error import HTTPError

import numpy as np
import torch
from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import download_file
from astropy.wcs import WCS
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader

from bliss.catalog import FullCatalog, SourceType
from bliss.surveys.sdss import SloanDigitalSkySurvey as SDSS
from bliss.surveys.sdss import column_to_tensor
from bliss.surveys.survey import Survey, SurveyDownloader
from bliss.utils.download_utils import download_file_to_dst


class DarkEnergyCameraLegacySurvey(Survey):
    BANDS = ("g", "r", "i", "z")
    PIXEL_SCALE = 0.262
    PIXEL_SIZE = 3600  # arcsec/degree

    @staticmethod
    def pix_to_deg(pix: int) -> float:
        return (
            pix * DarkEnergyCameraLegacySurvey.PIXEL_SCALE / DarkEnergyCameraLegacySurvey.PIXEL_SIZE
        )

    @staticmethod
    def brick_for_radec(ra: float, dec: float) -> str:
        """Get brick name for specified RA, Dec."""
        bricks = DecalsDownloader.survey_bricks()
        # ra1 - lower RA boundary; ra2 - upper RA boundary
        # dec1 - lower DEC boundary; dec2 - upper DEC boundary
        return bricks[
            (bricks["ra1"] <= ra)
            & (bricks["ra2"] >= ra)
            & (bricks["dec1"] <= dec)
            & (bricks["dec2"] >= dec)
        ]["brickname"][0]

    def __init__(
        self,
        decals_dir="data/decals",
        ra=336.635,
        dec=-0.96,
        # TODO: fix band-indexing after DECaLS E2E
        bands=(1, 2, 3, 4),  # SDSS.BANDS indexing, for SDSS-trained encoder
        reference_band: int = 1,  # r-band
    ):
        super().__init__()

        self.decals_path = Path(decals_dir)
        self.ra = ra
        self.dec = dec
        self.bands = bands
        self.brickname = DarkEnergyCameraLegacySurvey.brick_for_radec(ra, dec)

        self.downloader = DecalsDownloader(self.brickname, self.decals_path)

        self.prepare_data()

        self.catalog_cls = DecalsFullCatalog
        self._predict_batch = {"images": self[0]["image"], "background": self[0]["background"]}

    def prepare_data(self):
        for b, bl in enumerate(SDSS.BANDS):
            if b in self.bands:
                image_filename = self.decals_path / f"legacysurvey-{self.brickname}-image-{bl}.fits"
                if not Path(image_filename).exists():
                    self.downloader.download_image(bl)
        self.downloader.download_catalog()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.get_from_disk(idx)

    def image_id(self, idx) -> str:
        return self.brickname

    def idx(self, image_id: str) -> int:
        return 0

    def image_ids(self) -> List[str]:
        return [self.brickname]

    def get_from_disk(self, idx):
        image_list = []
        # first get structure of image data for present band
        first_target_band_img = self.read_image_for_band(SDSS.BANDS[self.bands[0]])

        # band-indexing important for encoder's filtering in Encoder::get_input_tensor,
        # so force to SDSS.BANDS indexing
        for b, bl in enumerate(SDSS.BANDS):
            if b in self.bands and b != self.bands[0]:
                image_list.append(self.read_image_for_band(bl))
            elif b == self.bands[0]:
                image_list.append(first_target_band_img)
            else:
                image_list.append(
                    {
                        "image": np.zeros_like(first_target_band_img["image"]).astype(np.float32),
                        "background": first_target_band_img["background"],
                        "wcs": first_target_band_img["wcs"],
                        "nelec_per_nmgy_list": first_target_band_img["nelec_per_nmgy_list"],
                    }
                )

        ret = {}
        for k in image_list[0]:
            data_per_band = [image[k] for image in image_list]
            if isinstance(data_per_band[0], np.ndarray):
                ret[k] = np.stack(data_per_band)
            else:
                ret[k] = data_per_band

        return ret

    def read_image_for_band(self, band):
        img_fits = fits.open(self.decals_path / f"legacysurvey-{self.brickname}-image-{band}.fits")
        image = img_fits[1].data  # pylint: disable=no-member
        hr = img_fits[1].header  # pylint: disable=no-member
        wcs = WCS(hr)

        return {
            "image": image,
            # random normal background, in double precision
            "background": np.random.normal(size=image.shape).astype(
                np.float32
            ),  # TODO: replace with actual background
            "wcs": wcs,
            "nelec_per_nmgy_list": np.ones(
                shape=(1, len(self.bands))
            ),  # TODO: replace with actual ratios
        }

    @property
    def predict_batch(self):
        if not self._predict_batch:
            self._predict_batch = {
                "images": self[0]["image"],
                "background": self[0]["background"],
            }
        return self._predict_batch

    @predict_batch.setter
    def predict_batch(self, value):
        self._predict_batch = value

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        assert self.predict_batch is not None, "predict_batch must be set."
        return DataLoader([self.predict_batch], batch_size=1)


class DecalsDownloader(SurveyDownloader):
    """Class for downloading DECaLS data."""

    URLBASE = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9"

    @staticmethod
    def download_catalog_from_filename(tractor_filename: str):
        """Download tractor catalog given tractor-<brick_name>.fits filename."""
        brickname = tractor_filename.split("-")[1].split(".")[0]
        download_file_to_dst(
            f"{DecalsDownloader.URLBASE}/south/tractor/{brickname[:3]}/{tractor_filename}",
            tractor_filename,
        )

    @classmethod
    def download_survey_bricks(cls):
        # Download and use survey-bricks table in memory
        survey_bricks_filename = download_file(
            f"{cls.URLBASE}/south/survey-bricks-dr9-south.fits.gz",
            cache=True,
            timeout=10,
        )
        cls._survey_bricks = Table.read(survey_bricks_filename)

    @classmethod
    def survey_bricks(cls) -> Table:
        """Get survey bricks table."""
        if not getattr(cls, "_survey_bricks", None):
            cls.download_survey_bricks()
        return cls._survey_bricks  # pylint: disable=no-member

    def __init__(self, brickname, download_dir):
        self.brickname = brickname
        self.download_dir = download_dir

    def download_image(self, band="r"):
        """Download image for specified band, for this brick."""
        image_filename = self.download_dir / f"legacysurvey-{self.brickname}-image-{band}.fits"
        try:
            download_file_to_dst(
                f"{DecalsDownloader.URLBASE}/south/coadd/{self.brickname[:3]}/{self.brickname}/"
                f"legacysurvey-{self.brickname}-image-{band}.fits.fz",
                image_filename,
            )
        except HTTPError as e:
            warnings.warn(
                f"No {band}-band image for brick {self.brickname}. Check cfg.datasets.decals.bands."
            )
            raise e

    def download_catalog(self) -> str:
        """Download tractor catalog for this brick.

        Returns:
            str: path to downloaded tractor catalog
        """
        tractor_filename = Path(self.download_dir) / f"tractor-{self.brickname}.fits"
        download_file_to_dst(
            f"{DecalsDownloader.URLBASE}/south/tractor/{self.brickname[:3]}/"
            f"tractor-{self.brickname}.fits",
            tractor_filename,
        )
        return tractor_filename


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
        cat_path,
        wcs: WCS,
        height,
        width,
        band: str = "r",
    ):
        """Loads DECaLS catalog from FITS file.

        Args:
            cat_path (str): Path to .fits file.
            band (str): Band to read from. Defaults to "r".
            wcs (WCS): WCS object for the image.
            height (int): Height of the image.
            width (int): Width of the image.

        Returns:
            A DecalsFullCatalog containing data from the provided file.
        """
        catalog_path = Path(cat_path)
        if not catalog_path.exists():
            DecalsDownloader.download_catalog_from_filename(catalog_path.name)
        assert catalog_path.exists(), f"File {catalog_path} does not exist"

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
        ras = column_to_tensor(table, "RA")
        decs = column_to_tensor(table, "DEC")
        fluxes = column_to_tensor(table, f"FLUX_{band}")
        mask = torch.from_numpy((bits & bitmask) == 0).bool()

        galaxy_bools = is_galaxy & mask & (fluxes > 0)
        star_bools = is_star & mask & (fluxes > 0)

        # true light source mask
        keep = galaxy_bools | star_bools

        # filter quantities
        objid = objid[keep]

        galaxy_bools = galaxy_bools[keep]
        star_bools = star_bools[keep]
        ras = ras[keep]
        decs = decs[keep]
        fluxes = fluxes[keep]
        mags = cls._flux_to_mag(fluxes)
        nobj = objid.shape[0]

        # get pixel coordinates
        plocs = cls.plocs_from_ra_dec(ras, decs, wcs)

        # Verify each tile contains either a star or a galaxy
        assert torch.all(star_bools + galaxy_bools)
        source_type = SourceType.STAR * star_bools + SourceType.GALAXY * galaxy_bools

        d = {
            "plocs": plocs.reshape(1, nobj, 2),
            "objid": objid.reshape(1, nobj, 1),
            "n_sources": torch.tensor((nobj,)),
            "source_type": source_type.reshape(1, nobj, 1),
            "fluxes": fluxes.reshape(1, nobj, 1),
            "mags": mags.reshape(1, nobj, 1),
            "ra": ras.reshape(1, nobj, 1),
            "dec": decs.reshape(1, nobj, 1),
        }

        return cls(height, width, d)
