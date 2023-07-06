import warnings
from pathlib import Path
from typing import Tuple
from urllib.error import HTTPError

import numpy as np
import pytorch_lightning as pl
import torch
from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import download_file
from astropy.wcs import WCS
from torch.utils.data import DataLoader, Dataset

from bliss.catalog import FullCatalog, SourceType
from bliss.surveys.sdss import SloanDigitalSkySurvey as SDSS
from bliss.surveys.sdss import column_to_tensor, prepare_batch, prepare_image
from bliss.utils.download_utils import download_file_to_dst, download_git_lfs_file


class DarkEnergyCameraLegacySurvey(pl.LightningDataModule, Dataset):
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
        survey_bricks = DecalsDownloader.survey_bricks()
        # ra1 - lower RA boundary; ra2 - upper RA boundary
        # dec1 - lower DEC boundary; dec2 - upper DEC boundary
        return survey_bricks[
            (survey_bricks["ra1"] <= ra)
            & (survey_bricks["ra2"] >= ra)
            & (survey_bricks["dec1"] <= dec)
            & (survey_bricks["dec2"] >= dec)
        ]["brickname"][0]

    def __init__(
        self,
        decals_dir="data/decals",
        ra=336.635,
        dec=-0.96,
        # TODO: fix band-indexing after DECaLS E2E
        bands=(1, 2, 3, 4),  # SDSS.BANDS indexing, for SDSS-trained encoder.
        predict_device=None,
        predict_crop=None,
    ):
        super().__init__()

        self.decals_path = Path(decals_dir)
        self.ra = ra
        self.dec = dec
        self.bands = bands
        self.brickname = DarkEnergyCameraLegacySurvey.brick_for_radec(ra, dec)

        self.downloader = DecalsDownloader(ra, dec, self.decals_path)

        self.prepare_data()

        self.predict_device = predict_device
        self.predict_crop = predict_crop

    def prepare_data(self):
        for b, bl in enumerate(SDSS.BANDS):
            if b in self.bands:
                image_filename = self.decals_path / f"legacysurvey-{self.brickname}-image-{bl}.fits"
                if not Path(image_filename).exists():
                    self.downloader.download_image(bl)
        self.downloader.download_catalog(self.brickname)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.get_from_disk(idx)

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
        }

    def predict_dataloader(self):
        img = prepare_image(self[0]["image"], device=self.predict_device)
        bg = prepare_image(self[0]["background"], device=self.predict_device)
        batch = prepare_batch(img, bg, self.predict_crop)
        return DataLoader([batch], batch_size=1)


class DecalsDownloader:
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

    def __init__(self, ra, dec, download_dir):
        self.ra = ra
        self.dec = dec
        self.download_dir = download_dir

        # get brick name from (ra, dec) via survey-bricks.fits
        self.brickname = DarkEnergyCameraLegacySurvey.brick_for_radec(ra, dec)

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

    def download_catalog(self, brickname):
        """Download tractor catalog for this brick."""
        tractor_filename = self.download_dir / f"tractor-{brickname}.fits"
        download_file_to_dst(
            f"{DecalsDownloader.URLBASE}/south/tractor/{brickname[:3]}/tractor-{brickname}.fits",
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


# TODO: 3366m010 is hardcoded brick now. Remove this when able to map SDSS RCF -> DECaLS brick.
def download_decals_base(download_dir: str):
    cutout_filename = "cutout_336.635_-0.9600.fits"
    tractor_filename = "tractor-3366m010.fits"
    cutout = download_git_lfs_file(
        f"https://api.github.com/repos/prob-ml/bliss/contents/data/decals/{cutout_filename}"
    )
    tractor = download_git_lfs_file(
        f"https://api.github.com/repos/prob-ml/bliss/contents/data/decals/{tractor_filename}"
    )
    cutout_path = Path(download_dir) / cutout_filename
    tractor_path = Path(download_dir) / tractor_filename
    cutout_path.write_bytes(cutout)
    tractor_path.write_bytes(tractor)
