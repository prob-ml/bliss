import gzip
import warnings
from pathlib import Path
from typing import List, Tuple
from urllib.error import HTTPError

import galsim
import numpy as np
import torch
from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import download_file
from astropy.wcs import WCS
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader

from bliss.catalog import FullCatalog, SourceType
from bliss.simulator.background import ImageBackground
from bliss.simulator.psf import ImagePSF, PSFConfig
from bliss.surveys.des import DarkEnergySurvey as DES  # pylint: disable=cyclic-import
from bliss.surveys.des import DESImageID, SkyCoord  # pylint: disable=cyclic-import
from bliss.surveys.sdss import column_to_tensor
from bliss.surveys.survey import Survey, SurveyDownloader
from bliss.utils.download_utils import download_file_to_dst


class DarkEnergyCameraLegacySurvey(Survey):
    BANDS = ("g", "r", "i", "z")

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

    @staticmethod
    def coadd_images(constituent_images) -> torch.Tensor:
        """Get coadd image for image_id, given constituent images."""
        _, n_bands, height, width = constituent_images.shape

        cow = np.zeros(n_bands)  # inverse-variance weights
        cowimg = np.zeros((n_bands, height, width))
        for image_aligned in constituent_images:
            # (2) weight by inverse-variance
            image_aligned = image_aligned.numpy()
            invvar = 1 / image_aligned.var(axis=(1, 2))
            cow += invvar
            cowimg += image_aligned * np.expand_dims(invvar, axis=(1, 2))

        tinyw = 1e-30
        cowimg /= np.expand_dims(np.maximum(cow, tinyw), axis=(1, 2))
        return torch.from_numpy(cowimg)

    @classmethod
    def create_constituent_objs(cls, dir_path, ccds_table_for_brick, pixel_shift: int) -> DES:
        if getattr(cls, "constituent_objs", None):
            return cls.constituent_objs

        des_image_ids = []
        for brickname, ccds_table in ccds_table_for_brick.items():
            for ccd in ccds_table:
                des_image_id = {
                    "sky_coord": {"ra": ccd["ra"], "dec": ccd["dec"]},
                    "decals_brickname": brickname,
                    "ccdname": ccd["ccdname"],
                    ccd["filter"]: ccd["image_filename"].split(".fits.fz")[0],
                }
                des_image_ids.append(des_image_id)
        cls.constituent_objs = DES(
            dir_path=dir_path,
            image_ids=tuple(des_image_ids),
            psf_config={"pixel_scale": 0.262, "psf_slen": 25},
            pixel_shift=pixel_shift,
        )
        return cls.constituent_objs

    @classmethod
    def constituent_obj(
        cls, brickname: str, sky_coord: SkyCoord, ccdname: str, bl: str, image_filename: str
    ):
        des_image_id: DESImageID = {
            "sky_coord": sky_coord,
            "decals_brickname": brickname,
            "ccdname": ccdname,
            bl: image_filename,
        }
        for fltr in DES.BANDS:
            des_image_id[fltr] = des_image_id.get(fltr, "")

        try:  # noqa: WPS229
            idx = cls.constituent_objs.idx(des_image_id)
            return cls.constituent_objs[idx], idx
        except AttributeError as e:
            warnings.warn(
                f"DES survey objects not created. Call "
                f"{cls.__name__}.create_constituent_objs() first."
            )
            raise e

    def __init__(
        self,
        psf_config: PSFConfig,
        dir_path="data/decals",
        sky_coords=({"ra": 336.6643042496718, "dec": -0.9316385797930247},),
        bands=(0, 1, 2, 3),
        pixel_shift: int = 2,
    ):
        super().__init__()

        self.decals_path = Path(dir_path)
        self.bands = bands
        self.bricknames = [DECaLS.brick_for_radec(c["ra"], c["dec"]) for c in sky_coords]
        self.pixel_shift = pixel_shift

        self.downloader = DecalsDownloader(self.bricknames, self.decals_path)
        self.prepare_data()

        self.background = ImageBackground(self, bands=tuple(range(len(self.BANDS))))

        self.downloader.download_psfsizes(self.bands)
        self.ccds_table_for_brick = self.single_exposure_ccds_for_bricks()
        target_wcs = self.target_wcs_for_brick()
        self.psf = DECaLS_PSF(
            dir_path,
            self.image_ids(),
            self.bands,
            psf_config,
            self.ccds_table_for_brick,
            target_wcs,
            pixel_shift,
        )

        self.nmgy_to_nelec_dict = self.nmgy_to_nelec()

        self.catalog_cls = DecalsFullCatalog
        self._predict_batch = {"images": self[0]["image"], "background": self[0]["background"]}

    def prepare_data(self):
        self.downloader.download_images(self.bands)
        self.downloader.download_catalogs()
        for brickname in self.bricknames:
            catalog_filename = (
                self.decals_path / brickname[:3] / brickname / f"tractor-{brickname}.fits"
            )
            assert Path(catalog_filename).exists(), f"Catalog file {catalog_filename} not found"
            for b, bl in enumerate(self.BANDS):
                if b in self.bands:
                    image_filename = (
                        self.decals_path
                        / brickname[:3]
                        / brickname
                        / f"legacysurvey-{brickname}-image-{bl}.fits"
                    )
                    assert Path(image_filename).exists(), f"Image file {image_filename} not found"

    def __len__(self):
        return len(self.bricknames)

    def __getitem__(self, idx):
        return self.get_from_disk(idx)

    def image_id(self, idx) -> str:
        return self.bricknames[idx]

    def idx(self, image_id: str) -> int:
        return self.bricknames.index(image_id)

    def image_ids(self) -> List[str]:
        return self.bricknames

    def get_from_disk(self, idx):
        brickname = self.bricknames[idx]

        image_list = [{} for _ in self.BANDS]
        # first get structure of image data for a present band
        # get first present band by checking des_image_id[bl] for bl in DES.BANDS
        first_present_bl = self.BANDS[self.bands[0]]
        first_present_bl_obj = self.read_image_for_band(brickname, first_present_bl)
        image_list[self.bands[0]] = first_present_bl_obj

        # band-indexing important for encoder's filtering in Encoder::get_input_tensor
        img_shape = first_present_bl_obj["image"].shape
        for b, bl in enumerate(self.BANDS):
            if bl != first_present_bl and b in self.bands:
                image_list[b] = self.read_image_for_band(brickname, bl)
            elif bl != first_present_bl:
                image_list[b] = {
                    "image": np.zeros(img_shape, dtype=np.float32),
                    "background": np.random.rand(*img_shape).astype(np.float32),
                    "wcs": first_present_bl_obj["wcs"],  # NOTE: junk; just for format
                    "nelec_per_nmgy_list": np.ones((1, 1, 1)),
                }

        ret = {}
        for k in image_list[0]:
            data_per_band = [image[k] for image in image_list]
            if isinstance(data_per_band[0], np.ndarray):
                ret[k] = np.stack(data_per_band)
            else:
                ret[k] = data_per_band

        return ret

    def read_image_for_band(self, brickname, bl):
        img_fits = fits.open(
            self.decals_path
            / brickname[:3]
            / brickname
            / f"legacysurvey-{brickname}-image-{bl}.fits"
        )
        image = img_fits[1].data  # pylint: disable=no-member
        hr = img_fits[1].header  # pylint: disable=no-member
        wcs = WCS(hr)

        return {
            "image": image.astype(np.float32),
            "background": np.full_like(
                image, fill_value=hr[f"COSKY_{bl.upper()}"], dtype=np.float32
            ),
            "wcs": wcs,
            "nelec_per_nmgy_list": np.array([[[DES.zpt_to_scale(hr["MAGZERO"])]]]),
        }

    def single_exposure_ccds_for_bricks(self):
        ccds_table_for_brick = {}  # indexed by brickname

        self.downloader.download_brick_ccds_all()
        for brickname in self.image_ids():
            brick_ccds_filename = (
                self.decals_path / brickname[:3] / brickname / f"legacysurvey-{brickname}-ccds.fits"
            )
            assert Path(brick_ccds_filename).exists(), f"CCDs file {brick_ccds_filename} not found"

            brick_ccds = Table.read(brick_ccds_filename)
            # uniformly sample one exposure per band
            all_expnums = {}
            expnums = {}
            for b in self.BANDS:
                all_expnums[b] = np.unique(brick_ccds["expnum"][brick_ccds["filter"] == b])
                expnums[b] = all_expnums[b][0] if len(all_expnums[b]) > 0 else None  # noqa: WPS507
            # NOTE: now using same hardcoded CCDs (regardless of brick) for co-adding PSFs to
            # ensure images are available.
            # TODO: remove this hardcoding.
            fixed_ccds = brick_ccds[
                (
                    brick_ccds["image_filename"]
                    == "decam/CP/V4.8.2a/CP20141020/c4d_141021_015854_ooi_g_ls9.fits.fz"
                )
                | (
                    brick_ccds["image_filename"]
                    == "decam/CP/V4.8.2a/CP20151107/c4d_151108_003333_ooi_r_ls9.fits.fz"
                )
                | (
                    brick_ccds["image_filename"]
                    == "decam/CP/V4.8.2a/CP20130912/c4d_130913_040652_ooi_z_ls9.fits.fz"
                )
            ]
            ccds_table_for_brick[brickname] = fixed_ccds

        return ccds_table_for_brick  # noqa: WPS331

    def target_wcs_for_brick(self):
        target_wcs_for_brick = {}  # indexed by brickname
        for brickname in self.bricknames:
            target_wcs_for_brick[brickname] = self[self.idx(brickname)]["wcs"]
        return target_wcs_for_brick

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


DECaLS = DarkEnergyCameraLegacySurvey


class DecalsDownloader(SurveyDownloader):
    """Class for downloading DECaLS data."""

    URLBASE = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9"

    @staticmethod
    def download_ccds_annotated(download_dir):
        """Download CCDs annotated table."""
        download_file_to_dst(
            f"{DecalsDownloader.URLBASE}/ccds-annotated-decam-dr9.fits.gz",
            Path(download_dir) / "ccds-annotated-decam-dr9.fits",
            gzip.decompress,
        )

    @staticmethod
    def download_catalog_from_filename(tractor_filename: str):
        """Download tractor catalog given tractor-<brick_name>.fits filename."""
        basename = Path(tractor_filename).name
        brickname = basename.split("-")[1].split(".")[0]
        download_file_to_dst(
            f"{DecalsDownloader.URLBASE}/south/tractor/{brickname[:3]}/{basename}",
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

    def __init__(self, bricknames, download_dir):
        self.bricknames = bricknames
        self.download_dir = download_dir

    def download_images(self, bands: List[int]):
        """Download images for all bands, for all bricks."""
        for brickname in self.bricknames:
            for b, bl in enumerate(DECaLS.BANDS):
                if b in bands:
                    self.download_image(brickname, bl)

    def download_image(self, brickname, band="r"):
        """Download image for specified band, for this brick."""
        image_filename = (
            self.download_dir
            / brickname[:3]
            / brickname
            / f"legacysurvey-{brickname}-image-{band}.fits"
        )
        try:
            download_file_to_dst(
                f"{DecalsDownloader.URLBASE}/south/coadd/{brickname[:3]}/{brickname}/"
                f"legacysurvey-{brickname}-image-{band}.fits.fz",
                image_filename,
            )
        except HTTPError as e:
            warnings.warn(
                f"No {band}-band image for brick {brickname}. Check cfg.datasets.decals.bands."
            )
            raise e

    def download_psfsizes(self, bands: List[int]):
        """Download psf sizes for all bricks."""
        for brickname in self.bricknames:
            for b, bl in enumerate(DECaLS.BANDS):
                if b in bands:
                    self.download_psfsize(brickname, bl)

    def download_psfsize(self, brickname, band="r"):
        """Download psf size for this brick."""
        psfsize_filename = (
            self.download_dir
            / brickname[:3]
            / brickname
            / f"legacysurvey-{brickname}-psfsize-{band}.fits.fz"
        )
        try:
            download_file_to_dst(
                f"{DecalsDownloader.URLBASE}/south/coadd/{brickname[:3]}/{brickname}/"
                f"legacysurvey-{brickname}-psfsize-{band}.fits.fz",
                psfsize_filename,
            )
        except HTTPError as e:
            warnings.warn(
                f"No {band}-band psfsize for brick {brickname}. Check cfg.datasets.decals.bands."
            )
            raise e

    def download_catalogs(self):
        """Download tractor catalogs for all bricks."""
        for brickname in self.bricknames:
            self.download_catalog(brickname)

    def download_catalog(self, brickname) -> str:
        """Download tractor catalog for this brick.

        Args:
            brickname (str): brick name

        Returns:
            str: path to downloaded tractor catalog
        """
        tractor_filename = (
            self.download_dir / brickname[:3] / brickname / f"tractor-{brickname}.fits"
        )
        download_file_to_dst(
            f"{DecalsDownloader.URLBASE}/south/tractor/{brickname[:3]}/"
            f"tractor-{brickname}.fits",
            tractor_filename,
        )
        return str(tractor_filename)

    def download_brick_ccds_all(self):
        """Download CCDs table for all bricks."""
        for brickname in self.bricknames:
            self.download_brick_ccds(brickname)

    def download_brick_ccds(self, brickname) -> str:
        ccds_filename = (
            self.download_dir / brickname[:3] / brickname / f"legacysurvey-{brickname}-ccds.fits"
        )
        download_file_to_dst(
            f"{DecalsDownloader.URLBASE}/south/coadd/{brickname[:3]}/{brickname}/"
            f"legacysurvey-{brickname}-ccds.fits",
            ccds_filename,
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


class DECaLS_PSF(ImagePSF):  # noqa: N801
    # PSF parameters for encoder to learn
    PARAM_NAMES = [
        "psf_mx2",
        "psf_my2",
        "psf_mxy",
        "psf_a",
        "psf_b",
        "psf_theta",
        "psf_ell",
        "psfnorm_mean",
        "psfnorm_std",
        "psfdepth",
        "galdepth",
        "gausspsfdepth",
        "gaussgaldepth",
        "psf_fwhm",
    ]

    @staticmethod
    def _get_fit_file_psf_params(
        psf_fit_file: str, bands: Tuple[int, ...], ccds_for_brick: Table, brick_fwhms
    ):
        msg = (
            f"{psf_fit_file} does not exist. "
            + "Make sure data files are available for bricks specified in config."
        )
        assert Path(psf_fit_file).exists(), msg

        ccds_annotated = Table.read(psf_fit_file)
        brick_ccds_mask = np.isin(ccds_annotated["ccdname"], ccds_for_brick)
        psf_params = torch.zeros(len(DECaLS.BANDS), len(DECaLS_PSF.PARAM_NAMES))
        psf_cols = [
            col
            for col in ccds_annotated.colnames
            if col.startswith("psf") or col.startswith("gal") or col.startswith("gauss")
        ]
        for b, bl in enumerate(DECaLS.BANDS):
            if b in bands:
                ccds_psf_band = ccds_annotated[brick_ccds_mask & (ccds_annotated["filter"] == bl)][
                    psf_cols
                ]
                band_params = [0.0 for _ in range(len(DECaLS_PSF.PARAM_NAMES))]
                for i, param in enumerate(DECaLS_PSF.PARAM_NAMES):
                    band_params[i] = (
                        np.mean(brick_fwhms[b])
                        if param == "psf_fwhm"
                        else np.mean(ccds_psf_band[param])
                    )
                psf_params[b] = torch.tensor(np.array(band_params))

        return psf_params

    def __init__(
        self,
        survey_data_dir,
        image_ids,
        bands,
        psf_config: PSFConfig,
        ccds_table_for_brick,
        target_wcs_for_brick,
        pixel_shift,
    ):
        super().__init__(bands, **psf_config)

        self.survey_data_dir = survey_data_dir
        self.psf_galsim = {}
        self.psf_params = {}

        DecalsDownloader.download_ccds_annotated(survey_data_dir)
        ccds_annotated_filename = f"{survey_data_dir}/ccds-annotated-decam-dr9.fits"
        for brickname in image_ids:
            ccds_for_brick = ccds_table_for_brick[brickname]["ccdname"]
            brick_fwhms = {}
            for b in self.bands:
                brick_fwhm_b_filename = (
                    f"{survey_data_dir}/{brickname[:3]}/{brickname}/"
                    + f"legacysurvey-{brickname}-psfsize-{DECaLS.BANDS[b]}.fits.fz"
                )
                brick_fwhm_b = fits.open(brick_fwhm_b_filename)
                brick_fwhms[b] = brick_fwhm_b[1].data  # pylint: disable=no-member

            self.psf_params[brickname] = self._get_fit_file_psf_params(
                ccds_annotated_filename, bands, ccds_for_brick, brick_fwhms
            )
            DECaLS.create_constituent_objs(self.survey_data_dir, ccds_table_for_brick, pixel_shift)
            self.psf_galsim[brickname] = self._get_psf_coadded(
                brickname, ccds_table_for_brick[brickname], target_wcs_for_brick
            )

    def _get_psf_coadded(self, brickname, brick_ccds, target_wcs_for_brick):
        """Get co-added PSF images for each band in brick, using CCDS in `brick_ccds`.
        cf. https://github.com/legacysurvey/legacypipe/blob/ba1ffd4969c1f920566e780118c542d103cbd9a5/py/legacypipe/coadds.py#L486-L519 # noqa: E501 # pylint: disable=line-too-long

        Args:
            brickname (str): brick name
            brick_ccds (Table): CCDs to use for coadd image simulation
            target_wcs_for_brick (WCS): target WCS for brick

        Returns:
            np.ndarray: co-added PSF images for each band in brick
        """
        coadd_psf_image = np.full((len(DECaLS.BANDS), self.psf_slen, self.psf_slen), 1e-30)
        for ccd in brick_ccds:
            band = DES.BANDS.index(ccd["filter"])
            des_obj, idx = DECaLS.constituent_obj(
                brickname,
                {"ra": ccd["ra"], "dec": ccd["dec"]},
                ccd["ccdname"],
                ccd["filter"],
                ccd["image_filename"].split(".fits.fz")[0],
            )
            des_img = des_obj["image"]
            _, image_h, image_w = des_img.shape
            px, py = image_w // 2, image_h // 2
            psf_patch = DECaLS.constituent_objs.psf.get_psf_via_despsfex(
                des_image_id=DECaLS.constituent_objs.image_id(idx), px=px, py=py
            )[band]

            # TODO: fix always all-0 psf
            coadd_psf_image[band] += psf_patch.original.image.array / (des_obj["sig1"][band] ** 2)

        # convert to image
        images = []
        for b, _bl in enumerate(DECaLS.BANDS):
            psf_image = galsim.Image(coadd_psf_image[b], scale=self.pixel_scale)
            images.append(galsim.InterpolatedImage(psf_image).withFlux(1.0))
        return images
