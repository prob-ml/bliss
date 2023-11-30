import warnings
from pathlib import Path
from typing import List, Tuple, TypedDict
from urllib.error import HTTPError

import galsim.des  # noqa: WPS301
import numpy as np
import torch
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS, FITSFixedWarning
from numpy.core import defchararray
from omegaconf import DictConfig
from pyvo.dal import sia
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import zoom

from bliss.catalog import FullCatalog, SourceType
from bliss.simulator.background import ImageBackground
from bliss.simulator.psf import ImagePSF, PSFConfig
from bliss.surveys.sdss import column_to_tensor
from bliss.surveys.survey import Survey, SurveyDownloader
from bliss.utils.download_utils import download_file_to_dst

SkyCoord = TypedDict(
    "SkyCoord",
    {"ra": float, "dec": float},
)
DESImageID = TypedDict(
    "DESImageID",
    {
        "sky_coord": SkyCoord,  # TODO: keep one of {this, decals_brickname}
        "decals_brickname": str,
        "ccdname": str,
        "g": str,
        "r": str,
        "i": str,
        "z": str,
    },
    total=False,
)


class DarkEnergySurvey(Survey):
    BANDS = ("g", "r", "i", "z")

    GAIN = 4.0  # e-/ADU (cf. https://noirlab.edu/science/programs/ctio/instruments/Dark-Energy-Camera/characteristics) # noqa: E501 # pylint: disable=line-too-long
    EXPTIME = 90.0  # s

    @staticmethod
    def zpt_to_scale(zpt):
        """Converts a magnitude zero point per sec to nelec/nmgy scale.

        See also https://github.com/dstndstn/tractor/blob/cdb82000422e85c9c97b134edadff31d68bced0c/tractor/brightness.py#L217C6-L217C6. # noqa: E501 # pylint: disable=line-too-long

        Args:
            zpt (float): magnitude zero point per sec

        Returns:
            float: nelec/nmgy scale
        """
        return 10.0 ** ((zpt - 22.5) / 2.5)

    def __init__(
        self,
        psf_config: PSFConfig,
        pixel_shift,
        dir_path="data/des",
        image_ids: Tuple[DESImageID] = (
            # TODO: maybe find a better/more general image_id representation?
            {
                "sky_coord": {"ra": 336.6643042496718, "dec": -0.9316385797930247},
                "decals_brickname": "3366m010",
                "ccdname": "S28",
                "g": "decam/CP/V4.8.2a/CP20171108/c4d_171109_002003_ooi_g_ls9",
                "r": "decam/CP/V4.8.2a/CP20170926/c4d_170927_025457_ooi_r_ls9",
                "i": "",
                "z": "decam/CP/V4.8.2a/CP20170926/c4d_170927_025655_ooi_z_ls9",
            },
        ),
        load_image_data: bool = False,
    ):
        super().__init__()

        self.des_path = Path(dir_path)

        self.load_image_data = load_image_data

        self.image_id_list = self.process_image_ids(image_ids)
        self.bands = tuple(range(len(self.BANDS)))
        self.n_bands = len(self.BANDS)
        self.pixel_shift = pixel_shift

        self.downloader = DESDownloader(self.image_id_list, self.des_path)
        self.prepare_data()

        self.background = ImageBackground(self, bands=self.bands)
        self.psf = DES_PSF(dir_path, self.image_ids(), self.bands, psf_config)
        self.flux_calibration_dict = self.get_flux_calibrations()

        self.catalog_cls = TractorFullCatalog
        if self.load_image_data:
            self._predict_batch = {"images": self[0]["image"], "background": self[0]["background"]}

    def prepare_data(self):
        self.downloader.download_images()
        self.downloader.download_backgrounds()
        self.downloader.download_psfexs()

    def __len__(self):
        return len(self.image_id_list)

    def __getitem__(self, idx):
        return self.get_from_disk(idx)

    def image_id(self, idx) -> DESImageID:
        return self.image_id_list[idx]

    def idx(self, image_id: DESImageID) -> int:
        return self.image_id_list.index(self.to_dictconfig(image_id))

    def image_ids(self) -> List[DESImageID]:
        return self.image_id_list

    def get_from_disk(self, idx):
        des_image_id = self.image_id(idx)

        image_list = [{} for _ in self.BANDS]
        # first get structure of image data for a present band
        # get first present band by checking des_image_id[bl] for bl in DES.BANDS
        first_present_bl = next(bl for bl in DES.BANDS if des_image_id[bl])
        first_present_bl_obj = self.read_image_for_band(des_image_id, first_present_bl)
        image_list[DES.BANDS.index(first_present_bl)] = first_present_bl_obj

        img_shape = first_present_bl_obj["background"].shape
        for b, bl in enumerate(self.BANDS):
            if bl != first_present_bl and des_image_id[bl]:
                image_list[b] = self.read_image_for_band(des_image_id, bl)
            elif bl != first_present_bl:
                image_list[b] = {
                    "background": np.random.rand(*img_shape).astype(np.float32),
                    "wcs": first_present_bl_obj["wcs"],  # NOTE: junk; just for format
                    "flux_calibration_list": np.ones((1, 1, 1)),
                }
                if self.load_image_data:
                    image_list[b].update(
                        {"image": np.zeros(img_shape).astype(np.float32), "sig1": 0.0}
                    )

        ret = {}
        for k in image_list[0]:
            data_per_band = [image[k] for image in image_list]
            if isinstance(data_per_band[0], np.ndarray):
                ret[k] = np.stack(data_per_band)
            else:
                ret[k] = data_per_band

        return ret

    def read_image_for_band(self, des_image_id, band):
        brickname = des_image_id["decals_brickname"]
        ccdname = des_image_id["ccdname"]
        image_basename = DESDownloader.image_basename_from_filename(des_image_id[band], band)
        img_fits_filename = self.des_path / brickname[:3] / brickname / f"{image_basename}.fits"
        hr = fits.getheader(img_fits_filename, 0)  # pylint: disable=no-member
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FITSFixedWarning)
            wcs = WCS(hr)
        image_shape = (hr["NAXIS2"], hr["NAXIS1"])

        flux_calibration = self.GAIN * hr["EXPTIME"]
        background_nelec = (
            self.splinesky_level_for_band(brickname, ccdname, des_image_id[band], image_shape)
            * flux_calibration
        )
        d = {
            "background": background_nelec,
            "wcs": wcs,
            "flux_calibration_list": np.array([[[flux_calibration]]]),
        }
        if self.load_image_data:
            # compute sig1 (cf. https://github.com/dstndstn/tractor/blob/cdb82000422e85c9c97b134edadff31d68bced0c/projects/desi/decam.py#L313-L333) # noqa: E501 # pylint: disable=line-too-long
            image = fits.getdata(img_fits_filename, 0)  # pylint: disable=no-member

            # TODO: don't use image data to compute sig1 - so DECaLS gen won't load DES images
            diffs = image[:-5:10, :-5:10] - image[5::10, 5::10]
            mad = np.median(np.abs(diffs).ravel())
            zpscale = DES.zpt_to_scale(hr["MAGZPT"])
            sig1 = (1.4826 * mad / np.sqrt(2.0)) / zpscale

            image_nelec = image.astype(np.float32) * flux_calibration
            d.update({"image": image_nelec, "sig1": sig1})
        return d

    def splinesky_level_for_band(self, brickname, ccdname, image_filename, image_shape):
        save_filename = DESDownloader.save_filename_from_image_filename(image_filename)
        background_fits_filename = (
            self.des_path / brickname[:3] / brickname / f"{save_filename}-splinesky.fits"
        )
        background_fits = fits.open(background_fits_filename)

        background_table_hdu = background_fits[1]
        background_table = Table.read(background_table_hdu)

        # Get `row` corresponding to DECam image (i.e., CCD)
        rows = np.where(background_table["ccdname"] == ccdname)[0]
        assert len(rows) == 1
        row = rows[0]
        splinesky_params = background_table[row]

        gridw = splinesky_params["gridw"]
        gridh = splinesky_params["gridh"]
        gridvals = splinesky_params["gridvals"]
        xgrid = splinesky_params["xgrid"]
        ygrid = splinesky_params["ygrid"]
        order = splinesky_params["order"]

        # Meshgrid for pixel coordinates on smaller grid
        x, y = np.meshgrid(np.arange(gridw), np.arange(gridh))
        # Initialize the B-spline sky model with the extracted parameters
        splinesky_x = RectBivariateSpline(ygrid, xgrid, gridvals, kx=order, ky=order)
        splinesky_y = RectBivariateSpline(ygrid, xgrid, gridvals, kx=order, ky=order)

        # Evaluate the sky model at the given pixel coordinates
        background_values_grid_x = splinesky_x(y.flatten(), x.flatten(), grid=False).reshape(
            gridh, gridw
        )
        background_values_grid_y = splinesky_y(y.flatten(), x.flatten(), grid=False).reshape(
            gridh, gridw
        )

        # Upscale the background values from the smaller grid to the original image size using
        # bi-`order` interpolation
        background_values_x = zoom(
            background_values_grid_x,
            zoom=(image_shape[0] / gridh, image_shape[1] / gridw),
            order=order,
            mode="nearest",
        ).astype(np.float32)
        background_values_y = zoom(
            background_values_grid_y,
            zoom=(image_shape[0] / gridh, image_shape[1] / gridw),
            order=order,
            mode="nearest",
        ).astype(np.float32)

        # Take the mean of the x and y components
        return (background_values_x + background_values_y) / 2

    def to_dictconfig(self, image_id):
        # convert sky_coord["ra"], sky_coord["dec"] to np.float32
        image_id["sky_coord"] = {
            "ra": float(image_id["sky_coord"]["ra"]),
            "dec": float(image_id["sky_coord"]["dec"]),
        }
        return DictConfig(image_id)

    def process_image_ids(self, image_ids) -> List[DictConfig]:
        im_ids = list(image_ids)
        for im_id in im_ids:
            for b in self.BANDS:
                im_id[b] = im_id.get(b, "")
        # convert to hashable DictConfig
        return [self.to_dictconfig(im_id) for im_id in im_ids]


DES = DarkEnergySurvey


class DESDownloader(SurveyDownloader):
    """Class for downloading DECaLS data."""

    URLBASE = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9"
    DEF_ACCESS_URL = "https://datalab.noirlab.edu/sia/calibrated_all"
    DECaLS_URLBASE = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9"

    @staticmethod
    def image_basename_from_filename(image_filename, bl):
        return image_filename.split("/")[-1].split(f"_{bl}")[0] + f"_{bl}"

    @staticmethod
    def save_filename_from_image_filename(image_filename):
        return image_filename.split("/")[-1]

    @staticmethod
    def download_catalog_from_filename(tractor_filename: str):
        """Download tractor catalog given tractor-<brick_name>.fits filename."""
        basename = Path(tractor_filename).name
        brickname = basename.split("-")[1].split(".")[0]
        download_file_to_dst(
            f"{DESDownloader.DECaLS_URLBASE}/south/tractor/{brickname[:3]}/{basename}",
            tractor_filename,
        )

    def __init__(self, image_ids: List[DESImageID], download_dir):
        self.band_image_filenames = image_ids
        self.bricknames = [image_id["decals_brickname"] for image_id in image_ids]
        self.download_dir = download_dir

        self.svc = sia.SIAService(self.DEF_ACCESS_URL)

    def download_images(self):
        """Download images for all bands, for all image_ids."""
        for image_id in self.band_image_filenames:
            brickname = image_id["decals_brickname"]
            for bl in DES.BANDS:
                if image_id[bl]:
                    image_basename = DESDownloader.image_basename_from_filename(image_id[bl], bl)
                    self.download_image(brickname, image_id["sky_coord"], image_basename)

    def download_image(self, brickname, sky_coord, image_basename):
        """Download image for specified band, for this brick/ccd."""
        image_table = self.svc.search((sky_coord["ra"], sky_coord["dec"])).to_table()
        sel = defchararray.find(image_table["access_url"].astype(str), image_basename) != -1
        image_dst_filename = (
            self.download_dir / brickname[:3] / brickname / f"{image_basename}.fits"
        )
        try:
            download_file_to_dst(image_table[sel][0]["access_url"], image_dst_filename)
        except IndexError as e:
            warnings.warn(
                f"Desired image with basename {image_basename} not found in SIA database."
            )
            raise e
        except HTTPError as e:
            warnings.warn(
                f"No {image_basename} image for brick {brickname} at sky position "
                f"({sky_coord['ra']}, {sky_coord['dec']}). Check cfg.datasets.des.image_ids.",
                stacklevel=2,
            )
            raise e

    def download_psfexs(self):
        """Download psfex files for all image_ids."""
        for image_id in self.band_image_filenames:
            brickname = image_id["decals_brickname"]
            for bl in DES.BANDS:
                if image_id[bl]:
                    self.download_psfex(brickname, image_id[bl])

    def download_psfex(self, brickname, image_filename_no_ext):
        """Download psfex file for specified band, for this brick/ccd."""
        save_filename = self.save_filename_from_image_filename(image_filename_no_ext)
        psfex_filename = (
            self.download_dir / brickname[:3] / brickname / f"{save_filename}-psfex.fits"
        )
        try:
            download_file_to_dst(
                f"{DESDownloader.URLBASE}/calib/psfex/{image_filename_no_ext}-psfex.fits",
                psfex_filename,
            )
        except HTTPError as e:
            warnings.warn(
                f"No {psfex_filename} image for brick {brickname}. Check "
                "cfg.datasets.des.image_ids.",
                stacklevel=2,
            )
            raise e
        return str(psfex_filename)

    def download_backgrounds(self):
        """Download sky params for all image_ids."""
        for image_id in self.band_image_filenames:
            brickname = image_id["decals_brickname"]
            for bl in DES.BANDS:
                if image_id[bl]:
                    self.download_background(brickname, image_id[bl])

    def download_background(self, brickname, image_filename_no_ext):
        """Download sky params for specified band, for this brick/ccd."""
        save_filename = self.save_filename_from_image_filename(image_filename_no_ext)
        background_filename = (
            self.download_dir / brickname[:3] / brickname / f"{save_filename}-splinesky.fits"
        )
        try:
            download_file_to_dst(
                f"{DESDownloader.URLBASE}/calib/sky/{image_filename_no_ext}-splinesky.fits",
                background_filename,
            )
        except HTTPError as e:
            warnings.warn(
                f"No {background_filename} image for brick {brickname}. Check "
                "cfg.datasets.des.image_ids.",
                stacklevel=2,
            )
            raise e
        return str(background_filename)

    def download_catalog(self, des_image_id) -> str:
        """Download tractor catalog given tractor-<brick_name>.fits filename."""
        brickname = des_image_id["decals_brickname"]
        tractor_filename = str(
            self.download_dir / brickname[:3] / brickname / f"tractor-{brickname}.fits"
        )
        basename = Path(tractor_filename).name
        download_file_to_dst(
            f"{DESDownloader.DECaLS_URLBASE}/south/tractor/{brickname[:3]}/{basename}",
            tractor_filename,
        )
        return tractor_filename


# NOTE: No DES catalog; re-use DecalsFullCatalog


class DES_PSF(ImagePSF):  # noqa: N801
    # PSF parameters for encoder to learn
    PARAM_NAMES = [
        "chi2",
        "fit_original",
        "moffat_alpha",
        "moffat_beta",
        "polscal1",
        "polscal2",
        "polzero1",
        "polzero2",
        "psf_fwhm",
        "sum_diff",
    ]

    def __init__(self, survey_data_dir, image_ids, bands, psf_config: PSFConfig):
        super().__init__(bands, **psf_config)

        self.survey_data_dir = survey_data_dir

        # NOTE: pass `method="no_pixel"` to galsim.drawImage to avoid double-convolution
        # see https://galsim-developers.github.io/GalSim/_build/html/des.html#des-psf-models
        self.psf_draw_method = "no_pixel"
        self.psf_galsim = {}
        self.psf_params = {}

        for image_id in image_ids:
            psf_params = torch.zeros(len(DES.BANDS), len(DES_PSF.PARAM_NAMES))
            for b, bl in enumerate(DES.BANDS):
                if image_id[bl]:
                    psf_params[b] = self._psf_params_for_band(image_id, bl)

            self.psf_params[image_id] = psf_params
            self.psf_galsim[image_id] = self.get_psf_via_despsfex(image_id)

    def _psfex_hdu_for_band_image(self, des_image_id, bl):
        brickname = des_image_id["decals_brickname"]
        ccdname = des_image_id["ccdname"]

        save_filename = DESDownloader.save_filename_from_image_filename(des_image_id[bl])
        psfex_fits_filename = (
            Path(self.survey_data_dir) / brickname[:3] / brickname / f"{save_filename}-psfex.fits"
        )
        psfex_fits = fits.open(psfex_fits_filename)
        psfex_table_hdu = psfex_fits[1]

        # Get `row` corresponding to DECam image (i.e., CCD)
        rows = np.where(psfex_table_hdu.data["ccdname"] == ccdname)[0]  # pylint: disable=no-member
        assert len(rows) == 1, f"Found {len(rows)} rows for ccdname {ccdname}; expected 1."
        psfex_table_hdu.data = psfex_table_hdu.data[  # pylint: disable=no-member
            rows[0] : rows[0] + 1
        ]
        return psfex_table_hdu

    def _psf_params_for_band(self, des_image_id, bl):
        band_psfex_table_hdu = self._psfex_hdu_for_band_image(des_image_id, bl)

        psf_params = np.zeros(len(DES_PSF.PARAM_NAMES))
        for i, param in enumerate(DES_PSF.PARAM_NAMES):
            psf_params[i] = band_psfex_table_hdu.data[param]  # pylint: disable=no-member

        return torch.tensor(psf_params, dtype=torch.float32)

    def get_psf_via_despsfex(self, des_image_id, px=0.0, py=0.0):  # noqa: W0237
        """Construct PSF image from PSFEx FITS files.

        Args:
            des_image_id (DESImageID): image_id for this image
            px (float): x image pixel coordinate for PSF center
            py (float): y image pixel coordinate for PSF center

        Returns:
            images (List[InterpolatedImage]): list of psf transformations for each band
        """

        brickname = des_image_id["decals_brickname"]

        # Filler PSF for bands not in `bands`
        fake_psf = galsim.InterpolatedImage(
            galsim.Image(np.random.rand(self.psf_slen, self.psf_slen), scale=1)
        ).withFlux(1)
        images = [fake_psf for _ in range(len(DES.BANDS))]
        for b, bl in enumerate(DES.BANDS):
            if des_image_id[bl]:
                psfex_table_hdu = self._psfex_hdu_for_band_image(des_image_id, bl)
                fmt_psfex_table_hdu = self._format_psfex_table_hdu_for_galsim(psfex_table_hdu)

                image_basename = DESDownloader.image_basename_from_filename(des_image_id[bl], bl)
                image_filename = (
                    Path(self.survey_data_dir)
                    / brickname[:3]
                    / brickname
                    / f"{image_basename}.fits"
                )
                des_psfex_band = galsim.des.DES_PSFEx(
                    fmt_psfex_table_hdu,
                    str(image_filename),
                )
                # TODO: use an appropriate image position for the PSF
                psf_image = des_psfex_band.getPSF(galsim.PositionD(px, py))
                images[b] = psf_image

        return images

    def _format_psfex_table_hdu_for_galsim(self, psfex_table_hdu):
        """Format PSFEx table HDU for use with `galsim.des`."""
        # Get single values for the following parameters
        param_names = [
            "polnaxis",
            "polzero1",
            "polzero2",
            "polscal1",
            "polscal2",
            "polname1",
            "polname2",
            "polngrp",
            "polgrp1",
            "polgrp2",
            "poldeg1",
            "psfnaxis",
            "psfaxis1",
            "psfaxis2",
            "psfaxis3",
            "psf_samp",
        ]

        # Add to HDU header
        for param in param_names:
            psfex_table_hdu.header[param.upper()] = psfex_table_hdu.data[0][param]

        psfex_table_hdu.header["NAXIS2"] = 1
        psfex_table_hdu.header["NAXIS1"] = len(psfex_table_hdu.columns)

        return psfex_table_hdu


class TractorFullCatalog(FullCatalog):
    """Class for the DECaLS Tractor Catalog.

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
            A TractorFullCatalog containing data from the provided file.
        """
        catalog_path = Path(cat_path)
        if not catalog_path.exists():
            DESDownloader.download_catalog_from_filename(catalog_path.name)
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
