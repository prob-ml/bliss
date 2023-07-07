import bz2
import gzip
import warnings
from pathlib import Path
from typing import List, Tuple, TypedDict

import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from einops import rearrange
from scipy.interpolate import RegularGridInterpolator
from torch import Tensor
from torch.utils.data import DataLoader

from bliss.catalog import FullCatalog, SourceType
from bliss.simulator.background import ImageBackground
from bliss.simulator.prior import ImagePrior, PriorConfig
from bliss.simulator.psf import ImagePSF, PSFConfig
from bliss.surveys.survey import Survey
from bliss.utils.download_utils import download_file_to_dst

SDSSFields = List[TypedDict("SDSSField", {"run": int, "camcol": int, "fields": List[int]})]


class SloanDigitalSkySurvey(Survey):
    BANDS = ("u", "g", "r", "i", "z")

    def __init__(
        self,
        psf_config,
        prior_config: PriorConfig,
        sdss_fields,
        prior_flux_ref_band: int = 2,  # r-band
        sdss_dir="data/sdss",
        predict_device=None,
        predict_crop=None,
    ):
        super().__init__(predict_device, predict_crop)

        self.sdss_path = Path(sdss_dir)
        self.rcfgcs = []

        self.sdss_fields = sdss_fields
        self.bands = tuple(range(len(self.BANDS)))

        self.prepare_data()

        self.prior = SDSSPrior(
            sdss_dir, self.image_ids(), self, self.bands, prior_flux_ref_band, prior_config
        )
        self.background = ImageBackground(items=self, bands=self.bands)
        self.psf = SDSSPSF(sdss_dir, self.image_ids(), self.bands, psf_config)

    def prepare_data(self):
        for rcf_conf in self.sdss_fields:
            run, camcol, fields = rcf_conf["run"], rcf_conf["camcol"], rcf_conf["fields"]
            downloader = SDSSDownloader(run, camcol, fields[0], download_dir=str(self.sdss_path))

            pf_file = f"photoField-{run:06d}-{camcol:d}.fits"
            camcol_path = self.sdss_path.joinpath(str(run), str(camcol))
            pf_path = camcol_path.joinpath(pf_file)
            if not Path(pf_path).exists():
                downloader.download_pf()
            msg = (
                f"{pf_path} does not exist. "
                + "Make sure data files are available for specified fields."
            )
            assert Path(pf_path).exists(), msg
            pf_fits = fits.getdata(pf_path)

            assert pf_fits is not None, f"Could not load fits file {pf_path}."
            fieldnums = pf_fits["FIELD"]
            fieldgains = pf_fits["GAIN"]

            # get desired field
            for i, field in enumerate(fieldnums):
                gain = fieldgains[i]
                if (not fields) or field in fields:
                    self.rcfgcs.append((run, camcol, field, gain))

        for rcfgc in self.rcfgcs:
            run, camcol, field, _ = rcfgc
            downloader = SDSSDownloader(run, camcol, field, download_dir=str(self.sdss_path))
            field_path = self.sdss_path.joinpath(str(run), str(camcol), str(field))
            for bl in SloanDigitalSkySurvey.BANDS:
                frame_name = f"frame-{bl}-{run:06d}-{camcol:d}-{field:04d}.fits"
                frame_path = str(field_path.joinpath(frame_name))
                if not Path(frame_path).exists():
                    downloader.download_frame(band=bl)
                assert Path(frame_path).exists(), f"{frame_path} does not exist."

        self.items = [None for _ in range(len(self.rcfgcs))]

    def __len__(self):
        return len(self.rcfgcs)

    def __getitem__(self, idx):
        if not self.items[idx]:
            self.items[idx] = self.get_from_disk(idx)
        return self.items[idx]

    def image_id(self, idx) -> Tuple[int, int, int]:
        """Return the image_id for the given index."""
        return self.rcfgcs[idx][0], self.rcfgcs[idx][1], self.rcfgcs[idx][2]

    def idx(self, image_id: Tuple[int, int, int]) -> int:
        """Return the index for the given image_id."""
        r, c, f = image_id
        # Return first index that matches r, c, f
        return next(
            i
            for i, (run, camcol, field, _) in enumerate(self.rcfgcs)
            if (run, camcol, field) == (r, c, f)
        )

    def image_ids(self) -> List[Tuple[int, int, int]]:
        """Return all image_ids.

        Note: Parallel to `rcfgcs`.

        Returns:
            List[Tuple[int, int, int]]: List of (run, camcol, field) image_ids.
        """
        return [self.image_id(i) for i in range(len(self))]

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
        download_file_to_dst(
            f"{SDSSDownloader.URLBASE}/photoObj/301/{self.run_stripped}/"
            f"photoField-{self.run6}-{self.camcol}.fits",
            f"{self.download_dir}/{self.subdir2}/" f"photoField-{self.run6}-{self.camcol}.fits",
        )

    def download_po(self):
        download_file_to_dst(
            f"{SDSSDownloader.URLBASE}/photoObj/301/{self.run_stripped}/{self.camcol}/"
            f"photoObj-{self.run6}-{self.camcol}-{self.field4}.fits",
            f"{self.download_dir}/{self.subdir3}/"
            f"photoObj-{self.run6}-{self.camcol}-{self.field4}.fits",
        )

    def download_frame(self, band="r"):
        download_file_to_dst(
            f"{SDSSDownloader.URLBASE}/photo/redux/301/{self.run_stripped}/objcs/{self.camcol}/"
            f"fpM-{self.run6}-{band}{self.camcol}-{self.field4}.fit.gz",
            f"{self.download_dir}/{self.subdir3}/"
            f"fpM-{self.run6}-{band}{self.camcol}-{self.field4}.fits",
            gzip.decompress,
        )

        download_file_to_dst(
            f"{SDSSDownloader.URLBASE}/photoObj/frames/301/{self.run_stripped}/{self.camcol}/"
            f"frame-{band}-{self.run6}-{self.camcol}-{self.field4}.fits.bz2",
            f"{self.download_dir}/{self.subdir3}/"
            f"frame-{band}-{self.run6}-{self.camcol}-{self.field4}.fits",
            bz2.decompress,
        )

    def download_psfield(self):
        download_file_to_dst(
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

        for band in SloanDigitalSkySurvey.BANDS:
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
        """Instantiates PhotoFullCatalog with RCF and WCS information from disk."""
        # Path to SDSS data directory
        sdss_path = Path(sdss_path)
        camcol_dir = sdss_path / str(run) / str(camcol) / str(field)
        # FITS file specific to RCF
        po_path = camcol_dir / f"photoObj-{run:06d}-{camcol:d}-{field:04d}.fits"
        if not Path(po_path).exists():
            SDSSDownloader(run, camcol, field, download_dir=str(sdss_path)).download_po()
        assert Path(po_path).exists(), f"File {po_path} does not exist, even after download."
        po_fits = fits.getdata(po_path)

        # Convert table entries to tensors
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

        # Combine light source parameters to one tensor
        fluxes = star_fluxes + galaxy_fluxes
        mags = star_mags + galaxy_mags

        # true light source mask
        keep = galaxy_bools | star_bools

        galaxy_bools = galaxy_bools[keep]
        star_bools = star_bools[keep]
        ras = ras[keep]
        decs = decs[keep]
        fluxes = fluxes[keep]
        mags = mags[keep]

        # We require all 5 bands for computing loss on predictions.
        n_bands = 5

        # Take WCS specific to r-band as standard.
        wcs: WCS = sdss_obj[0]["wcs"][2]

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

        # Verify each tile contains either a star or a galaxy
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

        # Collect required height, width by FullCatalog
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
            int(plocs[0, :, 0].max() - plocs[0, :, 0].min()),  # new height
            int(plocs[0, :, 1].max() - plocs[0, :, 1].min()),  # new width
            d,
        )


class SDSSPSF(ImagePSF):
    def __init__(self, survey_data_dir, image_ids, bands, psf_config: PSFConfig):
        super().__init__(bands, **psf_config)

        self.psf_galsim = {}
        self.psf_params = {}
        for run, camcol, field in image_ids:
            # load raw params from file
            field_dir = f"{survey_data_dir}/{run}/{camcol}/{field}"
            filename = f"{field_dir}/psField-{run:06}-{camcol}-{field:04}.fits"
            if not Path(filename).exists():
                SDSSDownloader(run, camcol, field, download_dir=survey_data_dir).download_psfield()
            psf_params = self._get_fit_file_psf_params(filename, bands)

            # load psf image from params
            self.psf_params[(run, camcol, field)] = psf_params
            self.psf_galsim[(run, camcol, field)] = self._get_psf(psf_params)

    @staticmethod
    def _get_fit_file_psf_params(psf_fit_file: str, bands: Tuple[int, ...]):
        """Load psf parameters from fits file.

        See https://data.sdss.org/datamodel/files/PHOTO_REDUX/RERUN/RUN/objcs/CAMCOL/psField.html
        for details on the parameters.

        Args:
            psf_fit_file (str): file to load from
            bands (Tuple[int, ...]): SDSS bands to load

        Returns:
            psf_params: tensor of parameters for each band
        """
        msg = (
            f"{psf_fit_file} does not exist. "
            + "Make sure data files are available for fields specified in config."
        )
        assert Path(psf_fit_file).exists(), msg
        # HDU 6 contains the PSF header (after primary and eigenimages)
        data = fits.open(psf_fit_file, ignore_missing_end=True).pop(6).data
        psf_params = torch.zeros(len(bands), 6)
        for i, band in enumerate(bands):
            sigma1 = data["psf_sigma1"][0][band] ** 2
            sigma2 = data["psf_sigma2"][0][band] ** 2
            sigmap = data["psf_sigmap"][0][band] ** 2
            beta = data["psf_beta"][0][band]
            b = data["psf_b"][0][band]
            p0 = data["psf_p0"][0][band]

            psf_params[i] = torch.tensor([sigma1, sigma2, sigmap, beta, b, p0])

        return psf_params

    def _psf_fun(self, r, sigma1, sigma2, sigmap, beta, b, p0):
        """Generate the PSF from the parameters using the power-law model.

        See https://data.sdss.org/datamodel/files/PHOTO_REDUX/RERUN/RUN/objcs/CAMCOL/psField.html
        for details on the parameters and the equation used.

        Args:
            r: radius
            sigma1: Inner gaussian sigma for the composite fit
            sigma2: Outer gaussian sigma for the composite fit
            sigmap: Width parameter for power law (pixels)
            beta: Slope of power law.
            b: Ratio of the outer PSF to the inner PSF at the origin
            p0: The value of the power law at the origin.

        Returns:
            The psf function evaluated at r.
        """

        term1 = torch.exp(-(r**2) / (2 * sigma1))
        term2 = b * torch.exp(-(r**2) / (2 * sigma2))
        term3 = p0 * (1 + r**2 / (beta * sigmap)) ** (-beta / 2)
        return (term1 + term2 + term3) / (1 + b + p0)


class SDSSPrior(ImagePrior):
    def __init__(
        self, survey_data_dir, image_ids, image_items, bands, ref_band, prior_config: PriorConfig
    ):
        super().__init__(bands, **prior_config)
        self.stars_fluxes, self.gals_fluxes = self._flux_ratios_against_b(
            survey_data_dir, image_ids, image_items, b=ref_band
        )

    def _flux_ratios_against_b(self, survey_data_dir, image_ids, items, b) -> Tuple[dict, dict]:
        stars_fluxes = {}
        gals_fluxes = {}

        for i, image_id in enumerate(image_ids):
            run, camcol, field = image_id
            image = items[i]

            # load project SDSS dir
            sdss_path = Path(survey_data_dir)

            # Set photoObj file path
            field_dir = sdss_path / str(run) / str(camcol) / str(field)
            po_path = field_dir / f"photoObj-{run:06d}-{camcol:d}-{field:04d}.fits"

            if not po_path.exists():
                SDSSDownloader(run, camcol, field, str(sdss_path)).download_po()
            msg = (
                f"{po_path} does not exist. "
                + "Make sure data files are available for fields specified in config."
            )
            assert Path(po_path).exists(), msg
            po_fits = fits.getdata(po_path)

            # retrieve object-specific information for ratio computing
            objc_type = column_to_tensor(po_fits, "objc_type").numpy()
            thing_id = column_to_tensor(po_fits, "thing_id").numpy()

            # mask fluxes based on object identity & validity
            galaxy_bools = (objc_type == 3) & (thing_id != -1)
            star_bools = (objc_type == 6) & (thing_id != -1)
            star_fluxes = column_to_tensor(po_fits, "psfflux") * star_bools.reshape(-1, 1)
            gal_fluxes = column_to_tensor(po_fits, "cmodelflux") * galaxy_bools.reshape(-1, 1)
            fluxes = star_fluxes + gal_fluxes

            # containers for light source ratios in current field
            star_fluxes_ratios = []
            gal_fluxes_ratios = []

            for obj, _ in enumerate(objc_type):
                if thing_id[obj] != -1 and torch.all(fluxes[obj] > 0):
                    ratios = self._flux_ratios_in_nelec(fluxes[obj], image, b)  # noqa: WPS220
                    if objc_type[obj] == 6:  # noqa: WPS220
                        star_fluxes_ratios.append(ratios)  # noqa: WPS220
                    elif objc_type[obj] == 3:  # noqa: WPS220
                        gal_fluxes_ratios.append(ratios)  # noqa: WPS220

            if star_fluxes_ratios:
                stars_fluxes[(run, camcol, field)] = star_fluxes_ratios
            if gal_fluxes_ratios:
                gals_fluxes[(run, camcol, field)] = gal_fluxes_ratios

        return stars_fluxes, gals_fluxes

    def _flux_ratios_in_nelec(self, obj_fluxes, image_item, b):
        """Query SDSS frames to get flux ratios in units of electron count.

        Args:
            obj_fluxes: tensor of electron counts for a particular SDSS object
            image_item: SDSS image item for a particular field
            b: index of reference band

        Returns:
            ratios (Tensor): Ratio of electron counts for each light source in field
            relative to `b`-band
        """
        ratios = torch.zeros(obj_fluxes.size())

        # distribution of nelec_per_nmgy very clustered around mean
        nelec_per_nmgys = image_item["nelec_per_nmgy_list"]
        b_rat = nelec_per_nmgys[b].mean()  # fix b-band (count:nmgy) ratio

        for i in range(self.max_bands):
            # distribution of nelec_per_nmgy very clustered around mean
            nelec_per_nmgy_rat = nelec_per_nmgys[i].mean() / b_rat

            # (nmgy ratio relative to r-band) * (nelec/nmgy ratio relative to r band)
            # result: ratio in units of electron counts
            ratios[i] = (obj_fluxes[i] / obj_fluxes[2]) * nelec_per_nmgy_rat

        return ratios


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
