import bz2
import gzip
import warnings
from pathlib import Path
from typing import List, Tuple, TypedDict

import numpy as np
import torch
from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import download_file
from astropy.wcs import WCS, FITSFixedWarning
from einops import rearrange
from scipy.interpolate import RegularGridInterpolator

from bliss.align import align, crop_to_mult16
from bliss.catalog import FullCatalog, SourceType
from bliss.simulator.background import ImageBackground
from bliss.simulator.psf import ImagePSF, PSFConfig
from bliss.surveys.download_utils import download_file_to_dst
from bliss.surveys.survey import Survey

SDSSFields = List[TypedDict("SDSSField", {"run": int, "camcol": int, "fields": List[int]})]


class SloanDigitalSkySurvey(Survey):
    BANDS = ("u", "g", "r", "i", "z")

    @staticmethod
    def radec_for_rcf(run, camcol, field) -> Tuple[float, float]:
        """Get center (RA, DEC) for a given run, camcol, field."""
        extents = SDSSDownloader.field_extents()
        row = extents[
            (extents["run"] == run) & (extents["camcol"] == camcol) & (extents["field"] == field)
        ][0]
        ra_center = row["ramin"] + (row["ramax"] - row["ramin"]) / 2
        dec_center = row["decmin"] + (row["decmax"] - row["decmin"]) / 2
        return (ra_center, dec_center)

    @staticmethod
    def rcf_for_radec(ra, dec) -> Tuple[int, int, int]:
        """Get run, camcol, field for a given RA, DEC."""
        extents = SDSSDownloader.field_extents()
        row = extents[
            (extents["ramin"] <= ra)
            & (extents["ramax"] >= ra)
            & (extents["decmin"] <= dec)
            & (extents["decmax"] >= dec)
        ][0]
        return (row["run"], row["camcol"], row["field"])

    def __init__(
        self,
        psf_config: PSFConfig,
        fields,
        pixel_shift=0.0,
        dir_path="data/sdss",
        load_image_data: bool = False,
        background_offset=0.0,
        align_to_band=None,
        crop_config=None,
    ):
        super().__init__()

        self.sdss_path = Path(dir_path)
        self.sdss_fields = fields
        self.bands = tuple(range(len(self.BANDS)))
        self.n_bands = len(self.BANDS)
        self.pixel_shift = pixel_shift
        self.load_image_data = load_image_data
        self.background_offset = background_offset
        self.align_to_band = align_to_band
        self.crop_config = crop_config

        num_frames = sum(len(rcf_conf["fields"]) for rcf_conf in fields)
        self.items = [None for _ in range(num_frames)]
        self.rcfgcs = []

        self.downloader = SDSSDownloader(self.image_ids(), download_dir=str(self.sdss_path))

        self.psf = SDSS_PSF(dir_path, self.image_ids(), self.bands, psf_config)

        self.catalog_cls = PhotoFullCatalog

    def prepare_data(self):
        self.downloader.download_pfs()

        for rcf_conf in self.sdss_fields:
            run, camcol, fields = rcf_conf["run"], rcf_conf["camcol"], rcf_conf["fields"]

            pf_file = f"photoField-{run:06d}-{camcol:d}.fits"
            pf_path = self.sdss_path / str(run) / str(camcol) / pf_file
            msg = (
                f"{pf_path} does not exist. "
                + "Make sure data files are available for specified (run, camcol)."
            )
            assert Path(pf_path).exists(), msg
            pf_fits = fits.getdata(pf_path)
            assert pf_fits is not None, f"Could not load fits file {pf_path}."

            fieldnums = pf_fits["FIELD"]
            fieldgains = pf_fits["GAIN"]

            if fields:
                for field in fields:
                    gain = fieldgains[fieldnums == field][0]
                    self.rcfgcs.append((run, camcol, field, gain))
            else:
                for field, gain in zip(fieldnums, fieldgains):
                    self.rcfgcs.append((run, camcol, field, gain))

        self.downloader.download_images()
        for rcfgc in self.rcfgcs:
            run, camcol, field, _ = rcfgc
            field_path = self.sdss_path / f"{run}/{camcol}/{field}"
            for bl in SloanDigitalSkySurvey.BANDS:
                frame_name = f"frame-{bl}-{run:06d}-{camcol:d}-{field:04d}.fits"
                frame_path = field_path / frame_name
                assert Path(frame_path).exists(), f"{frame_path} does not exist."

        self.background = ImageBackground(self, bands=self.bands)
        self.flux_calibration_dict = self.get_flux_calibrations()

    def __len__(self):
        return len(self.rcfgcs)

    def __getitem__(self, idx):
        if not self.items[idx]:
            item = self.get_from_disk(idx)
            item["background"] += self.background_offset
            if not self.load_image_data:
                # we're just using the background/metadata, so no need to align or crop
                return item
            for k in ("image", "background"):
                if k not in item:
                    continue
                if self.align_to_band is not None:
                    item[k] = align(item[k], wcs_list=item["wcs"], ref_band=self.align_to_band)
                if self.crop_config:
                    r1, r2, c1, c2 = self.crop_config
                    item[k] = item[k][:, r1:r2, c1:c2]
                item[k] = self._crop_image(item[k])
            item["psf_params"] = self.psf.psf_params[self.image_id(idx)]
            self.items[idx] = item
        return self.items[idx]

    def _crop_image(self, x):
        return crop_to_mult16(x)

    def image_id(self, idx) -> Tuple[int, int, int]:
        """Return the image_id for the given index."""
        return self.rcfgcs[idx][:3]

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
        rcfs = []
        for rcf_conf in self.sdss_fields:
            run, camcol, fields = rcf_conf["run"], rcf_conf["camcol"], rcf_conf["fields"]
            for field in fields:
                rcfs.append((run, camcol, field))
        return rcfs

    def get_from_disk(self, idx):
        if self.rcfgcs[idx] is None:
            return None
        run, camcol, field, gain = self.rcfgcs[idx]

        camcol_dir = self.sdss_path.joinpath(str(run), str(camcol))
        field_dir = camcol_dir.joinpath(str(field))
        frame_list = []

        for b, bl in enumerate(self.BANDS):
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
        calibration = fits.getdata(frame_path, 1)
        nelec_per_nmgy = gain / calibration

        sky_data = fits.getdata(frame_path, 2)
        sky_small = sky_data["ALLSKY"][0]
        sky_x = sky_data["XINTERP"][0]
        sky_y = sky_data["YINTERP"][0]

        small_rows = np.mgrid[0 : sky_small.shape[0]]
        small_cols = np.mgrid[0 : sky_small.shape[1]]
        small_rcs = (small_rows, small_cols)
        sky_interp = RegularGridInterpolator(small_rcs, sky_small, method="nearest")

        sky_y = sky_y.clip(0, sky_small.shape[0] - 1)
        sky_x = sky_x.clip(0, sky_small.shape[1] - 1)
        large_points = rearrange(np.meshgrid(sky_y, sky_x), "n x y -> y x n")
        large_sky = sky_interp(large_points)
        large_sky_nelec = large_sky * gain

        if self.load_image_data:
            pixels_ss_nmgy = fits.getdata(frame_path, 0)
            pixels_ss_nelec = pixels_ss_nmgy * nelec_per_nmgy
            pixels_nelec = pixels_ss_nelec + large_sky_nelec

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FITSFixedWarning)
            wcs = WCS(fits.getheader(frame_path, 0))

        d = {
            "background": large_sky_nelec,
            "gain": np.array(gain),
            "flux_calibration_list": nelec_per_nmgy,
            "calibration": calibration,
            "wcs": wcs,
        }
        if self.load_image_data:
            d.update({"image": pixels_nelec})
        return d


class SDSSDownloader:
    """Class for downloading SDSS data."""

    URLBASE = "https://data.sdss.org/sas/dr12/boss"

    @staticmethod
    def stripped(val):
        return str(val).lstrip("0")

    @staticmethod
    def run6(run) -> str:
        return f"{int(SDSSDownloader.stripped(run)):06d}"

    @staticmethod
    def field4(field) -> str:
        return f"{int(SDSSDownloader.stripped(field)):04d}"

    @staticmethod
    def subdir2(run, camcol) -> str:
        return f"{SDSSDownloader.stripped(run)}/{camcol}"

    @staticmethod
    def subdir3(run, camcol, field) -> str:
        return f"{SDSSDownloader.subdir2(run, camcol)}/{SDSSDownloader.stripped(field)}"

    def __init__(self, image_ids, download_dir):
        self.image_ids = image_ids
        self.download_dir = download_dir

    @classmethod
    def download_field_extents(cls):
        # Download and use field-extents in memory
        field_extents_filename = download_file(
            "https://portal.nersc.gov/project/dasrepo/celeste/field_extents.fits",
            cache=True,
            timeout=10,
        )
        cls._field_extents = Table.read(field_extents_filename)

    @classmethod
    def field_extents(cls) -> Table:
        """Get field extents table."""
        if not getattr(cls, "_field_extents", None):
            cls.download_field_extents()
        return cls._field_extents

    def download_pfs(self):
        for image_id in self.image_ids:
            run, camcol, _ = image_id
            self.download_pf(run, camcol)

    def download_pf(self, run, camcol):
        download_file_to_dst(
            f"{SDSSDownloader.URLBASE}/photoObj/301/{self.stripped(run)}/"
            f"photoField-{self.run6(run)}-{camcol}.fits",
            f"{self.download_dir}/{self.subdir2(run, camcol)}/"
            f"photoField-{self.run6(run)}-{camcol}.fits",
        )

    def download_catalogs(self):
        for image_id in self.image_ids:
            run, camcol, field = image_id
            self.download_catalog((run, camcol, field))

    def download_catalog(self, rcf) -> str:
        run, camcol, field = rcf
        cat_path = (
            f"{self.download_dir}/{self.subdir3(run, camcol, field)}/"
            + f"photoObj-{self.run6(run)}-{camcol}-{self.field4(field)}.fits"
        )
        download_file_to_dst(
            f"{SDSSDownloader.URLBASE}/photoObj/301/{self.stripped(run)}/{camcol}/"
            f"photoObj-{self.run6(run)}-{camcol}-{self.field4(field)}.fits",
            cat_path,
        )
        return cat_path

    def download_images(self):
        for image_id in self.image_ids:
            run, camcol, field = image_id
            for bl in SloanDigitalSkySurvey.BANDS:
                self.download_image(run, camcol, field, bl)

    def download_image(self, run, camcol, field, band="r"):
        download_file_to_dst(
            f"{SDSSDownloader.URLBASE}/photo/redux/301/{self.stripped(run)}/objcs/{camcol}/"
            f"fpM-{self.run6(run)}-{band}{camcol}-{self.field4(field)}.fit.gz",
            f"{self.download_dir}/{self.subdir3(run, camcol, field)}/"
            f"fpM-{self.run6(run)}-{band}{camcol}-{self.field4(field)}.fits",
            gzip.decompress,
        )

        download_file_to_dst(
            f"{SDSSDownloader.URLBASE}/photoObj/frames/301/{self.stripped(run)}/{camcol}/"
            f"frame-{band}-{self.run6(run)}-{camcol}-{self.field4(field)}.fits.bz2",
            f"{self.download_dir}/{self.subdir3(run, camcol, field)}/"
            f"frame-{band}-{self.run6(run)}-{camcol}-{self.field4(field)}.fits",
            bz2.decompress,
        )

    def download_psfields(self):
        for image_id in self.image_ids:
            run, camcol, field = image_id
            self.download_psfield(run, camcol, field)

    def download_psfield(self, run, camcol, field):
        download_file_to_dst(
            f"{SDSSDownloader.URLBASE}/photo/redux/301/{self.stripped(run)}/objcs/{camcol}/"
            f"psField-{self.run6(run)}-{camcol}-{self.field4(field)}.fit",
            f"{self.download_dir}/{self.subdir3(run, camcol, field)}/"
            f"psField-{self.run6(run)}-{camcol}-{self.field4(field)}.fits",
        )

    def download_all(self):
        if not Path(self.download_dir).exists():
            # create download directory
            Path(self.download_dir).mkdir(parents=True, exist_ok=True)

        self.download_pfs()
        self.download_catalogs()
        self.download_images()
        self.download_psfields()


class PhotoFullCatalog(FullCatalog):
    """Class for the SDSS PHOTO Catalog.

    Some resources:
    - https://www.sdss.org/dr12/algorithms/classify/
    - https://www.sdss.org/dr12/algorithms/resolve/
    """

    @classmethod
    def from_file(cls, cat_path, wcs: WCS, height, width):
        """Instantiates PhotoFullCatalog with RCF and WCS information from disk."""
        assert Path(cat_path).exists(), f"File {cat_path} does not exist"

        table = fits.getdata(cat_path)

        # Convert table entries to tensors
        objc_type = column_to_tensor(table, "objc_type")
        thing_id = column_to_tensor(table, "thing_id")
        ras = column_to_tensor(table, "ra")
        decs = column_to_tensor(table, "dec")
        galaxy_bools = (objc_type == 3) & (thing_id != -1)
        star_bools = (objc_type == 6) & (thing_id != -1)
        star_fluxes = column_to_tensor(table, "psfflux") * star_bools.reshape(-1, 1)
        star_mags = column_to_tensor(table, "psfmag") * star_bools.reshape(-1, 1)
        galaxy_fluxes = column_to_tensor(table, "cmodelflux") * galaxy_bools.reshape(-1, 1)
        galaxy_mags = column_to_tensor(table, "cmodelmag") * galaxy_bools.reshape(-1, 1)

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
        nobj = ras.shape[0]

        # We require all 5 bands for computing loss on predictions.
        n_bands = len(SloanDigitalSkySurvey.BANDS)

        # get pixel coordinates
        plocs = cls.plocs_from_ra_dec(ras, decs, wcs)

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

        return cls(height, width, d)

    def restrict_by_ra_dec(self, ra_lim, dec_lim):
        """Helper function to restrict photo catalog to within RA and DEC limits."""
        ra = self["ra"].squeeze()
        dec = self["dec"].squeeze()

        keep = (ra > ra_lim[0]) & (ra < ra_lim[1]) & (dec >= dec_lim[0]) & (dec <= dec_lim[1])
        plocs = self["plocs"][:, keep]
        n_sources = torch.tensor([plocs.size()[1]])

        d = {"n_sources": n_sources}
        for key, val in self.items():
            if key != "n_sources":
                d[key] = val[:, keep]

        return PhotoFullCatalog(
            int(plocs[0, :, 0].max() - plocs[0, :, 0].min()),  # new height
            int(plocs[0, :, 1].max() - plocs[0, :, 1].min()),  # new width
            d,
        )


class SDSS_PSF(ImagePSF):  # noqa: N801
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

    def __init__(self, survey_data_dir, image_ids, bands, psf_config: PSFConfig):
        super().__init__(bands, **psf_config)

        self.psf_galsim = {}
        self.psf_params = {}
        SDSSDownloader(image_ids, download_dir=survey_data_dir).download_psfields()
        for run, camcol, field in image_ids:
            # load raw params from file
            field_dir = f"{survey_data_dir}/{run}/{camcol}/{field}"
            filename = f"{field_dir}/psField-{run:06}-{camcol}-{field:04}.fits"
            assert Path(filename).exists(), f"psField file {filename} not found"
            psf_params = self._get_fit_file_psf_params(filename, bands)

            # load psf image from params
            self.psf_params[(run, camcol, field)] = psf_params
            self.psf_galsim[(run, camcol, field)] = self._get_psf(psf_params)

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


def nelec_to_nmgy_for_catalog(est_cat, nelec_per_nmgy_per_band):
    fluxes_suffix = "_fluxes"
    # reshape nelec_per_nmgy_per_band to (1, 1, 1, 1, {n_bands}) to broadcast
    nelec_per_nmgy_per_band = torch.tensor(nelec_per_nmgy_per_band, device=est_cat.device)
    nelec_per_nmgy_per_band = nelec_per_nmgy_per_band.view(1, 1, 1, 1, -1)
    for key in est_cat.keys():
        if key.endswith(fluxes_suffix):
            est_cat[key] = est_cat[key] / nelec_per_nmgy_per_band
    return est_cat


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
