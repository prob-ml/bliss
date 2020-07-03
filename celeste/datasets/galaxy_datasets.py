import numpy as np
import random
import inspect
import sys

import descwl
import h5py
import galsim
from astropy.table import Column, Table
from torch.utils.data import Dataset


bands_dict = {
    1: ("i",),
    6: ("y", "z", "i", "r", "g", "u"),
}


def save_images(
    dataset, file_path, n_images=1,
):
    """Generate images from dataset cls and save num_images into h5py file.
    """

    with h5py.File(file_path, "w") as images_file:
        hds_shape = (n_images, dataset.n_bands, dataset.slen, dataset.slen)
        hds = images_file.create_dataset("images", hds_shape, dtype=dataset.dtype)
        for i in range(n_images):
            random_idx = random.randrange(len(dataset))
            image = dataset[random_idx]["image"]
            hds[i, :, :, :] = image
            hds.flush()
        hds.attrs["background"] = dataset.background
        hds.flush()


class H5Catalog(Dataset):
    def __init__(self, h5_file="images.hdf5", slen=51, n_bands=1):
        """ A dataset created from single galaxy images in a h5py file.
        """
        super().__init__()

        self.file = h5py.File(h5_file, "r")

        assert "images" in self.file, "The dataset is not in this file"

        self.dset = self.file["images"]
        self.n_bands = self.dset.shape[1]
        self.slen = self.dset.shape[2]
        assert (
            self.slen == slen == self.dset.shape[3]
        ), "slen does not match expected values."
        assert (
            self.n_bands == n_bands
        ), "Number of bands in training and in dataset do not match."

        assert "background" in self.dset.attrs, "Background is not in file"
        self.background = self.dset.attrs["background"]

    def __len__(self):
        """Number of images saved in the file.
        """
        return self.dset.shape[0]

    def __getitem__(self, idx):
        return {
            "image": self.dset[idx],  # shape = (n_bands, slen, slen)
            "background": self.background,
            "num_galaxies": 1,
        }

    def print_props(self, prop_file=sys.stdout):
        pass

    def __exit__(self):
        self.file.close()

    @staticmethod
    def add_args(parser):
        parser.add_argument("--h5-file", type=str, default=None, help="file path")

    @classmethod
    def from_args(cls, args):
        assert args.h5_file, "Specify h5_file if using this dataset."

        args_dict = vars(args)
        parameters = inspect.signature(cls).parameters
        args_dict = {param: args_dict[param] for param in parameters}
        return cls(**args_dict)


def get_pixel_scale(survey_name):
    return descwl.survey.Survey.get_defaults(survey_name, "*")["pixel_scale"]


def filter_bounds(cat, param, min_val=-np.inf, max_val=np.inf):
    return cat[(cat[param] >= min_val) & (cat[param] <= max_val)]


class SurveyObs(object):
    def __init__(self, renderer):
        """Returns a list of :class:`Survey` objects, each of them has an image attribute which is
        where images are written to by iso_render_engine.render_galaxy.
        """
        self.pixel_scale = renderer.pixel_scale
        self.bands = renderer.bands
        self.survey_name = renderer.survey_name
        self.image_size = renderer.image_size
        self.dtype = renderer.dtype

        obs = []
        for band in self.bands:
            # dictionary of default values.
            survey_dict = descwl.survey.Survey.get_defaults(
                survey_name=self.survey_name, filter_band=band
            )

            assert (
                self.pixel_scale == survey_dict["pixel_scale"]
            ), "Pixel scale does not match particular band?"
            survey_dict["image_width"] = self.image_size  # pixels
            survey_dict["image_height"] = self.image_size

            descwl_survey = descwl.survey.Survey(
                no_analysis=True,
                survey_name=self.survey_name,
                filter_band=band,
                **survey_dict,
            )
            obs.append(descwl_survey)

        self.obs = obs

    def __enter__(self):
        return self.obs

    def __exit__(self, exc_type, exc_val, exc_tb):
        for single_obs in self.obs:
            single_obs.image = galsim.Image(
                bounds=single_obs.image_bounds, scale=self.pixel_scale, dtype=self.dtype
            )


# ToDo: More flexibility than drawing randomly centered in central pixel.
class CatsimRenderer(object):
    def __init__(
        self,
        survey_name,
        bands,
        stamp_size,
        pixel_scale,
        snr=200,
        dtype=np.float32,
        min_snr=0.05,
        truncate_radius=30,
        add_noise=True,
        preserve_flux=False,
        verbose=False,
        deviate_center=False,
    ):
        """
        Can draw a single entry in CATSIM in the given bands.

        NOTE: Background is constant given the band, survey_name, image size, and default
        survey_dict, so it can be obtained in advance only once.
        """
        self.survey_name = survey_name
        self.bands = bands
        self.n_bands = len(self.bands)
        self.stamp_size = stamp_size  # arcsecs
        self.pixel_scale = pixel_scale
        self.image_size = int(self.stamp_size / self.pixel_scale)  # pixels.
        self.snr = snr
        self.min_snr = min_snr
        self.truncate_radius = truncate_radius
        self.add_noise = add_noise
        self.preserve_flux = preserve_flux  # when changing SNR.
        self.deviate_center = deviate_center
        self.verbose = verbose
        self.dtype = dtype
        self.background = self.get_background()

    def get_background(self):
        background = np.zeros(
            (self.n_bands, self.image_size, self.image_size), dtype=self.dtype
        )

        with SurveyObs(self) as obs:
            for i, single_obs in enumerate(obs):
                background[i, :, :] = single_obs.mean_sky_level
        return background

    # noinspection DuplicatedCode
    def get_size(self, cat):
        """
        Return a astropy.Column with the size of each entry of the catalog given the current
        rendering observing conditions / survey.
        """
        assert "i" in self.bands, "Requires the i band to be present."
        with SurveyObs(self) as obs:
            i_obs = obs[self.bands.index("i")]

            f = cat["fluxnorm_bulge"] / (cat["fluxnorm_disk"] + cat["fluxnorm_bulge"])
            hlr_d = np.sqrt(cat["a_d"] * cat["b_d"])
            hlr_b = np.sqrt(cat["a_b"] * cat["b_b"])
            r_sec = np.hypot(hlr_d * (1 - f) ** 0.5 * 4.66, hlr_b * f ** 0.5 * 1.46)
            psf = i_obs.psf_model
            psf_r_sec = psf.calculateMomentRadius()
            size = (
                np.sqrt(r_sec ** 2 + psf_r_sec ** 2) / self.pixel_scale
            )  # size is in pixels.
        return Column(size, name="size")

    def center_deviation(self, entry):
        # random deviation from exactly in center of center pixel, in arcsecs.
        deviation_ra = (np.random.rand() - 0.5) if self.deviate_center else 0.0
        deviation_dec = (np.random.rand() - 0.5) if self.deviate_center else 0.0
        entry["ra"] = deviation_ra * self.pixel_scale  # arcsecs
        entry["dec"] = deviation_dec * self.pixel_scale
        return entry

    def draw(self, entry):
        """
        Return a multi-band image corresponding to the entry from the catalog given.

        * The final image includes its background based on survey's sky level.
        * If deviate_center==True, then galaxy not aligned between bands.
        """
        image = np.zeros(
            (self.n_bands, self.image_size, self.image_size), dtype=self.dtype
        )

        with SurveyObs(self) as obs:
            for i, band in enumerate(self.bands):
                entry = self.center_deviation(entry)
                image_no_background = self.single_band(entry, obs[i], band)
                image[i, :, :] = image_no_background + self.background[i]

        return image, self.background

    def single_band(self, entry, single_obs, band):
        """Builds galaxy from a single entry in the catalog. With no background sky level added.
        """

        galaxy_builder = descwl.model.GalaxyBuilder(
            single_obs, no_disk=False, no_bulge=False, no_agn=False, verbose_model=False
        )

        galaxy = galaxy_builder.from_catalog(entry, entry["ra"], entry["dec"], band)

        iso_render_engine = descwl.render.Engine(
            survey=single_obs,
            min_snr=self.min_snr,
            truncate_radius=self.truncate_radius,
            no_margin=False,
            verbose_render=False,
        )

        # Up to this point, single_obs has not been changed by the previous 3 statements.

        try:
            # this line draws the given galaxy image onto single_obs.image,
            # this is the only change in single_obs.
            iso_render_engine.render_galaxy(
                galaxy,
                variations_x=None,
                variations_s=None,
                variations_g=None,
                no_fisher=True,
                calculate_bias=False,
                no_analysis=True,
            )  # saves image in single_obs

        except descwl.render.SourceNotVisible:
            if self.verbose:
                print("Source not visible")
            raise descwl.render.SourceNotVisible  # pass it on with a warning.

        image_temp = galsim.Image(self.image_size, self.image_size)
        image_temp += single_obs.image

        if self.add_noise:
            generator = galsim.random.BaseDeviate(seed=np.random.randint(99999999))
            noise = galsim.PoissonNoise(
                rng=generator, sky_level=single_obs.mean_sky_level
            )  # remember PoissonNoise assumes background already subtracted off.

            # Both of the adding noise methods add noise on the image consisting of the
            # (galaxy flux + background), but then remove the background at the end so we need
            # to add it later.
            if self.snr:
                image_temp.addNoiseSNR(
                    noise, snr=self.snr, preserve_flux=self.preserve_flux
                )
            else:
                image_temp.addNoise(noise)

        return image_temp.array


class CatsimGalaxies(Dataset):
    def __init__(
        self,
        catalog_file="OneDegSq.fits",
        survey_name="LSST",
        slen=51,
        snr=200,
        n_bands=6,
        dtype=np.float32,
        deviate_center=False,
        preserve_flux=False,
        add_noise=True,
    ):
        """
        This class reads a random entry from the OneDegSq.fits file (sample from the Catsim catalog)
         and returns a galaxy drawn from the catalog with realistic seeing conditions using
         functions from WeakLensingDeblending.

        For now, only one galaxy can be returned at once.

        :param snr: The SNR of the galaxy to draw, if None uses the actually seeing SNR from LSST
                    survey.
        """
        super().__init__()
        assert survey_name == "LSST", "Only using default survey name for now is LSST."
        assert n_bands in [1, 6], "Only 1 or 6 bands are supported."
        assert (
            slen >= 51
        ), "Does not seem to work well if the number of pixels is too low."
        assert slen % 2 == 1, "Odd number of pixels is preferred."
        assert dtype is np.float32, "Only float32 is supported for now."
        assert (
            preserve_flux is False
        ), "Otherwise variance of the noise will change which is not desirable."
        # ToDo: Create a test or assertion to check that mean == variance approx.

        self.survey_name = survey_name
        self.n_bands = n_bands
        self.bands = bands_dict[self.n_bands]

        self.slen = slen
        self.pixel_scale = get_pixel_scale(self.survey_name)
        self.stamp_size = self.pixel_scale * self.slen  # arcsecs.
        self.snr = snr
        self.dtype = dtype
        self.preserve_flux = preserve_flux
        self.add_noise = add_noise
        self.deviate_center = deviate_center

        self.renderer = CatsimRenderer(
            self.survey_name,
            self.bands,
            self.stamp_size,
            self.pixel_scale,
            snr=self.snr,
            dtype=self.dtype,
            preserve_flux=self.preserve_flux,
            add_noise=self.add_noise,
            deviate_center=deviate_center,
        )
        self.background = self.renderer.background

        # prepare catalog table.
        self.table = Table.read(catalog_file)  # full path
        self.table = self.table[
            np.random.permutation(len(self.table))
        ]  # shuffle in case that order matters.

        # TODO: Add support for other dicts.
        self.filter_dict = self.get_default_filters()
        self.cat = self.get_filtered_table()

    def __len__(self):
        return len(self.cat)

    # ToDo: Remove all non-visible sources from catalogue directly?
    def __getitem__(self, idx):

        while True:  # loop until visible galaxy is selected.
            try:
                entry = self.cat[idx]
                final, background = self.renderer.draw(entry)
                break

            except descwl.render.SourceNotVisible:
                idx = np.random.choice(
                    np.arange(len(self))
                )  # select some other random galaxy to return.

        return {"image": final, "background": background, "num_galaxies": 1}

    def print_props(self, output=sys.stdout):
        print(
            f"len(cat): {len(self.cat)} \n",
            f"slen: {self.slen} \n"
            f"snr: {self.snr} \n"
            f"survey name: {self.survey_name} \n"
            f"bands: {self.bands}\n"
            f"pixel scale: {self.pixel_scale}\n"
            f"filter_dict: Default\n"
            f"min_snr: {self.renderer.min_snr}\n"
            f"truncate_radius: {self.renderer.truncate_radius}\n"
            f"add_noise: {self.renderer.add_noise}\n"
            f"preserve_flux: {self.renderer.preserve_flux}\n"
            f"deviate center: {self.deviate_center}\n"
            f"dtype: {self.renderer.dtype}",
            file=output,
        )

    def get_filtered_table(self):
        cat = self.table.copy()
        for param, bounds in self.filter_dict.items():
            min_val, max_val = bounds
            cat = filter_bounds(cat, param, min_val, max_val)
        return cat

    @staticmethod
    def get_default_filters():
        # ToDo: Make a cut on the size? Something like:
        # sizes = self.renderer.get_size(self.cat); cat = cat[sizes < 30] (size in pixels)
        # cut on magnitude same as BTK does (gold sample)
        filters = dict(i_ab=(-np.inf, 25.3))
        return filters

    @staticmethod
    def add_args(parser):
        # add arguments to configure dataset directly from parser
        parser.add_argument(
            "--catalog-file",
            type=str,
            default=None,
            help="Catalog file to load entries from.",
        )
        parser.add_argument(
            "--survey-name", type=str, default="LSST", help="Survey to use."
        )
        parser.add_argument(
            "--deviate-center",
            action="store_true",
            help="Randomly deviate galaxies from center.",
        )

    @classmethod
    def from_args(cls, args):
        assert args.catalog_file, "Need to specify catalog file in catsim dataset."

        args_dict = vars(args)
        parameters = inspect.signature(cls).parameters
        args_dict = {param: args_dict[param] for param in parameters}
        return cls(**args_dict)
